"""
SpanKey 实验 Demo：key 注入 + ResNet18 + CIFAR-10 分类

训练：仅用正确 key（span 内动态 key）训练。
评估：无 key / 正确 key / 错误 key（span 外）。
注入方式：add / mul；支持多层注入、per_layer_key、per_layer_basis。

默认使用 CUDA，可用 --cpu 强制 CPU。
"""

import os
import warnings
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

warnings.filterwarnings("ignore", message=".*gemm_and_bias error.*CUBLAS.*", category=UserWarning)


def _tensor_stats(t: torch.Tensor):
    t = t.detach().float()
    return {"min": t.min().item(), "max": t.max().item(), "mean": t.mean().item(), "std": t.std().item()}


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of softmax distribution (per sample)."""
    p = torch.softmax(logits, dim=-1)
    return -(p * (p.clamp_min(1e-12)).log()).sum(dim=-1)


# ---------- Basis / Key ----------

def make_basis(m: int, d: int, seed: int, orthonormalize: bool = True, scale_to_data: bool = True, device=None):
    """B: (m, d). scale_to_data=True => B in [-1/sqrt(m), 1/sqrt(m)]."""
    g = torch.Generator().manual_seed(seed)
    if scale_to_data:
        scale = (m ** -0.5) if m >= 1 else 1.0
        B = (2 * torch.rand(m, d, generator=g) - 1) * scale
    else:
        B = torch.randn(m, d, generator=g)
        if orthonormalize and m <= d:
            B, _ = torch.linalg.qr(B.T)
            B = B.T
    if device is not None:
        B = B.to(device)
    return B


def key_in_span(B: torch.Tensor, alpha: torch.Tensor, device=None) -> torch.Tensor:
    if device is not None and B.is_cuda:
        return (alpha.cpu() @ B.cpu()).squeeze(-1).to(device)
    return (alpha @ B).squeeze(-1)


def sample_key_outside_span(d: int, device=None) -> torch.Tensor:
    return torch.randn(d, device=device)


def rescale_key_to_std(k, target_std: float = 0.5, eps: float = 1e-8):
    if isinstance(k, (list, tuple)):
        out = []
        for t in k:
            out.append(None if t is None else rescale_key_to_std(t, target_std, eps))
        return out
    s = k.std().item() + eps
    if s <= 0:
        return k
    return k * (target_std / s)


# ---------- Injection ----------

def inject_add(x: torch.Tensor, k: torch.Tensor, gamma: float) -> torch.Tensor:
    return x + gamma * k


def inject_mul(x: torch.Tensor, k: torch.Tensor, gamma: float) -> torch.Tensor:
    scale = (1 + gamma * torch.tanh(k)).clamp(0.1, 2.0)
    return x * scale


def get_inject_fn(mode: str):
    fns = {"add": inject_add, "mul": inject_mul}
    if mode not in fns:
        raise ValueError(f"--inject 可选: {list(fns.keys())}")
    return fns[mode]


# ---------- Data ----------

def get_cifar10_loaders(data_dir: str, batch_size: int, seed: int, train_augment: bool = True):
    set_seed(seed)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    if train_augment:
        train_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_trans)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_trans)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ---------- Model ----------

def build_resnet18_cifar10():
    m = resnet18(num_classes=10)
    # CIFAR-10: 32x32，使用 3x3 conv1 + stride1 更合适
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def build_resnet18_cifar10_with_reject():
    """Same as build_resnet18_cifar10 but with an extra 'reject' class (index=10)."""
    m = resnet18(num_classes=11)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


# Inject points:
# 0: input (3,32,32)
# 1: after stem (conv1/bn/relu) => (64,32,32)
# 2: after layer2 => (128,16,16)
# 3: after layer3 => (256,8,8)
# 4: after layer4 => (512,4,4)
LAYER_SHAPES = [(3, 32, 32), (64, 32, 32), (128, 16, 16), (256, 8, 8), (512, 4, 4)]
LAYER_DIMS = [c * h * w for (c, h, w) in LAYER_SHAPES]


def _get_k_for_layer(k_input, layer_idx: int):
    if isinstance(k_input, (list, tuple)):
        return k_input[layer_idx]
    return k_input[..., : LAYER_DIMS[layer_idx]]


def forward_with_injection(model, x: torch.Tensor, k_input,
                           gamma: float, inject_fn, inject_layers: list,
                           debug_list: list = None):
    h = x
    if 0 in inject_layers:
        k0 = _get_k_for_layer(k_input, 0).view(h.size(0), *LAYER_SHAPES[0])
        if debug_list is not None:
            h_after = inject_fn(h, k0, gamma)
            debug_list.append({"layer": 0, "shape": list(h.shape), "h_before": _tensor_stats(h),
                               "k": _tensor_stats(k0), "gamma_k": _tensor_stats(gamma * k0), "h_after": _tensor_stats(h_after)})
        h = inject_fn(h, k0, gamma)

    # stem
    h = model.conv1(h)
    h = model.bn1(h)
    h = model.relu(h)
    if 1 in inject_layers:
        k1 = _get_k_for_layer(k_input, 1).view(h.size(0), *LAYER_SHAPES[1])
        if debug_list is not None:
            h_after = inject_fn(h, k1, gamma)
            debug_list.append({"layer": 1, "shape": list(h.shape), "h_before": _tensor_stats(h),
                               "k": _tensor_stats(k1), "gamma_k": _tensor_stats(gamma * k1), "h_after": _tensor_stats(h_after)})
        h = inject_fn(h, k1, gamma)

    h = model.layer1(h)
    h = model.layer2(h)
    if 2 in inject_layers:
        k2 = _get_k_for_layer(k_input, 2).view(h.size(0), *LAYER_SHAPES[2])
        if debug_list is not None:
            h_after = inject_fn(h, k2, gamma)
            debug_list.append({"layer": 2, "shape": list(h.shape), "h_before": _tensor_stats(h),
                               "k": _tensor_stats(k2), "gamma_k": _tensor_stats(gamma * k2), "h_after": _tensor_stats(h_after)})
        h = inject_fn(h, k2, gamma)

    h = model.layer3(h)
    if 3 in inject_layers:
        k3 = _get_k_for_layer(k_input, 3).view(h.size(0), *LAYER_SHAPES[3])
        if debug_list is not None:
            h_after = inject_fn(h, k3, gamma)
            debug_list.append({"layer": 3, "shape": list(h.shape), "h_before": _tensor_stats(h),
                               "k": _tensor_stats(k3), "gamma_k": _tensor_stats(gamma * k3), "h_after": _tensor_stats(h_after)})
        h = inject_fn(h, k3, gamma)

    h = model.layer4(h)
    if 4 in inject_layers:
        k4 = _get_k_for_layer(k_input, 4).view(h.size(0), *LAYER_SHAPES[4])
        if debug_list is not None:
            h_after = inject_fn(h, k4, gamma)
            debug_list.append({"layer": 4, "shape": list(h.shape), "h_before": _tensor_stats(h),
                               "k": _tensor_stats(k4), "gamma_k": _tensor_stats(gamma * k4), "h_after": _tensor_stats(h_after)})
        h = inject_fn(h, k4, gamma)

    h = model.avgpool(h)
    h = torch.flatten(h, 1)
    return model.fc(h)


# ---------- Train / Eval ----------

def train(model, basis, loader, inject_fn, gamma, inject_layers, per_layer_key,
          lr, epochs, device, debug_file=None, target_key_std: float = 0.5,
          optimizer_name: str = "adam", weight_decay: float = 0.0, milestones=None,
          deny_mode: str = "none", deny_weight: float = 0.0, deny_margin: float = 1.0,
          deny_on: str = "wrong"):
    milestones = milestones or []
    oname = (optimizer_name or "adam").lower()
    if oname == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
        )
        if milestones:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = None
    use_per_layer_basis = isinstance(basis, (list, tuple))
    if use_per_layer_basis:
        B_list = basis
        m = B_list[inject_layers[0]].shape[0]
    else:
        B = basis
        m, d_max = B.shape

    debug_collect = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            n = x.size(0)

            if use_per_layer_basis:
                k_list = [None] * len(LAYER_DIMS)
                if not per_layer_key:
                    alpha = torch.rand(n, B_list[inject_layers[0]].shape[0], device=device)
                for li in inject_layers:
                    B_i = B_list[li]
                    m_i = B_i.shape[0]
                    alpha_i = torch.rand(n, m_i, device=device) if per_layer_key else alpha
                    k_list[li] = key_in_span(B_i, alpha_i, device)
                k_input = k_list
            elif per_layer_key:
                k_list = [None] * len(LAYER_DIMS)
                for li in inject_layers:
                    alpha_i = torch.rand(n, m, device=device)
                    k_list[li] = key_in_span(B, alpha_i, device)[..., : LAYER_DIMS[li]]
                k_input = k_list
            else:
                alpha = torch.rand(n, m, device=device)
                k_input = key_in_span(B, alpha, device)

            if target_key_std and target_key_std > 0:
                k_input = rescale_key_to_std(k_input, target_key_std)

            # forward with correct key (primary objective)
            if debug_file and epoch == 0 and batch_idx == 0:
                inj_dbg = []
                logits = forward_with_injection(model, x, k_input, gamma, inject_fn, inject_layers, debug_list=inj_dbg)
                debug_collect = {
                    "config": {
                        "gamma": gamma,
                        "inject_layers": inject_layers,
                        "per_layer_key": per_layer_key,
                        "per_layer_basis": use_per_layer_basis,
                        "m": m,
                        "target_key_std": target_key_std,
                    },
                    "train_first_batch": {"x": _tensor_stats(x), "injections": inj_dbg},
                }
                if use_per_layer_basis:
                    debug_collect["config"]["B_stats"] = [_tensor_stats(basis[i]) for i in inject_layers]
                else:
                    debug_collect["config"]["B_stats"] = [_tensor_stats(basis)]
            else:
                logits = forward_with_injection(model, x, k_input, gamma, inject_fn, inject_layers)

            loss = F.cross_entropy(logits, y)

            # Optional deny loss: encourage wrong/no-key to be unhelpful.
            dmode = (deny_mode or "none").lower()
            if dmode != "none" and deny_weight and deny_weight > 0:
                # Build a wrong key for this batch (out-of-span).
                if use_per_layer_basis or per_layer_key:
                    kw_list = [None] * len(LAYER_DIMS)
                    for li in inject_layers:
                        kw_list[li] = sample_key_outside_span(LAYER_DIMS[li], device).unsqueeze(0).expand(n, -1)
                    k_wrong = kw_list
                else:
                    k_wrong = sample_key_outside_span(d_max, device).unsqueeze(0).expand(n, -1)
                if target_key_std and target_key_std > 0:
                    k_wrong = rescale_key_to_std(k_wrong, target_key_std)
                logits_wrong = forward_with_injection(model, x, k_wrong, gamma, inject_fn, inject_layers)

                # Optional no-key path
                logits_no = None
                if deny_on in ("wrong+no", "no+wrong", "both"):
                    logits_no = model(x)

                deny_loss = 0.0
                if dmode == "a":
                    # A: maximize entropy on invalid keys => minimize negative entropy
                    deny_loss = -_entropy_from_logits(logits_wrong).mean()
                    if logits_no is not None:
                        deny_loss = deny_loss - _entropy_from_logits(logits_no).mean()
                        deny_loss = deny_loss / 2.0
                elif dmode == "b":
                    # B: add an explicit reject class. Enforce invalid keys -> reject (class index 10).
                    reject_y = torch.full((n,), 10, device=device, dtype=y.dtype)
                    deny_loss = F.cross_entropy(logits_wrong, reject_y)
                    if logits_no is not None:
                        deny_loss = 0.5 * (deny_loss + F.cross_entropy(logits_no, reject_y))
                elif dmode == "c":
                    # C: margin ranking on true class logit: z_ok[y] should exceed z_wrong[y] by margin.
                    z_ok_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
                    z_w_y = logits_wrong.gather(1, y.view(-1, 1)).squeeze(1)
                    deny_loss = torch.relu(deny_margin - (z_ok_y - z_w_y)).mean()
                else:
                    deny_loss = 0.0

                loss = loss + deny_weight * deny_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"  epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}")

    if debug_file and debug_collect is not None:
        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump(debug_collect, f, indent=2, ensure_ascii=False)
        print(f"  Debug 数值已写入: {debug_file}")


@torch.no_grad()
def evaluate(model, basis, loader, inject_fn, gamma, inject_layers, per_layer_key,
             device, target_key_std: float = 0.5):
    model.eval()
    use_per_layer_basis = isinstance(basis, (list, tuple))
    if use_per_layer_basis:
        B_list = basis
        m = B_list[inject_layers[0]].shape[0]
    else:
        B = basis
        m, d_max = B.shape

    n_classes = 10
    all_y, all_no, all_ok, all_wrong = [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        n = x.size(0)
        all_y.append(y)
        all_no.append(model(x).argmax(1))

        # correct
        if use_per_layer_basis or per_layer_key:
            k_list = [None] * len(LAYER_DIMS)
            for li in inject_layers:
                B_i = B_list[li] if use_per_layer_basis else B
                alpha_i = torch.rand(1, B_i.shape[0], device=device)
                k_i = key_in_span(B_i, alpha_i, device).squeeze(0)[: LAYER_DIMS[li]]
                k_list[li] = k_i.unsqueeze(0).expand(n, -1)
            k_correct = k_list
        else:
            alpha = torch.rand(1, m, device=device)
            k_correct = key_in_span(B, alpha, device).expand(n, -1)
        if target_key_std and target_key_std > 0:
            k_correct = rescale_key_to_std(k_correct, target_key_std)
        logits_ok = forward_with_injection(model, x, k_correct, gamma, inject_fn, inject_layers)
        all_ok.append(logits_ok.argmax(1))

        # wrong
        if use_per_layer_basis or per_layer_key:
            kw_list = [None] * len(LAYER_DIMS)
            for li in inject_layers:
                kw_list[li] = sample_key_outside_span(LAYER_DIMS[li], device).unsqueeze(0).expand(n, -1)
            k_wrong = kw_list
        else:
            k_wrong = sample_key_outside_span(d_max, device).unsqueeze(0).expand(n, -1)
        if target_key_std and target_key_std > 0:
            k_wrong = rescale_key_to_std(k_wrong, target_key_std)
        logits_wrong = forward_with_injection(model, x, k_wrong, gamma, inject_fn, inject_layers)
        all_wrong.append(logits_wrong.argmax(1))

    y_all = torch.cat(all_y)
    acc_no = (torch.cat(all_no) == y_all).float().mean().item()
    acc_ok = (torch.cat(all_ok) == y_all).float().mean().item()
    acc_wrong = (torch.cat(all_wrong) == y_all).float().mean().item()
    return {"acc_no_key": acc_no, "acc_correct_key": acc_ok, "acc_wrong_key": acc_wrong, "random_baseline": 1.0 / n_classes}


def main():
    p = argparse.ArgumentParser(description="SpanKey ResNet18 + CIFAR-10")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--inject", type=str, default="mul", choices=["add", "mul"])
    p.add_argument("--inject_layers", type=str, default="0", help="0..4 逗号分隔")
    p.add_argument("--per_layer_key", action="store_true")
    p.add_argument("--per_layer_basis", action="store_true")
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--target_key_std", type=float, default=1.0, help="0 表示不缩放")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1, help="SGD 默认 0.1；若 --optimizer adam 请改用约 1e-3")
    p.add_argument("--weight_decay", type=float, default=5e-4, help="SGD 时常用 5e-4；Adam 可设 0")
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"],
                   help="CIFAR 强基线常用 SGD+momentum；沿用旧行为可用 adam")
    p.add_argument("--milestones", type=str, default="50,75",
                   help="SGD 时 MultiStepLR 的 epoch 列表（逗号分隔）；为空则对 SGD 使用 CosineAnnealing")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no_scale_b_to_data", action="store_true")
    p.add_argument("--no_train_aug", action="store_true", help="禁用 RandomCrop/RandomFlip 训练增强")
    p.add_argument("--debug_file", type=str, default=None)
    p.add_argument("--deny_mode", type=str, default="none", choices=["none", "A", "B", "C"],
                   help="拒绝策略：A=wrong/no key 高熵；B=增加 reject 类；C=correct-vs-wrong 真类 logit margin")
    p.add_argument("--deny_weight", type=float, default=0.0, help="拒绝损失权重（0 表示禁用）")
    p.add_argument("--deny_margin", type=float, default=1.0, help="C 策略的 margin")
    p.add_argument("--deny_on", type=str, default="wrong", choices=["wrong", "wrong+no"],
                   help="拒绝损失施加在哪些无效输入上：wrong 或 wrong+no")
    args = p.parse_args()

    inject_layers = [int(s.strip()) for s in args.inject_layers.split(",") if s.strip() != ""]
    if not all(0 <= i <= 4 for i in inject_layers):
        raise ValueError("--inject_layers 每项应为 0..4")

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        _ = torch.cuda.device_count()
        if hasattr(torch.backends.cuda, "preferred_blas_library"):
            torch.backends.cuda.preferred_blas_library("cublas")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    inject_fn = get_inject_fn(args.inject)
    per_layer_basis = args.per_layer_basis and len(inject_layers) > 1
    print(f"Inject: {args.inject}, inject_layers: {inject_layers}, per_layer_key: {args.per_layer_key}, per_layer_basis: {per_layer_basis}, device: {device}")

    train_loader, test_loader = get_cifar10_loaders(
        args.data_dir, args.batch_size, args.seed + 1, train_augment=not args.no_train_aug
    )
    ms = [int(x.strip()) for x in args.milestones.split(",") if x.strip() != ""]
    if args.optimizer == "sgd" and not ms:
        ms = None  # train() 内对 SGD 使用 Cosine
    scale_to_data = not args.no_scale_b_to_data

    if per_layer_basis:
        B_list = [None] * len(LAYER_DIMS)
        for li in inject_layers:
            B_list[li] = make_basis(args.m, LAYER_DIMS[li], args.seed + 2 + li, scale_to_data=scale_to_data, device=device)
        basis = B_list
        print("Per-layer basis: " + ", ".join(f"B[{i}].shape={basis[i].shape}" for i in inject_layers) + (", scale_to_data=True" if scale_to_data else ""))
    else:
        d_max = max(LAYER_DIMS[i] for i in inject_layers)
        basis = make_basis(args.m, d_max, args.seed + 2, scale_to_data=scale_to_data, device=device)
        print(f"Basis B: shape {basis.shape} (d_max={d_max})" + (", scale_to_data=True" if scale_to_data else ""))

    deny_mode = (args.deny_mode or "none").lower()
    if deny_mode == "b":
        model = build_resnet18_cifar10_with_reject().to(device)
    else:
        model = build_resnet18_cifar10().to(device)
    if device.type == "cuda":
        try:
            with torch.no_grad():
                _ = model(torch.randn(2, 3, 32, 32, device=device))
            torch.cuda.synchronize()
        except RuntimeError as e:
            if "CUBLAS" in str(e) or "cublas" in str(e).lower():
                print("Warning: CUDA 初始化失败，回退到 CPU")
                device = torch.device("cpu")
                model = model.cpu()
                basis = [b.cpu() if b is not None else None for b in basis] if per_layer_basis else basis.cpu()

    print(f"Training (optimizer={args.optimizer}, lr={args.lr}, milestones={ms}, deny={args.deny_mode}, w={args.deny_weight})...")
    train(
        model, basis, train_loader, inject_fn, args.gamma, inject_layers, args.per_layer_key,
        args.lr, args.epochs, device, debug_file=args.debug_file, target_key_std=args.target_key_std,
        optimizer_name=args.optimizer, weight_decay=args.weight_decay, milestones=ms,
        deny_mode=args.deny_mode, deny_weight=args.deny_weight, deny_margin=args.deny_margin,
        deny_on=args.deny_on,
    )

    def _print_spankey_metrics(metrics: dict):
        print(f"  No key:        acc = {metrics['acc_no_key']:.4f}")
        print(f"  Correct key:  acc = {metrics['acc_correct_key']:.4f}")
        print(f"  Wrong key:    acc = {metrics['acc_wrong_key']:.4f}")
        print(f"  Random:       1/10 = {metrics['random_baseline']:.4f}")

    print("\nEvaluation (train set):")
    metrics_train = evaluate(
        model, basis, train_loader, inject_fn, args.gamma, inject_layers, args.per_layer_key,
        device, target_key_std=args.target_key_std,
    )
    _print_spankey_metrics(metrics_train)

    print("\nEvaluation (test set):")
    metrics_test = evaluate(
        model, basis, test_loader, inject_fn, args.gamma, inject_layers, args.per_layer_key,
        device, target_key_std=args.target_key_std,
    )
    _print_spankey_metrics(metrics_test)


if __name__ == "__main__":
    main()

