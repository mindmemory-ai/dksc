"""
SpanKey 实验 Demo：key 注入 + ResNet-like + FashionMNIST 分类

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

warnings.filterwarnings("ignore", message=".*gemm_and_bias error.*CUBLAS.*", category=UserWarning)


def _tensor_stats(t: torch.Tensor):
    t = t.detach().float()
    return {"min": t.min().item(), "max": t.max().item(), "mean": t.mean().item(), "std": t.std().item()}


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- Basis / Key ----------

def make_basis(m: int, d: int, seed: int, orthonormalize: bool = True, scale_to_data: bool = True, device=None):
    """B: (m, d).
    scale_to_data=True: B in [-1/sqrt(m), 1/sqrt(m)] so key std ~ constant; no orthonormalize.
    """
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

def get_fashion_mnist_loaders(data_dir: str, batch_size: int, seed: int, train_augment: bool = True):
    set_seed(seed)
    norm = [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    if train_augment:
        train_trans = transforms.Compose([
            transforms.RandomAffine(degrees=8, translate=(0.06, 0.06)),
            *norm,
        ])
    else:
        train_trans = transforms.Compose(norm)
    test_trans = transforms.Compose(norm)
    train_ds = datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_trans)
    test_ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_trans)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ---------- Model (ResNet-like) ----------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.down is None else self.down(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class SmallResNet(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 28x28
        self.layer1 = nn.Sequential(ResidualBlock(32, 32, stride=1), ResidualBlock(32, 32, stride=1))
        # 14x14
        self.layer2 = nn.Sequential(ResidualBlock(32, 64, stride=2), ResidualBlock(64, 64, stride=1))
        # 7x7
        self.layer3 = nn.Sequential(ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128, stride=1))
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.mean(dim=(2, 3))  # GAP
        return self.head(x)


# Inject points:
# 0: input (1,28,28)
# 1: after stem (32,28,28)
# 2: after layer2 (64,14,14)
# 3: after layer3 (128,7,7)
LAYER_SHAPES = [(1, 28, 28), (32, 28, 28), (64, 14, 14), (128, 7, 7)]
LAYER_DIMS = [c * h * w for (c, h, w) in LAYER_SHAPES]


def _get_k_for_layer(k_input, layer_idx: int):
    if isinstance(k_input, (list, tuple)):
        return k_input[layer_idx]
    return k_input[..., : LAYER_DIMS[layer_idx]]


def forward_with_injection(model: SmallResNet, x: torch.Tensor, k_input,
                           gamma: float, inject_fn, inject_layers: list,
                           debug_list: list = None) -> torch.Tensor:
    h = x
    if 0 in inject_layers:
        k0 = _get_k_for_layer(k_input, 0).view(h.size(0), *LAYER_SHAPES[0])
        if debug_list is not None:
            h_after = inject_fn(h, k0, gamma)
            debug_list.append({"layer": 0, "shape": list(h.shape), "h_before": _tensor_stats(h),
                               "k": _tensor_stats(k0), "gamma_k": _tensor_stats(gamma * k0), "h_after": _tensor_stats(h_after)})
        h = inject_fn(h, k0, gamma)

    h = model.stem(h)
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

    h = h.mean(dim=(2, 3))
    return model.head(h)


# ---------- Train / Eval ----------

def train(model, basis, loader, inject_fn, gamma, inject_layers, per_layer_key,
          lr, epochs, device, debug_file=None, target_key_std: float = 0.5):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
                k_list = [None, None, None, None]
                if not per_layer_key:
                    alpha = torch.rand(n, B_list[inject_layers[0]].shape[0], device=device)
                for li in inject_layers:
                    B_i = B_list[li]
                    m_i = B_i.shape[0]
                    alpha_i = torch.rand(n, m_i, device=device) if per_layer_key else alpha
                    k_list[li] = key_in_span(B_i, alpha_i, device)
                k_input = k_list
            elif per_layer_key:
                k_list = [None, None, None, None]
                for li in inject_layers:
                    alpha_i = torch.rand(n, m, device=device)
                    k_list[li] = key_in_span(B, alpha_i, device)[..., : LAYER_DIMS[li]]
                k_input = k_list
            else:
                alpha = torch.rand(n, m, device=device)
                k_input = key_in_span(B, alpha, device)

            if target_key_std and target_key_std > 0:
                k_input = rescale_key_to_std(k_input, target_key_std)

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
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

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
            k_list = [None, None, None, None]
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
            kw_list = [None, None, None, None]
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
    p = argparse.ArgumentParser(description="SpanKey ResNet-like + FashionMNIST")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--inject", type=str, default="mul", choices=["add", "mul"])
    p.add_argument("--inject_layers", type=str, default="0", help="0..3 逗号分隔")
    p.add_argument("--per_layer_key", action="store_true")
    p.add_argument("--per_layer_basis", action="store_true")
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--target_key_std", type=float, default=1.0, help="0 表示不缩放")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no_scale_b_to_data", action="store_true")
    p.add_argument("--debug_file", type=str, default=None)
    p.add_argument("--no_train_aug", action="store_true", help="禁用训练集上的 RandomAffine 增强")
    args = p.parse_args()

    inject_layers = [int(s.strip()) for s in args.inject_layers.split(",") if s.strip() != ""]
    if not all(0 <= i <= 3 for i in inject_layers):
        raise ValueError("--inject_layers 每项应为 0..3")

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

    train_loader, test_loader = get_fashion_mnist_loaders(
        args.data_dir, args.batch_size, args.seed + 1, train_augment=not args.no_train_aug
    )
    scale_to_data = not args.no_scale_b_to_data

    if per_layer_basis:
        B_list = [None, None, None, None]
        for li in inject_layers:
            B_list[li] = make_basis(args.m, LAYER_DIMS[li], args.seed + 2 + li, scale_to_data=scale_to_data, device=device)
        basis = B_list
        print("Per-layer basis: " + ", ".join(f"B[{i}].shape={basis[i].shape}" for i in inject_layers) + (", scale_to_data=True" if scale_to_data else ""))
    else:
        d_max = max(LAYER_DIMS[i] for i in inject_layers)
        basis = make_basis(args.m, d_max, args.seed + 2, scale_to_data=scale_to_data, device=device)
        print(f"Basis B: shape {basis.shape} (d_max={d_max})" + (", scale_to_data=True" if scale_to_data else ""))

    model = SmallResNet().to(device)
    if device.type == "cuda":
        try:
            with torch.no_grad():
                _ = model(torch.randn(2, 1, 28, 28, device=device))
            torch.cuda.synchronize()
        except RuntimeError as e:
            if "CUBLAS" in str(e) or "cublas" in str(e).lower():
                print("Warning: CUDA 初始化失败，回退到 CPU")
                device = torch.device("cpu")
                model = model.cpu()
                basis = [b.cpu() if b is not None else None for b in basis] if per_layer_basis else basis.cpu()

    print("Training...")
    train(
        model, basis, train_loader, inject_fn, args.gamma, inject_layers, args.per_layer_key,
        args.lr, args.epochs, device, debug_file=args.debug_file, target_key_std=args.target_key_std,
    )

    def _print_res_metrics(metrics: dict):
        print(f"  No key:        acc = {metrics['acc_no_key']:.4f}")
        print(f"  Correct key:  acc = {metrics['acc_correct_key']:.4f}")
        print(f"  Wrong key:    acc = {metrics['acc_wrong_key']:.4f}")
        print(f"  Random:       1/10 = {metrics['random_baseline']:.4f}")

    print("\nEvaluation (train set):")
    metrics_tr = evaluate(
        model, basis, train_loader, inject_fn, args.gamma, inject_layers, args.per_layer_key,
        device, target_key_std=args.target_key_std,
    )
    _print_res_metrics(metrics_tr)

    print("\nEvaluation (test set):")
    metrics_te = evaluate(
        model, basis, test_loader, inject_fn, args.gamma, inject_layers, args.per_layer_key,
        device, target_key_std=args.target_key_std,
    )
    _print_res_metrics(metrics_te)


if __name__ == "__main__":
    main()

