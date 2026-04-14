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
import math
import time
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


def _k_input_stats(k_input, gamma: float) -> dict:
    """Scalar summaries of key tensor(s) for logging."""
    if isinstance(k_input, (list, tuple)):
        parts = [t for t in k_input if t is not None]
        if not parts:
            return {}
        ma = sum(p.detach().float().abs().mean().item() for p in parts) / len(parts)
        st = sum(p.detach().float().std().item() for p in parts) / len(parts)
    else:
        k = k_input.detach().float()
        ma = k.abs().mean().item()
        st = k.std().item()
    return {
        "k_mean_abs": ma,
        "k_std": st,
        "gamma_k_mean_abs": abs(gamma) * ma,
    }


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().data.norm(2).item() ** 2)
    return math.sqrt(total)


def _append_jsonl(path: str, row: dict):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of softmax distribution (per sample)."""
    p = torch.softmax(logits, dim=-1)
    return -(p * (p.clamp_min(1e-12)).log()).sum(dim=-1)


def _deny_a_soft(logits_inv: torch.Tensor, n_classes: int, gap: float) -> torch.Tensor:
    """Penalize only when entropy is below log(C) - gap (softer than Mode A)."""
    h = _entropy_from_logits(logits_inv)
    h_tgt = math.log(n_classes)
    shortfall = torch.relu(h_tgt - h - gap)
    return (shortfall ** 2).mean()


def _deny_c_margin(logits_ok: torch.Tensor, logits_wrong: torch.Tensor, y: torch.Tensor, m: float) -> torch.Tensor:
    z_ok_y = logits_ok.gather(1, y.view(-1, 1)).squeeze(1)
    z_w_y = logits_wrong.gather(1, y.view(-1, 1)).squeeze(1)
    return torch.relu(m - (z_ok_y - z_w_y)).mean()


def _deny_c_plus(logits_ok: torch.Tensor, logits_wrong: torch.Tensor, y: torch.Tensor, m1: float, m2: float) -> torch.Tensor:
    """Mode C plus: also push wrong-key max logit over classes c!=y above z_wrong[y]."""
    z_ok_y = logits_ok.gather(1, y.view(-1, 1)).squeeze(1)
    z_w_y = logits_wrong.gather(1, y.view(-1, 1)).squeeze(1)
    loss_c = torch.relu(m1 - (z_ok_y - z_w_y)).mean()
    n = logits_wrong.size(0)
    mask = torch.ones_like(logits_wrong, dtype=torch.bool)
    mask[torch.arange(n, device=y.device), y] = False
    neg_inf = torch.finfo(logits_wrong.dtype).min / 2
    max_other = logits_wrong.masked_fill(~mask, neg_inf).max(dim=1).values
    loss_top = torch.relu(m2 + z_w_y - max_other).mean()
    return loss_c + loss_top


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


def build_resnet18_cifar10_with_aux_reject():
    """C-way semantic head + separate scalar 'reject' logit on penultimate features (Mode B_aux)."""
    m = build_resnet18_cifar10()
    m.add_module("fc_aux", nn.Linear(512, 1))
    return m


def forward_logits_penultimate_noinject(model: nn.Module, x: torch.Tensor):
    """No-key forward: logits (n,C) and penultimate h (n,512)."""
    penult = []

    def _hook(_mod, _inp, out):
        penult.append(torch.flatten(out, 1))

    h = model.avgpool.register_forward_hook(_hook)
    try:
        logits = model(x)
    finally:
        h.remove()
    return logits, penult[0]


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
                           debug_list: list = None, return_penultimate: bool = False):
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
    logits = model.fc(h)
    if return_penultimate:
        return logits, h
    return logits


# ---------- Train / Eval ----------

def train(model, basis, loader, inject_fn, gamma, inject_layers, per_layer_key,
          lr, epochs, device, debug_file=None, target_key_std: float = 0.5,
          optimizer_name: str = "adam", weight_decay: float = 0.0, milestones=None,
          deny_mode: str = "none", deny_weight: float = 0.0, deny_margin: float = 1.0,
          deny_margin2: float = 0.5, a_soft_gap: float = 0.3, deny_warmup_epochs: int = 0,
          deny_on: str = "wrong",
          train_log: str = None, log_interval: int = 1, log_grad_norm: bool = True):
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
    global_step = 0
    t0 = time.perf_counter()
    if train_log:
        if os.path.isfile(train_log):
            os.remove(train_log)
        _append_jsonl(train_log, {
            "type": "run_start",
            "ts": time.time(),
            "epochs": epochs,
            "batches_per_epoch": len(loader),
            "gamma": gamma,
            "inject_layers": inject_layers,
            "per_layer_key": per_layer_key,
            "deny_mode": (deny_mode or "none").lower(),
            "deny_weight": deny_weight,
            "deny_warmup_epochs": deny_warmup_epochs,
            "deny_margin": deny_margin,
            "deny_margin2": deny_margin2,
            "a_soft_gap": a_soft_gap,
            "optimizer": oname,
            "lr": lr,
        })

    dmode_l = (deny_mode or "none").lower()
    use_aux_head = dmode_l == "b_aux"
    n_cls = 10

    for epoch in range(epochs):
        eff_deny_w = float(deny_weight)
        if deny_warmup_epochs and deny_warmup_epochs > 0:
            eff_deny_w = deny_weight * min(1.0, (epoch + 1) / float(deny_warmup_epochs))

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
                if use_aux_head:
                    logits, h_pre = forward_with_injection(
                        model, x, k_input, gamma, inject_fn, inject_layers,
                        debug_list=inj_dbg, return_penultimate=True,
                    )
                else:
                    logits = forward_with_injection(model, x, k_input, gamma, inject_fn, inject_layers, debug_list=inj_dbg)
                    h_pre = None
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
            elif use_aux_head:
                logits, h_pre = forward_with_injection(
                    model, x, k_input, gamma, inject_fn, inject_layers, return_penultimate=True
                )
            else:
                logits = forward_with_injection(model, x, k_input, gamma, inject_fn, inject_layers)
                h_pre = None

            ce_loss = F.cross_entropy(logits, y)
            loss = ce_loss
            deny_loss_val = None

            # Optional deny loss: encourage wrong/no-key to be unhelpful.
            if dmode_l != "none" and eff_deny_w > 0:
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

                logits_no = None
                h_no = None
                if deny_on in ("wrong+no", "no+wrong", "both"):
                    logits_no, h_no = forward_logits_penultimate_noinject(model, x)

                deny_loss = torch.tensor(0.0, device=device)
                if use_aux_head:
                    # B_aux: scalar reject logit; authorized path -> 0, invalid -> 1
                    aux_ok = model.fc_aux(h_pre).squeeze(-1)
                    l_auth_ok = F.binary_cross_entropy_with_logits(
                        aux_ok, torch.zeros(n, device=device, dtype=aux_ok.dtype)
                    )
                    logits_w, h_w = forward_with_injection(
                        model, x, k_wrong, gamma, inject_fn, inject_layers, return_penultimate=True
                    )
                    aux_w = model.fc_aux(h_w).squeeze(-1)
                    l_w = F.binary_cross_entropy_with_logits(
                        aux_w, torch.ones(n, device=device, dtype=aux_w.dtype)
                    )
                    parts = [l_w, l_auth_ok]
                    if logits_no is not None:
                        aux_n = model.fc_aux(h_no).squeeze(-1)
                        l_n = F.binary_cross_entropy_with_logits(
                            aux_n, torch.ones(n, device=device, dtype=aux_n.dtype)
                        )
                        parts.append(l_n)
                    deny_loss = sum(parts) / len(parts)
                else:
                    logits_wrong = forward_with_injection(model, x, k_wrong, gamma, inject_fn, inject_layers)

                    if dmode_l == "a":
                        deny_loss = -_entropy_from_logits(logits_wrong).mean()
                        if logits_no is not None:
                            deny_loss = (deny_loss - _entropy_from_logits(logits_no).mean()) / 2.0
                    elif dmode_l == "a_soft":
                        deny_loss = _deny_a_soft(logits_wrong, n_cls, a_soft_gap)
                        if logits_no is not None:
                            deny_loss = (deny_loss + _deny_a_soft(logits_no, n_cls, a_soft_gap)) / 2.0
                    elif dmode_l == "b":
                        reject_y = torch.full((n,), 10, device=device, dtype=y.dtype)
                        deny_loss = F.cross_entropy(logits_wrong, reject_y)
                        if logits_no is not None:
                            deny_loss = 0.5 * (deny_loss + F.cross_entropy(logits_no, reject_y))
                    elif dmode_l == "c":
                        deny_loss = _deny_c_margin(logits, logits_wrong, y, deny_margin)
                        if logits_no is not None:
                            deny_loss = (
                                deny_loss + _deny_c_margin(logits, logits_no, y, deny_margin)
                            ) / 2.0
                    elif dmode_l == "cplus":
                        deny_loss = _deny_c_plus(logits, logits_wrong, y, deny_margin, deny_margin2)
                        if logits_no is not None:
                            deny_loss = (
                                deny_loss + _deny_c_plus(logits, logits_no, y, deny_margin, deny_margin2)
                            ) / 2.0
                    elif dmode_l == "ac":
                        la = _deny_a_soft(logits_wrong, n_cls, a_soft_gap)
                        lc = _deny_c_margin(logits, logits_wrong, y, deny_margin)
                        deny_loss = 0.5 * (la + lc)
                        if logits_no is not None:
                            la2 = _deny_a_soft(logits_no, n_cls, a_soft_gap)
                            lc2 = _deny_c_margin(logits, logits_no, y, deny_margin)
                            deny_loss = (deny_loss + 0.5 * (la2 + lc2)) / 2.0
                    else:
                        deny_loss = torch.tensor(0.0, device=device)

                deny_loss_val = float(deny_loss.detach().item()) if isinstance(deny_loss, torch.Tensor) else float(deny_loss)
                loss = ce_loss + eff_deny_w * deny_loss

            opt.zero_grad()
            loss.backward()
            gn = _grad_norm(model) if log_grad_norm else None
            opt.step()

            with torch.no_grad():
                batch_acc = (logits.argmax(1) == y).float().mean().item()
            lr_now = opt.param_groups[0]["lr"]

            if train_log and log_interval > 0 and (global_step % log_interval == 0):
                row = {
                    "type": "train_step",
                    "ts": time.time(),
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "lr": lr_now,
                    "loss": float(loss.item()),
                    "ce_loss": float(ce_loss.item()),
                    "deny_loss": deny_loss_val,
                    "batch_acc": batch_acc,
                    "grad_norm": gn,
                    **_k_input_stats(k_input, gamma),
                }
                _append_jsonl(train_log, row)
            global_step += 1
            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        if train_log:
            _append_jsonl(train_log, {
                "type": "epoch_end",
                "epoch": epoch,
                "mean_loss": total_loss / len(loader),
                "elapsed_sec": time.perf_counter() - t0,
            })

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"  epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}")

    if train_log:
        _append_jsonl(train_log, {
            "type": "run_end",
            "ts": time.time(),
            "total_steps": global_step,
            "elapsed_sec": time.perf_counter() - t0,
        })

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
    aux_no, aux_ok, aux_wrong = [], [], []
    has_aux = getattr(model, "fc_aux", None) is not None

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        n = x.size(0)
        all_y.append(y)
        if has_aux:
            logits_no, h_no = forward_logits_penultimate_noinject(model, x)
            all_no.append(logits_no[:, :n_classes].argmax(1))
            aux_no.append(torch.sigmoid(model.fc_aux(h_no).squeeze(-1)))
        else:
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
        if has_aux:
            logits_ok, h_ok = forward_with_injection(
                model, x, k_correct, gamma, inject_fn, inject_layers, return_penultimate=True
            )
            all_ok.append(logits_ok[:, :n_classes].argmax(1))
            aux_ok.append(torch.sigmoid(model.fc_aux(h_ok).squeeze(-1)))
        else:
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
        if has_aux:
            logits_w, h_w = forward_with_injection(
                model, x, k_wrong, gamma, inject_fn, inject_layers, return_penultimate=True
            )
            all_wrong.append(logits_w[:, :n_classes].argmax(1))
            aux_wrong.append(torch.sigmoid(model.fc_aux(h_w).squeeze(-1)))
        else:
            logits_wrong = forward_with_injection(model, x, k_wrong, gamma, inject_fn, inject_layers)
            all_wrong.append(logits_wrong.argmax(1))

    y_all = torch.cat(all_y)
    acc_no = (torch.cat(all_no) == y_all).float().mean().item()
    acc_ok = (torch.cat(all_ok) == y_all).float().mean().item()
    acc_wrong = (torch.cat(all_wrong) == y_all).float().mean().item()
    out = {"acc_no_key": acc_no, "acc_correct_key": acc_ok, "acc_wrong_key": acc_wrong, "random_baseline": 1.0 / n_classes}
    if has_aux:
        out["aux_reject_prob_no_key"] = torch.cat(aux_no).mean().item()
        out["aux_reject_prob_correct_key"] = torch.cat(aux_ok).mean().item()
        out["aux_reject_prob_wrong_key"] = torch.cat(aux_wrong).mean().item()
    return out


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
    p.add_argument(
        "--train_log",
        type=str,
        default=None,
        help="训练过程 JSONL 日志路径：每步一行（loss、ce、deny、batch_acc、lr、grad_norm、密钥统计等）",
    )
    p.add_argument("--log_interval", type=int, default=1, help="每多少 global_step 写一条 train_step（1=每批）")
    p.add_argument("--no_log_grad_norm", action="store_true", help="不在日志中计算 grad_norm（略快）")
    p.add_argument(
        "--deny_mode",
        type=str,
        default="none",
        choices=["none", "A", "B", "C", "A_soft", "cplus", "AC", "B_aux"],
        help=(
            "拒绝策略：A=高熵；A_soft=仅当熵低于 log(C)-gap 时惩罚；"
            "C=真类 logit margin；cplus=C+错钥时非真类最大 logit 间隔；"
            "AC=A_soft 与 C 各 0.5 混合；B=(C+1) reject 类；"
            "B_aux=保留 C 类主头 + 独立标量拒绝 logit（不扩展 softmax 语义维）"
        ),
    )
    p.add_argument("--deny_weight", type=float, default=0.0, help="拒绝损失权重（0 表示禁用）")
    p.add_argument("--deny_margin", type=float, default=1.0, help="C / cplus / AC 中 C 项的 margin")
    p.add_argument("--deny_margin2", type=float, default=0.5, help="cplus：错钥时 max_{c!=y} z_c - z_y 的下界 margin")
    p.add_argument("--a_soft_gap", type=float, default=0.3, help="A_soft：目标熵为 log(C)-gap（nats）")
    p.add_argument("--deny_warmup_epochs", type=int, default=0, help="deny 权重从 0 线性升到 deny_weight 的 epoch 数；0 表示关闭")
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
    elif deny_mode == "b_aux":
        model = build_resnet18_cifar10_with_aux_reject().to(device)
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
        deny_margin2=args.deny_margin2, a_soft_gap=args.a_soft_gap, deny_warmup_epochs=args.deny_warmup_epochs,
        deny_on=args.deny_on,
        train_log=args.train_log,
        log_interval=max(1, args.log_interval),
        log_grad_norm=not args.no_log_grad_norm,
    )

    def _print_spankey_metrics(metrics: dict):
        print(f"  No key:        acc = {metrics['acc_no_key']:.4f}")
        print(f"  Correct key:  acc = {metrics['acc_correct_key']:.4f}")
        print(f"  Wrong key:    acc = {metrics['acc_wrong_key']:.4f}")
        print(f"  Random:       1/10 = {metrics['random_baseline']:.4f}")
        if "aux_reject_prob_no_key" in metrics:
            print(f"  B_aux σ(reject) no/wrong/correct: {metrics['aux_reject_prob_no_key']:.4f} / "
                  f"{metrics['aux_reject_prob_wrong_key']:.4f} / {metrics['aux_reject_prob_correct_key']:.4f}")

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

