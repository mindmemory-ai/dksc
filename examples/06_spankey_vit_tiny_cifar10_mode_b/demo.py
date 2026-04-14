"""
示例 06：CIFAR-10 + timm ViT-Tiny（patch 4 @32×32）+ 模式 B（显式拒绝类）。

用于在 Transformer 视觉骨干上验证 SpanKey / 拒损之可迁移性；算力需求低于 ResNet-18
全量训练，24GB 显存下以 batch 64 可舒适运行。

注入点：索引 0 = patch+pos 编码后之 token 序列；索引 1 = 前半段 Transformer block 之后（默认
depth//2 号 block 输出后）。两层 token 形状相同，默认 --per_layer_key 使两层使用独立 in-span 系数。
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

warnings.filterwarnings("ignore", message=".*gemm_and_bias error.*CUBLAS.*", category=UserWarning)

NUM_CLASSES = 10
REJECT_IDX = 10


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_basis(m: int, d: int, seed: int, orthonormalize: bool = True, scale_to_data: bool = True, device=None):
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


def sample_key_outside_span(_B: torch.Tensor, d: int, device=None) -> torch.Tensor:
    return torch.randn(d, device=device)


def rescale_key_to_std(k, target_std: float = 0.5, eps: float = 1e-8):
    if isinstance(k, (list, tuple)):
        return [None if t is None else rescale_key_to_std(t, target_std, eps) for t in k]
    s = k.std().item() + eps
    if s <= 0:
        return k
    return k * (target_std / s)


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


def build_vit_tiny_mode_b(
    img_size: int = 32,
    patch_size: int = 4,
    num_classes_semantic: int = 10,
):
    """ViT-Tiny on CIFAR resolution; num_classes = 11 (Mode B)."""
    m = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        num_classes=num_classes_semantic + 1,
    )
    return m


def _token_shape(model: nn.Module) -> tuple[int, int, int]:
    """Return (1, T, C) for token tensor after pos_embed — infer without full blocks."""
    device = next(model.parameters()).device
    isz = getattr(model.patch_embed, "img_size", 32)
    if isinstance(isz, int):
        h = w = isz
    else:
        h, w = int(isz[0]), int(isz[1])
    with torch.no_grad():
        x = torch.zeros(1, 3, h, w, device=device)
        t = model.patch_embed(x)
        if model.cls_token is not None:
            t = torch.cat((model.cls_token.expand(1, -1, -1), t), dim=1)
        t = t + model.pos_embed
        return 1, t.shape[1], t.shape[2]


def compute_layer_dims(model: nn.Module) -> list[int]:
    _, t, c = _token_shape(model)
    d_token = t * c
    return [d_token, d_token]


def forward_with_injection(
    model: nn.Module,
    x: torch.Tensor,
    k_input,
    gamma: float,
    inject_fn,
    inject_layers: list,
    inject_after_block_idx: int,
):
    x = model.patch_embed(x)
    if model.cls_token is not None:
        x = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = model.pos_drop(x + model.pos_embed)

    def _get_k(layer_idx: int):
        if isinstance(k_input, (list, tuple)):
            return k_input[layer_idx]
        d = x.shape[1] * x.shape[2]
        return k_input[..., :d]

    if 0 in inject_layers:
        k0 = _get_k(0).view(x.shape[0], x.shape[1], x.shape[2])
        x = inject_fn(x, k0, gamma)

    for i, blk in enumerate(model.blocks):
        x = blk(x)
        if 1 in inject_layers and i == inject_after_block_idx:
            k1 = _get_k(1).view(x.shape[0], x.shape[1], x.shape[2])
            x = inject_fn(x, k1, gamma)

    x = model.norm(x)
    return model.forward_head(x)


def get_cifar10_loaders(data_dir: str, batch_size: int, seed: int, train_augment: bool = True):
    set_seed(seed)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    if train_augment:
        train_trans = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_trans)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_trans)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_mode_b(
    model: nn.Module,
    basis,
    train_loader: DataLoader,
    inject_fn,
    gamma: float,
    inject_layers: list,
    per_layer_key: bool,
    lr: float,
    weight_decay: float,
    epochs: int,
    device,
    target_key_std: float,
    deny_weight: float,
    deny_on: str,
    layer_dims: list[int],
    inject_after_block_idx: int,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_per_layer_basis = isinstance(basis, (list, tuple))
    if use_per_layer_basis:
        B_list = basis
        m = B_list[inject_layers[0]].shape[0]
    else:
        B = basis
        m, d_max = B.shape

    n_slots = max(inject_layers) + 1
    deny_on_l = (deny_on or "wrong").lower().replace("+", ",")
    use_no = deny_on_l in ("wrong,no", "no,wrong", "both", "wrong+no", "no+wrong")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for _batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            if use_per_layer_basis:
                k_per_layer = [None] * n_slots
                if not per_layer_key:
                    alpha = torch.rand(batch_size, B_list[inject_layers[0]].shape[0], device=device)
                for layer_idx in inject_layers:
                    B_i = B_list[layer_idx]
                    m_i = B_i.shape[0]
                    alpha_i = torch.rand(batch_size, m_i, device=device) if per_layer_key else alpha
                    k_per_layer[layer_idx] = key_in_span(B_i, alpha_i, device)
                k_input = k_per_layer
            elif per_layer_key:
                k_per_layer = [None] * n_slots
                for layer_idx in inject_layers:
                    alpha_i = torch.rand(batch_size, m, device=device)
                    k_i = key_in_span(B, alpha_i, device)[..., : layer_dims[layer_idx]]
                    k_per_layer[layer_idx] = k_i
                k_input = k_per_layer
            else:
                alpha = torch.rand(batch_size, m, device=device)
                k_input = key_in_span(B, alpha, device)

            if target_key_std and target_key_std > 0:
                k_input = rescale_key_to_std(k_input, target_key_std)

            logits_ok = forward_with_injection(
                model, x, k_input, gamma, inject_fn, inject_layers, inject_after_block_idx
            )
            loss = F.cross_entropy(logits_ok, y)

            if deny_weight > 0:
                if use_per_layer_basis or per_layer_key:
                    k_wrong_list = [None] * n_slots
                    for layer_idx in inject_layers:
                        d_i = layer_dims[layer_idx]
                        k_wrong_list[layer_idx] = sample_key_outside_span(B, d_i, device).unsqueeze(0).expand(
                            batch_size, -1
                        )
                    k_wrong = k_wrong_list
                else:
                    k_wrong = sample_key_outside_span(B, d_max, device).unsqueeze(0).expand(batch_size, -1)
                if target_key_std and target_key_std > 0:
                    k_wrong = rescale_key_to_std(k_wrong, target_key_std)
                logits_wrong = forward_with_injection(
                    model, x, k_wrong, gamma, inject_fn, inject_layers, inject_after_block_idx
                )
                reject_y = torch.full((batch_size,), REJECT_IDX, device=device, dtype=y.dtype)
                deny_loss = F.cross_entropy(logits_wrong, reject_y)
                if use_no:
                    logits_no = model(x)
                    deny_loss = 0.5 * (deny_loss + F.cross_entropy(logits_no, reject_y))
                loss = loss + deny_weight * deny_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs} loss={total_loss/len(train_loader):.4f}")
    return model


@torch.no_grad()
def evaluate_mode_b(
    model: nn.Module,
    basis,
    data_loader: DataLoader,
    inject_fn,
    gamma: float,
    inject_layers: list,
    per_layer_key: bool,
    device,
    target_key_std: float,
    layer_dims: list[int],
    inject_after_block_idx: int,
):
    model.eval()
    use_per_layer_basis = isinstance(basis, (list, tuple))
    if use_per_layer_basis:
        B_list = basis
        m = B_list[inject_layers[0]].shape[0]
    else:
        B = basis
        m, d_max = B.shape

    n_slots = max(inject_layers) + 1
    all_y = []
    pred_no, pred_ok, pred_wrong = [], [], []

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        n = x.size(0)
        all_y.append(y)
        logits_no = model(x)
        pred_no.append(logits_no.argmax(dim=1))

        if use_per_layer_basis or per_layer_key:
            k_list = [None] * n_slots
            for layer_idx in inject_layers:
                B_i = B_list[layer_idx] if use_per_layer_basis else B
                m_i = B_i.shape[0]
                alpha_i = torch.rand(1, m_i, device=device)
                k_i = key_in_span(B_i, alpha_i, device).squeeze(0)[: layer_dims[layer_idx]]
                k_list[layer_idx] = k_i.unsqueeze(0).expand(n, -1)
            k_correct = k_list
        else:
            alpha = torch.rand(1, m, device=device)
            k_correct = key_in_span(B, alpha, device).expand(n, -1)
        if target_key_std and target_key_std > 0:
            k_correct = rescale_key_to_std(k_correct, target_key_std)

        logits_ok = forward_with_injection(
            model, x, k_correct, gamma, inject_fn, inject_layers, inject_after_block_idx
        )
        pred_ok.append(logits_ok.argmax(dim=1))

        if use_per_layer_basis or per_layer_key:
            k_wrong_list = [None] * n_slots
            for layer_idx in inject_layers:
                d_i = layer_dims[layer_idx]
                k_wrong_list[layer_idx] = sample_key_outside_span(B, d_i, device).unsqueeze(0).expand(n, -1)
            k_wrong = k_wrong_list
        else:
            k_wrong = sample_key_outside_span(B, d_max, device).unsqueeze(0).expand(n, -1)
        if target_key_std and target_key_std > 0:
            k_wrong = rescale_key_to_std(k_wrong, target_key_std)
        logits_wrong = forward_with_injection(
            model, x, k_wrong, gamma, inject_fn, inject_layers, inject_after_block_idx
        )
        pred_wrong.append(logits_wrong.argmax(dim=1))

    y_cat = torch.cat(all_y)
    pn = torch.cat(pred_no)
    pok = torch.cat(pred_ok)
    pw = torch.cat(pred_wrong)

    def semantic_acc(p: torch.Tensor, yv: torch.Tensor) -> float:
        return ((p < NUM_CLASSES) & (p == yv)).float().mean().item()

    def reject_frac(p: torch.Tensor) -> float:
        return (p == REJECT_IDX).float().mean().item()

    return {
        "acc_semantic_no_key": semantic_acc(pn, y_cat),
        "acc_semantic_correct_key": semantic_acc(pok, y_cat),
        "acc_semantic_wrong_key": semantic_acc(pw, y_cat),
        "reject_frac_no_key": reject_frac(pn),
        "reject_frac_correct_key": reject_frac(pok),
        "reject_frac_wrong_key": reject_frac(pw),
        "random_baseline": 1.0 / NUM_CLASSES,
    }


def main():
    p = argparse.ArgumentParser(description="SpanKey 示例06：CIFAR-10 ViT-Tiny + 模式 B")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m", type=int, default=16, help="子空间维数 m（每层同一 B 时共享）")
    p.add_argument("--inject", type=str, default="mul", choices=["add", "mul"])
    p.add_argument("--inject_layers", type=str, default="0,1", help="0=pos 后 token；1=前半段 block 后")
    p.add_argument("--no_per_layer_key", action="store_true", help="关闭 per_layer_key（默认两层独立 α）")
    p.add_argument("--per_layer_basis", action="store_true")
    p.add_argument("--gamma", type=float, default=0.35)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no_scale_b_to_data", action="store_true")
    p.add_argument("--target_key_std", type=float, default=0.45)
    p.add_argument("--deny_weight", type=float, default=0.1)
    p.add_argument("--deny_on", type=str, default="wrong", choices=["wrong", "wrong+no"])
    p.add_argument("--json_out", type=str, default=None)
    args = p.parse_args()

    inject_layers = sorted({int(s.strip()) for s in args.inject_layers.split(",") if s.strip()})
    if not inject_layers or not all(i in (0, 1) for i in inject_layers):
        raise ValueError("--inject_layers 应为 0、1 或 0,1")

    per_layer_key = not args.no_per_layer_key  # default True

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_vit_tiny_mode_b().to(device)
    LAYER_DIMS = compute_layer_dims(model)
    inject_after_block_idx = max(0, len(model.blocks) // 2 - 1)

    inject_fn = get_inject_fn(args.inject)
    scale_to_data = not args.no_scale_b_to_data

    train_loader, test_loader = get_cifar10_loaders(
        args.data_dir, args.batch_size, args.seed + 1, train_augment=True
    )

    if args.per_layer_basis:
        B_list = [None, None]
        for i in inject_layers:
            B_list[i] = make_basis(
                args.m, LAYER_DIMS[i], args.seed + 2 + i, orthonormalize=True, scale_to_data=scale_to_data, device=device
            )
        basis = B_list
    else:
        d_max = max(LAYER_DIMS[i] for i in inject_layers)
        basis = make_basis(args.m, d_max, args.seed + 2, orthonormalize=True, scale_to_data=scale_to_data, device=device)

    print(
        f"Example 06 ViT-Tiny Mode B | inject={args.inject} γ={args.gamma} m={args.m} "
        f"layers={inject_layers} per_layer_key={per_layer_key} deny_w={args.deny_weight} "
        f"token_dim={LAYER_DIMS[0]} inject_after_block={inject_after_block_idx} device={device}"
    )

    train_mode_b(
        model,
        basis,
        train_loader,
        inject_fn,
        args.gamma,
        inject_layers,
        per_layer_key,
        args.lr,
        args.weight_decay,
        args.epochs,
        device,
        args.target_key_std,
        args.deny_weight,
        args.deny_on,
        LAYER_DIMS,
        inject_after_block_idx,
    )

    metrics_tr = evaluate_mode_b(
        model,
        basis,
        train_loader,
        inject_fn,
        args.gamma,
        inject_layers,
        per_layer_key,
        device,
        args.target_key_std,
        LAYER_DIMS,
        inject_after_block_idx,
    )
    metrics_te = evaluate_mode_b(
        model,
        basis,
        test_loader,
        inject_fn,
        args.gamma,
        inject_layers,
        per_layer_key,
        device,
        args.target_key_std,
        LAYER_DIMS,
        inject_after_block_idx,
    )

    def _print(tag: str, met: dict):
        print(f"\n=== {tag} ===")
        print(
            f"  semantic acc  no / ok / wrong: {met['acc_semantic_no_key']:.4f} / "
            f"{met['acc_semantic_correct_key']:.4f} / {met['acc_semantic_wrong_key']:.4f}"
        )
        print(
            f"  reject frac     no / ok / wrong: {met['reject_frac_no_key']:.4f} / "
            f"{met['reject_frac_correct_key']:.4f} / {met['reject_frac_wrong_key']:.4f}"
        )

    _print("train", metrics_tr)
    _print("test", metrics_te)

    payload = {
        "model": "vit_tiny_patch16_224",
        "img_size": 32,
        "patch_size": 4,
        "config": {
            "m": args.m,
            "gamma": args.gamma,
            "inject_layers": inject_layers,
            "inject": args.inject,
            "per_layer_key": per_layer_key,
            "deny_weight": args.deny_weight,
            "deny_on": args.deny_on,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "inject_after_block_idx": inject_after_block_idx,
            "layer_dims": LAYER_DIMS,
        },
        "metrics_train": metrics_tr,
        "metrics_test": metrics_te,
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
