"""
示例 05：MNIST + 与 02 同规模之 SmallCNN + 模式 B（显式拒绝类，logits 维数为 11）。

用于在轻量设定下快速验证拒绝损失 B 与消融：子空间维数 m、注入层、注入强度 γ。
仅用正确密钥路径训练主任务；非法路径（错钥，及可选无钥）监督至第 10 类（拒绝）。
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

warnings.filterwarnings("ignore", message=".*gemm_and_bias error.*CUBLAS.*", category=UserWarning)

NUM_CLASSES = 10
REJECT_IDX = 10  # 第 11 维，模式 B


def _tensor_stats(t: torch.Tensor):
    t = t.detach().float()
    return {"min": t.min().item(), "max": t.max().item(), "mean": t.mean().item(), "std": t.std().item()}


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


def rescale_key_to_std(k, target_std: float = 0.5, eps: float = 1e-8):
    if isinstance(k, (list, tuple)):
        return [None if t is None else rescale_key_to_std(t, target_std, eps) for t in k]
    s = k.std().item() + eps
    if s <= 0:
        return k
    return k * (target_std / s)


def sample_key_outside_span(_B: torch.Tensor, d: int, device=None) -> torch.Tensor:
    return torch.randn(d, device=device)


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


def get_mnist_loaders(data_dir: str, batch_size: int, seed: int, train_augment: bool = True):
    set_seed(seed)
    norm = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if train_augment:
        train_trans = transforms.Compose([
            transforms.RandomAffine(degrees=6, translate=(0.06, 0.06)),
            *norm,
        ])
    else:
        train_trans = transforms.Compose(norm)
    test_trans = transforms.Compose(norm)
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=train_trans)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=test_trans)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


LAYER_DIMS = [1 * 28 * 28, 16 * 14 * 14, 32 * 7 * 7]
LAYER_SHAPES = [(1, 28, 28), (16, 14, 14), (32, 7, 7)]


def _get_k_for_layer(k_input, layer_idx: int, inject_layers: list):
    if isinstance(k_input, (list, tuple)):
        return k_input[layer_idx]
    d = LAYER_DIMS[layer_idx]
    return k_input[..., :d]


def forward_with_injection(
    model: nn.Module,
    x: torch.Tensor,
    k_input,
    gamma: float,
    inject_fn,
    inject_layers: list,
) -> torch.Tensor:
    h = x
    if 0 in inject_layers:
        k0 = _get_k_for_layer(k_input, 0, inject_layers)
        k0 = k0.view(k0.size(0), *LAYER_SHAPES[0])
        h = inject_fn(h, k0, gamma)
    h = model.pool(F.relu(model.conv1(h)))
    if 1 in inject_layers:
        k1 = _get_k_for_layer(k_input, 1, inject_layers)
        k1 = k1.view(k1.size(0), *LAYER_SHAPES[1])
        h = inject_fn(h, k1, gamma)
    h = model.pool(F.relu(model.conv2(h)))
    if 2 in inject_layers:
        k2 = _get_k_for_layer(k_input, 2, inject_layers)
        k2 = k2.view(k2.size(0), *LAYER_SHAPES[2])
        h = inject_fn(h, k2, gamma)
    h = h.view(h.size(0), -1)
    return model.fc(h)


class SmallCNNModeB(nn.Module):
    """与 02 同骨干；输出 11 维（0–9 语义 + 10 拒绝）。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 7 * 7, NUM_CLASSES + 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train_mode_b(
    model: nn.Module,
    basis,
    train_loader: DataLoader,
    inject_fn,
    gamma: float,
    inject_layers: list,
    per_layer_key: bool,
    lr: float,
    epochs: int,
    device,
    target_key_std: float,
    deny_weight: float,
    deny_on: str,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_per_layer_basis = isinstance(basis, (list, tuple))
    if use_per_layer_basis:
        B_list = basis
        m = B_list[inject_layers[0]].shape[0]
    else:
        B = basis
        m, d_max = B.shape

    deny_on_l = (deny_on or "wrong").lower().replace("+", ",")
    use_no = deny_on_l in ("wrong,no", "no,wrong", "both", "wrong+no", "no+wrong")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for _batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            if use_per_layer_basis:
                k_per_layer = [None, None, None]
                if not per_layer_key:
                    alpha = torch.rand(batch_size, B_list[inject_layers[0]].shape[0], device=device)
                for layer_idx in inject_layers:
                    B_i = B_list[layer_idx]
                    m_i = B_i.shape[0]
                    alpha_i = torch.rand(batch_size, m_i, device=device) if per_layer_key else alpha
                    k_per_layer[layer_idx] = key_in_span(B_i, alpha_i, device)
                k_input = k_per_layer
            elif per_layer_key:
                k_per_layer = [None, None, None]
                for layer_idx in inject_layers:
                    alpha_i = torch.rand(batch_size, m, device=device)
                    k_i = key_in_span(B, alpha_i, device)[..., : LAYER_DIMS[layer_idx]]
                    k_per_layer[layer_idx] = k_i
                k_input = k_per_layer
            else:
                alpha = torch.rand(batch_size, m, device=device)
                k_input = key_in_span(B, alpha, device)

            if target_key_std and target_key_std > 0:
                k_input = rescale_key_to_std(k_input, target_key_std)

            logits_ok = forward_with_injection(model, x, k_input, gamma, inject_fn, inject_layers)
            loss = F.cross_entropy(logits_ok, y)

            if deny_weight > 0:
                if use_per_layer_basis or per_layer_key:
                    k_wrong_list = [None, None, None]
                    for layer_idx in inject_layers:
                        B_i = B_list[layer_idx] if use_per_layer_basis else B
                        d_i = LAYER_DIMS[layer_idx]
                        k_wrong_list[layer_idx] = sample_key_outside_span(B_i, d_i, device).unsqueeze(0).expand(
                            batch_size, -1
                        )
                    k_wrong = k_wrong_list
                else:
                    k_wrong = sample_key_outside_span(B, d_max, device).unsqueeze(0).expand(batch_size, -1)
                if target_key_std and target_key_std > 0:
                    k_wrong = rescale_key_to_std(k_wrong, target_key_std)
                logits_wrong = forward_with_injection(model, x, k_wrong, gamma, inject_fn, inject_layers)
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
):
    """语义 top-1：若 argmax 为拒绝维 10，则计为语义错误；另报告拒绝维胜率。"""
    model.eval()
    use_per_layer_basis = isinstance(basis, (list, tuple))
    if use_per_layer_basis:
        B_list = basis
        m = B_list[inject_layers[0]].shape[0]
    else:
        B = basis
        m, d_max = B.shape

    all_y = []
    pred_no, pred_ok, pred_wrong = [], [], []

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        n = x.size(0)
        all_y.append(y)
        logits_no = model(x)
        pred_no.append(logits_no.argmax(dim=1))

        if use_per_layer_basis or per_layer_key:
            k_list = [None, None, None]
            for layer_idx in inject_layers:
                B_i = B_list[layer_idx] if use_per_layer_basis else B
                m_i = B_i.shape[0]
                alpha_i = torch.rand(1, m_i, device=device)
                k_i = key_in_span(B_i, alpha_i, device).squeeze(0)[: LAYER_DIMS[layer_idx]]
                k_list[layer_idx] = k_i.unsqueeze(0).expand(n, -1) if k_i.dim() == 1 else k_i
            k_correct = k_list
        else:
            alpha = torch.rand(1, m, device=device)
            k_correct = key_in_span(B, alpha, device).expand(n, -1)
        if target_key_std and target_key_std > 0:
            k_correct = rescale_key_to_std(k_correct, target_key_std)

        logits_ok = forward_with_injection(model, x, k_correct, gamma, inject_fn, inject_layers)
        pred_ok.append(logits_ok.argmax(dim=1))

        if use_per_layer_basis or per_layer_key:
            k_wrong_list = [None, None, None]
            for layer_idx in inject_layers:
                B_i = B_list[layer_idx] if use_per_layer_basis else B
                d_i = LAYER_DIMS[layer_idx]
                k_wrong_list[layer_idx] = sample_key_outside_span(B_i, d_i, device).unsqueeze(0).expand(n, -1)
            k_wrong = k_wrong_list
        else:
            k_wrong = sample_key_outside_span(B, d_max, device).unsqueeze(0).expand(n, -1)
        if target_key_std and target_key_std > 0:
            k_wrong = rescale_key_to_std(k_wrong, target_key_std)
        logits_wrong = forward_with_injection(model, x, k_wrong, gamma, inject_fn, inject_layers)
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
    p = argparse.ArgumentParser(description="SpanKey 示例05：MNIST SmallCNN + 模式 B（拒绝类）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m", type=int, default=8, help="子空间维数 m")
    p.add_argument("--inject", type=str, default="mul", choices=["add", "mul"])
    p.add_argument("--inject_layers", type=str, default="0,2", help="注入层：0 输入 1 pool1后 2 pool2后，逗号分隔")
    p.add_argument("--per_layer_key", action="store_true")
    p.add_argument("--per_layer_basis", action="store_true")
    p.add_argument("--gamma", type=float, default=0.5, help="注入强度 γ")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no_scale_b_to_data", action="store_true")
    p.add_argument("--target_key_std", type=float, default=0.5)
    p.add_argument("--no_train_aug", action="store_true")
    p.add_argument("--deny_weight", type=float, default=0.1, help="拒损权重 λ")
    p.add_argument("--deny_on", type=str, default="wrong", choices=["wrong", "wrong+no"],
                   help="非法监督：仅错钥或错钥+无钥")
    p.add_argument("--json_out", type=str, default=None, help="将指标与配置写入该 JSON 路径（便于消融汇总）")
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="将训练后的权重与基矩阵等保存到该路径（.pt），供 security_attacks_05.py 做攻击评估",
    )
    args = p.parse_args()

    inject_layers = [int(s.strip()) for s in args.inject_layers.split(",") if s.strip()]
    if not inject_layers or not all(0 <= i <= 2 for i in inject_layers):
        raise ValueError("--inject_layers 每项应为 0、1 或 2 之非空子集")

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    inject_fn = get_inject_fn(args.inject)
    per_layer_key = args.per_layer_key
    per_layer_basis = args.per_layer_basis and len(inject_layers) > 1
    scale_to_data = not args.no_scale_b_to_data

    train_loader, test_loader = get_mnist_loaders(
        args.data_dir, args.batch_size, args.seed + 1, train_augment=not args.no_train_aug
    )

    if per_layer_basis:
        B_list = [None, None, None]
        for i in inject_layers:
            B_list[i] = make_basis(args.m, LAYER_DIMS[i], args.seed + 2 + i, orthonormalize=True,
                                   scale_to_data=scale_to_data, device=device)
        basis = B_list
    else:
        d_max = max(LAYER_DIMS[i] for i in inject_layers)
        basis = make_basis(args.m, d_max, args.seed + 2, orthonormalize=True,
                           scale_to_data=scale_to_data, device=device)

    model = SmallCNNModeB().to(device)
    print(
        f"Mode B | inject={args.inject} γ={args.gamma} m={args.m} layers={inject_layers} "
        f"deny_w={args.deny_weight} deny_on={args.deny_on} device={device}"
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
        args.epochs,
        device,
        args.target_key_std,
        args.deny_weight,
        args.deny_on,
    )

    metrics_tr = evaluate_mode_b(
        model, basis, train_loader, inject_fn, args.gamma, inject_layers, per_layer_key, device, args.target_key_std
    )
    metrics_te = evaluate_mode_b(
        model, basis, test_loader, inject_fn, args.gamma, inject_layers, per_layer_key, device, args.target_key_std
    )

    def _print(tag: str, met: dict):
        print(f"\n=== {tag} ===")
        print(f"  semantic acc  no / ok / wrong: {met['acc_semantic_no_key']:.4f} / "
              f"{met['acc_semantic_correct_key']:.4f} / {met['acc_semantic_wrong_key']:.4f}")
        print(f"  reject frac     no / ok / wrong: {met['reject_frac_no_key']:.4f} / "
              f"{met['reject_frac_correct_key']:.4f} / {met['reject_frac_wrong_key']:.4f}")
        print(f"  random semantic baseline 1/10 = {met['random_baseline']:.4f}")

    _print("train", metrics_tr)
    _print("test", metrics_te)

    payload = {
        "config": {
            "m": args.m,
            "gamma": args.gamma,
            "inject_layers": inject_layers,
            "inject": args.inject,
            "deny_weight": args.deny_weight,
            "deny_on": args.deny_on,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "metrics_train": metrics_tr,
        "metrics_test": metrics_te,
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.json_out}")

    if args.ckpt:
        ckpt = {
            "model_state": model.state_dict(),
            "config": {
                "m": args.m,
                "gamma": args.gamma,
                "inject_layers": inject_layers,
                "inject": args.inject,
                "per_layer_key": per_layer_key,
                "per_layer_basis": per_layer_basis,
                "target_key_std": args.target_key_std,
                "seed": args.seed,
                "scale_to_data": scale_to_data,
            },
        }
        if per_layer_basis:
            ckpt["basis"] = [None if b is None else b.detach().cpu() for b in basis]
        else:
            ckpt["basis"] = basis.detach().cpu()
        torch.save(ckpt, args.ckpt)
        print(f"\nSaved checkpoint to {args.ckpt}")


if __name__ == "__main__":
    main()
