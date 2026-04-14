#!/usr/bin/env python3
"""
示例 05（Mode B / MNIST）安全性评估：三类攻击均为「评估性」实验，不构成形式化安全证明。

1) adaptive   — 在子空间内随机采样系数 alpha，取验证集上语义准确率最高的密钥（adaptive key search）。
2) gradient   — 白盒：对 alpha 做梯度下降，最小化一批样本上的 CE(logits, y)（gradient-based key search）。
3) blackbox   — 仅前向查询：随机试探密钥，预算可与 adaptive 对齐（black-box query attack）。

使用前须先用 demo.py 训练并保存 checkpoint，例如：
  python demo.py ... --ckpt ./mode_b.pt

仅支持 checkpoint 中 per_layer_key=False 且 per_layer_basis=False（单一共享 B 与单一 alpha）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

# 与 demo.py 同目录运行
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from demo import (  # noqa: E402
    LAYER_DIMS,
    NUM_CLASSES,
    REJECT_IDX,
    SmallCNNModeB,
    forward_with_injection,
    get_inject_fn,
    get_mnist_loaders,
    key_in_span,
    rescale_key_to_std,
    sample_key_outside_span,
    set_seed,
)


def rescale_key_to_std_tensor(k: torch.Tensor, target_std: float, eps: float = 1e-8) -> torch.Tensor:
    """与 demo.rescale_key_to_std 统计一致，但保持可微（用于梯度攻击）。"""
    s = k.std().clamp_min(eps)
    return k * (target_std / s)


def load_checkpoint(path: str, device: torch.device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    if cfg.get("per_layer_key") or cfg.get("per_layer_basis"):
        raise ValueError(
            "此脚本当前仅支持 per_layer_key=False 且 per_layer_basis=False 的 checkpoint；"
            "请用共享 alpha + 单一 B 训练并保存。"
        )
    model = SmallCNNModeB().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    basis = ckpt["basis"]
    if isinstance(basis, list):
        raise ValueError("unexpected list basis without per_layer_basis flag")
    basis = basis.to(device)
    return model, basis, cfg


@torch.no_grad()
def evaluate_key(
    model: torch.nn.Module,
    B: torch.Tensor,
    data_loader,
    alpha_1m: torch.Tensor,
    gamma: float,
    inject_fn,
    inject_layers: list,
    device: torch.device,
    target_key_std: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    """alpha_1m: shape (1, m)，单密钥重复用于整批。"""
    m = B.shape[0]
    assert alpha_1m.shape == (1, m)
    k = key_in_span(B, alpha_1m, device)
    if target_key_std and target_key_std > 0:
        k = rescale_key_to_std(k, target_key_std)

    n_batches = 0
    tot_sem = 0.0
    tot_rej = 0.0
    tot_n = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        n = x.size(0)
        logits = forward_with_injection(
            model, x, k.expand(n, -1), gamma, inject_fn, inject_layers
        )
        pred = logits.argmax(dim=1)
        sem = ((pred < NUM_CLASSES) & (pred == y)).float()
        rej = (pred == REJECT_IDX).float()
        tot_sem += sem.sum().item()
        tot_rej += rej.sum().item()
        tot_n += n
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return {
        "semantic_acc": tot_sem / max(tot_n, 1),
        "reject_frac": tot_rej / max(tot_n, 1),
        "n_samples": tot_n,
    }


def attack_adaptive_random_search(
    model: torch.nn.Module,
    B: torch.Tensor,
    test_loader,
    gamma: float,
    inject_fn,
    inject_layers: list,
    device: torch.device,
    target_key_std: float,
    trials: int,
    seed: int,
    eval_max_batches: int | None,
) -> dict[str, float]:
    m = B.shape[0]
    best = {"semantic_acc": -1.0, "reject_frac": 1.0, "trial": -1}
    best_alpha: torch.Tensor | None = None
    for t in range(trials):
        torch.manual_seed(seed + t)
        alpha = torch.rand(1, m, device=device)
        met = evaluate_key(
            model,
            B,
            test_loader,
            alpha,
            gamma,
            inject_fn,
            inject_layers,
            device,
            target_key_std,
            max_batches=eval_max_batches,
        )
        if met["semantic_acc"] > best["semantic_acc"]:
            best = {**met, "trial": t}
            best_alpha = alpha.detach().clone()
    out: dict[str, float | int | None] = {
        "attack": "adaptive_random_search",
        "trials": trials,
        "best_semantic_acc": best["semantic_acc"],
        "reject_frac_at_best": best["reject_frac"],
        "best_trial": best["trial"],
        "eval_max_batches": eval_max_batches,
    }
    # 若试探阶段仅用部分 batch，则对最优 alpha 在完整测试集上再评一次（供论文报告）
    if best_alpha is not None:
        full = evaluate_key(
            model,
            B,
            test_loader,
            best_alpha,
            gamma,
            inject_fn,
            inject_layers,
            device,
            target_key_std,
            max_batches=None,
        )
        out["best_semantic_acc_full_test"] = full["semantic_acc"]
        out["reject_frac_at_best_full_test"] = full["reject_frac"]
    return out


def attack_blackbox_random_queries(
    model: torch.nn.Module,
    B: torch.Tensor,
    test_loader,
    gamma: float,
    inject_fn,
    inject_layers: list,
    device: torch.device,
    target_key_std: float,
    queries: int,
    seed: int,
    eval_max_batches: int | None,
) -> dict[str, float]:
    """与 adaptive 相同逻辑，单独命名以便在报告中区分「仅前向、无梯度」叙事。"""
    return {
        **attack_adaptive_random_search(
            model,
            B,
            test_loader,
            gamma,
            inject_fn,
            inject_layers,
            device,
            target_key_std,
            trials=queries,
            seed=seed + 999,
            eval_max_batches=eval_max_batches,
        ),
        "attack": "blackbox_query",
        "queries": queries,
    }


def attack_gradient_key_search(
    model: torch.nn.Module,
    B: torch.Tensor,
    data_loader,
    test_loader,
    gamma: float,
    inject_fn,
    inject_layers: list,
    device: torch.device,
    target_key_std: float,
    steps: int,
    lr: float,
    grad_seed: int,
    batch_size_cap: int,
) -> dict[str, float]:
    """在一批固定图像上优化 alpha，使 CE 最小；再在整集上评估。"""
    m = B.shape[0]
    torch.manual_seed(grad_seed)
    xs, ys = [], []
    for x, y in data_loader:
        xs.append(x)
        ys.append(y)
        if sum(t.size(0) for t in xs) >= batch_size_cap:
            break
    xb = torch.cat(xs, dim=0)[:batch_size_cap].to(device)
    yb = torch.cat(ys, dim=0)[:batch_size_cap].to(device)

    alpha = torch.empty(1, m, device=device).uniform_(1e-3, 1 - 1e-3)
    alpha.requires_grad_(True)
    opt = torch.optim.Adam([alpha], lr=lr)

    model.train()
    for _ in range(steps):
        opt.zero_grad()
        k = key_in_span(B, alpha, device)
        if target_key_std and target_key_std > 0:
            k = rescale_key_to_std_tensor(k, target_key_std)
        logits = forward_with_injection(
            model, xb, k.expand(xb.size(0), -1), gamma, inject_fn, inject_layers
        )
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        with torch.no_grad():
            alpha.clamp_(1e-6, 1.0 - 1e-6)

    model.eval()
    with torch.no_grad():
        alpha_eval = alpha.detach()
        met_full = evaluate_key(
            model,
            B,
            test_loader,
            alpha_eval,
            gamma,
            inject_fn,
            inject_layers,
            device,
            target_key_std,
            max_batches=None,
        )
    return {
        "attack": "gradient_key_search",
        "grad_steps": steps,
        "lr": lr,
        "train_batch_for_grad": xb.size(0),
        "best_semantic_acc": met_full["semantic_acc"],
        "reject_frac_at_best": met_full["reject_frac"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="示例05：子空间密钥攻击评估（adaptive / gradient / blackbox）")
    p.add_argument("--ckpt", type=str, required=True, help="demo.py --ckpt 保存的 .pt")
    p.add_argument(
        "--attack",
        type=str,
        default="all",
        choices=["all", "adaptive", "gradient", "blackbox"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--trials", type=int, default=400, help="adaptive：随机密钥试探次数")
    p.add_argument("--blackbox_queries", type=int, default=None, help="blackbox：查询次数（默认与 --trials 相同）")
    p.add_argument("--grad_steps", type=int, default=300)
    p.add_argument("--grad_lr", type=float, default=0.15)
    p.add_argument("--grad_batch", type=int, default=256, help="梯度攻击使用的样本数上限")
    p.add_argument(
        "--eval_max_batches",
        type=int,
        default=15,
        help="adaptive/blackbox 每次试探最多评估的 batch 数（加速）；最终梯度攻击仍全测试集",
    )
    p.add_argument("--json_out", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)
    model, B, cfg = load_checkpoint(args.ckpt, device)
    gamma = float(cfg["gamma"])
    inject_layers = list(cfg["inject_layers"])
    target_key_std = float(cfg.get("target_key_std", 0.5))
    inject_fn = get_inject_fn(cfg["inject"])

    _, test_loader = get_mnist_loaders(args.data_dir, batch_size=128, seed=args.seed + 1, train_augment=False)
    train_loader, _ = get_mnist_loaders(args.data_dir, batch_size=128, seed=args.seed + 2, train_augment=True)

    bq = args.blackbox_queries if args.blackbox_queries is not None else args.trials
    results: dict = {"checkpoint": args.ckpt, "config": cfg}

    # 基线：无密钥 / 单次随机子空间密钥 / 单次随机子空间外密钥（与 demo 一致）
    with torch.no_grad():
        m = B.shape[0]
        d_max = B.shape[1]
        n0, n_ok, n_wrong = 0, 0, 0
        c0, c_ok, c_wrong = 0.0, 0.0, 0.0
        r0, r_ok, r_wrong = 0.0, 0.0, 0.0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            n = x.size(0)
            logits_no = model(x)
            p0 = logits_no.argmax(dim=1)
            alpha_ok = torch.rand(1, m, device=device)
            k_ok = rescale_key_to_std(key_in_span(B, alpha_ok, device), target_key_std)
            logits_ok = forward_with_injection(
                model, x, k_ok.expand(n, -1), gamma, inject_fn, inject_layers
            )
            p_ok = logits_ok.argmax(dim=1)
            k_wrong = rescale_key_to_std(
                sample_key_outside_span(B, d_max, device).unsqueeze(0), target_key_std
            )
            logits_w = forward_with_injection(
                model, x, k_wrong.expand(n, -1), gamma, inject_fn, inject_layers
            )
            pw = logits_w.argmax(dim=1)

            def _sem(pred, yv):
                return ((pred < NUM_CLASSES) & (pred == yv)).float()

            def _rej(pred):
                return (pred == REJECT_IDX).float()

            c0 += _sem(p0, y).sum().item()
            r0 += _rej(p0).sum().item()
            c_ok += _sem(p_ok, y).sum().item()
            r_ok += _rej(p_ok).sum().item()
            c_wrong += _sem(pw, y).sum().item()
            r_wrong += _rej(pw).sum().item()
            n0 += n
        results["baseline"] = {
            "semantic_acc_no_key": c0 / n0,
            "semantic_acc_random_in_span_key": c_ok / n0,
            "semantic_acc_random_out_span_key": c_wrong / n0,
            "reject_frac_no_key": r0 / n0,
            "reject_frac_random_in_span": r_ok / n0,
            "reject_frac_random_out_span": r_wrong / n0,
        }

    if args.attack in ("all", "adaptive"):
        results["adaptive"] = attack_adaptive_random_search(
            model,
            B,
            test_loader,
            gamma,
            inject_fn,
            inject_layers,
            device,
            target_key_std,
            trials=args.trials,
            seed=args.seed + 11,
            eval_max_batches=args.eval_max_batches,
        )
    if args.attack in ("all", "blackbox"):
        results["blackbox"] = attack_blackbox_random_queries(
            model,
            B,
            test_loader,
            gamma,
            inject_fn,
            inject_layers,
            device,
            target_key_std,
            queries=bq,
            seed=args.seed + 22,
            eval_max_batches=args.eval_max_batches,
        )
    if args.attack in ("all", "gradient"):
        results["gradient"] = attack_gradient_key_search(
            model,
            B,
            train_loader,
            test_loader,
            gamma,
            inject_fn,
            inject_layers,
            device,
            target_key_std,
            steps=args.grad_steps,
            lr=args.grad_lr,
            grad_seed=args.seed + 33,
            batch_size_cap=args.grad_batch,
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
