"""
SpanKey 实验 Demo：key 注入 + MLP + 合成数据

训练：仅用正确 key（span 内动态 key）训练。
目标：无 key 或错误 key 时模型输出错误；有正确 key 时输出正确。
注入方式（--inject 二选一）：add=对输入做加法 x+γk，mul=对输入做乘法 x·(1+γ·tanh(k))；--inject_layers 指定注入层。
多层注入时若启用 --per_layer_basis，则每层使用独立向量空间（每层一个 B）。
"""

# 在 import torch 之前设置，避免 WSL 下 cublasLt 未初始化导致 CUBLAS_STATUS_NOT_INITIALIZED
import os
import warnings
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")
# 部分环境下 cuBLAS workspace 分配导致 init 失败，强制不占用 workspace，使 unfused cublas 路径可用
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# 在 CUBLAS_WORKSPACE_CONFIG=:0:0 下 cublasLt 会报错并自动走 unfused 路径，屏蔽该 UserWarning
warnings.filterwarnings("ignore", message=".*gemm_and_bias error.*CUBLAS.*", category=UserWarning)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- 基向量与 Key 生成 ----------

def make_basis(m: int, d: int, seed: int, orthonormalize: bool = True, scale_to_data: bool = False, device=None):
    """生成基向量矩阵 B: (m, d)。
    scale_to_data=True 时：B 取值在 [-1/m, 1/m]，使 k=alpha@B 与 alpha∈[0,1] 时分量在 [-1,1]，
    与归一化数据同量级且不随 m 放大；不做正交化。
    scale_to_data=False 时：沿用 randn + 可选正交化。
    """
    g = torch.Generator().manual_seed(seed)
    if scale_to_data:
        scale = (m ** -0.5) if m >= 1 else 1.0
        B = (2 * torch.rand(m, d, generator=g) - 1) * scale
    else:
        B = torch.randn(m, d, generator=g)
        if orthonormalize and m <= d:
            B, _ = torch.linalg.qr(B.T)
            B = B.T  # (m, d)
    if device is not None:
        B = B.to(device)
    return B


def key_in_span(B: torch.Tensor, alpha: torch.Tensor, device=None) -> torch.Tensor:
    """k = alpha @ B，返回 (d,) 或 (batch, d)。在 CPU 上计算后移至 device，避免 WSL 下 CUBLAS 未初始化。"""
    if device is not None and B.is_cuda:
        alpha_cpu = alpha.cpu()
        B_cpu = B.cpu()
        out = (alpha_cpu @ B_cpu).squeeze(-1).to(device)
        return out
    return (alpha @ B).squeeze(-1)


def sample_key_outside_span(B: torch.Tensor, d: int, device=None) -> torch.Tensor:
    """采样 span 外的 key：随机高斯向量（当 d > m 时几乎必然不在 span 内）。"""
    return torch.randn(d, device=device)


# ---------- 注入 ----------

def inject_add(x: torch.Tensor, k: torch.Tensor, gamma: float) -> torch.Tensor:
    """加法注入：x' = x + gamma * k。x (B, d), k (d,) 或 (B, d)。"""
    return x + gamma * k


def inject_mul(x: torch.Tensor, k: torch.Tensor, gamma: float) -> torch.Tensor:
    """乘法注入：x' = x * (1 + gamma * tanh(k))，缩放有界避免爆炸。"""
    scale = (1 + gamma * torch.tanh(k)).clamp(0.1, 2.0)
    return x * scale


def get_inject_fn(mode: str):
    """根据 mode 返回注入函数 (x, k, gamma) -> x_prime。"""
    fns = {"add": inject_add, "mul": inject_mul}
    if mode not in fns:
        raise ValueError(f"--inject 可选: {list(fns.keys())}，当前为 {mode!r}")
    return fns[mode]


# ---------- 数据 ----------

def make_synthetic_data(n_samples: int, d: int, n_classes: int, seed: int, device=None):
    """简单可分的合成数据：随机特征 + 线性投影得 logits，取 argmax 为标签。"""
    set_seed(seed)
    X = torch.randn(n_samples, d) * 0.5
    W = torch.randn(d, n_classes) * 0.5
    logits = X @ W
    y = logits.argmax(dim=1)
    if device is not None:
        X, y = X.to(device), y.to(device)
    return X, y


# ---------- 模型 ----------

def get_layer_dims(model: nn.Module) -> list:
    """MLP 各注入点对应的维度：0=输入层, 1=第一层线性输出, 2=第二层线性输出。"""
    net = model.net
    # Sequential: [Linear(d,64), ReLU, Linear(64,32), ReLU, Linear(32,n_classes)]
    return [net[0].in_features, net[0].out_features, net[2].out_features]


def _get_k_for_layer(k_input, layer_idx: int, inject_layers: list, layer_dims: list):
    """k_input 为 (batch, d_max) 或 list of 3 tensors；返回该层用的 key 切片。"""
    if isinstance(k_input, (list, tuple)):
        return k_input[layer_idx]
    return k_input[..., : layer_dims[layer_idx]]


def forward_with_injection(model: nn.Module, x: torch.Tensor, k_input,
                          gamma: float, inject_fn, inject_layers: list, layer_dims: list) -> torch.Tensor:
    """按 inject_layers 在指定层做 key 注入并前向。
    k_input: 单 key 时为 (batch, d_max) 或 (d_max,)；每层不同 key 时为 list of 3 tensors，k_input[i] 形状 (..., layer_dims[i])。
    """
    net = model.net
    h = x
    if 0 in inject_layers:
        k0 = _get_k_for_layer(k_input, 0, inject_layers, layer_dims)
        h = inject_fn(h, k0, gamma)
    h = net[0](h)
    if 1 in inject_layers:
        k1 = _get_k_for_layer(k_input, 1, inject_layers, layer_dims)
        h = inject_fn(h, k1, gamma)
    h = net[1](h)
    h = net[2](h)
    if 2 in inject_layers:
        k2 = _get_k_for_layer(k_input, 2, inject_layers, layer_dims)
        h = inject_fn(h, k2, gamma)
    h = net[3](h)
    h = net[4](h)
    return h


class MLP(nn.Module):
    def __init__(self, d: int, n_classes: int, hidden: list = (64, 32)):
        super().__init__()
        layers = []
        prev = d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------- 训练与评估 ----------

def train(
    model: nn.Module,
    basis,
    train_loader: DataLoader,
    inject_fn,
    gamma: float,
    inject_layers: list,
    layer_dims: list,
    per_layer_key: bool,
    lr: float,
    epochs: int,
    device,
):
    """basis 为单个 B 或 B_list（每层一个 B）。per_layer_key 为 True 时每层用不同 α。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_per_layer_basis = isinstance(basis, (list, tuple))
    if use_per_layer_basis:
        B_list = basis
        m = B_list[inject_layers[0]].shape[0]
    else:
        B = basis
        m, d_max = B.shape
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
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
                    k_i = key_in_span(B_i, alpha_i, device)
                    k_per_layer[layer_idx] = k_i
                k_input = k_per_layer
            elif per_layer_key:
                k_per_layer = [None, None, None]
                for layer_idx in inject_layers:
                    alpha_i = torch.rand(batch_size, m, device=device)
                    k_i = key_in_span(B, alpha_i, device)[..., : layer_dims[layer_idx]]
                    k_per_layer[layer_idx] = k_i
                k_input = k_per_layer
            else:
                alpha = torch.rand(batch_size, m, device=device)
                k_input = key_in_span(B, alpha, device)
            logits = forward_with_injection(
                model, x, k_input, gamma, inject_fn, inject_layers, layer_dims
            )
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs} loss={total_loss/len(train_loader):.4f}")
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    basis,
    X: torch.Tensor,
    y: torch.Tensor,
    inject_fn,
    gamma: float,
    inject_layers: list,
    layer_dims: list,
    per_layer_key: bool,
    device,
):
    model.eval()
    use_per_layer_basis = isinstance(basis, (list, tuple))
    if use_per_layer_basis:
        B_list = basis
        m = B_list[inject_layers[0]].shape[0]
    else:
        B = basis
        m, d_max = B.shape
    n_classes = model.net[-1].out_features

    # 无 key
    acc_no_key = (model(X).argmax(1) == y).float().mean().item()

    # 正确 key（span 内）
    if use_per_layer_basis or per_layer_key:
        k_correct_list = [None, None, None]
        for layer_idx in inject_layers:
            B_i = B_list[layer_idx] if use_per_layer_basis else B
            m_i = B_i.shape[0]
            alpha_i = torch.rand(1, m_i, device=device)
            k_i = key_in_span(B_i, alpha_i, device).squeeze(0)
            if not use_per_layer_basis:
                k_i = k_i[: layer_dims[layer_idx]]
            k_correct_list[layer_idx] = k_i
        k_correct = k_correct_list
    else:
        alpha_unseen = torch.rand(1, m, device=device)
        k_correct = key_in_span(B, alpha_unseen, device).squeeze(0)
    logits_correct = forward_with_injection(
        model, X, k_correct, gamma, inject_fn, inject_layers, layer_dims
    )
    acc_correct = (logits_correct.argmax(1) == y).float().mean().item()

    # 错误 key（span 外）
    if use_per_layer_basis or per_layer_key:
        k_wrong_list = [None, None, None]
        for layer_idx in inject_layers:
            d_i = layer_dims[layer_idx]
            k_wrong_list[layer_idx] = sample_key_outside_span(B_list[layer_idx] if use_per_layer_basis else B, d_i, device)
        k_wrong = k_wrong_list
    else:
        k_wrong = sample_key_outside_span(B, d_max, device)
    logits_wrong = forward_with_injection(
        model, X, k_wrong, gamma, inject_fn, inject_layers, layer_dims
    )
    pred_wrong = logits_wrong.argmax(1)
    acc_wrong = (pred_wrong == y).float().mean().item()
    # 随机猜测准确率约为 1/n_classes
    random_baseline = 1.0 / n_classes

    return {
        "acc_no_key": acc_no_key,
        "acc_correct_key": acc_correct,
        "acc_wrong_key": acc_wrong,
        "random_baseline": random_baseline,
    }


def main():
    p = argparse.ArgumentParser(description="SpanKey demo: --inject add|mul, --inject_layers 指定注入层")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d", type=int, default=20, help="input dim")
    p.add_argument("--m", type=int, default=5, help="num basis vectors, m << d")
    p.add_argument("--n_classes", type=int, default=5)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--inject", type=str, default="add", choices=["add", "mul"],
                   help="注入方式（二选一）: add=对输入加法 x+γk, mul=对输入乘法 x·(1+γ·tanh(k))")
    p.add_argument("--inject_layers", type=str, default="0", help="注入层索引逗号分隔: 0=输入 1=第一隐层 2=第二隐层")
    p.add_argument("--per_layer_key", action="store_true", help="每层用不同 α 生成 key，减少冗余、降低吸收")
    p.add_argument("--per_layer_basis", action="store_true", help="多层注入时每层使用独立向量空间（每层一个 B）")
    p.add_argument("--gamma", type=float, default=1.0, help="注入强度")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--train_frac", type=float, default=0.8,
                   help="合成数据中用于训练的比例，其余为测试集（用于 train/test 双集评估）")
    p.add_argument("--cuda", action="store_true", help="使用 GPU")
    p.add_argument("--scale_b_to_data", action="store_true",
                   help="B 按归一化数据尺度与 m 缩放，使 key 与样本同量级")
    args = p.parse_args()

    inject_layers = [int(s.strip()) for s in args.inject_layers.split(",")]
    if not all(0 <= i <= 2 for i in inject_layers):
        raise ValueError("--inject_layers 中每项应为 0、1 或 2")

    set_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        _ = torch.cuda.device_count()
        if hasattr(torch.backends.cuda, "preferred_blas_library"):
            torch.backends.cuda.preferred_blas_library("cublas")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if use_cuda else "cpu")

    inject_fn = get_inject_fn(args.inject)
    per_layer_key = args.per_layer_key
    per_layer_basis = args.per_layer_basis and len(inject_layers) > 1
    print(f"Inject: {args.inject}, inject_layers: {inject_layers}, per_layer_key: {per_layer_key}, per_layer_basis: {per_layer_basis}")

    # 数据：划分训练 / 测试（固定随机置换）
    X, y = make_synthetic_data(
        args.n_samples, args.d, args.n_classes, args.seed + 2, device=None
    )
    g_split = torch.Generator().manual_seed(args.seed + 99)
    n = X.size(0)
    cut = max(1, min(n - 1, int(n * args.train_frac)))
    perm = torch.randperm(n, generator=g_split)
    train_idx, test_idx = perm[:cut], perm[cut:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    print(f"Data split: train={len(train_idx)}, test={len(test_idx)} (train_frac={args.train_frac})")

    # 模型与各层维度；多层且 per_layer_basis 时每层一个 B，否则单一 B
    model = MLP(args.d, args.n_classes).to(device)
    layer_dims = get_layer_dims(model)
    scale_to_data = args.scale_b_to_data
    if per_layer_basis:
        B_list = [None, None, None]
        for i in inject_layers:
            B_list[i] = make_basis(args.m, layer_dims[i], args.seed + 1 + i,
                                   orthonormalize=True, scale_to_data=scale_to_data, device=device)
        basis = B_list
        print(f"Per-layer basis: B[i] shape (m, layer_dims[i]) for i in {inject_layers}" +
              (", scale_to_data=True" if scale_to_data else ""))
    else:
        d_max = max(layer_dims[i] for i in inject_layers)
        basis = make_basis(args.m, d_max, args.seed + 1,
                          orthonormalize=True, scale_to_data=scale_to_data, device=device)
        print(f"Basis B: shape {basis.shape} (d_max={d_max} for layers {inject_layers})" +
              (", scale_to_data=True" if scale_to_data else ""))

    # 使用 CUDA 时先做一次前向预热；若出现 WSL 下 CUBLAS 未初始化则自动回退到 CPU
    if device.type == "cuda":
        try:
            with torch.no_grad():
                _dummy = torch.randn(args.batch_size, args.d, device=device)
                _ = model(_dummy)
            torch.cuda.synchronize()
        except RuntimeError as e:
            if "CUBLAS" in str(e) or "cublas" in str(e).lower():
                print("Warning: CUDA CUBLAS 初始化失败，已自动回退到 CPU 运行。")
                device = torch.device("cpu")
                model = model.cpu()
                if per_layer_basis:
                    basis = [b.cpu() if b is not None else None for b in basis]
                else:
                    basis = basis.cpu()
            else:
                raise

    print("Training...")
    train(
        model, basis, train_loader,
        inject_fn=inject_fn,
        gamma=args.gamma,
        inject_layers=inject_layers,
        layer_dims=layer_dims,
        per_layer_key=per_layer_key,
        lr=args.lr,
        epochs=args.epochs,
        device=device,
    )

    def _print_mlp_metrics(metrics: dict, label: str):
        print(label)
        print(f"  No key:        acc = {metrics['acc_no_key']:.4f}")
        print(f"  Correct key:   acc = {metrics['acc_correct_key']:.4f}  (span 内，未见过的 alpha)")
        print(f"  Wrong key:     acc = {metrics['acc_wrong_key']:.4f}  (span 外)")
        print(f"  Random guess:  1/n_classes = {metrics['random_baseline']:.4f}")

    print("\nEvaluation (train split):")
    metrics_tr = evaluate(
        model, basis, X_train.to(device), y_train.to(device),
        inject_fn, args.gamma, inject_layers, layer_dims, per_layer_key, device
    )
    _print_mlp_metrics(metrics_tr, "")

    print("\nEvaluation (test split):")
    metrics_te = evaluate(
        model, basis, X_test.to(device), y_test.to(device),
        inject_fn, args.gamma, inject_layers, layer_dims, per_layer_key, device
    )
    _print_mlp_metrics(metrics_te, "")
    print("\n预期：无 key / 错误 key 时准确率低（计算错误）；正确 key 时准确率高（span 泛化）。")


if __name__ == "__main__":
    main()
