"""
SpanKey 实验 Demo：key 注入 + CNN + MNIST 手写数字识别

与 01 同模式：仅用正确 key（span 内）训练；--inject_layers 指定注入层；--per_layer_key / --per_layer_basis 同 01。
注入方式（--inject）：add=对输入加法，mul=对输入乘法。默认使用 CUDA，可用 --cpu 强制 CPU。
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
    """返回张量的 min/max/mean/std（float），便于写入调试文件。"""
    t = t.detach().float()
    return {
        "min": t.min().item(),
        "max": t.max().item(),
        "mean": t.mean().item(),
        "std": t.std().item(),
    }


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- 基向量与 Key ----------

def make_basis(m: int, d: int, seed: int, orthonormalize: bool = True, scale_to_data: bool = True, device=None):
    """生成基向量矩阵 B: (m, d)。
    scale_to_data=True 时：B 取值在 [-1/sqrt(m), 1/sqrt(m)]，使 k=alpha@B 的 std 与 m 无关（约 0.3~0.5），
    与归一化后样本同量级；不做正交化。
    scale_to_data=False 时：沿用 randn + 可选正交化。
    """
    g = torch.Generator().manual_seed(seed)
    if scale_to_data:
        # B 元素 [-1/sqrt(m), 1/sqrt(m)] => k 分量 std ~ 常数，不随 m 增大而变小
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
        out = (alpha.cpu() @ B.cpu()).squeeze(-1).to(device)
        return out
    return (alpha @ B).squeeze(-1)


def rescale_key_to_std(k, target_std: float = 0.5, eps: float = 1e-8):
    """将 key 张量（或每层 key 的 list）整体缩放到 target_std，使与归一化数据同量级。"""
    if isinstance(k, (list, tuple)):
        out = []
        for t in k:
            out.append(None if t is None else rescale_key_to_std(t, target_std, eps))
        return out
    s = k.std().item() + eps
    if s <= 0:
        return k
    return k * (target_std / s)


def sample_key_outside_span(B: torch.Tensor, d: int, device=None) -> torch.Tensor:
    return torch.randn(d, device=device)


# ---------- 注入 ----------

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


# ---------- 数据 ----------

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


# ---------- 模型 ----------

# CNN 各注入点维度：0=输入 1×28×28，1=pool1 后 16×14×14，2=pool2 后 32×7×7
LAYER_DIMS = [1 * 28 * 28, 16 * 14 * 14, 32 * 7 * 7]
LAYER_SHAPES = [(1, 28, 28), (16, 14, 14), (32, 7, 7)]


def _get_k_for_layer(k_input, layer_idx: int, inject_layers: list):
    if isinstance(k_input, (list, tuple)):
        return k_input[layer_idx]
    d = LAYER_DIMS[layer_idx]
    return k_input[..., :d]


def forward_with_injection(model: nn.Module, x: torch.Tensor, k_input,
                          gamma: float, inject_fn, inject_layers: list,
                          debug_list: list = None) -> torch.Tensor:
    """按 inject_layers 在指定层做 key 注入；k 按层 reshape 为对应特征图形状。
    若 debug_list 非 None，会向其中追加每层注入前的 h、k、gamma*k、注入后的 h 的统计量。"""
    h = x
    if 0 in inject_layers:
        k0 = _get_k_for_layer(k_input, 0, inject_layers)
        k0 = k0.view(k0.size(0), *LAYER_SHAPES[0])
        if debug_list is not None:
            h_after = inject_fn(h, k0, gamma)
            debug_list.append({
                "layer": 0,
                "shape": list(h.shape),
                "h_before": _tensor_stats(h),
                "k": _tensor_stats(k0),
                "gamma_k": _tensor_stats(gamma * k0),
                "h_after": _tensor_stats(h_after),
            })
        h = inject_fn(h, k0, gamma)
    h = model.pool(F.relu(model.conv1(h)))
    if 1 in inject_layers:
        k1 = _get_k_for_layer(k_input, 1, inject_layers)
        k1 = k1.view(k1.size(0), *LAYER_SHAPES[1])
        if debug_list is not None:
            h_after = inject_fn(h, k1, gamma)
            debug_list.append({
                "layer": 1,
                "shape": list(h.shape),
                "h_before": _tensor_stats(h),
                "k": _tensor_stats(k1),
                "gamma_k": _tensor_stats(gamma * k1),
                "h_after": _tensor_stats(h_after),
            })
        h = inject_fn(h, k1, gamma)
    h = model.pool(F.relu(model.conv2(h)))
    if 2 in inject_layers:
        k2 = _get_k_for_layer(k_input, 2, inject_layers)
        k2 = k2.view(k2.size(0), *LAYER_SHAPES[2])
        if debug_list is not None:
            h_after = inject_fn(h, k2, gamma)
            debug_list.append({
                "layer": 2,
                "shape": list(h.shape),
                "h_before": _tensor_stats(h),
                "k": _tensor_stats(k2),
                "gamma_k": _tensor_stats(gamma * k2),
                "h_after": _tensor_stats(h_after),
            })
        h = inject_fn(h, k2, gamma)
    h = h.view(h.size(0), -1)
    return model.fc(h)


class SmallCNN(nn.Module):
    """MNIST 用小型 CNN：1x28x28 -> 10 类。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------- 训练与评估 ----------

def train(
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
    debug_file: str = None,
    target_key_std: float = 0.5,
):
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
        for batch_idx, (x, y) in enumerate(train_loader):
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
            if debug_file and epoch == 0 and batch_idx == 0:
                injections_debug = []
                logits = forward_with_injection(
                    model, x, k_input, gamma, inject_fn, inject_layers, debug_list=injections_debug
                )
                debug_collect = {
                    "config": {
                        "gamma": gamma,
                        "inject_layers": inject_layers,
                        "per_layer_key": per_layer_key,
                        "per_layer_basis": use_per_layer_basis,
                        "m": m,
                    },
                    "train_first_batch": {
                        "x": _tensor_stats(x),
                        "injections": injections_debug,
                    },
                }
                if use_per_layer_basis:
                    debug_collect["config"]["B_stats"] = [
                        _tensor_stats(basis[i]) for i in inject_layers
                    ]
                else:
                    debug_collect["config"]["B_stats"] = [_tensor_stats(basis)]
            else:
                logits = forward_with_injection(model, x, k_input, gamma, inject_fn, inject_layers)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs} loss={total_loss/len(train_loader):.4f}")
    if debug_file and debug_collect is not None:
        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump(debug_collect, f, indent=2, ensure_ascii=False)
        print(f"  Debug 数值已写入: {debug_file}")
    return model


@torch.no_grad()
def _append_eval_debug(debug_file: str, model, basis, data_loader, inject_fn, gamma,
                      inject_layers, per_layer_key, device, target_key_std: float = 0.5):
    """取评估集第一个 batch，计算 no_key / correct_key / wrong_key 的 logits 统计并追加到 debug 文件。"""
    model.eval()
    use_per_layer_basis = isinstance(basis, (list, tuple))
    B_list = basis if use_per_layer_basis else None
    B = None if use_per_layer_basis else basis
    m = B_list[inject_layers[0]].shape[0] if use_per_layer_basis else B.shape[0]
    d_max = B.shape[1] if B is not None else None
    it = iter(data_loader)
    x, y = next(it)
    x, y = x.to(device), y.to(device)
    n = x.size(0)
    logits_no = model(x)
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
    with open(debug_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["eval_first_batch"] = {
        "no_key_logits": _tensor_stats(logits_no),
        "correct_key_logits": _tensor_stats(logits_ok),
        "wrong_key_logits": _tensor_stats(logits_wrong),
    }
    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Debug 已追加 eval 首 batch logits 统计: {debug_file}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    basis,
    data_loader: DataLoader,
    inject_fn,
    gamma: float,
    inject_layers: list,
    per_layer_key: bool,
    device,
    target_key_std: float = 0.5,
):
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
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        n = x.size(0)
        all_y.append(y)
        all_no.append(model(x).argmax(1))
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
        all_ok.append(logits_ok.argmax(1))
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
        all_wrong.append(logits_wrong.argmax(1))
    y = torch.cat(all_y)
    acc_no = (torch.cat(all_no) == y).float().mean().item()
    acc_ok = (torch.cat(all_ok) == y).float().mean().item()
    acc_wrong = (torch.cat(all_wrong) == y).float().mean().item()
    return {
        "acc_no_key": acc_no,
        "acc_correct_key": acc_ok,
        "acc_wrong_key": acc_wrong,
        "random_baseline": 1.0 / n_classes,
    }


def main():
    p = argparse.ArgumentParser(description="SpanKey CNN + MNIST 数字识别")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m", type=int, default=10, help="基向量个数 m << 784")
    p.add_argument("--inject", type=str, default="add", choices=["add", "mul"],
                   help="注入方式: add=对输入加法, mul=对输入乘法")
    p.add_argument("--inject_layers", type=str, default="0",
                   help="注入层逗号分隔: 0=输入 1=pool1后 2=pool2后，如 0 或 0,1,2")
    p.add_argument("--per_layer_key", action="store_true", help="每层用不同 α 生成 key")
    p.add_argument("--per_layer_basis", action="store_true", help="多层时每层独立向量空间（每层一个 B）")
    p.add_argument("--gamma", type=float, default=0.5, help="注入强度")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data_dir", type=str, default="./data", help="MNIST 数据目录")
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU（默认启用 CUDA）")
    p.add_argument("--no_scale_b_to_data", action="store_true",
                   help="禁用：B 不按归一化数据尺度与 m 缩放（沿用 randn+正交化）")
    p.add_argument("--debug_file", type=str, default=None,
                   help="将首轮首个 batch 的 x/k/注入前后统计量写入该文件（JSON），便于调试数值")
    p.add_argument("--target_key_std", type=float, default=0.5,
                   help="key 缩放目标 std，与归一化数据同量级（0 表示不缩放）")
    p.add_argument("--no_train_aug", action="store_true", help="禁用训练集上的 RandomAffine 增强")
    args = p.parse_args()

    inject_layers = [int(s.strip()) for s in args.inject_layers.split(",")]
    if not all(0 <= i <= 2 for i in inject_layers):
        raise ValueError("--inject_layers 每项应为 0、1 或 2")

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        _ = torch.cuda.device_count()
        if hasattr(torch.backends.cuda, "preferred_blas_library"):
            torch.backends.cuda.preferred_blas_library("cublas")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    inject_fn = get_inject_fn(args.inject)
    per_layer_key = args.per_layer_key
    per_layer_basis = args.per_layer_basis and len(inject_layers) > 1
    print(f"Inject: {args.inject}, inject_layers: {inject_layers}, per_layer_key: {per_layer_key}, per_layer_basis: {per_layer_basis}, device: {device}")

    train_loader, test_loader = get_mnist_loaders(
        args.data_dir, args.batch_size, args.seed + 1, train_augment=not args.no_train_aug
    )
    scale_to_data = not args.no_scale_b_to_data
    if per_layer_basis:
        B_list = [None, None, None]
        for i in inject_layers:
            B_list[i] = make_basis(args.m, LAYER_DIMS[i], args.seed + 2 + i,
                                   orthonormalize=True, scale_to_data=scale_to_data, device=device)
        basis = B_list
        print(f"Per-layer basis: " + ", ".join(f"B[{i}].shape={basis[i].shape}" for i in inject_layers) +
              (", scale_to_data=True" if scale_to_data else ""))
    else:
        d_max = max(LAYER_DIMS[i] for i in inject_layers)
        basis = make_basis(args.m, d_max, args.seed + 2,
                           orthonormalize=True, scale_to_data=scale_to_data, device=device)
        print(f"Basis B: shape {basis.shape} (d_max={d_max})" +
              (", scale_to_data=True" if scale_to_data else ""))

    model = SmallCNN().to(device)
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
    train(model, basis, train_loader, inject_fn, args.gamma, inject_layers, per_layer_key,
          args.lr, args.epochs, device, debug_file=args.debug_file, target_key_std=args.target_key_std)

    def _print_cnn_metrics(metrics: dict):
        print(f"  No key:        acc = {metrics['acc_no_key']:.4f}")
        print(f"  Correct key:  acc = {metrics['acc_correct_key']:.4f}")
        print(f"  Wrong key:    acc = {metrics['acc_wrong_key']:.4f}")
        print(f"  Random:       1/10 = {metrics['random_baseline']:.4f}")

    print("\nEvaluation (train set):")
    metrics_tr = evaluate(model, basis, train_loader, inject_fn, args.gamma, inject_layers, per_layer_key,
                          device, target_key_std=args.target_key_std)
    if args.debug_file and os.path.isfile(args.debug_file):
        _append_eval_debug(
            args.debug_file, model, basis, train_loader, inject_fn, args.gamma,
            inject_layers, per_layer_key, device, target_key_std=args.target_key_std,
        )
    _print_cnn_metrics(metrics_tr)

    print("\nEvaluation (test set):")
    metrics_te = evaluate(model, basis, test_loader, inject_fn, args.gamma, inject_layers, per_layer_key,
                          device, target_key_std=args.target_key_std)
    _print_cnn_metrics(metrics_te)
    print("\n预期：无 key / 错误 key 准确率低，正确 key 准确率高。")


if __name__ == "__main__":
    main()
