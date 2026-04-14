# SpanKey (dksc): experiments for dynamic key-space conditioning

Public code for **SpanKey**: you fix a low-dimensional secret subspace $\mathrm{Span}(B)$, sample **dynamic keys** $k=\alpha^\top B$ (in-span at train time; out-of-span for “wrong key” tests), and **inject** $k$ into chosen layers (additive or multiplicative). Training usually uses **only valid (in-span) keys**; evaluation compares **no key / correct key / wrong key** accuracies. The goal is to study how much behavior depends on the key versus plain input features—**not** to provide cryptographic guarantees.

Repository: [github.com/mindmemory-ai/dksc](https://github.com/mindmemory-ai/dksc)

---

## How experiments are organized

| Stage | What we do |
|--------|------------|
| **Data & model** | Examples 01–04 use synthetic vectors, MNIST, Fashion-MNIST, or CIFAR-10 with MLP / small CNN / ResNet-style or ResNet-18 backbones. |
| **Key & basis** | A fixed random orthonormal basis $B$ spans an $m$-dimensional subspace; coefficients $\alpha$ are sampled per step or per epoch to form keys. “Wrong” keys are sampled **outside** $\mathrm{Span}(B)$ (e.g. Gaussian in the orthogonal complement). |
| **Injection** | Keys are injected at selected layers via `--inject add` or `--inject mul` with strength `--gamma`, optionally `--per_layer_key` / `--per_layer_basis` / `--inject_layers …`. |
| **Training** | By default, supervision uses **correct-key** forward passes only (baseline). Example **04** additionally supports **deny losses** (`--deny_mode` A / B / C and extensions) on wrong-key (and optionally no-key) batches. |
| **Metrics** | Each `demo.py` prints **train** then **test** accuracies for **no key / correct key / wrong key** (and extra lines for Mode B reject probability when applicable). |
| **Batch scripts** | Under `examples/`, helper scripts print or run sweeps (baseline add vs mul, deny modes, MNIST Mode B ablations); see below. |

Paper-facing tables were produced from GPU runs logged alongside these demos; you can keep an optional aggregated note (e.g. `examples/results_summary.md`) for your own train/test figures.

---

## Installation

**Requirements:** Python **3.10+** recommended; **PyTorch** with CUDA for examples **02–05** and for full **04** training (use `--cpu` on 04 only for smoke tests).

1. **Clone**

   ```bash
   git clone https://github.com/mindmemory-ai/dksc.git
   cd dksc
   ```

2. **Virtual environment (recommended)**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies (whole repo)**

   From the repository **root**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Each numbered folder under `examples/<NN>_…/` also ships a minimal `requirements.txt` if you prefer to install **per example only** (`pip install -r examples/01_spankey_mlp/requirements.txt`, etc.).

4. **GPU note:** Install a PyTorch build that matches your CUDA version from [pytorch.org](https://pytorch.org) if the default `pip install torch` wheel is not suitable.

---

## Layout

| Path | Role |
|------|------|
| `requirements.txt` | Root stack: `torch`, `torchvision`, `numpy`; `matplotlib` for optional plotting. |
| `examples/01_spankey_mlp/` | MLP on synthetic vectors; multi-layer injection; optional per-layer keys/bases. |
| `examples/02_spankey_cnn_mnist/` | Small CNN on MNIST; add/mul injection. |
| `examples/03_spankey_resnet_fashionmnist/` | ResNet-style model on Fashion-MNIST. |
| `examples/04_spankey_resnet18_cifar10/` | ResNet-18 on CIFAR-10; augmentation + SGD schedule; **deny modes** A/B/C and extensions (`--deny_mode`, …). |
| `examples/05_spankey_cnn_mnist_mode_b/` | MNIST small CNN with **Mode B** (reject class); `security_attacks_05.py` runs adaptive / gradient / black-box key-search **evaluations** on a saved `--ckpt`. |
| `examples/06_spankey_vit_tiny_cifar10_mode_b/` | CIFAR-10 **ViT-Tiny** (timm) with Mode B; token-level `mul` injection; needs `timm` (see root `requirements.txt`). |
| `examples/run_baseline_dual_inject.py` | Runs 01–04 with default `demo.py` (add & mul paths in script; see file). |
| `examples/run_deny04_abc_modes.py` | Prints or runs CIFAR **04** sweeps for deny modes aligned with paper Modes A/B/C. |
| `examples/run_deny04_enhanced_modes.py` | Extended deny modes (e.g. `A_soft`, `cplus`, `AC`, `B_aux`). |
| `examples/run_spankey05_mode_b_ablation.py` | Single-factor grids for Mode B on MNIST (m / layers / γ). |
| `examples/plot_spankey05_ablation.py` | Plots JSON ablations (needs `matplotlib`). |
| `examples/05_spankey_ablation_json/` | Example logged JSON outputs for 05-style runs. |
| `paper/` | Draft LaTeX (may be omitted in minimal clones); not required to run `examples/`. |

---

## Quick start (single demo)

From repo root, after `pip install -r requirements.txt`:

```bash
cd examples/01_spankey_mlp
python demo.py
```

Repeat with `02_spankey_cnn_mnist`, `03_…`, `04_…` as needed. **04** defaults to CUDA; pass `--cpu` for a short CPU run.

**Stronger CIFAR-10 baseline (closer to paper tables):**

```bash
cd examples/04_spankey_resnet18_cifar10
python demo.py --inject mul --gamma 2.0 --target_key_std 1.0 \
  --inject_layers 0,2,3,4 --per_layer_key --per_layer_basis \
  --epochs 100 --optimizer sgd --lr 0.1 --weight_decay 5e-4 --milestones 50,75
```

See `examples/04_spankey_resnet18_cifar10/README.md` for deny-loss flags and logging.

---

## Batch / reproducibility scripts

Run from `examples/` (or adjust paths):

```bash
cd examples

# Parse stdout from 01–04 default demos into JSON (see script for inject modes)
python run_baseline_dual_inject.py

# Deny-mode sweeps on 04 (default: print commands; add --run to execute)
python run_deny04_abc_modes.py
python run_deny04_enhanced_modes.py

# MNIST Mode B ablations (default: print; --run to execute)
python run_spankey05_mode_b_ablation.py --run --out-dir ./spankey05_runs
```

---

## Adding another example

Copy an existing `examples/<NN>_…` folder, keep the same injection pattern (`forward_with_injection`, basis sampling), and ship `demo.py`, `requirements.txt`, and a short `README` next to it.

---

## Citation

If you use this code, please cite the SpanKey paper (see `paper/` draft or the project page) and link this repository: [https://github.com/mindmemory-ai/dksc](https://github.com/mindmemory-ai/dksc).
