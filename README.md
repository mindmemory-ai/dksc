# SpanKey examples

This folder holds small, runnable demos for **SpanKey**: you fix a low-dimensional key subspace (a basis), sample keys inside or outside that subspace, and **inject** them into chosen layers of a network. Training normally uses only **in-span** keys; at evaluation you can measure accuracy with no key, a valid key, or a wrong (out-of-span) key. The point is to study how much behavior depends on the key versus plain input features—not to provide cryptographic guarantees.

The write-up for the project lives in `paper/` (English draft) and the public repo root; this README only orients you to **this directory**.

## Layout

| Directory | Contents |
|-----------|----------|
| `01_spankey_mlp/` | MLP on synthetic vectors; multi-layer injection; optional per-layer keys and per-layer bases. |
| `02_spankey_cnn_mnist/` | Small CNN on MNIST; add or multiply injection at the input-side stack. |
| `03_spankey_resnet_fashionmnist/` | ResNet-style model on FashionMNIST; harder task than MNIST. |
| `04_spankey_resnet18_cifar10/` | ResNet-18 on CIFAR-10; stronger training setup (augmentation, SGD schedule). Optional **deny** losses on wrong-key batches (modes A/B/C) are implemented here only. |

Each numbered folder has its own `demo.py`, `requirements.txt`, and a short `README` with flags and defaults.

**`results_summary.md`** in this same directory collects train/test numbers for examples 01–04 (and deny variants for 04) from GPU runs we logged for the paper tables.

## How to run

```bash
cd 01_spankey_mlp
pip install -r requirements.txt
python demo.py
```

Repeat with the path to `02_…`, `03_…`, or `04_…` as needed. Example 04 expects a CUDA device unless you pass `--cpu`; see `04_spankey_resnet18_cifar10/README.md` for deny options and training switches.

## Adding another example

Copy an existing folder, wire injection the same way (`forward_with_injection`, basis sampling), and keep `README` + `requirements.txt` next to `demo.py` so the pattern stays obvious.
