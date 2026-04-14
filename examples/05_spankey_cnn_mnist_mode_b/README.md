# 05 — MNIST SmallCNN + Mode B（拒绝类）

与仓库根 `README` 一致：训练后用 `--ckpt` 保存权重与基矩阵，便于复现与**安全性评估脚本**。

## 训练

```bash
pip install -r ../02_spankey_cnn_mnist/requirements.txt   # 或与 02 相同依赖
python demo.py --epochs 20 --ckpt ./mode_b.pt
```

常用参数见 `demo.py --help`（`--m`、`--gamma`、`--inject_layers`、拒损等）。

## 攻击评估（`security_attacks_05.py`）

在**已知** $B$、注入层与 $\gamma$ 的白盒设定下，对**子空间系数** $\alpha$ 做三类试探（用于论文中的**实证**讨论，非形式化证明）：

| 模式 | 含义 |
|------|------|
| `adaptive` | 子空间内随机搜索多组 $\alpha$，取测试集上语义准确率最高者（adaptive key search）。 |
| `gradient` | 白盒梯度：对 $\alpha$ 最小化一批样本上的 `CrossEntropy(logits, y)`（gradient-based attack）。 |
| `blackbox` | 仅前向、无反传：随机查询若干 $\alpha$，预算可与 adaptive 对齐（black-box query attack）。 |

```bash
python security_attacks_05.py --ckpt ./mode_b.pt --attack all --json_out ./attacks.json
```

**限制：** checkpoint 须为 `per_layer_key=False` 且 `per_layer_basis=False`（单一共享 $B$ 与单一 $\alpha$）。

**解读提示：** 若模型对几乎所有子空间内密钥都给出高语义准确率，则 adaptive / random in-span 都会偏高——此时应同时对照「无钥 / 子空间外随机钥」与论文中的威胁模型叙述。

论文「威胁模型与安全讨论」节中示例~05 探针表之数值，可由本目录 `security_attacks_05.py` 在固定 checkpoint 上复现；若本地存在 `paper/artifacts/spankey05_attacks.json` 则为一次完整运行之原始输出。
