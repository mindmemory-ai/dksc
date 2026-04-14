# Example 04 deny 对比（40 epoch 对齐实验）

设定：`--inject mul --gamma 2.0 --target_key_std 1.0 --inject_layers 0,2,3,4 --per_layer_key --per_layer_basis --seed 42`，SGD `lr=0.1, wd=5e-4`，**40 epochs**，`MultiStepLR` milestones **20,30**（与 100ep/50,75 成比例）。`λ_deny=0.1`（baseline 为 0）。

| Variant | Train: No / OK / Wrong | Test: No / OK / Wrong | 备注 |
|---------|------------------------|----------------------|------|
| Baseline | 0.8300 / 0.9386 / 0.9146 | 0.8178 / 0.8825 / 0.8593 | 无 deny |
| A_soft | 0.4976 / 0.8875 / 0.6973 | 0.5467 / 0.8517 / 0.6837 | gap=0.3 |
| cplus | 0.4345 / 0.9199 / 0.5351 | 0.4410 / 0.8649 / **0.5094** | m=1, m2=0.5, warmup=4 |
| AC | 0.6417 / 0.9047 / 0.7780 | 0.6785 / 0.8596 / 0.7629 | A_soft+C 各半 |
| B_aux | 0.7841 / 0.9373 / 0.9121 | 0.7860 / 0.8821 / 0.8615 | 见下 σ(reject) |

**B_aux** 辅助头（test）：`σ(reject)` — no key **0.9795**，wrong **0.9802**，correct key **0.0061**。语义 top-1 仍为 10 类 argmax，故 headline acc 与 baseline 接近；拒绝信号见辅助 logit。

日志：`experiment/logs/train_*.jsonl`，脚本：`experiment/run_deny04_sweep.sh`、`run_remaining_variants.sh`。
