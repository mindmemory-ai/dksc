# 04_spankey_resnet18_cifar10：key 注入 + ResNet18 + CIFAR-10

比 03 更复杂的示例：采用 **CIFAR-10（3×32×32）** 与 **ResNet18**，并在多个 stage 做 SpanKey 注入，观察无 key / 正确 key / 错误 key 的差异。

与 01/02/03 一致：
- 训练时仅用 **正确 key（span 内）**；
- 评估对比 **No key / Correct key / Wrong key（span 外）**；
- 支持 `add` / `mul`、多层注入、`per_layer_key`、`per_layer_basis`；
- 默认使用 CUDA（可用 `--cpu` 强制 CPU）。

## 依赖

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python demo.py
```

更强访问控制（建议从 mul 开始）。默认 **100 epoch、SGD（momentum）+ MultiStepLR、训练集 RandomCrop/Flip 增强**，与论文表一致：

```bash
python demo.py --inject mul --gamma 2.0 --target_key_std 1.0 \
  --inject_layers 0,2,3,4 --per_layer_key --per_layer_basis \
  --epochs 100 --optimizer sgd --lr 0.1 --weight_decay 5e-4 --milestones 50,75
```

若需旧版 Adam 行为，可 `--optimizer adam --lr 1e-3`。`--no_train_aug` 可关掉数据增强。

训练结束后先打印 **训练集**、再打印 **测试集** 上的 No key / Correct key / Wrong key。

## 拒绝损失（`--deny_mode`）

| 模式 | 含义 |
|------|------|
| `none` | 无拒绝项 |
| `A` | 错钥（及可选无钥）softmax 熵最大化（与正钥尖峰易冲突） |
| `A_soft` | 仅当熵低于 `log(C)-gap` 时惩罚（`--a_soft_gap`），减轻与主任务的拉扯 |
| `C` |  hinge：正钥与错钥在**真类** logit 上拉开 |
| `cplus` | `C` 再加一项：错钥时要求 **max_{c≠y} z_c** 高于 `z_y`（`--deny_margin2`） |
| `AC` | `A_soft` 与 `C` 各 0.5 混合 |
| `B` | 分类头改为 **11 类**，第 11 维为 reject（与论文 Mode B 一致） |
| `B_aux` | **仍为 10 类语义头**，另加 `fc_aux: 512→1` 标量拒绝 logit；评估时额外打印三条路径上的 `σ(reject)` |

通用参数：`--deny_weight`、`--deny_margin`、`--deny_on wrong|wrong+no`、`--deny_warmup_epochs`（deny 权重从 0 线性升满）。

示例（与论文表类似的强设定 + 改进版 deny）：

```bash
python demo.py --inject mul --gamma 2.0 --target_key_std 1.0 \
  --inject_layers 0,2,3,4 --per_layer_key --per_layer_basis \
  --epochs 100 --optimizer sgd --lr 0.1 --weight_decay 5e-4 --milestones 50,75 \
  --deny_mode cplus --deny_weight 0.1 --deny_margin 1.0 --deny_margin2 0.5 --deny_warmup_epochs 10
```

## 训练过程日志（`--train_log`）

将每个 optimizer step 的标量写入 **JSONL**（一行一个 JSON），便于画曲线或检查「密钥尺度 / 梯度 / batch 准确率」是否异常。

```bash
python demo.py --epochs 20 --train_log ./logs/metrics.jsonl
# 若文件过大：--log_interval 10 表示每 10 个 global_step 记一条
# 略快、省磁盘：--no_log_grad_norm
```

典型字段：`epoch`, `batch`, `global_step`, `lr`, `loss`, `ce_loss`, `deny_loss`（启用拒绝损失时）, `batch_acc`, `grad_norm`, `k_mean_abs`, `gamma_k_mean_abs`。另有 `run_start` / `epoch_end` / `run_end` 元数据行。

## 注入层（`--inject_layers`）

- 0：输入图像 `(3, 32, 32)`
- 1：stem 后特征（conv1/bn/relu）`(64, 32, 32)`
- 2：layer2 后特征 `(128, 16, 16)`
- 3：layer3 后特征 `(256, 8, 8)`
- 4：layer4 后特征 `(512, 4, 4)`

