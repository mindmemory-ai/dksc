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

## 注入层（`--inject_layers`）

- 0：输入图像 `(3, 32, 32)`
- 1：stem 后特征（conv1/bn/relu）`(64, 32, 32)`
- 2：layer2 后特征 `(128, 16, 16)`
- 3：layer3 后特征 `(256, 8, 8)`
- 4：layer4 后特征 `(512, 4, 4)`

