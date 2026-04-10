# 03_spankey_resnet_fashionmnist：key 注入 + ResNet-like + FashionMNIST

更复杂网络（带残差块）+ 更复杂数据（FashionMNIST），用于对比 01/02 在更高表达能力模型下的 SpanKey 注入表现。

与 01/02 一致：
- 训练时 **只用正确 key（span 内）**；
- 评估对比 **无 key / 正确 key / 错误 key（span 外）**；
- 支持 `add` / `mul` 注入，多层注入、`per_layer_key`、`per_layer_basis`；
- 默认启用 CUDA，可用 `--cpu` 强制 CPU。

## 依赖

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python demo.py
```

更强访问控制（建议从 mul 开始）：

```bash
python demo.py --inject mul --gamma 2.0 --target_key_std 1.0 \
  --inject_layers 0,1,2,3 --per_layer_key --per_layer_basis --epochs 20
```

写出调试统计（首轮首 batch）：

```bash
python demo.py --debug_file debug_values.json
```

## 注入层说明（`--inject_layers`）

- 0：输入图像 `(1, 28, 28)`
- 1：stem 后特征 `(32, 28, 28)`
- 2：layer2 后特征 `(64, 14, 14)`
- 3：layer3 后特征 `(128, 7, 7)`

key 会按各层形状 reshape 后逐元素 add/mul。

