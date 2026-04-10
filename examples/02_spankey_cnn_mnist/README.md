# 02_spankey_cnn_mnist：key 注入 + CNN + MNIST 数字识别

与 01 同模式：基向量 B、动态 key（span 内）、在指定层做 key 注入，用 CNN 做手写数字分类，评估无 key / 正确 key / 错误 key 的准确率。**默认使用 CUDA**，可用 `--cpu` 强制 CPU。

## 依赖

```bash
pip install -r requirements.txt
```

## 数据

自动下载 MNIST 到 `--data_dir`（默认 `./data`）。

## 运行

```bash
python demo.py
```

可选参数示例：

```bash
# 注入方式：add（对输入加法）| mul（对输入乘法）
python demo.py --inject add --gamma 0.5 --epochs 20
python demo.py --inject mul

# 指定注入层：0=输入 1=pool1 后 2=pool2 后（逗号分隔）
python demo.py --inject_layers 0,1,2 --epochs 30

# 每层不同 key（不同 α）| 每层独立基空间（每层一个 B）
python demo.py --inject_layers 0,1,2 --per_layer_key --epochs 40
python demo.py --inject_layers 0,1,2 --per_layer_basis --epochs 40

# 强制 CPU（默认启用 CUDA）
python demo.py --cpu

# B 不按数据尺度缩放（沿用 randn+正交化，key 可能偏小）
python demo.py --no_scale_b_to_data

# key 与归一化数据同量级：--target_key_std 1.0；更强访问控制可试 mul + gamma 2
python demo.py --inject mul --gamma 2.0 --target_key_std 1.0 --inject_layers 0,1,2 --per_layer_key --per_layer_basis --epochs 40
```

## 基向量 B 与 key 尺度

默认 **scale_to_data=True**：B 元素在 `[-1/√m, 1/√m]`，使 k 的 std 与 m 无关（约 0.3~0.5）。生成 k 后再按 **`--target_key_std`**（默认 0.5）缩放到与归一化数据同量级；可设为 1.0 增强 key 作用。用 `--no_scale_b_to_data` 可恢复为 randn+正交化。

## 模型与注入

- **模型**：小型 CNN（2 层 Conv + MaxPool + Linear），输入 1×28×28，输出 10 类。
- **注入层**（`--inject_layers`）：0=输入 1×28×28，1=pool1 后 16×14×14，2=pool2 后 32×7×7；key 按层维度 reshape 后与激活做 add/mul。
- **`--per_layer_key`**：每层用不同 α 生成 key，减少“记一层代替所有层”。
- **`--per_layer_basis`**：多层注入时每层使用独立基矩阵 B（不同子空间），与 01 行为一致。

## 预期现象

- **正确 key**（span 内）：准确率高。
- **无 key / 错误 key**：准确率明显低于正确 key，体现访问控制。
