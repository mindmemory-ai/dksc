# 01_spankey_mlp：key 注入 + MLP + 合成数据

基向量 B、动态 key（span 内）、**注入方式** add/mul，**可指定注入层**（0=输入 1=第一隐层 2=第二隐层），在合成数据上训练并评估无 key / 正确 key / 错误 key 的准确率。

## 依赖

```bash
pip install -r requirements.txt
```

## 运行

```bash
python demo.py
```

可选参数示例：

```bash
# 注入方式：add（默认）| mul
python demo.py --inject add --gamma 0.3 --epochs 80
python demo.py --inject mul

# 指定注入层：0=仅输入层，0,1=输入+第一隐层，0,1,2=全部三层
python demo.py --inject_layers 0
python demo.py --inject_layers 0,1
# 每层用不同 α 生成 key
python demo.py --inject_layers 0,1 --per_layer_key
# 多层注入时每层独立向量空间（每层一个 B）
python demo.py --inject_layers 0,1,2 --per_layer_basis --per_layer_key

python demo.py --d 20 --m 5 --gamma 0.3 --epochs 80
python demo.py --cuda
```

## 注入方式与注入层

| `--inject`（二选一） | 说明 |
|---------------------|------|
| `add` | 对输入做**加法**：x' = x + γ·k |
| `mul` | 对输入做**乘法**：x' = x · (1 + γ·tanh(k))，缩放有界 |

`--inject_layers`：逗号分隔的层索引。0=输入层，1=第一隐层（64 维），2=第二隐层（32 维）。例如 `0,1` 在输入与第一隐层输出处各做一次 key 注入。

`--per_layer_key`：每一注入层用**不同的随机 α** 生成 key，减少冗余、降低吸收。

`--per_layer_basis`：**多层注入**时加此参数，则每层使用**独立向量空间**（每层一个 B，形状 (m, layer_dims[i])），层与层 key 不在同一子空间，进一步降低吸收。

## 预期现象

- **正确 key**（span 内、且为训练时未见过的系数）：准确率应明显高于随机。
- **错误 key**（span 外）：准确率应接近随机猜测 `1/n_classes`。
- **无 key**：未注入 key 时模型未在训练中见过，准确率通常接近随机或较低，体现「无 key 无访问」。

## 输出说明

脚本会打印训练 loss 与评估结果（no key / correct key / wrong key 的 acc 及 random baseline）。
