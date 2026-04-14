#!/usr/bin/env python3
"""
Sweep Example 04 (CIFAR-10 ResNet-18) with deny modes: none (baseline), a, b, c.

对照论文「模式 A / B / C」；数据已写入正文表（见 paper/backups 归档）。本脚本仅生成可复现命令行，默认不执行训练（DRY_RUN）。

用法:
  python run_deny04_abc_modes.py           # 打印命令
  python run_deny04_abc_modes.py --run     # 实际依次 subprocess（耗时长）

与归档表 tab:deny04_* 对齐的典型超参（见 main_zh 表注）:
  --inject mul --gamma 2.0 --target_key_std 1.0
  --inject_layers 0,2,3,4 --per_layer_key --per_layer_basis
  --epochs 100 --milestones 50,75   # 与默认 04 demo 一致；归档拒绝实验亦在此规程下采集
  --deny_weight 0.1 --deny_on wrong
  模式 B 需网络为 (C+1) 类头: --deny_mode b（demo 内自动换 build_resnet18_cifar10_with_reject）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(ROOT, "04_spankey_resnet18_cifar10")
DEMO = os.path.join(DEMO_DIR, "demo.py")

# 与开源 demo 及归档实验可比的公共前缀（可按机器改 --epochs / --cuda）
COMMON = [
    "python",
    "demo.py",
    "--inject",
    "mul",
    "--gamma",
    "2.0",
    "--target_key_std",
    "1.0",
    "--inject_layers",
    "0,2,3,4",
    "--per_layer_key",
    "--per_layer_basis",
    "--seed",
    "42",
    "--epochs",
    "100",
    "--milestones",
    "50,75",
    "--optimizer",
    "sgd",
    "--lr",
    "0.1",
    "--weight_decay",
    "5e-4",
    "--deny_weight",
    "0.1",
    "--deny_on",
    "wrong",
    "--deny_margin",
    "1.0",
]

RUNS: list[tuple[str, list[str]]] = [
    ("baseline", COMMON + ["--deny_mode", "none"]),
    ("mode_a", COMMON + ["--deny_mode", "a"]),
    ("mode_b", COMMON + ["--deny_mode", "b"]),
    ("mode_c", COMMON + ["--deny_mode", "c"]),
]


def main() -> None:
    p = argparse.ArgumentParser(description="Example 04: baseline vs deny modes A/B/C")
    p.add_argument(
        "--run",
        action="store_true",
        help="actually execute each command (long GPU run)",
    )
    args = p.parse_args()

    for name, cmd in RUNS:
        line = "cd {} && {}".format(DEMO_DIR, " ".join(cmd))
        print(f"\n=== {name} ===\n{line}\n", flush=True)
        if args.run:
            r = subprocess.run(cmd, cwd=DEMO_DIR)
            if r.returncode != 0:
                print(f"FAILED: {name} exit {r.returncode}", file=sys.stderr)
                sys.exit(r.returncode)


if __name__ == "__main__":
    main()
