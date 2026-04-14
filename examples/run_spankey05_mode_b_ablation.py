#!/usr/bin/env python3
"""
示例 05（MNIST SmallCNN + 模式 B）消融脚本。

按 QS「一点补充建议」组织三组单因素扫描（其余超参固定，便于看曲线）：
  1) 子空间维数 m
  2) 注入层 / 位置（--inject_layers）
  3) 注入强度 γ

默认仅打印命令（与 run_deny04_*.py 一致）；加 --run 则依次执行，并将指标写入 --out-dir 下 JSON。

用法:
  python run_spankey05_mode_b_ablation.py
  python run_spankey05_mode_b_ablation.py --run --out-dir ./spankey05_runs
  python run_spankey05_mode_b_ablation.py --quick --run   # 更少网格点，更快
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(ROOT, "05_spankey_cnn_mnist_mode_b")

# 与 02 可比：mul、默认 20 epoch、较小 CNN；B 方案固定
def _base(data_dir: str) -> list[str]:
    return [
        "python",
        "demo.py",
        "--inject",
        "mul",
        "--epochs",
        "20",
        "--batch_size",
        "128",
        "--lr",
        "1e-3",
        "--seed",
        "42",
        "--deny_weight",
        "0.1",
        "--deny_on",
        "wrong",
        "--target_key_std",
        "0.5",
        "--data_dir",
        data_dir,
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="SpanKey 05 Mode B ablation (m, layers, gamma)")
    ap.add_argument("--run", action="store_true", help="execute each command")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="if set with --run, pass --json_out DEMO_DIR/<name>.json for each run",
    )
    ap.add_argument("--data_dir", type=str, default="./data", help="MNIST root (same as demo)")
    ap.add_argument("--cpu", action="store_true", help="pass --cpu to demo")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="smaller grids (fewer runs)",
    )
    args = ap.parse_args()

    # 固定「锚点」超参（单因素扫描时其余量保持不变）
    anchor_gamma = 0.5
    anchor_layers = "0,2"
    anchor_m = 8

    if args.quick:
        grid_m = [4, 16]
        grid_layers = ["0", "0,2", "0,1,2"]
        grid_gamma = [0.25, 1.0]
    else:
        grid_m = [4, 8, 16, 32]
        grid_layers = ["0", "1", "2", "0,1", "0,2", "0,1,2"]
        grid_gamma = [0.25, 0.5, 1.0, 2.0]

    suites: list[tuple[str, list[tuple[str, list[str]]]]] = []

    # 1) sweep m
    runs_m: list[tuple[str, list[str]]] = []
    for m in grid_m:
        name = f"ab_m{m}_layers{anchor_layers.replace(',', '-')}_g{anchor_gamma}"
        cmd = _base(args.data_dir) + [
            "--m",
            str(m),
            "--inject_layers",
            anchor_layers,
            "--gamma",
            str(anchor_gamma),
        ]
        if args.cpu:
            cmd.append("--cpu")
        runs_m.append((name, cmd))
    suites.append(("A_sweep_subspace_dim_m", runs_m))

    # 2) sweep inject_layers
    runs_l: list[tuple[str, list[str]]] = []
    for layers in grid_layers:
        safe = layers.replace(",", "-")
        name = f"ab_layers{safe}_m{anchor_m}_g{anchor_gamma}"
        cmd = _base(args.data_dir) + [
            "--m",
            str(anchor_m),
            "--inject_layers",
            layers,
            "--gamma",
            str(anchor_gamma),
        ]
        if args.cpu:
            cmd.append("--cpu")
        runs_l.append((name, cmd))
    suites.append(("B_sweep_inject_layers", runs_l))

    # 3) sweep gamma
    runs_g: list[tuple[str, list[str]]] = []
    for g in grid_gamma:
        name = f"ab_gamma{g}_m{anchor_m}_layers{anchor_layers.replace(',', '-')}"
        cmd = _base(args.data_dir) + [
            "--m",
            str(anchor_m),
            "--inject_layers",
            anchor_layers,
            "--gamma",
            str(g),
        ]
        if args.cpu:
            cmd.append("--cpu")
        runs_g.append((name, cmd))
    suites.append(("C_sweep_inject_strength_gamma", runs_g))

    os.makedirs(DEMO_DIR, exist_ok=True)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    for suite_title, runs in suites:
        print(f"\n######## {suite_title} ########\n", flush=True)
        for name, cmd in runs:
            line = "cd {} && {}".format(DEMO_DIR, " ".join(cmd))
            if args.out_dir and args.run:
                json_path = os.path.join(os.path.abspath(args.out_dir), f"{name}.json")
                cmd = cmd + ["--json_out", json_path]
                line = "cd {} && {}".format(DEMO_DIR, " ".join(cmd))
            print(f"=== {name} ===\n{line}\n", flush=True)
            if args.run:
                r = subprocess.run(cmd, cwd=DEMO_DIR)
                if r.returncode != 0:
                    print(f"FAILED: {name} exit {r.returncode}", file=sys.stderr)
                    sys.exit(r.returncode)

    if not args.run:
        print(
            "\n提示: 单因素扫描时固定锚点 "
            f"m={anchor_m}, inject_layers={anchor_layers}, γ={anchor_gamma}（各组对应一项变化）。"
            " 使用 --run 执行；GPU 可去掉 demo 的 --cpu。",
            flush=True,
        )


if __name__ == "__main__":
    main()
