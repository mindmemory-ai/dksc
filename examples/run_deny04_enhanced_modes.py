#!/usr/bin/env python3
"""
Sweep Example 04 with *extended* deny modes: A_soft, cplus, AC, B_aux (+ baseline).

对应 QS「第六部分」与正文 \\ref{sec:enhanced_deny}；表数据见 deny04_results_40ep.md。
典型设置: 40 epoch，MultiStepLR milestones 20,30（与 100ep/50,75 成比例），λ_deny=0.1。

用法:
  python run_deny04_enhanced_modes.py
  python run_deny04_enhanced_modes.py --run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(ROOT, "04_spankey_resnet18_cifar10")

# 与 deny04_results_40ep.md 对齐的短程训练
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
    "40",
    "--milestones",
    "20,30",
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
    "--deny_margin2",
    "0.5",
    "--a_soft_gap",
    "0.3",
]

RUNS: list[tuple[str, list[str]]] = [
    ("baseline", COMMON + ["--deny_mode", "none"]),
    ("a_soft", COMMON + ["--deny_mode", "a_soft"]),
    ("cplus", COMMON + ["--deny_mode", "cplus", "--deny_warmup_epochs", "4"]),
    ("ac", COMMON + ["--deny_mode", "ac", "--deny_warmup_epochs", "0"]),
    ("b_aux", COMMON + ["--deny_mode", "b_aux"]),
]


def main() -> None:
    p = argparse.ArgumentParser(description="Example 04: extended deny variants")
    p.add_argument("--run", action="store_true", help="execute subprocess runs")
    args = p.parse_args()

    demo_py = os.path.join(DEMO_DIR, "demo.py")
    if not os.path.isfile(demo_py):
        print(f"Missing {demo_py}", file=sys.stderr)
        sys.exit(1)

    for name, cmd in RUNS:
        line = f"cd {DEMO_DIR} && " + " ".join(cmd)
        print(f"\n=== {name} ===\n{line}\n", flush=True)
        if args.run:
            r = subprocess.run(cmd, cwd=DEMO_DIR)
            if r.returncode != 0:
                print(f"FAILED: {name}", file=sys.stderr)
                sys.exit(r.returncode)


if __name__ == "__main__":
    main()
