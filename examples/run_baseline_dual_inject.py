#!/usr/bin/env python3
"""Run 01–04 demos with --inject add and --inject mul; parse train/test no/correct/wrong acc."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any

ROOT = os.path.dirname(os.path.abspath(__file__))

RUNS: list[tuple[str, list[str]]] = [
    ("01_spankey_mlp", ["python", "demo.py", "--cuda"]),
    ("02_spankey_cnn_mnist", ["python", "demo.py"]),
    ("03_spankey_resnet_fashionmnist", ["python", "demo.py"]),
    ("04_spankey_resnet18_cifar10", ["python", "demo.py"]),
]


def parse_metrics(stdout: str) -> dict[str, Any]:
    matches = re.findall(r"acc = ([0-9.]+)", stdout)
    if len(matches) < 6:
        raise ValueError(f"expected >=6 'acc =' lines, got {len(matches)}")
    last6 = [float(x) for x in matches[-6:]]
    return {
        "train": {
            "acc_no_key": last6[0],
            "acc_correct_key": last6[1],
            "acc_wrong_key": last6[2],
        },
        "test": {
            "acc_no_key": last6[3],
            "acc_correct_key": last6[4],
            "acc_wrong_key": last6[5],
        },
    }


def main() -> None:
    out: dict[str, Any] = {"runs": []}
    for name, base_cmd in RUNS:
        for inj in ("add", "mul"):
            cmd = base_cmd + ["--inject", inj]
            cwd = os.path.join(ROOT, name)
            print(f"\n=== {name} --inject {inj} ===", flush=True)
            r = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=7200,
            )
            log = (r.stdout or "") + (r.stderr or "")
            if r.returncode != 0:
                print(log[-4000:], file=sys.stderr)
                raise SystemExit(f"{name} {inj} failed with code {r.returncode}")
            m = parse_metrics(log)
            row = {
                "example": name.split("_")[0],
                "dir": name,
                "inject": inj,
                **m,
            }
            out["runs"].append(row)
            print(json.dumps(row, indent=2), flush=True)

    path = os.path.join(ROOT, "baseline_dual_inject_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {path}", flush=True)


if __name__ == "__main__":
    main()
