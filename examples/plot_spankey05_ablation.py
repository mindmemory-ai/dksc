#!/usr/bin/env python3
"""
从 05_spankey_ablation_json/*.json 读取消融结果，生成论文用多子图（科研风格）。

输出（默认写入 ../../paper/figures/）:
  spankey05_ablation.png  — 位图（paper/.gitignore 忽略 *.pdf 时仍可用 png 纳入版本库）
  spankey05_ablation.pdf  — 矢量；需时 git add -f

用法:
  python plot_spankey05_ablation.py
  python plot_spankey05_ablation.py --json-dir ./05_spankey_ablation_json --out-dir ../../paper/figures
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "05_spankey_ablation_json"),
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "paper", "figures"),
    )
    args = ap.parse_args()
    jdir = Path(args.json_dir).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    by_m: dict[int, dict] = {}
    by_gamma: dict[float, dict] = {}
    by_layers: dict[str, dict] = {}

    for p in sorted(jdir.glob("*.json")):
        data = load_json(p)
        name = p.stem
        te = data["metrics_test"]
        if name.startswith("ab_m") and "_layers" in name:
            m = int(re.match(r"ab_m(\d+)_", name).group(1))
            by_m[m] = te
        elif name.startswith("ab_gamma"):
            g = float(re.match(r"ab_gamma([\d.]+)_", name).group(1))
            by_gamma[g] = te
        elif name.startswith("ab_layers"):
            rest = name[len("ab_layers") :].split("_m")[0]
            by_layers[rest] = te

    ms = sorted(by_m.keys())
    gammas = sorted(by_gamma.keys())
    layer_order = ["0", "1", "2", "0-1", "0-2", "0-1-2"]
    layer_labels = ["0", "1", "2", "0,1", "0,2", "0,1,2"]
    layers_present = [k for k in layer_order if k in by_layers]

    y1 = "acc_semantic_correct_key"
    y2 = "acc_semantic_no_key"
    y3 = "reject_frac_wrong_key"

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.25))
    cols = ["#0173B2", "#DE8F05", "#029E73"]

    # (A) m
    ax = axes[0]
    ax.plot(ms, [by_m[m][y1] for m in ms], "o-", color=cols[0], lw=1.8, ms=5, label="Semantic acc. (correct key)")
    ax.plot(ms, [by_m[m][y2] for m in ms], "s--", color=cols[1], lw=1.5, ms=4, label="Semantic acc. (no key)")
    ax.plot(ms, [by_m[m][y3] for m in ms], "^-", color=cols[2], lw=1.5, ms=4, label="Reject rate (wrong key)")
    ax.set_xlabel(r"Subspace dimension $m$")
    ax.set_ylabel("Rate / accuracy")
    ax.set_title(r"(A) Vary $m$ ($\gamma=0.5$, layers $=0,2$)")
    ax.set_xticks(ms)
    ax.set_ylim(-0.05, 1.05)

    # (B) layers
    ax = axes[1]
    xs = list(range(len(layers_present)))
    ax.plot(xs, [by_layers[k][y1] for k in layers_present], "o-", color=cols[0], lw=1.8, ms=5)
    ax.plot(xs, [by_layers[k][y2] for k in layers_present], "s--", color=cols[1], lw=1.5, ms=4)
    ax.plot(xs, [by_layers[k][y3] for k in layers_present], "^-", color=cols[2], lw=1.5, ms=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([layer_labels[layer_order.index(k)] for k in layers_present], rotation=22, ha="right")
    ax.set_xlabel(r"Injection layer indices")
    ax.set_ylabel("Rate / accuracy")
    ax.set_title(r"(B) Vary layers ($m=8$, $\gamma=0.5$)")
    ax.set_ylim(-0.05, 1.05)

    # (C) gamma
    ax = axes[2]
    ax.plot(gammas, [by_gamma[g][y1] for g in gammas], "o-", color=cols[0], lw=1.8, ms=5)
    ax.plot(gammas, [by_gamma[g][y2] for g in gammas], "s--", color=cols[1], lw=1.5, ms=4)
    ax.plot(gammas, [by_gamma[g][y3] for g in gammas], "^-", color=cols[2], lw=1.5, ms=4)
    ax.set_xlabel(r"Injection strength $\gamma$")
    ax.set_ylabel("Rate / accuracy")
    ax.set_title(r"(C) Vary $\gamma$ ($m=8$, layers $=0,2$)")
    ax.set_xticks(gammas)
    ax.set_ylim(-0.05, 1.05)

    handles = [
        plt.Line2D([0], [0], color=cols[0], marker="o", ls="-", lw=1.8, ms=5),
        plt.Line2D([0], [0], color=cols[1], marker="s", ls="--", lw=1.5, ms=4),
        plt.Line2D([0], [0], color=cols[2], marker="^", ls="-", lw=1.5, ms=4),
    ]
    labels = [
        "Semantic acc. (correct key)",
        "Semantic acc. (no key)",
        "Reject rate (wrong key)",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.subplots_adjust(bottom=0.28, wspace=0.28)

    out_png = out / "spankey05_ablation.png"
    out_pdf = out / "spankey05_ablation.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out_png}\nWrote {out_pdf}")


if __name__ == "__main__":
    main()
