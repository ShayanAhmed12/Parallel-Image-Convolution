#!/usr/bin/env python3
"""Plot speedup (T1/Tp) and efficiency (speedup/p) from pdc_conv CSV output.

usage: python3 scripts/plot_results.py [results_bench.csv] [results_batch.csv] [outdir]
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bench_csv = sys.argv[1] if len(sys.argv) > 1 else "results_bench.csv"
batch_csv = sys.argv[2] if len(sys.argv) > 2 else "results_batch.csv"
outdir = sys.argv[3] if len(sys.argv) > 3 else "plots"
os.makedirs(outdir, exist_ok=True)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def bench_plots(rows):
    # data[kernel][variant] -> list of (threads, speedup, efficiency)
    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r["kernel"]][r["variant"]].append(
            (int(r["threads"]), float(r["speedup_vs_seq"]), float(r["efficiency"])))
    kernels = list(data)
    for metric, idx, ylabel, fname in (
        ("Speedup  T1 / Tp", 1, "speedup", "speedup.png"),
        ("Efficiency  speedup / p", 2, "efficiency", "efficiency.png"),
    ):
        fig, axes = plt.subplots(1, len(kernels), figsize=(5.2 * len(kernels), 4.4), squeeze=False)
        for ax, k in zip(axes[0], kernels):
            all_p = sorted({t for v in data[k].values() for t, *_ in v if t >= 1})
            for variant, pts in data[k].items():
                pts = sorted(pts)
                if variant == "seq":
                    continue
                if len(pts) == 1:  # sequential variants (seq_sep): horizontal reference line
                    ax.axhline(pts[0][idx], ls=":", lw=1.2, label=f"{variant} (1 thread)")
                else:
                    ax.plot([p[0] for p in pts], [p[idx] for p in pts], marker="o", label=variant)
            if idx == 1:
                ax.plot(all_p, all_p, "k--", lw=1, label="ideal")
            else:
                ax.axhline(1.0, color="k", ls="--", lw=1, label="ideal")
            ax.set_title(k)
            ax.set_xlabel("threads")
            ax.set_ylabel(ylabel)
            ax.set_xticks(all_p)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(metric)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close(fig)
        print("wrote", os.path.join(outdir, fname))


def batch_plot(rows):
    data = defaultdict(list)
    for r in rows:
        if r["strategy"] == "seq":
            continue
        data[r["strategy"]].append((int(r["threads"]), float(r["speedup_vs_seq"]),
                                    float(r["images_per_s"])))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ps = sorted({t for v in data.values() for t, *_ in v})
    for strat, pts in data.items():
        pts = sorted(pts)
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=strat)
        axes[1].plot([p[0] for p in pts], [p[2] for p in pts], marker="o", label=strat)
    axes[0].plot(ps, ps, "k--", lw=1, label="ideal")
    axes[0].set_ylabel("speedup vs sequential")
    axes[1].set_ylabel("images / second")
    for ax in axes:
        ax.set_xlabel("threads")
        ax.set_xticks(ps)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Batch of small images: per-image vs per-row parallelism")
    fig.tight_layout()
    path = os.path.join(outdir, "batch.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


if os.path.exists(bench_csv):
    bench_plots(read(bench_csv))
else:
    print("skip: no", bench_csv)
if os.path.exists(batch_csv):
    batch_plot(read(batch_csv))
else:
    print("skip: no", batch_csv)
