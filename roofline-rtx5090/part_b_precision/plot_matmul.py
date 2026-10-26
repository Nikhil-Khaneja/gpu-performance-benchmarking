"""Part B: plot achieved TFLOPS vs matrix size, one line per precision, on a
single figure. Also emits a %-of-peak version of the same plot.

Usage: python part_b_precision/plot_matmul.py [--csv results/matmul_precision.csv]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "results", "matmul_precision.csv")
FIG_DIR = os.path.join(ROOT, "figures")


def load_rows(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r["status"] == "ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        print(f"No 'ok' rows found in {args.csv}. Run benchmark_matmul.py first.")
        sys.exit(1)

    by_precision = defaultdict(list)
    for r in rows:
        by_precision[r["precision"]].append((int(r["n"]), float(r["tflops"]), float(r["pct_of_peak"])))
    for k in by_precision:
        by_precision[k].sort()

    gpu_name = rows[0]["gpu_name"]
    gpu_uuid = rows[0]["gpu_uuid"]

    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for precision, points in sorted(by_precision.items()):
        ns = [p[0] for p in points]
        tf = [p[1] for p in points]
        ax.plot(ns, tf, marker="o", label=precision)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Matrix size N")
    ax.set_ylabel("Achieved TFLOPS")
    ax.set_title(f"Matmul throughput vs N -- {gpu_name}\nUUID {gpu_uuid}", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3)
    out1 = os.path.join(FIG_DIR, f"partB_tflops_vs_n_{gpu_uuid.replace('-', '')}.png")
    fig.tight_layout()
    fig.savefig(out1, dpi=150)
    print(f"wrote {out1}")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for precision, points in sorted(by_precision.items()):
        ns = [p[0] for p in points]
        pct = [p[2] for p in points]
        ax2.plot(ns, pct, marker="o", label=precision)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Matrix size N")
    ax2.set_ylabel("% of theoretical peak")
    ax2.set_title(f"Matmul efficiency vs N -- {gpu_name}\nUUID {gpu_uuid}", fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    out2 = os.path.join(FIG_DIR, f"partB_pct_peak_vs_n_{gpu_uuid.replace('-', '')}.png")
    fig2.tight_layout()
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")

    print("\nPlateau check (report in METRICS.md): for each precision, the N at")
    print("which TFLOPS stops increasing materially is its plateau. Small N never")
    print("reaches it because launch/kernel overhead and imperfect tile utilization")
    print("dominate before the GEMM is large enough to saturate all SMs.")


if __name__ == "__main__":
    main()
