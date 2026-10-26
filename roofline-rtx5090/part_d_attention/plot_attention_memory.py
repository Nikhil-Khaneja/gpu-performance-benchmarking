"""Part D: plot peak memory vs seq_len for naive vs fused attention, fit a
quadratic to the naive curve and report the measured coefficient (rather
than asserting O(seq^2) a priori), and print latency speedup fused-vs-naive
at each seq_len both implementations completed.

Usage: python part_d_attention/plot_attention_memory.py
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "results", "attention.csv")
FIG_DIR = os.path.join(ROOT, "figures")


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        print(f"No 'ok' rows in {args.csv}. Run benchmark_attention.py first.")
        sys.exit(1)

    def config_key(r):
        return (r["batch"], r["heads"], r["head_dim"], r["dtype"])

    # results/attention.csv is append-only, so reruns with different
    # --batch/--heads/--head-dim/--dtype leave old rows in the file. Only
    # plot the most recently appended config -- mixing configs silently
    # wrecks the quadratic fit.
    latest_config = config_key(ok_rows[-1])
    dropped = [r for r in ok_rows if config_key(r) != latest_config]
    ok_rows = [r for r in ok_rows if config_key(r) == latest_config]
    if dropped:
        b, h, hd, dt = latest_config
        print(f"Note: ignoring {len(dropped)} row(s) from other batch/heads/head_dim/dtype "
              f"configs in {args.csv}; plotting only the latest config "
              f"(batch={b} heads={h} head_dim={hd} dtype={dt}).\n")

    by_impl = defaultdict(list)
    for r in ok_rows:
        by_impl[r["impl"]].append((int(r["seq_len"]), float(r["peak_mem_bytes"]), float(r["median_latency_ms"])))
    for k in by_impl:
        by_impl[k].sort()

    gpu_name = ok_rows[0]["gpu_name"]
    gpu_uuid = ok_rows[0]["gpu_uuid"]
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    coeffs_report = {}
    for impl, points in sorted(by_impl.items()):
        seqs = np.array([p[0] for p in points], dtype=float)
        mem_gb = np.array([p[1] for p in points], dtype=float) / 1e9
        ax.plot(seqs, mem_gb, marker="o", label=f"{impl} (measured)")

        if len(seqs) >= 3:
            coeffs = np.polyfit(seqs, mem_gb, 2)  # a*seq^2 + b*seq + c, GB
            a, b, c = coeffs
            fit_seqs = np.linspace(seqs.min(), seqs.max(), 100)
            fit_mem = np.polyval(coeffs, fit_seqs)
            ax.plot(fit_seqs, fit_mem, linestyle="--", alpha=0.6, label=f"{impl} quadratic fit")
            coeffs_report[impl] = (a, b, c)

    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Peak memory (GB)")
    ax.set_title(f"Attention peak memory vs seq_len -- {gpu_name}\nUUID {gpu_uuid}", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(FIG_DIR, f"partD_memory_vs_seqlen_{gpu_uuid.replace('-', '')}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}\n")

    for impl, (a, b, c) in coeffs_report.items():
        print(f"[{impl}] quadratic fit: memory_GB = {a:.3e}*seq^2 + {b:.3e}*seq + {c:.3e}")
        if impl == "naive":
            print(f"  -> measured quadratic coefficient a = {a:.3e} GB per (token^2); "
                  f"report this as your confirmation of the O(seq^2) term.")

    print("\nLatency speedup (fused vs naive), at seq_lens both completed:")
    naive_by_seq = {s: lat for s, _, lat in by_impl.get("naive", [])}
    fused_by_seq = {s: lat for s, _, lat in by_impl.get("fused", [])}
    for seq in sorted(set(naive_by_seq) & set(fused_by_seq)):
        speedup = naive_by_seq[seq] / fused_by_seq[seq]
        print(f"  seq={seq}: naive={naive_by_seq[seq]:.3f}ms  fused={fused_by_seq[seq]:.3f}ms  "
              f"speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
