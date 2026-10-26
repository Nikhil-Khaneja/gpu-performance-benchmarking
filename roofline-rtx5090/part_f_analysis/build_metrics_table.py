"""Part F: pull the headline number for each row of Table HW2.5.1 out of the
CSVs produced by Parts B-E, so every cell is traceable to a run rather than
hand-copied. Prints a Markdown table; paste it into METRICS.md (or redirect
this script's stdout there).

Usage: python part_f_analysis/build_metrics_table.py
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def load(path):
    p = os.path.join(RESULTS, path)
    if not os.path.isfile(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def main():
    matmul = [r for r in load("matmul_precision.csv") if r["status"] == "ok"]
    bw = load("bandwidth_compute.csv")
    attn_oom = load("attention_oom_boundary.csv")
    thermal_summary_path = os.path.join(RESULTS, "thermal_throughput.csv")

    rows = {}

    bf16_rows = [r for r in matmul if r["precision"] == "bf16"]
    if bf16_rows:
        best = max(bf16_rows, key=lambda r: float(r["tflops"]))
        rows["Peak achieved TFLOPS (BF16)"] = (f"{float(best['tflops']):.2f}", f"at N={best['n']}")
        if best["pct_of_peak"]:
            rows["% of theoretical peak (BF16)"] = (f"{float(best['pct_of_peak']):.1f}%", f"at N={best['n']}")
        else:
            rows["% of theoretical peak (BF16)"] = ("--", "no peak spec on file for this GPU")
    else:
        rows["Peak achieved TFLOPS (BF16)"] = ("--", "run benchmark_matmul.py")
        rows["% of theoretical peak (BF16)"] = ("--", "run benchmark_matmul.py")

    bw_rows = [r for r in bw if r["kernel"] == "elementwise_add"]
    if bw_rows:
        r = bw_rows[-1]
        rows["Effective bandwidth (GB/s)"] = (
            f"{float(r['bandwidth_gbps']):.1f}",
            f"{float(r['pct_of_peak_bandwidth']):.1f}% of spec, {r['dtype']}",
        )
    else:
        rows["Effective bandwidth (GB/s)"] = ("--", "run benchmark_bandwidth_compute.py")

    # attention_oom_boundary.csv is append-only; take the most recently
    # appended row per impl, not the first (a rerun with different
    # batch/heads/head_dim/dtype should supersede an earlier one).
    naive_b = next((r for r in reversed(attn_oom) if r["impl"] == "naive"), None)
    fused_b = next((r for r in reversed(attn_oom) if r["impl"] == "fused"), None)
    rows["Naive attention OOM length"] = (
        (f"{naive_b['smallest_failing_seq_len']}", f"largest ok={naive_b['largest_ok_seq_len']}")
        if naive_b else ("--", "run benchmark_attention.py")
    )
    rows["Fused attention OOM length"] = (
        (f"{fused_b['smallest_failing_seq_len']}", f"largest ok={fused_b['largest_ok_seq_len']}")
        if fused_b else ("--", "no OOM observed / not run")
    )

    if os.path.isfile(thermal_summary_path):
        tp = load("thermal_throughput.csv")
        end_t = max(float(r["wall_time_s"]) for r in tp)
        first_30 = [float(r["tflops"]) for r in tp if float(r["wall_time_s"]) <= 30]
        last_5min = [float(r["tflops"]) for r in tp if float(r["wall_time_s"]) >= end_t - 300]
        if first_30 and last_5min:
            peak = max(first_30)
            steady = sum(last_5min) / len(last_5min)
            rows["Steady-state / peak throughput"] = (f"{100.0 * steady / peak:.1f}%", "")
        else:
            rows["Steady-state / peak throughput"] = ("--", "insufficient duration")
    else:
        rows["Steady-state / peak throughput"] = ("--", "run sustained_load.py")

    thermal_log_path = os.path.join(RESULTS, "thermal_log.csv")
    if os.path.isfile(thermal_log_path):
        log = load("thermal_log.csv")
        onset = None
        for r in log:
            active = any(
                r.get(k, "").strip().lower() in ("active", "yes", "1")
                for k in ("throttle_sw_power_cap", "throttle_hw_slowdown", "throttle_hw_thermal_slowdown")
            )
            if active:
                onset = float(r["wall_time_s"])
                break
        rows["Throttle onset (s, or none)"] = (f"{onset:.0f}" if onset is not None else "none", "")
    else:
        rows["Throttle onset (s, or none)"] = ("--", "run sustained_load.py")

    gpu_uuid = ""
    for src in (matmul, bw):
        if src:
            gpu_uuid = src[0].get("gpu_uuid", "")
            break

    print(f"## Table HW2.5.1 -- summary for UUID `{gpu_uuid}`\n")
    print("| Measurement | Your GPU | Notes |")
    print("|---|---|---|")
    order = [
        "Peak achieved TFLOPS (BF16)", "% of theoretical peak (BF16)",
        "Effective bandwidth (GB/s)", "Naive attention OOM length",
        "Fused attention OOM length", "Steady-state / peak throughput",
        "Throttle onset (s, or none)",
    ]
    for k in order:
        val, note = rows[k]
        print(f"| {k} | {val} | {note} |")


if __name__ == "__main__":
    main()
