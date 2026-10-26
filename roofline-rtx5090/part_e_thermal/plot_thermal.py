"""Part E: plot SM clock and temperature vs time on one figure, detect
throttling from nvidia-smi's own throttle-reason flags (not a heuristic),
and report peak throughput (first 30s) vs steady-state throughput (final
5 minutes) as a percentage.

Usage: python part_e_thermal/plot_thermal.py
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THERMAL_LOG = os.path.join(ROOT, "results", "thermal_log.csv")
THROUGHPUT_LOG = os.path.join(ROOT, "results", "thermal_throughput.csv")
FIG_DIR = os.path.join(ROOT, "figures")


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thermal-csv", default=THERMAL_LOG)
    ap.add_argument("--throughput-csv", default=THROUGHPUT_LOG)
    args = ap.parse_args()

    thermal = load_rows(args.thermal_csv)
    throughput = load_rows(args.throughput_csv)
    if not thermal or not throughput:
        print("Missing thermal or throughput log. Run sustained_load.py first.")
        sys.exit(1)

    t = [float(r["wall_time_s"]) for r in thermal]
    sm_clock = [float(r["sm_clock_mhz"]) for r in thermal]
    temp = [float(r["temp_c"]) for r in thermal]
    gpu_name = thermal[0].get("gpu_uuid", "")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(t, sm_clock, color="tab:blue", label="SM clock (MHz)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("SM clock (MHz)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(t, temp, color="tab:red", label="Temperature (C)")
    ax2.set_ylabel("Temperature (C)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title(f"Sustained load: clock & temperature vs time\nUUID {gpu_name}", fontsize=9)
    ax1.grid(True, alpha=0.3)
    out = os.path.join(FIG_DIR, f"partE_thermal_{gpu_name.replace('-', '')}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}\n")

    # Throttle detection from nvidia-smi's own reason flags.
    throttle_onset_s = None
    throttle_reasons_seen = set()
    for r in thermal:
        active = [
            k for k in ("throttle_sw_power_cap", "throttle_hw_slowdown", "throttle_hw_thermal_slowdown")
            if r.get(k, "").strip().lower() in ("active", "yes", "1")
        ]
        if active:
            throttle_reasons_seen.update(active)
            if throttle_onset_s is None:
                throttle_onset_s = float(r["wall_time_s"])

    if throttle_onset_s is not None:
        print(f"Throttling detected at t={throttle_onset_s:.0f}s. Reasons: {sorted(throttle_reasons_seen)}")
        idx = next(i for i, r in enumerate(thermal) if float(r["wall_time_s"]) >= throttle_onset_s)
        print(f"  temp at onset: {temp[idx]:.0f}C   sm_clock at onset: {sm_clock[idx]:.0f}MHz")
    else:
        print("No throttling flags observed by nvidia-smi during this run.")

    # Peak (first 30s) vs steady-state (final 5 min) throughput.
    end_t = max(float(r["wall_time_s"]) for r in throughput)
    first_30 = [float(r["tflops"]) for r in throughput if float(r["wall_time_s"]) <= 30]
    last_5min = [float(r["tflops"]) for r in throughput if float(r["wall_time_s"]) >= end_t - 300]

    if first_30 and last_5min:
        peak = max(first_30)
        steady = sum(last_5min) / len(last_5min)
        pct = 100.0 * steady / peak
        print(f"\nPeak throughput (first 30s): {peak:.2f} TFLOPS")
        print(f"Steady-state throughput (final 5 min, mean): {steady:.2f} TFLOPS")
        print(f"Steady-state / peak: {pct:.1f}%")
    else:
        print("\nNot enough data in first-30s / final-5min windows; "
              "check the run duration and throughput log.")


if __name__ == "__main__":
    main()
