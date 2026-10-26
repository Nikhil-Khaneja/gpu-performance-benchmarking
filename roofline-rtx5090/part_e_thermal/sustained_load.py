"""Part E: run a sustained compute load for 20 minutes, sampling GPU state
(clock, memory clock, temperature, power, utilization, throttle reasons)
every 5 seconds via nvidia-smi, on a background thread, while the main
thread hammers the GPU with a fixed-size matmul in a tight loop.

Two logs come out of this:
  - results/thermal_log.csv          (nvidia-smi samples every 5s)
  - results/thermal_throughput.csv   (achieved TFLOPS per loop iteration,
                                       used to compute peak-first-30s vs
                                       steady-state-last-5min in Part F)

Usage: python part_e_thermal/sustained_load.py --minutes 20
"""
import argparse
import csv
import os
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.gpu_info import get_gpu_identity
from common.csv_utils import append_row
from common.run_log import log_run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THERMAL_LOG = os.path.join(ROOT, "results", "thermal_log.csv")
THROUGHPUT_LOG = os.path.join(ROOT, "results", "thermal_throughput.csv")

SMI_FIELDS = [
    "timestamp", "uuid", "clocks.sm", "clocks.mem", "temperature.gpu",
    "power.draw", "power.limit", "utilization.gpu",
    "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
]


def sample_nvidia_smi(index=0):
    cmd = ["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader,nounits", "-i", str(index)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [v.strip() for v in out.stdout.strip().split("\n")[0].split(",")]


def sampler_thread_fn(stop_event, interval_s, index=0):
    while not stop_event.is_set():
        t0 = time.time()
        try:
            vals = sample_nvidia_smi(index)
            row = dict(zip(
                ["smi_timestamp", "gpu_uuid", "sm_clock_mhz", "mem_clock_mhz", "temp_c",
                 "power_draw_w", "power_limit_w", "util_pct",
                 "throttle_sw_power_cap", "throttle_hw_slowdown", "throttle_hw_thermal_slowdown"],
                vals,
            ))
            row["wall_time_s"] = round(t0 - START_TIME, 1)
            append_row(THERMAL_LOG, row)
        except Exception as e:  # noqa: BLE001 -- sampler must never kill the run
            print(f"[sampler] warning: {e}", file=sys.stderr)
        stop_event.wait(max(0.0, interval_s - (time.time() - t0)))


START_TIME = None


def main():
    global START_TIME
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--sample-interval-s", type=float, default=5.0)
    ap.add_argument("--matmul-n", type=int, default=8192)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    identity = get_gpu_identity(0)
    print(f"GPU: {identity['name']}  UUID: {identity['uuid']}")
    print(f"Running sustained load for {args.minutes} min, sampling every {args.sample_interval_s}s\n")

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    a = torch.randn(args.matmul_n, args.matmul_n, device="cuda", dtype=dtype)
    b = torch.randn(args.matmul_n, args.matmul_n, device="cuda", dtype=dtype)
    flops = 2 * (args.matmul_n ** 3)

    START_TIME = time.time()
    stop_event = threading.Event()
    sampler = threading.Thread(target=sampler_thread_fn, args=(stop_event, args.sample_interval_s), daemon=True)
    sampler.start()

    end_time = START_TIME + args.minutes * 60
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    n_iters = 0
    last_print = 0
    while time.time() < end_time:
        iter_start.record()
        torch.matmul(a, b)
        iter_end.record()
        torch.cuda.synchronize()
        elapsed_ms = iter_start.elapsed_time(iter_end)
        tflops = flops / (elapsed_ms / 1000) / 1e12
        wall_t = time.time() - START_TIME
        append_row(THROUGHPUT_LOG, {
            "gpu_uuid": identity["uuid"], "wall_time_s": round(wall_t, 2),
            "iter": n_iters, "iter_ms": elapsed_ms, "tflops": tflops,
        })
        n_iters += 1
        if wall_t - last_print > 30:
            print(f"t={wall_t:6.0f}s  iter={n_iters:6d}  {tflops:7.2f} TFLOPS")
            last_print = wall_t

    stop_event.set()
    sampler.join(timeout=args.sample_interval_s + 5)

    print(f"\nDone. {n_iters} matmul iterations logged to {THROUGHPUT_LOG}")
    print(f"Thermal samples logged to {THERMAL_LOG}")

    log_run(
        "sustained_load.py", identity["uuid"],
        {"minutes": args.minutes, "matmul_n": args.matmul_n, "dtype": args.dtype},
        [THERMAL_LOG, THROUGHPUT_LOG],
    )


if __name__ == "__main__":
    main()
