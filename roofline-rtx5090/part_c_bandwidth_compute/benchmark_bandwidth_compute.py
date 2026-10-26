"""Part C: one memory-bound kernel (large elementwise add) and one
compute-bound kernel (large square matmul), reporting:
  - memory-bound: effective bandwidth in GB/s and % of the card's spec'd bandwidth
  - both: arithmetic intensity in FLOPs/byte, and which side of the roofline
    each lands on for this card

Usage: python part_c_bandwidth_compute/benchmark_bandwidth_compute.py
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.gpu_info import get_gpu_identity
from common.peak_specs import get_peak_specs
from common.timing import time_cuda
from common.csv_utils import append_row
from common.run_log import log_run

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bandwidth_compute.csv"
)


def bench_elementwise_add(num_elements, dtype, warmup, iters, trials, device="cuda"):
    a = torch.randn(num_elements, device=device, dtype=dtype)
    b = torch.randn(num_elements, device=device, dtype=dtype)
    c = torch.empty_like(a)

    def fn():
        torch.add(a, b, out=c)

    stats = time_cuda(fn, warmup=warmup, iters=iters, trials=trials)

    itemsize = a.element_size()
    bytes_moved = 3 * num_elements * itemsize  # read a, read b, write c
    seconds = stats["median_ms"] / 1000
    bandwidth_gbps = (bytes_moved / seconds) / 1e9

    flops = num_elements  # one add per element
    arithmetic_intensity = flops / bytes_moved  # FLOPs/byte

    del a, b, c
    torch.cuda.empty_cache()
    return stats, bandwidth_gbps, arithmetic_intensity, bytes_moved


def bench_matmul_compute(n, dtype, warmup, iters, trials, device="cuda"):
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)

    def fn():
        torch.matmul(a, b)

    stats = time_cuda(fn, warmup=warmup, iters=iters, trials=trials)

    flops = 2 * (n ** 3)
    seconds = stats["median_ms"] / 1000
    tflops = flops / seconds / 1e12

    itemsize = a.element_size()
    # Bytes touched from/to HBM if A, B, C are each read/written once (a
    # lower bound that ignores on-chip reuse/tiling -- the point here is
    # that this ratio grows with N, unlike the elementwise kernel).
    bytes_touched = 3 * (n ** 2) * itemsize
    arithmetic_intensity = flops / bytes_touched

    del a, b
    torch.cuda.empty_cache()
    return stats, tflops, arithmetic_intensity, bytes_touched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-elements", type=int, default=1 << 28, help="elementwise add size")
    ap.add_argument("--matmul-n", type=int, default=8192)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--peak-bandwidth-gbps", type=float, default=None,
                     help="override, for hardware not in common/peak_specs.py")
    ap.add_argument("--peak-tflops", type=float, default=None,
                     help="override, for hardware not in common/peak_specs.py")
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    assert torch.cuda.is_available(), "CUDA GPU required"
    identity = get_gpu_identity(0)
    if args.peak_bandwidth_gbps is not None and args.peak_tflops is not None:
        peak_bandwidth = args.peak_bandwidth_gbps
        peak_compute_tflops = args.peak_tflops
    else:
        spec = get_peak_specs(identity["name"])  # raises KeyError with guidance if not on file
        peak_bandwidth = spec["bandwidth_gbps"]
        peak_compute_tflops = spec[f"{args.dtype}_tflops"] if args.dtype != "fp32" else spec["fp32_tflops"]
    ridge_point = (peak_compute_tflops * 1e12) / (peak_bandwidth * 1e9)  # FLOPs/byte

    print(f"GPU: {identity['name']}  UUID: {identity['uuid']}")
    print(f"Roofline ridge point for {args.dtype}: {ridge_point:.2f} FLOPs/byte "
          f"(peak {peak_compute_tflops} TFLOPS / peak {peak_bandwidth} GB/s)\n")

    # --- memory-bound: elementwise add ---
    print(f"[memory-bound] elementwise add, N={args.num_elements} ({args.dtype}) ...", end=" ", flush=True)
    mstats, bw_gbps, ai_mem, bytes_moved = bench_elementwise_add(
        args.num_elements, dtype, args.warmup, args.iters, args.trials
    )
    pct_of_bw = 100.0 * bw_gbps / peak_bandwidth
    side_mem = "memory-bound" if ai_mem < ridge_point else "compute-bound"
    print(f"{bw_gbps:.1f} GB/s ({pct_of_bw:.1f}% of spec), AI={ai_mem:.4f} FLOPs/byte -> {side_mem}")

    append_row(RESULTS_PATH, {
        "gpu_uuid": identity["uuid"], "gpu_name": identity["name"], "kernel": "elementwise_add",
        "dtype": args.dtype, "size": args.num_elements, "median_ms": mstats["median_ms"],
        "bandwidth_gbps": bw_gbps, "pct_of_peak_bandwidth": pct_of_bw,
        "tflops": "", "arithmetic_intensity": ai_mem, "roofline_side": side_mem,
        "ridge_point": ridge_point,
    })

    # --- compute-bound: square matmul ---
    print(f"[compute-bound] matmul N={args.matmul_n} ({args.dtype}) ...", end=" ", flush=True)
    cstats, tflops, ai_compute, bytes_touched = bench_matmul_compute(
        args.matmul_n, dtype, args.warmup, args.iters, args.trials
    )
    side_compute = "memory-bound" if ai_compute < ridge_point else "compute-bound"
    pct_of_peak_compute = 100.0 * tflops / peak_compute_tflops
    print(f"{tflops:.2f} TFLOPS ({pct_of_peak_compute:.1f}% of peak), "
          f"AI={ai_compute:.2f} FLOPs/byte -> {side_compute}")

    append_row(RESULTS_PATH, {
        "gpu_uuid": identity["uuid"], "gpu_name": identity["name"], "kernel": "matmul",
        "dtype": args.dtype, "size": args.matmul_n, "median_ms": cstats["median_ms"],
        "bandwidth_gbps": "", "pct_of_peak_bandwidth": "",
        "tflops": tflops, "arithmetic_intensity": ai_compute, "roofline_side": side_compute,
        "ridge_point": ridge_point,
    })

    log_run(
        "benchmark_bandwidth_compute.py", identity["uuid"],
        {"num_elements": args.num_elements, "matmul_n": args.matmul_n, "dtype": args.dtype},
        RESULTS_PATH,
    )


if __name__ == "__main__":
    main()
