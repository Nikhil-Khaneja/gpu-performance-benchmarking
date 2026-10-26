"""Part B: dense matmul throughput across N in {1024, 4096, 8192, 16384} and
precision in {fp32, tf32, fp16, bf16}, plus a best-effort fp8 attempt.

Usage:
    python part_b_precision/benchmark_matmul.py
    python part_b_precision/benchmark_matmul.py --sizes 1024 4096 --precisions fp16 bf16
    python part_b_precision/benchmark_matmul.py --try-fp8
"""
import argparse
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.gpu_info import get_gpu_identity
from common.peak_specs import get_peak_specs
from common.timing import time_cuda, is_oom_error
from common.csv_utils import append_row
from common.run_log import log_run

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "matmul_precision.csv"
)

DEFAULT_SIZES = [1024, 4096, 8192, 16384]
DEFAULT_PRECISIONS = ["fp32", "tf32", "fp16", "bf16"]


def configure_precision(precision):
    """Returns the storage dtype to allocate tensors in, and sets the global
    TF32 matmul flag (TF32 is a compute mode on fp32-stored tensors, not a
    separate dtype)."""
    if precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]


def bench_matmul(n, precision, warmup, iters, trials, device="cuda"):
    dtype = configure_precision(precision)
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)

    def fn():
        torch.matmul(a, b)

    stats = time_cuda(fn, warmup=warmup, iters=iters, trials=trials)
    flops = 2 * (n ** 3)
    tflops = flops / (stats["median_ms"] / 1000) / 1e12
    del a, b
    torch.cuda.empty_cache()
    return stats, tflops


def try_fp8_matmul(n, warmup, iters, trials, device="cuda"):
    """Best-effort fp8 GEMM via torch._scaled_mm. This is exactly the kind
    of thing Part B asks you to attempt and report on: on many stacks this
    either isn't exposed, requires row/col-major-specific layouts, or needs
    a specific PyTorch build. Returns (stats_or_None, tflops_or_None, note).
    """
    if not hasattr(torch, "_scaled_mm"):
        return None, None, "torch._scaled_mm not present in this PyTorch build"
    try:
        fp8_dtype = torch.float8_e4m3fn
        a = torch.randn(n, n, device=device, dtype=torch.float32).to(fp8_dtype)
        b = torch.randn(n, n, device=device, dtype=torch.float32).to(fp8_dtype).t().contiguous().t()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)

        def fn():
            torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)

        stats = time_cuda(fn, warmup=warmup, iters=iters, trials=trials)
        flops = 2 * (n ** 3)
        tflops = flops / (stats["median_ms"] / 1000) / 1e12
        return stats, tflops, "ok"
    except Exception as e:  # noqa: BLE001 -- deliberately broad, this is a probe
        return None, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--precisions", type=str, nargs="+", default=DEFAULT_PRECISIONS)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20, help="matmuls per timed trial")
    ap.add_argument("--trials", type=int, default=5, help="independent timed trials")
    ap.add_argument("--try-fp8", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    identity = get_gpu_identity(0)
    try:
        spec = get_peak_specs(identity["name"])
    except KeyError as e:
        print(f"WARNING: {e}\nProceeding without %-of-peak numbers.", file=sys.stderr)
        spec = None
    print(f"GPU: {identity['name']}  UUID: {identity['uuid']}")

    peak_key = {"fp32": "fp32_tflops", "tf32": "tf32_tflops", "fp16": "fp16_tflops", "bf16": "bf16_tflops"}

    for precision in args.precisions:
        for n in args.sizes:
            print(f"[{precision}] N={n} ...", end=" ", flush=True)
            try:
                stats, tflops = bench_matmul(n, precision, args.warmup, args.iters, args.trials)
            except RuntimeError as e:
                if is_oom_error(e):
                    print("OOM")
                    row = {
                        "gpu_uuid": identity["uuid"], "gpu_name": identity["name"],
                        "precision": precision, "n": n, "status": "oom",
                        "median_ms": "", "std_ms": "", "tflops": "", "pct_of_peak": "",
                        "iters_per_trial": args.iters, "trials": args.trials, "note": "",
                    }
                    append_row(RESULTS_PATH, row)
                    torch.cuda.empty_cache()
                    continue
                raise
            if spec is not None:
                pct_of_peak = 100.0 * tflops / spec[peak_key[precision]]
                pct_str = f"({pct_of_peak:.1f}% of peak)"
            else:
                pct_of_peak = ""
                pct_str = "(no peak spec on file)"
            print(f"median={stats['median_ms']:.3f}ms  {tflops:.2f} TFLOPS  {pct_str}")
            row = {
                "gpu_uuid": identity["uuid"], "gpu_name": identity["name"],
                "precision": precision, "n": n, "status": "ok",
                "median_ms": stats["median_ms"], "std_ms": stats["std_ms"],
                "tflops": tflops, "pct_of_peak": pct_of_peak,
                "iters_per_trial": args.iters, "trials": args.trials, "note": "",
            }
            append_row(RESULTS_PATH, row)

    if args.try_fp8:
        for n in args.sizes:
            print(f"[fp8, best-effort] N={n} ...", end=" ", flush=True)
            stats, tflops, note = try_fp8_matmul(n, args.warmup, args.iters, args.trials)
            fp8_pct = (100.0 * tflops / spec["fp8_tflops"]) if (stats and spec is not None) else ""
            if stats is None:
                print(f"FAILED: {note}")
            else:
                pct_str = f"({fp8_pct:.1f}% of peak)" if fp8_pct != "" else "(no peak spec on file)"
                print(f"median={stats['median_ms']:.3f}ms  {tflops:.2f} TFLOPS  {pct_str}")
            row = {
                "gpu_uuid": identity["uuid"], "gpu_name": identity["name"],
                "precision": "fp8", "n": n,
                "status": "ok" if stats else "failed",
                "median_ms": stats["median_ms"] if stats else "",
                "std_ms": stats["std_ms"] if stats else "",
                "tflops": tflops if stats else "",
                "pct_of_peak": fp8_pct,
                "iters_per_trial": args.iters, "trials": args.trials,
                "note": note,
            }
            append_row(RESULTS_PATH, row)

    log_run(
        "benchmark_matmul.py", identity["uuid"],
        {"sizes": args.sizes, "precisions": args.precisions, "iters": args.iters, "trials": args.trials},
        RESULTS_PATH,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
