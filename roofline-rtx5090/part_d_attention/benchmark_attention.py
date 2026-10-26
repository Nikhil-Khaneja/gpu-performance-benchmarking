"""Part D: cost of attention.

1. Coarse sweep over seq_len in {512,1024,2048,4096,8192,16384}, naive and
   fused attention, recording peak memory and forward latency.
2. Refine the OOM boundary (per implementation) by bisecting between the
   largest seq_len that succeeded and the smallest that failed.

Batch size and head_dim are fixed and printed/logged, per the assignment's
"choose one head dimension and one batch size and state it."

Usage: python part_d_attention/benchmark_attention.py
"""
import argparse
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.gpu_info import get_gpu_identity
from common.timing import time_cuda, peak_memory_of, is_oom_error
from common.csv_utils import append_row
from common.run_log import log_run
from attention_ops import naive_attention, fused_attention, make_qkv

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "attention.csv"
)

SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384]
DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_once(impl, batch, heads, seq, head_dim, dtype, device="cuda"):
    """Returns True on success (also frees memory before returning), raises
    the RuntimeError back up on failure so the caller decides OOM vs real bug."""
    scale = 1.0 / (head_dim ** 0.5)
    q, k, v = make_qkv(batch, heads, seq, head_dim, dtype, device)
    fn = naive_attention if impl == "naive" else fused_attention
    out = fn(q, k, v, scale)
    torch.cuda.synchronize()
    del q, k, v, out
    clear_gpu()
    return True


def measure(impl, batch, heads, seq, head_dim, dtype, warmup, iters, trials, device="cuda"):
    scale = 1.0 / (head_dim ** 0.5)
    fn = naive_attention if impl == "naive" else fused_attention

    q, k, v = make_qkv(batch, heads, seq, head_dim, dtype, device)
    peak_bytes = peak_memory_of(lambda: fn(q, k, v, scale), device=device)

    stats = time_cuda(lambda: fn(q, k, v, scale), warmup=warmup, iters=iters, trials=trials)

    del q, k, v
    clear_gpu()
    return peak_bytes, stats


def refine_oom_boundary(low_ok, high_fail, test_fn, resolution=1):
    """Bisect to `resolution` token precision. test_fn(seq_len) -> bool
    (True = ran without OOM)."""
    while high_fail - low_ok > resolution:
        mid = (low_ok + high_fail) // 2
        clear_gpu()
        try:
            test_fn(mid)
            low_ok = mid
        except RuntimeError as e:
            if not is_oom_error(e):
                raise
            high_fail = mid
            clear_gpu()
    return low_ok, high_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=SEQ_LENS)
    ap.add_argument("--refine-resolution", type=int, default=1)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    identity = get_gpu_identity(0)
    dtype = DTYPE_MAP[args.dtype]
    print(f"GPU: {identity['name']}  UUID: {identity['uuid']}")
    print(f"Fixed config: batch={args.batch} heads={args.heads} head_dim={args.head_dim} dtype={args.dtype}\n")

    for impl in ["naive", "fused"]:
        print(f"=== {impl} attention ===")
        last_ok, first_fail = None, None
        for seq in args.seq_lens:
            print(f"seq={seq} ...", end=" ", flush=True)
            try:
                peak_bytes, stats = measure(
                    impl, args.batch, args.heads, seq, args.head_dim, dtype,
                    args.warmup, args.iters, args.trials,
                )
                print(f"peak_mem={peak_bytes / 1e9:.3f} GB  latency={stats['median_ms']:.3f} ms")
                last_ok = seq
                append_row(RESULTS_PATH, {
                    "gpu_uuid": identity["uuid"], "gpu_name": identity["name"], "impl": impl,
                    "batch": args.batch, "heads": args.heads, "head_dim": args.head_dim,
                    "dtype": args.dtype, "seq_len": seq, "status": "ok",
                    "peak_mem_bytes": peak_bytes, "median_latency_ms": stats["median_ms"],
                    "std_latency_ms": stats["std_ms"],
                })
            except RuntimeError as e:
                if not is_oom_error(e):
                    raise
                print("OOM")
                first_fail = seq
                clear_gpu()
                append_row(RESULTS_PATH, {
                    "gpu_uuid": identity["uuid"], "gpu_name": identity["name"], "impl": impl,
                    "batch": args.batch, "heads": args.heads, "head_dim": args.head_dim,
                    "dtype": args.dtype, "seq_len": seq, "status": "oom",
                    "peak_mem_bytes": "", "median_latency_ms": "", "std_latency_ms": "",
                })
                break  # larger seq lens will also OOM; stop the coarse sweep here

        if first_fail is not None and last_ok is not None:
            print(f"Refining OOM boundary between {last_ok} (ok) and {first_fail} (fail) ...")

            def test_fn(seq):
                return run_once(impl, args.batch, args.heads, seq, args.head_dim, dtype)

            boundary_ok, boundary_fail = refine_oom_boundary(
                last_ok, first_fail, test_fn, resolution=args.refine_resolution
            )
            print(f"Refined: largest OK = {boundary_ok}, smallest failing = {boundary_fail}")
            append_row(os.path.join(os.path.dirname(RESULTS_PATH), "attention_oom_boundary.csv"), {
                "gpu_uuid": identity["uuid"], "gpu_name": identity["name"], "impl": impl,
                "batch": args.batch, "heads": args.heads, "head_dim": args.head_dim,
                "dtype": args.dtype, "largest_ok_seq_len": boundary_ok,
                "smallest_failing_seq_len": boundary_fail, "resolution": args.refine_resolution,
            })
        elif first_fail is None:
            print(f"No OOM observed up to seq_len={args.seq_lens[-1]}; "
                  f"extend --seq-lens to find the boundary on this card.")
        print()

    log_run(
        "benchmark_attention.py", identity["uuid"],
        {"batch": args.batch, "heads": args.heads, "head_dim": args.head_dim, "dtype": args.dtype},
        [RESULTS_PATH, os.path.join(os.path.dirname(RESULTS_PATH), "attention_oom_boundary.csv")],
    )


if __name__ == "__main__":
    main()
