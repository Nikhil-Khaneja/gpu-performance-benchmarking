"""CUDA-event based timing. All benchmarks in this repo warm up before
timing and time a batch of repetitions between two events (rather than
synchronizing every iteration) so launch overhead doesn't dominate small
problem sizes.
"""
import torch


def time_cuda(fn, warmup=10, iters=30, trials=5):
    """Time `fn()` (a no-arg callable that issues GPU work) on CUDA.

    Returns dict with per-call latency stats in milliseconds, computed over
    `trials` independent batches of `iters` repetitions each. Reporting
    across trials (rather than across raw iterations) gives a defensible
    variance estimate that isn't inflated by CPU dispatch jitter within a
    single tight loop.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    trial_avg_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(trials):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        trial_avg_ms.append(start.elapsed_time(end) / iters)

    trial_avg_ms.sort()
    mean = sum(trial_avg_ms) / len(trial_avg_ms)
    var = sum((x - mean) ** 2 for x in trial_avg_ms) / len(trial_avg_ms)
    return {
        "mean_ms": mean,
        "std_ms": var ** 0.5,
        "median_ms": trial_avg_ms[len(trial_avg_ms) // 2],
        "min_ms": trial_avg_ms[0],
        "max_ms": trial_avg_ms[-1],
        "trials": trials,
        "iters_per_trial": iters,
        "total_reps": trials * iters,
    }


def peak_memory_of(fn, device="cuda"):
    """Run fn() once and return peak allocated memory in bytes, isolated
    from whatever ran before it on this device."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device)


def is_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg
