# RTX 5090 Metrics and Analysis

## Part A — Provenance

- **GPU:** NVIDIA GeForce RTX 5090 (Blackwell GB202), UUID `GPU-41051861-72ca-e198-fd60-2bf89fd2a67b`
- **Memory:** 32 GB GDDR7 (31.8 GiB reported), 512-bit bus, 1792 GB/s spec
- **Driver:** 610.60 · **CUDA (torch):** 12.8 · **PyTorch:** 2.7.0+cu128 · **Power limit:** 575 W
- **Environment:** Windows 11 host, Docker Desktop (WSL2 backend), image `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`, `--gpus all`. The GPU was also driving the display (~1.9 GB VRAM in use by the desktop before runs).
- **Vendor specs source:** NVIDIA RTX Blackwell GPU Architecture whitepaper.
  Peak numbers used: FP32 104.8 TFLOPS, TF32 tensor 104.8, FP16/BF16 tensor 209.5, FP8 tensor 419 (all dense, **FP32 accumulate** — the whitepaper also lists 2× higher FP16-accumulate figures, which PyTorch matmuls do not use).
- **GPU time used:** 21:06 → 22:02 UTC, 2026-09-22 ≈ **0.93 GPU-hours** (1× RTX 5090).

## Part B — Matmul throughput vs. N and precision

| Precision | N=1024 | N=4096 | N=8192 | N=16384 | Plateau N |
|---|---|---|---|---|---|
| FP32 | 41.4 | 62.8 | 63.6 | 64.3 (61%) | 4096 |
| TF32 | 71.9 | 98.9 | 112.8 | 108.2 (103%) | 8192 |
| FP16 | 94.2 | 162.7 | 204.6 | 224.8 (107%) | 8192 (still +10% at 16384) |
| BF16 | 94.3 | 180.8 | 222.8 | 229.0 (109%) | 8192 |
| FP8 (`torch._scaled_mm`) | 182.8 | 389.8 | 440.8 | 448.0 (107%) | 8192 |

(TFLOPS; % of vendor peak in parentheses.)

- **Small N:** at N=1024 every precision sits at 40–70% of peak — the GEMM is too small to fill all 170 SMs with tiles, and launch/dispatch overhead is a large fraction of a sub-0.1 ms kernel.
- **Plateau:** once N ≥ 4096–8192 each precision saturates its execution units; lower precisions plateau later because each tile finishes faster, so more work is needed to hide overhead.
- **>100% of peak:** vendor peaks are computed at the official ~2.41 GHz boost clock. Part E logged the SM clock at **2835–2857 MHz** under sustained load (~18% higher), so 105–109% of the spec-sheet number is expected.
- **FP32 at ~61%:** plain FP32 runs on CUDA cores, not tensor cores; ~60–70% of headline is typical for SGEMM on GeForce parts.
- **FP8:** worked via `torch._scaled_mm` on this build (no failure to report), reaching ~448 TFLOPS — ~2× BF16, as expected from the tensor-core rate.

## Part C — Memory-bound vs. compute-bound (roofline)

- **Ridge point (FP32):** 104.8 TFLOPS / 1792 GB/s = **58.5 FLOPs/byte**.
- **Elementwise add, 2²⁸ fp32 elements:** 1 FLOP per 12 bytes (read A, read B, write C) → AI = **0.083 FLOPs/byte**, far left of the ridge → memory-bound. Achieved **1563.7 GB/s = 87.3% of spec**, a normal ceiling for GDDR7 streaming.
- **Matmul N=8192, fp32:** AI = 2N³ / (3·N²·4 B) = N/6 = **1365 FLOPs/byte** (lower bound, ignores on-chip reuse), far right of the ridge → compute-bound. Achieved **65.2 TFLOPS = 62.2%** of FP32 peak, matching Part B.
- The two kernels sit ~4 orders of magnitude apart in arithmetic intensity; each is limited by the resource the roofline predicts. The elementwise kernel's AI is constant in N, while matmul's grows linearly with N.

## Part D — Naive vs. fused attention

**Config:** batch=1, heads=16, head_dim=64, bf16. Fused = PyTorch `scaled_dot_product_attention`.

| seq_len | Naive mem (GB) | Naive (ms) | Fused mem (GB) | Fused (ms) | Speedup |
|---|---|---|---|---|---|
| 512 | 0.029 | 0.046 | 0.017 | 0.024 | 1.9× |
| 2048 | 0.294 | 0.706 | 0.025 | 0.106 | 6.6× |
| 8192 | 4.371 | 9.405 | 0.076 | 1.398 | 6.7× |
| 16384 | 17.32 | 36.52 | 0.144 | 5.106 | 7.2× |
| 20480 | 27.02 | 57.27 | 0.178 | 7.919 | 7.2× |
| 22438 | **OOM** | — | ~0.19 | — | — |

- **O(seq²) confirmation:** quadratic fit of naive peak memory gives **a = 6.40×10⁻⁸ GB/token² = 64 bytes/token²** = 16 heads × 2 bytes (bf16) × 2 materialized seq×seq tensors (scores + softmax output). Fused memory grows ~linearly (0.017 → 0.28 GB up to seq 32768).
- **OOM boundary:** naive attention's largest successful length is **22,437**; **22,438 OOMs** (bisected to single-token resolution). Check: 6.4×10⁻⁸ × 22437² ≈ 32.2 GB = 30.0 GiB, matching the 30.20 GiB cap below.
- **Windows/WSL2 caveat (important):** by default the driver's *system-memory fallback* silently spills over-capacity allocations into host RAM over PCIe instead of raising OOM. Uncapped runs "succeeded" at seq 24576 (38.9 GB) and 32768 (69.0 GB) on a 32 GB card, but latency jumped **57 ms → 1670 ms (29×)** for only 1.4× more memory — a performance cliff rather than a crash. Disabling the fallback in NVIDIA Control Panel did not take effect inside WSL2, so the boundary was measured by capping PyTorch at the free VRAM (`torch.cuda.set_per_process_memory_fraction(free/total)` = 30.20 GiB of 31.84 GiB; see `run_d_capped.py`). Speedups of 165–245× seen at 24576/32768 are from the spilled runs and are **not** representative of on-GPU performance.
- **Fused attention never OOM'd** up to 32768 (0.28 GB).

**What the fused kernel avoids:** The naive implementation materializes the full seq×seq score matrix and its softmax in GPU memory, writing and re-reading O(seq²) bytes per head, which dominates both memory and time. The fused (FlashAttention-style) kernel tiles Q, K and V into on-chip SRAM and computes softmax incrementally with a running max and normalizer (online softmax), so the seq×seq matrix never exists in global memory. Memory therefore scales linearly in sequence length, and HBM/GDDR traffic drops sharply, which is where the ~7× speedup comes from. The backward pass (not measured here) recomputes the tiles instead of storing them, trading cheap FLOPs for expensive memory traffic.

## Part E — Sustained load (20 min, BF16 matmul)

- **Throughput:** peak (first 30 s) **236.6 TFLOPS**; steady state (final 5 min) **231.0 TFLOPS** → **97.6%** of peak. No downward trend across 20 min.
- **Temperature:** max **70 °C**, final-5-min mean **67.3 °C**.
- **Power:** mean **413 W**, max **463 W** vs. 575 W limit.
- **SM clock:** **2835–2857 MHz** (mean 2841) — essentially flat.
- **Throttling:** nvidia-smi reported `SW power cap`, `HW slowdown` and `HW thermal slowdown` as **Not Active in all 233 samples** → **throttle onset: none**.
- **Two transient dips** to ~200 TFLOPS (t≈330 s and 810 s) recovered by the next sample; with no throttle flag and flat clocks, these are attributed to host/display activity (this GPU also drives the desktop), not thermals.
- **Caveat:** mean nvidia-smi utilization was **56%** — the loop has gaps between iterations (host-side timing/sync and sampling), so the card never reached its 575 W limit or thermal ceiling. The result shows the card sustains full throughput *at this duty cycle*; a gap-free load would draw more power and might eventually hit the power cap.

## Table HW2.5.1 (auto-generated by Part F)

<!-- BEGIN AUTO TABLE -->
| Measurement | Your GPU | Notes |
|---|---|---|
| Peak achieved TFLOPS (BF16) | 229.01 | at N=16384 |
| % of theoretical peak (BF16) | 109.3% | at N=16384 |
| Effective bandwidth (GB/s) | 1563.7 | 87.3% of spec, fp32 |
| Naive attention OOM length | 22438 | largest ok=22437 |
| Fused attention OOM length | -- | no OOM observed / not run |
| Steady-state / peak throughput | 97.6% |  |
| Throttle onset (s, or none) | none |  |
<!-- END AUTO TABLE -->
