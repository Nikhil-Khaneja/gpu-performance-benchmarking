# GPU Performance Benchmarking

Measuring how fast GPU code really runs, and why: a hand-written CUDA C kernel profiled with nvprof,
and a roofline / precision / attention / thermal study of an NVIDIA RTX 5090 in PyTorch.

| Study | Hardware | Headline result |
|---|---|---|
| [CUDA C matrix multiply](cuda-matmul/) | Tesla T4 | **1,527x** faster than a CPU loop at N=4096 (end-to-end, including transfers) |
| [Roofline model](roofline-rtx5090/METRICS.md#part-c--memory-bound-vs-compute-bound-roofline) | RTX 5090 | FP32 ridge point **58.5 FLOPs/byte**; streaming add at **1,564 GB/s** (87% of spec) |
| [GEMM throughput by precision](roofline-rtx5090/METRICS.md#part-b--matmul-throughput-vs-n-and-precision) | RTX 5090 | FP32 64, TF32 113, FP16 225, BF16 229, FP8 **448 TFLOPS** |
| [Naive vs. fused attention](roofline-rtx5090/METRICS.md#part-d--naive-vs-fused-attention) | RTX 5090 | Fused SDPA up to **7.2x** faster; naive attention OOMs at **22,438** tokens (bisected) |
| [Sustained load](roofline-rtx5090/METRICS.md#part-e--sustained-load-20-min-bf16-matmul) | RTX 5090 | **97.6%** of peak after 20 min, no throttling |

## Layout

```
cuda-matmul/        matmul.cu (CUDA C kernel + CPU reference), cuda.ipynb (Colab run with nvprof), RESULTS.md
roofline-rtx5090/   PyTorch benchmark suite, results/ CSVs, figures/, METRICS.md
  common/             GPU identity, CUDA-event timing, CSV and run-log helpers, vendor peak specs
  part_a_provenance/  nvidia-smi capture and spec cross-check
  part_b_precision/   matmul TFLOPS vs. N for FP32 / TF32 / FP16 / BF16 / FP8
  part_c_bandwidth_compute/  memory-bound vs. compute-bound kernels on the roofline
  part_d_attention/   naive vs. fused attention, OOM boundary bisection, memory scaling
  part_e_thermal/     20-minute sustained load with clocks, power and throttle flags
  part_f_analysis/    summary table builder
```

## Running

**CUDA matmul** (any NVIDIA GPU with nvcc):

```bash
cd cuda-matmul
nvcc -O3 matmul.cu -o matmul
./matmul 1024            # N defaults to 256
nvprof ./matmul 1024     # or: nsys profile ./matmul 1024
```

**RTX 5090 suite** (PyTorch with CUDA):

```bash
cd roofline-rtx5090
pip install -r requirements.txt
bash run_all.sh          # read it first: Part E runs for 20 minutes and Part D OOMs on purpose
```

Every CSV row and run-log line is tagged with the GPU UUID, so results from different cards can be merged.

## Notes

- **Timing:** warm-up first, then CUDA events around batches of repetitions, over several trials, so
  per-iteration dispatch jitter does not inflate the variance.
- **TF32** is a compute mode on FP32 tensors (`torch.backends.cuda.matmul.allow_tf32`), not a storage type.
- **Above 100% of peak:** vendor peaks use the ~2.41 GHz boost clock; the card ran at 2,835-2,857 MHz
  under load, so 105-109% of the spec-sheet number is expected.
- **Roofline:** the matmul arithmetic intensity assumes each of A, B and C crosses DRAM once (a lower
  bound that ignores on-chip reuse). It grows with N; the elementwise kernel's stays flat.
- **WSL2 caveat:** the Windows driver silently spills over-capacity allocations to host RAM instead of
  raising OOM (57 ms -> 1,670 ms), so the OOM boundary was measured with PyTorch capped to free VRAM
  (`run_d_capped.py`).

## How this was built

Started as GPU coursework for SJSU DATA266 (Fall 2026). The PyTorch benchmark harness was drafted with
Claude Code; I ran everything on the hardware, checked the vendor specs, debugged the WSL2 memory
fallback, and wrote the analysis in `METRICS.md`.

## License

MIT
