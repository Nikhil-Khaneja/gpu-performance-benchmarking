# CUDA matmul: results

GPU: Tesla T4 (Google Colab), driver 580.82.07, CUDA 13.0. Built with `nvcc -O3 matmul.cu -o matmul`.
Kernel: one thread per output element, 2D grid of 16x16 thread blocks (no shared-memory tiling yet).
Timing: CUDA events; host-to-device and device-to-host copies are timed apart from the kernel.
Every run checks the GPU result against the CPU reference (max abs diff <= 3.7e-4).

| N | CPU (ms) | GPU kernel (ms) | H2D+D2H (ms) | End-to-end speedup |
|---|---|---|---|---|
| 256 | 20.32 | 0.23 | 0.39 | 32x |
| 1024 | 3,353.29 | 7.90 | 4.82 | 264x |
| 4096 | 603,595.49 | 319.19 | 76.15 | 1,527x |

**nvprof (N=1024):** the kernel takes 65.9% of GPU time and the H2D+D2H copies take 34.1%, so compute,
not data transfer, dominates at this size.

**Crossover:** the GPU already wins at N=256. It never wins at N close to 0, because every GPU run pays
fixed costs (kernel launch overhead and ~0.4-0.6 ms of transfers) before doing any work. Below a few
tens of elements per side, the O(n^3) work the GPU parallelizes is smaller than that fixed cost and a
single CPU thread is faster.
