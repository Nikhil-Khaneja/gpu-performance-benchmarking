### GPU provenance

- Name: NVIDIA GeForce RTX 5090
- UUID: `GPU-41051861-72ca-e198-fd60-2bf89fd2a67b`
- Driver version: 610.60
- CUDA runtime (torch): 12.8
- PyTorch version: 2.7.0+cu128
- VRAM: 31.8 GiB (reported by nvidia-smi)
- Power limit: 575 W

### Vendor-documented specs -- NVIDIA GeForce RTX 5090

- Architecture: Blackwell (GB202)
- Tensor core generation: 5
- Memory: GDDR7, 512-bit bus, 1792 GB/s
- Peak FP32: 104.8 TFLOPS
- Peak TF32 (tensor core, dense): 104.75 TFLOPS
- Peak FP16 (tensor core, dense): 209.5 TFLOPS
- Peak BF16 (tensor core, dense): 209.5 TFLOPS
- Peak FP8 (tensor core, dense): 419.0 TFLOPS
- Source: https://www.spheron.network/blog/nvidia-rtx-5090-specs/
  (VERIFY against NVIDIA's official spec sheet / architecture whitepaper
  before citing in your submission -- see common/peak_specs.py docstring.)
