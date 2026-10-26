"""Theoretical peak numbers for the two cards this assignment targets.

These are dense (no structured-sparsity) tensor-core numbers, since Part B
benchmarks dense GEMM. Populated from vendor spec pages found at write time;
the assignment requires you to cite vendor documentation yourself, so treat
these as a starting point and verify against NVIDIA's official spec sheet /
architecture whitepaper before you write them into METRICS.md.

Sources consulted when this file was written:
- RTX 4090: https://www.runpod.io/articles/guides/nvidia-rtx-4090
            (NVIDIA "Ada Lovelace" architecture, 4th-gen Tensor Cores)
- RTX 5090: https://www.spheron.network/blog/nvidia-rtx-5090-specs/
            (NVIDIA "Blackwell" architecture, 5th-gen Tensor Cores)
Replace these with the NVIDIA GeForce RTX 40/50-series spec sheets or the
Ada Lovelace / Blackwell architecture whitepapers for the citation you hand in.
"""

CARD_SPECS = {
    "4090": {
        "full_name": "NVIDIA GeForce RTX 4090",
        "architecture": "Ada Lovelace (AD102)",
        "tensor_core_gen": 4,
        "memory_type": "GDDR6X",
        "memory_bus_bits": 384,
        "bandwidth_gbps": 1008.0,
        "vram_gb": 24,
        "fp32_tflops": 82.6,
        "tf32_tflops": 82.6,
        "fp16_tflops": 330.3,
        "bf16_tflops": 330.3,
        "fp8_tflops": 660.6,
        "source": "https://www.runpod.io/articles/guides/nvidia-rtx-4090",
    },
    "5090": {
        "full_name": "NVIDIA GeForce RTX 5090",
        "architecture": "Blackwell (GB202)",
        "tensor_core_gen": 5,
        "memory_type": "GDDR7",
        "memory_bus_bits": 512,
        "bandwidth_gbps": 1792.0,
        "vram_gb": 32,
        "fp32_tflops": 104.8,
        "tf32_tflops": 104.75,
        "fp16_tflops": 209.5,
        "bf16_tflops": 209.5,
        "fp8_tflops": 419.0,  # not independently confirmed at write time; re-verify
        "source": "https://www.spheron.network/blog/nvidia-rtx-5090-specs/",
    },
}


def get_peak_specs(gpu_name: str) -> dict:
    """Fuzzy-match a torch/nvidia-smi device name ('NVIDIA GeForce RTX 4090')
    to a CARD_SPECS entry."""
    for key, spec in CARD_SPECS.items():
        if key in gpu_name:
            return spec
    raise KeyError(
        f"No peak-spec entry for GPU '{gpu_name}'. Add one to CARD_SPECS in "
        f"common/peak_specs.py with a cited vendor source before running Part B."
    )
