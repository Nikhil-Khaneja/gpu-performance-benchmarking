"""Part A: cross-check the live GPU identity (via nvidia-smi/torch) against
the vendor-documented peak specs in common/peak_specs.py, and emit a
Markdown block ready to paste into METRICS.md / PROVENANCE.md.

Usage: python part_a_provenance/record_card_specs.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.gpu_info import get_gpu_identity
from common.peak_specs import get_peak_specs


def _fmt(v, suffix="", ndigits=1):
    return f"{v:.{ndigits}f}{suffix}" if v is not None else "N/A (not reported by nvidia-smi on this card)"


def main():
    identity = get_gpu_identity(0)

    md = f"""### GPU provenance

- Name: {identity['name']}
- UUID: `{identity['uuid']}`
- Driver version: {identity['driver_version']}
- CUDA runtime (torch): {identity['cuda_runtime_version']}
- PyTorch version: {identity['torch_version']}
- VRAM: {_fmt(identity['vram_total_mib'] / 1024 if identity['vram_total_mib'] is not None else None, ' GiB')} (reported by nvidia-smi)
- Power limit: {_fmt(identity['power_limit_w'], ' W', 0)}
"""

    try:
        spec = get_peak_specs(identity["name"])
        md += f"""
### Vendor-documented specs -- {spec['full_name']}

- Architecture: {spec['architecture']}
- Tensor core generation: {spec['tensor_core_gen']}
- Memory: {spec['memory_type']}, {spec['memory_bus_bits']}-bit bus, {spec['bandwidth_gbps']:.0f} GB/s
- Peak FP32: {spec['fp32_tflops']} TFLOPS
- Peak TF32 (tensor core, dense): {spec['tf32_tflops']} TFLOPS
- Peak FP16 (tensor core, dense): {spec['fp16_tflops']} TFLOPS
- Peak BF16 (tensor core, dense): {spec['bf16_tflops']} TFLOPS
- Peak FP8 (tensor core, dense): {spec['fp8_tflops']} TFLOPS
- Source: {spec['source']}
  (VERIFY against NVIDIA's official spec sheet / architecture whitepaper
  before citing in your submission -- see common/peak_specs.py docstring.)
"""
    except KeyError as e:
        md += f"\n### Vendor-documented specs\n\n(none on file: {e})\n"

    print(md)

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "provenance", f"card_specs_{identity['uuid'].replace('-', '')}.md",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"Also wrote to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
