"""GPU identity helpers. Every result row produced by this repo is labelled
with the UUID returned by get_gpu_uuid() so measurements from different
machines (the 4090 box and the 5090 box) never get mixed in analysis.
"""
import subprocess
import torch


def _nvidia_smi_query(fields, index=0):
    """Run nvidia-smi --query-gpu and return the requested fields as strings."""
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
        "-i", str(index),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    row = out.stdout.strip().split("\n")[0]
    return [v.strip() for v in row.split(",")]


def get_gpu_uuid(index=0):
    return _nvidia_smi_query(["uuid"], index=index)[0]


def _parse_float_or_none(v):
    """nvidia-smi reports '[N/A]' for fields a card doesn't expose -- e.g.
    memory.total on unified-memory (ATS) systems that have no discrete VRAM
    counter. Discrete cards like the 4090/5090 always report a number here;
    this just keeps provenance capture from crashing when they don't."""
    try:
        return float(v)
    except ValueError:
        return None


def get_gpu_identity(index=0):
    """Full provenance block for Part A: uuid, name, driver, VRAM, power limit."""
    fields = ["uuid", "name", "driver_version", "memory.total", "power.limit"]
    uuid, name, driver, mem_total_mib, power_limit_w = _nvidia_smi_query(fields, index=index)
    return {
        "uuid": uuid,
        "name": name,
        "driver_version": driver,
        "vram_total_mib": _parse_float_or_none(mem_total_mib),
        "power_limit_w": _parse_float_or_none(power_limit_w),
        "cuda_runtime_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


if __name__ == "__main__":
    import json
    n = torch.cuda.device_count()
    print(f"visible CUDA devices: {n}")
    for i in range(n):
        print(json.dumps(get_gpu_identity(i), indent=2))
