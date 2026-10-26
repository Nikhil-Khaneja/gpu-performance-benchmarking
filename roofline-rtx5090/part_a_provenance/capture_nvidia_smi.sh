#!/usr/bin/env bash
# Part A: capture full nvidia-smi -q output plus the UUID/driver/CUDA/VRAM/
# power-limit fields the assignment asks you to record explicitly.
# Run this once per machine (RTX 4090 box, RTX 5090 box) and commit both
# outputs -- do not overwrite one machine's capture with the other's.
set -euo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results/provenance"
mkdir -p "$OUT_DIR"

HOST="$(hostname -s 2>/dev/null || hostname)"
TS="$(date +%Y%m%d_%H%M%S)"
UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits -i 0)"
SAFE_UUID="${UUID//-/}"

FULL_OUT="${OUT_DIR}/nvidia_smi_q_${HOST}_${SAFE_UUID}_${TS}.txt"
SUMMARY_OUT="${OUT_DIR}/gpu_summary_${HOST}_${SAFE_UUID}_${TS}.csv"

nvidia-smi -q > "$FULL_OUT"

nvidia-smi --query-gpu=uuid,name,driver_version,memory.total,power.limit \
  --format=csv > "$SUMMARY_OUT"

echo "Wrote:"
echo "  $FULL_OUT"
echo "  $SUMMARY_OUT"
echo
echo "Commit both files. GPU UUID for this machine: $UUID"
