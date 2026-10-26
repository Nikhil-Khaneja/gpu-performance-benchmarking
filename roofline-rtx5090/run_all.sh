#!/usr/bin/env bash
# Convenience driver for the full HW2.5 suite on a GPU machine. Run each
# part deliberately rather than blindly firing this end to end the first
# time -- Part E alone takes 20 minutes, and Part D's largest sizes can OOM
# by design (that's the point). Meant to be read and run part by part.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "== Part A: provenance =="
bash part_a_provenance/capture_nvidia_smi.sh
python part_a_provenance/record_card_specs.py

echo "== Part B: precision throughput =="
python part_b_precision/benchmark_matmul.py --try-fp8
python part_b_precision/plot_matmul.py

echo "== Part C: bandwidth vs compute bound =="
python part_c_bandwidth_compute/benchmark_bandwidth_compute.py

echo "== Part D: cost of attention =="
python part_d_attention/benchmark_attention.py
python part_d_attention/plot_attention_memory.py

echo "== Part E: sustained load (20 min) =="
python part_e_thermal/sustained_load.py --minutes 20
python part_e_thermal/plot_thermal.py

echo "== Part F: summary table =="
python part_f_analysis/build_metrics_table.py

echo "Done. Review results/, figures/, RUN_LOG.txt, and fill in METRICS.md."
