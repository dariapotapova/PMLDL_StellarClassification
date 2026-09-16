#!/bin/sh
# References: no external implementation was used in this original orchestration script.
set -eu

python -m src.verify_environment
python -m src.prepare_data
python -m pytest -q
python -m src.baseline_benchmark
python -m src.verify_results
python scripts/audit_anonymity.py

echo "Reproduction completed successfully."
echo "Machine-readable metrics: reproduction_outputs/verified_metrics.json"
