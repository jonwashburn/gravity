#!/usr/bin/env bash
set -euo pipefail

# Move to this script's directory
cd "$(dirname "$0")"

echo "[make_figs] Building SPARC master table..."
python build_sparc_master_table.py

echo "[make_figs] Plotting example rotation curve..."
python plot_example_rc.py

echo "[make_figs] Done. See ../results/example_rc.pdf"


