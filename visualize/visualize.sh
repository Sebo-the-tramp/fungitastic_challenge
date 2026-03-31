#!/usr/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT_DIR"

echo "Creating plots"
uv run --with plotly --with matplotlib python visualize_results/plot_mlp_vs_baseline.py
