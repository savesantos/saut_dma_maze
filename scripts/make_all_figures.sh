#!/usr/bin/env bash
# Regenerate every figure in `data/figures/` from runs under `data/`.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m maze_mdp.analysis.plot_convergence "$@"
python3 -m maze_mdp.analysis.plot_policy_heatmap "$@"
python3 -m maze_mdp.analysis.plot_comparison "$@"
echo "Figures written to data/figures/"
