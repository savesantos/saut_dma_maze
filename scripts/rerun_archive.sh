#!/usr/bin/env bash
# Rerun every archived scenario sweep and persist the full reproducible
# bundle (sweep YAML + policies + metrics + figures + summary) under
# `data/archive/<scenario>/`.
#
# Each scenario corresponds to one YAML under
# `src/maze_bringup/config/sweeps/archive/`. The figures use the current
# plotting / analysis code (post-fix) so every archive is regenerated with
# a consistent visual style. The narrative `README.md` already in each
# archive directory is preserved.
#
# Usage:
#   bash scripts/rerun_archive.sh                 # all scenarios
#   bash scripts/rerun_archive.sh reward10        # one scenario by name
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
SWEEP_DIR="src/maze_bringup/config/sweeps/archive"
ARCHIVE_DIR="data/archive"

export PYTHONPATH="src/maze_mdp${PYTHONPATH:+:$PYTHONPATH}"

run_one() {
    local scenario="$1"
    local yaml="${SWEEP_DIR}/${scenario}.yaml"
    local dest="${ARCHIVE_DIR}/${scenario}"
    if [[ ! -f "$yaml" ]]; then
        echo "ERROR: sweep YAML not found: $yaml" >&2
        return 1
    fi
    echo "=================================================================="
    echo "[scenario] ${scenario}"
    echo "[config]   ${yaml}"
    echo "[archive]  ${dest}"
    echo "=================================================================="

    # Clean workspace `data/` slate so the sweep output is unambiguous.
    rm -rf data/training data/figures
    mkdir -p data/training data/figures

    # Train every (algo, maze, seed) triple in the YAML.
    python3 -m maze_mdp.experiments.sweep --config "$yaml"

    # Pick the best policy per (algo, maze) for the heatmap / deployment.
    python3 -m maze_mdp.analysis.select_best_run

    # Generate the four canonical figures.
    bash scripts/make_all_figures.sh

    # Lay out the archive directory. Keep the existing README.md (narrative
    # for the report); replace everything else with fresh, reproducible
    # artifacts.
    mkdir -p "${dest}/training" "${dest}/figures"
    if [[ -f "${dest}/README.md" ]]; then
        cp "${dest}/README.md" "${dest}/README.md.bak"
    fi
    # Wipe old generated content but keep README.md / README.md.bak.
    find "${dest}" -mindepth 1 -maxdepth 1 \
        ! -name 'README.md' ! -name 'README.md.bak' -exec rm -rf {} +

    # Pin the exact sweep config used.
    cp "$yaml" "${dest}/sweep.yaml"

    # Full training artifacts (policy.npz, params.yaml, metrics.csv,
    # summary.json, selected.json). Use cp -a so symlinks / timestamps
    # stay consistent.
    mkdir -p "${dest}/training"
    cp -a data/training/. "${dest}/training/"

    # Figures: keep both PNGs at archive root (so existing report
    # references to `comparison.png` etc. keep working) AND a clean
    # copy under figures/ alongside the PDFs.
    cp data/figures/*.png "${dest}/" || true
    cp -a data/figures/. "${dest}/figures/"

    # Restore the original README.md.bak as README.md if no README is
    # present (in case the user wiped it manually). Otherwise discard the
    # backup since the original is already in place.
    if [[ ! -f "${dest}/README.md" && -f "${dest}/README.md.bak" ]]; then
        mv "${dest}/README.md.bak" "${dest}/README.md"
    else
        rm -f "${dest}/README.md.bak"
    fi

    echo "[done] ${scenario} -> ${dest}"
}

if [[ $# -gt 0 ]]; then
    for s in "$@"; do
        run_one "$s"
    done
else
    for yaml in "${SWEEP_DIR}"/*.yaml; do
        scenario="$(basename "$yaml" .yaml)"
        run_one "$scenario"
    done
fi

echo "All archived scenarios regenerated."
