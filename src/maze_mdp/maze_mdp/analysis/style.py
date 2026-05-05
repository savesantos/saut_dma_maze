"""Shared matplotlib styling for the report figures."""

from __future__ import annotations

import matplotlib as mpl

ALGO_COLORS = {
    'vi': '#1f77b4',
    'sarsa': '#ff7f0e',
    'qlearning': '#2ca02c',
}

ALGO_LABELS = {
    'vi': 'Value Iteration',
    'sarsa': 'SARSA',
    'qlearning': 'Q-Learning',
}


def apply() -> None:
    """Apply a consistent style to matplotlib's global rc."""
    mpl.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': 200,
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.frameon': False,
    })
