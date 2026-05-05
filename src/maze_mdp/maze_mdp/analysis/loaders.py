"""
Load training and deployment artifacts from ``data/`` into pandas DataFrames.

Each ``data/training/<algo>/<maze>/<run_id>/`` produces one row in
``load_training_runs`` (with arrays inlined as object columns).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _read_metrics_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_training_runs(root: Path | str = 'data/training') -> pd.DataFrame:
    """Walk ``data/training`` and return one row per run with metadata + arrays."""
    root = Path(root)
    rows: list[dict] = []
    if not root.exists():
        return pd.DataFrame(rows)
    for summary_path in sorted(root.rglob('summary.json')):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text())
        params_path = run_dir / 'params.yaml'
        params = yaml.safe_load(params_path.read_text()) if params_path.exists() else {}
        metrics = _read_metrics_csv(run_dir / 'metrics.csv')
        policy_path = run_dir / 'policy.npz'
        policy = (
            {k: np.asarray(v) for k, v in np.load(policy_path, allow_pickle=True).items()}
            if policy_path.exists()
            else {}
        )
        rows.append({
            **summary,
            'params': params,
            'metrics': metrics,
            'policy': policy,
        })
    return pd.DataFrame(rows)


def load_deployment_runs(root: Path | str = 'data/deployment') -> pd.DataFrame:
    """Walk ``data/deployment`` and return one row per recorded deployment."""
    root = Path(root)
    rows: list[dict] = []
    if not root.exists():
        return pd.DataFrame(rows)
    for summary_path in sorted(root.rglob('summary.json')):
        rows.append(json.loads(summary_path.read_text()))
    return pd.DataFrame(rows)


__all__ = ['load_training_runs', 'load_deployment_runs']
