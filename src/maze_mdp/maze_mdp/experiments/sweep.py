"""
Sweep driver: run :func:`maze_mdp.experiments.runner.train_one` over a YAML grid.

Sweep YAML schema:

.. code-block:: yaml

    algos: [vi, sarsa, qlearning]
    mazes: [fixture_3x3, fixture_5x5_corridor, fixture_7x7_loop]
    seeds: [0, 1, 2, 3, 4]
    mdp_config: {gamma: 0.99, slip_prob: 0.1}
    td_config:  {n_episodes: 2000}
    out: data
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path

import yaml

from maze_mdp._td import TDConfig
from maze_mdp.experiments.runner import train_one
from maze_mdp.mdp import MDPConfig


def _build_configs(spec: dict) -> tuple[MDPConfig, TDConfig]:
    mdp_overrides = spec.get('mdp_config') or {}
    td_overrides = spec.get('td_config') or {}
    return MDPConfig(**mdp_overrides), TDConfig(**td_overrides)


def _td_for_maze(spec: dict, maze: str) -> TDConfig:
    """Apply ``td_config_overrides[<maze>]`` on top of the global ``td_config``."""
    base = (spec.get('td_config') or {}).copy()
    overrides = (spec.get('td_config_overrides') or {}).get(maze, {})
    base.update(overrides)
    return TDConfig(**base)


def run_sweep(spec: dict) -> list[dict]:
    """Execute every (algo, maze, seed) triple in ``spec`` sequentially."""
    algos = spec['algos']
    mazes = spec['mazes']
    seeds = spec['seeds']
    out_root = Path(spec.get('out', 'data'))
    mdp_cfg, _ = _build_configs(spec)
    summaries: list[dict] = []
    triples = list(product(algos, mazes, seeds))
    total = len(triples)
    for i, (algo, maze, seed) in enumerate(triples, 1):
        td_cfg = _td_for_maze(spec, maze)
        print(f'[{i}/{total}] {algo} {maze} seed={seed} '
              f'episodes={td_cfg.n_episodes}', flush=True)
        s = train_one(
            algo=algo,
            maze_name=maze,
            seed=int(seed),
            out_root=out_root,
            mdp_config=mdp_cfg,
            td_config=td_cfg,
        )
        summaries.append(s)
    return summaries


def main(argv: list[str] | None = None) -> None:
    """CLI entry point ``sweep`` — execute all (algo, maze, seed) triples."""
    parser = argparse.ArgumentParser(description='Run a parameter sweep.')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    spec = yaml.safe_load(args.config.read_text())
    summaries = run_sweep(spec)
    # Echo configuration + summary for reproducibility.
    print(json.dumps({
        'spec': spec,
        'mdp_config': asdict(_build_configs(spec)[0]),
        'td_config': asdict(_build_configs(spec)[1]),
        'n_runs': len(summaries),
    }, indent=2))


if __name__ == '__main__':
    main()
