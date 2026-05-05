"""Schema test for the experiment harness."""

import json
from pathlib import Path

import yaml

from maze_mdp._td import TDConfig
from maze_mdp.experiments.runner import train_one
from maze_mdp.mdp import MDPConfig


def test_train_one_vi_writes_canonical_layout(tmp_path: Path):
    summary = train_one(
        algo='vi',
        maze_name='fixture_3x3',
        seed=0,
        out_root=tmp_path,
        mdp_config=MDPConfig(),
    )
    run_dir = Path(summary['run_dir'])
    assert (run_dir / 'policy.npz').exists()
    assert (run_dir / 'params.yaml').exists()
    assert (run_dir / 'summary.json').exists()
    saved = json.loads((run_dir / 'summary.json').read_text())
    assert saved['algo'] == 'vi'
    assert saved['converged'] is True


def test_train_one_qlearning_writes_metrics_csv(tmp_path: Path):
    summary = train_one(
        algo='qlearning',
        maze_name='fixture_3x3',
        seed=1,
        out_root=tmp_path,
        td_config=TDConfig(n_episodes=10),
    )
    run_dir = Path(summary['run_dir'])
    metrics = (run_dir / 'metrics.csv').read_text().splitlines()
    assert metrics[0] == 'episode,return,steps,epsilon,mean_td_error'
    assert len(metrics) == 1 + 10  # header + episodes
    params = yaml.safe_load((run_dir / 'params.yaml').read_text())
    assert params['algo'] == 'qlearning'
    assert params['td_config']['n_episodes'] == 10
