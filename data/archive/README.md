# Archived experimental scenarios

Each subdirectory is a **fully reproducible snapshot** of one
training sweep used to debug or motivate a design choice in the
project. Five scenarios are archived here; they form the experimental
narrative documented in the report.

| Scenario                                  | MDP `goal_reward` | TD `q_init`    | 7x7 episodes | 7x7 ε-anneal frac |
| ----------------------------------------- | ----------------- | -------------- | ------------ | ----------------- |
| `reward0_baseline`                        | 0.0               | `zeros`        | 30 000       | 0.5               |
| `reward10`                                | 10.0              | `zeros`        | 30 000       | 0.5               |
| `reward10_7x7_tuned`                      | 10.0              | `zeros`        | 80 000       | 0.8               |
| `reward10_pessimistic_init`               | 10.0              | `pessimistic`  | 30 000       | 0.5               |
| `reward10_pessimistic_init_7x7_tuned`     | 10.0              | `pessimistic`  | 80 000       | 0.8               |

Each archive directory contains:

- `README.md` &mdash; narrative discussion of what the run shows and
  why it was added (the scientific log entry).
- `sweep.yaml` &mdash; the exact sweep configuration consumed by
  [maze_mdp.experiments.sweep](../../src/maze_mdp/maze_mdp/experiments/sweep.py).
  This is the single source of truth for the scenario's
  hyperparameters &mdash; in particular `goal_reward`, `q_init`, and
  the per-maze episode budgets.
- `training/<algo>/<maze>/<run_id>/` &mdash; full per-run training
  artifacts (`policy.npz`, `params.yaml`, `metrics.csv`,
  `summary.json`) for every `(algo, maze, seed)` triple, plus the
  `selected.json` produced by
  [select_best_run](../../src/maze_mdp/maze_mdp/analysis/select_best_run.py)
  identifying which seed was promoted for the heatmap / deployment.
- `figures/` &mdash; PNG **and** PDF versions of the four canonical
  plots (`convergence`, `steps_curve`, `policy_heatmap`,
  `comparison`). The PNGs are also duplicated at the archive root
  so existing references in the report keep working.

## Regenerating an archive

To rebuild **one** scenario from scratch (retraining + plotting):

```bash
cd /home/salva/saut_dma_maze
bash scripts/rerun_archive.sh <scenario>
```

To rebuild **all** scenarios at once (long &mdash; on the order of
1&ndash;2 hours wall clock):

```bash
cd /home/salva/saut_dma_maze
bash scripts/rerun_archive.sh
```

## Running an archived policy

The archived `policy.npz` files are deserialised with
[maze_mdp.policy.load_policy](../../src/maze_mdp/maze_mdp/policy.py).
For example, to roll out the SARSA policy promoted for the 7x7 maze
in the best-result scenario:

```bash
cd /home/salva/saut_dma_maze
PYTHONPATH=src/maze_mdp python3 -c "
from maze_mdp.policy import load_policy
import json, pathlib
sel = json.loads(pathlib.Path(
    'data/archive/reward10_pessimistic_init_7x7_tuned/'
    'training/sarsa/fixture_7x7_loop/selected.json'
).read_text())
print(load_policy(sel['selected']['policy_path'])['pi'])
"
```

Or replay it on the ROS-free simulator / inside RViz via the
`policy_runner` and `compare_node` nodes documented in
[docs/visualization.md](../../docs/visualization.md).
