# `goal_reward = 10.0` run

Re-run of the same sweep as `reward0_baseline/`, this time with the
positive goal reward applied uniformly to VI / SARSA / Q-Learning.

## Conditions

- Date generated: 2026-05-11.
- Code state: `MDPConfig.goal_reward = 10.0`. All other hyperparameters
  unchanged — see [docs/mdp_design.md](../../../docs/mdp_design.md).
- Sweep config: identical to baseline
  ([src/maze_bringup/config/sweeps/default.yaml](../../../src/maze_bringup/config/sweeps/default.yaml)),
  same algos / mazes / seeds / episode budgets, same seeds.

## What changed vs. `reward0_baseline/`

`policy_heatmap.png`:

- The corner `(0, 0)` of fixture_3x3 under SARSA no longer shows a
  `N ↔ W` spin loop. With the goal acting as a positive attractor,
  surrounding states acquire `Q > 0` from goal back-propagation,
  dwarfing the optimistic zeros at under-visited corners and
  removing the "unvisited zero is the argmax" tie-break failure.
- VI policies are essentially unchanged in topology — VI sweeps every
  state regardless of reward sign, so the goal-reward only rescales
  values, not arg-maxes.
- Q-Learning's policies on fixture_5x5_corridor and fixture_7x7_loop
  are visibly more consistent across cells (fewer "decorative" turns
  in already-aligned states) because the value gradient toward the
  goal is steeper.

`convergence.png` / `steps_curve.png`:

- Returns are now positive once the agent reaches the goal; the
  curves trend upward instead of toward zero from below.
- Episode length to convergence is similar — the goal reward changes
  the *sign* of the value function, not how many TD updates are
  needed for it to stabilise.

`comparison.png`:

- Final steps-to-goal and success rates are comparable to baseline
  on the 5x5 / 7x7 mazes; the headline change is the disappearance
  of the corner-loop pathology on the 3x3 maze.

## Why this is still a valid VI vs SARSA vs QL comparison

The reward change was applied to the **MDP**, which all three
algorithms consume identically through `MDPConfig`. We are comparing
three planners / learners on the *same* (new) MDP rather than tuning
SARSA-only knobs to make it look better. The baseline archive
documents the previous MDP and its artifact for the report's
"reward shaping" discussion.

## Reproducing

Everything needed to re-run this scenario is pinned in the archive:

- `sweep.yaml` — exact sweep configuration consumed by the trainer.
- `training/<algo>/<maze>/<run_id>/` — `policy.npz`, `params.yaml`,
  `metrics.csv`, `summary.json` for every (algo, maze, seed) triple,
  plus `selected.json` identifying the policy picked by
  `select_best_run` for downstream consumers.
- `figures/` — PNG + PDF versions of all four canonical plots.

To regenerate the figures from the archived policies (no retraining):

```bash
cd /home/salva/saut_dma_maze
rm -rf data/training data/figures
cp -a data/archive/reward10/training data/
PYTHONPATH=src/maze_mdp bash scripts/make_all_figures.sh
```

To retrain end-to-end from the pinned sweep config:

```bash
cd /home/salva/saut_dma_maze
bash scripts/rerun_archive.sh reward10
```
