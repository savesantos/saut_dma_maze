# Baseline run — `goal_reward = 0.0`

Snapshot of the figures produced before changing the reward function.
Kept for the report so we can compare with the new run under
`goal_reward = 10.0`.

## Conditions

- Date generated: 2026-05-11.
- Code state: `MDPConfig.goal_reward = 0.0`; everything else as in
  [docs/mdp_design.md](../../../docs/mdp_design.md).
- Sweep config: [src/maze_bringup/config/sweeps/default.yaml](../../../src/maze_bringup/config/sweeps/default.yaml)
  (algos = vi, sarsa, qlearning; mazes = 3x3, 5x5 corridor, 7x7 loop;
  seeds 0–4; per-maze episode budgets 1k / 5k / 30k).
- `MDPConfig`: `slip_prob = 0.1`, `turn_fail_prob = 0.0`,
  `forward_cost = 1.0`, `turn_cost = 1.2`, `bump_cost = 2.5`,
  `goal_reward = 0.0`, `gamma = 0.99`.
- `TDConfig` (SARSA / Q-learning): `alpha0 = 0.1`, `alpha_min = 1e-3`,
  `alpha_tau = 50`, `epsilon0 = 1.0`, `epsilon_min = 0.05`,
  `epsilon_anneal_frac = 0.5`. Q initialised to zeros.

## Files

- `convergence.png` — return vs. episode (mean ± std over 5 seeds), per algo / maze.
- `steps_curve.png` — steps-to-goal vs. episode.
- `policy_heatmap.png` — value heatmap + per-heading policy arrows.
- `comparison.png` — final steps-to-goal & success-rate bars, VI vs SARSA vs QL.

## What the baseline shows

Symptom in `policy_heatmap.png` (visible most clearly on
SARSA / fixture_3x3, top-left cell):

- At the corner cell `(row=0, col=0)`, the greedy policy for
  heading `N` and heading `W` selects turns that flip the agent
  between the two wall-facing headings, producing a `N ↔ W`
  spin loop instead of progressing toward the goal at `(2, 2)`.

Why it happens with `goal_reward = 0.0`:

1. **All rewards are ≤ 0** (forward / turn / bump costs are
   the only nonzero rewards), so every Q-value the agent has
   ever updated is strictly negative.
2. **Q is initialised to zeros**, so any state-action pair that
   has not been visited keeps `Q = 0`, which is the
   *optimistic* upper bound. The ε-greedy argmax in
   `_td.epsilon_greedy` therefore systematically picks the
   *less-tried* action at under-visited states.
3. The corner `(0, 0, N|W)` is sampled rarely as a start under
   the uniform `GridMaze.reset()`, and almost never *traversed*
   by trajectories that already learned to go S/E. So its
   Q-row stays near zero for both turn actions, and the
   ε-greedy tie-break picks whichever turn was visited *less*.
4. SARSA's on-policy bootstrap `Q[s', a']` reuses the same
   under-visited entries from the next state, so the negative
   information from "spinning forever costs `-1.2 / (1-γ)`"
   never fully back-propagates to the corner.

VI does not exhibit the loop because it sweeps all
`(s, a)` entries equally — the artifact is a property of the
RL learner, not of the MDP.

## How we plan to fix it

Two complementary options were discussed:

- **Pessimistic Q init** (algorithm-side fix, keeps the MDP
  unchanged → cleanest for the apples-to-apples comparison).
- **`goal_reward = 10.0`** (environment-side fix): makes the
  goal a positive attractor so back-propagated Q-values can be
  positive. This eliminates the "unvisited zero is the argmax"
  trap because surrounding states acquire `Q > 0` from the
  goal reward, dwarfing the optimistic zeros at corners.
  Applied uniformly to VI / SARSA / QL so the comparison
  remains valid.

The companion archive `data/archive/reward10/` (next run)
contains the same plots with `goal_reward = 10.0` for the
side-by-side comparison in the report.

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
cp -a data/archive/reward0_baseline/training data/
PYTHONPATH=src/maze_mdp bash scripts/make_all_figures.sh
```

To retrain end-to-end from the pinned sweep config:

```bash
cd /home/salva/saut_dma_maze
bash scripts/rerun_archive.sh reward0_baseline
```
