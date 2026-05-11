# `goal_reward = 10.0`, 7x7 SARSA budget tuned

Same MDP and algorithms as `reward10/`. Only the SARSA / Q-learning
training schedule for the largest maze was changed, in response to
the diagnosis in `reward10/README.md` ("SARSA on `fixture_7x7_loop`
is materially broken: V*-Vπ ≈ 24.5, success 0.75").

## What changed vs. `reward10/`

Only [src/maze_bringup/config/sweeps/default.yaml](../../../src/maze_bringup/config/sweeps/default.yaml),
inside `td_config_overrides.fixture_7x7_loop`:

- `n_episodes`: 30 000 → **80 000**.
- `epsilon_anneal_frac`: 0.5 → **0.8** (slower ε decay so the agent
  keeps exploring further into training).

Everything else — `MDPConfig`, the other two mazes, all seeds, all
other TD hyperparameters — is identical to the `reward10/` archive.
VI is unaffected (it does not consume `TDConfig`).

## What was achieved

Read off [comparison.png](comparison.png) (final policy evaluation,
mean over 5 seeds, stochastic rollouts):

| metric                      | `reward10/` 7x7 | `reward10_7x7_tuned/` 7x7 |
| --------------------------- | --------------- | ------------------------- |
| SARSA `V*` − `Vπ`           | ≈ 24.5          | **≈ 4.0**                 |
| SARSA success rate          | 0.75            | **0.94**                  |
| Q-Learning `V*` − `Vπ`      | ≈ 3.1           | ≈ 0.0                     |
| Q-Learning success rate     | 0.97            | 1.00                      |

SARSA's value-function gap on `fixture_7x7_loop` collapsed by ~6×
and its rollout success rate jumped from 0.75 to 0.94, confirming
that the previous gap was *under-training* (not enough ε late in
the schedule to fix the wobble between the two homotopy paths
around the central island), not a structural pathology.

Q-Learning becomes essentially optimal on the 7x7 loop with the
extended budget too — the few non-greedy cells left in `reward10/`
were also visit-distribution artifacts.

## What is still suboptimal

Visible in [policy_heatmap.png](policy_heatmap.png) and
[comparison.png](comparison.png):

- **SARSA on `fixture_5x5_corridor`** still has `V*` − `Vπ` ≈ 2.8
  and success ≈ 0.96 (vs 1.00 for VI and QL). Budget there is
  already 5 000 episodes and was not changed; the residual gap is
  the visit-distribution / argmax-noise issue (zero-initialised Q
  + ε-greedy tie-breaks at rarely-visited states), *not*
  under-training. The next planned fix is **pessimistic Q init**
  in [_td.py](../../../src/maze_mdp/maze_mdp/_td.py) which would
  remove the optimistic-zero tie-break without changing the MDP.
- **Unreachable interior cells of `fixture_7x7_loop`** still show
  arbitrary policy arrows in the heatmap. Cosmetic — the agent
  cannot enter them; their Q rows are all zeros. Worth masking in
  the plot before the report.
- **Wide SARSA confidence band** on the 7x7 loop in
  [convergence.png](convergence.png) — even with 80 k episodes and
  slower ε decay, run-to-run variance is roughly 2× Q-learning's.
  This is intrinsic to on-policy bootstrapping and persistent ε,
  not something more episodes will close further.

## Reproducing

```bash
cd /home/salva/saut_dma_maze
PYTHONPATH=src/maze_mdp python3 -m maze_mdp.experiments.sweep \
    --config src/maze_bringup/config/sweeps/default.yaml
PYTHONPATH=src/maze_mdp bash scripts/make_all_figures.sh
```
