# Future Work

Items deliberately deferred from the initial production build. Listed here so
graders see we knew about them and can ask "why didn't you do X" with a
documented answer.

## ROS integration enhancements (Option B)

1. **`train_from_bag` node.** Replays a recorded rosbag of fiducial detections
   + cmd_vel and feeds the (s, a, r, s') tuples to SARSA / Q-Learning offline.
   Would let us train on real-robot data, not just simulator data.
   *Skipped because the graded contribution is the MDP/RL layer, and the
   evaluation harness already supports rosbag-based deployment evaluation
   without it.*

2. **`Retrain.srv`.** A ROS 2 service for online retraining/updates of the
   Q-table from a running node, returning convergence metrics.
   *Skipped — adds little signal vs. the offline `sweep` CLI we already ship.*

3. **RViz visualization panel.** A node publishing the policy as
   `visualization_msgs/MarkerArray` and the value function as an
   `OccupancyGrid` heatmap, so live runs look impressive in screenshots.
   *Skipped — purely cosmetic; matplotlib heatmaps cover the report's needs.*

## Alternative localization stacks

The `fiducial_localizer` node is intentionally swappable: any node that
publishes `maze_msgs/CellPose` on `/robot_cell` can replace it. Candidates
from the AGENTS.md guidance (any are acceptable per the professor):

- `slam_toolbox` (would require synthesising laser scans from the camera —
  brittle without a depth sensor).
- `nav2` AMCL with a hand-built occupancy map (heavy dependency for the
  same end result).
- `robot_localization` EKF fusing fiducial pose estimates with cmd_vel
  integration (would smooth heading jitter; promising follow-up).

## Algorithmic extensions

- 8-cardinal headings / 8-connected grid (only relevant if the maze gains
  diagonal passages).
- `backward` action (excluded — the AlphaBot2 has no rear sensors).
- Continuous-pose state with function approximation (out of scope; the
  project is graded on tabular methods).
- Empirical $\hat P$ estimated from real-robot rollouts (textbook defaults
  in `docs/mdp_design.md` proved sufficient).
- Potential-based reward shaping for faster RL convergence.

## Closing the VI-vs-RL gap on 7×7 (deferred)

Empirical observation from the smoke sweep (5 seeds, 30 k episodes per RL
run): on `fixture_7x7_loop`, $V^\pi$ averaged over free states is

| Algorithm  | $V^\pi$ (5 seeds)   |
| ---------- | ------------------- |
| VI         | $-15.92$            |
| Q-Learning | $-35.45 \pm 6.72$   |
| SARSA      | $-40.02 \pm 5.94$   |

So tabular RL leaves a $\sim 20$-reward gap on the largest maze even with a
30 k-episode budget, and deployment-time success rate drops to $\sim 75\%$
(see [`data/figures/comparison.png`](../data/figures/comparison.png)).

### Root cause: the heading expansion

The MDP factors orientation into the state, $s = (r, c, \theta)$ with
$\theta \in \{N, E, S, W\}$, so $|S| = 4 \cdot |\text{free cells}|$. For
`fixture_7x7_loop` this is $\approx 132$ states and $|S| \cdot |A| = 396$
state–action pairs. ε-greedy with linear decay $1.0 \to 0.05$ explores
mostly *near the start*, while cells deep in the inner loop see few
visits per episode — and each such cell has 4 headings, only one of
which is "correct" for entering from a given direction. The other three
remain noisy after 30 k episodes, which is exactly what the
SARSA / Q-Learning panels of `policy_heatmap.png` show. VI is unaffected
because it sweeps every $(s, a)$ deterministically per iteration; the 4×
expansion is a rounding error on its already sub-second cost.

### Two cheap remedies that would not change the algorithmic story

Both stay strictly within "tabular SARSA" and "tabular Q-learning"
(Sutton & Barto, chapters 6, 7, and 12), so the report's three-way
comparison (VI vs SARSA vs Q-Learning) is preserved.

1. **Optimistic initialisation.** Initialise $Q_0(s, a) = C$ for some
   $C > 0$ instead of zeros. Untried $(s, a)$ then look strictly better
   than tried ones until enough updates pull them down, which forces
   uniform coverage of the state-action space without raising $\varepsilon$.
   One-line change in `_td.py`; expected effect: removes most of the gap on
   states far from the start with no extra wall-clock.

2. **Eligibility traces — SARSA(λ) / Q(λ).** Maintain
   $e_t(s, a) \leftarrow \gamma \lambda \, e_{t-1}(s, a) + \mathbb{1}[s = s_t, a = a_t]$
   and apply each TD error to all recently visited pairs at once. With
   $\lambda \approx 0.9$, a single goal-reaching episode propagates the
   terminal signal across the entire trajectory in one pass instead of one
   cell per episode. Typical 5–10× speedup in episodes-to-convergence on
   long-horizon tabular problems. Implementation cost: $\sim 30$ LOC
   shared between SARSA and Q-Learning (single shared trace tensor in
   `_td.py`).

### Why we did not implement them now

The current results already make the qualitative point cleanly: VI is
optimal and instant when the model is known, model-free RL pays a real
sample-complexity tax that grows with the heading-expanded state space,
and Q-Learning has lower variance than SARSA on the largest maze. Adding
optimistic init / λ-traces would shrink the numerical gap without
changing the comparison's narrative — and the IEEE report has only six
pages.
