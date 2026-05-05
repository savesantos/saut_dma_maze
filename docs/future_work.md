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
