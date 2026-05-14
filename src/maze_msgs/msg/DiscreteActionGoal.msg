std_msgs/Header header
# Monotonically increasing id chosen by the policy runner so that the
# matching DiscreteActionResult can be correlated unambiguously.
uint32 goal_id
# Action to execute. Values 0..2 match maze_mdp.mdp.Action; 3 is the
# final-approach primitive used at the goal cell (drive straight until the
# goal fiducial fills enough of the camera frame, then stop).
uint8 action

uint8 FORWARD=0
uint8 TURN_LEFT=1
uint8 TURN_RIGHT=2
uint8 DRIVE_UNTIL_MARKER=3
