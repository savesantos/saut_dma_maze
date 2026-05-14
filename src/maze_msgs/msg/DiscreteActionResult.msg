std_msgs/Header header
# Echoes the goal_id of the DiscreteActionGoal this result corresponds to.
uint32 goal_id
# True if the action terminated by hitting its success condition
# (e.g. reached the next intersection for FORWARD, perpendicular line
# re-acquired for TURN_*). False on any failure_mode below.
bool success
# Failure taxonomy. Only meaningful when success == False.
# 0=NONE, 1=LINE_LOST, 2=TIMEOUT, 3=COLLISION, 4=ABORTED
uint8 failure_mode

uint8 NONE=0
uint8 LINE_LOST=1
uint8 TIMEOUT=2
uint8 COLLISION=3
uint8 ABORTED=4
