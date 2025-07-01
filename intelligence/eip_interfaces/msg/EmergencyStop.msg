# Emergency stop signal message
builtin_interfaces/Time timestamp
string reason
int32 safety_level
string[] violations
bool requires_human_intervention
geometry_msgs/Pose robot_pose_at_stop 