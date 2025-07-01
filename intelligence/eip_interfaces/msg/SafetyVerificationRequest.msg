# Safety verification request message
builtin_interfaces/Time timestamp
string requester_id
string task_description
geometry_msgs/Pose robot_pose
geometry_msgs/Twist cmd_vel
string[] environmental_constraints
float32 urgency_level 