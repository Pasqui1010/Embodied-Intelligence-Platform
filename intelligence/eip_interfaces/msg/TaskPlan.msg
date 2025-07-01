# LLM-generated task plan message
builtin_interfaces/Time timestamp
string plan_id
string goal_description
TaskStep[] steps
int32 estimated_duration_seconds
string[] required_capabilities
string[] safety_considerations

# Task step definition
MSG: intelligence/eip_interfaces/TaskStep
string action_type
string description
geometry_msgs/Pose target_pose
string[] parameters
float32 estimated_duration
string[] preconditions
string[] postconditions 