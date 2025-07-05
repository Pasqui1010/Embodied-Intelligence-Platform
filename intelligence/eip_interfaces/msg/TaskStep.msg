# TaskStep.msg
# Single step in an LLM-generated task plan

string action_type             # Verb/action (e.g. 'move', 'pick')
string description             # Natural-language description
geometry_msgs/Pose target_pose # Desired pose in map/world frame
string[] parameters            # Additional parameters (e.g. 'speed=0.2')
float32 estimated_duration     # Expected duration in seconds
string[] preconditions         # Preconditions before execution
string[] postconditions        # Expected conditions after execution 