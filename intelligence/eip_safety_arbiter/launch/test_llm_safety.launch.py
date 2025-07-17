from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        # Launch the Safety Monitor
        Node(
            package='eip_safety_arbiter',
            executable='safety_monitor',
            name='safety_monitor',
            output='screen',
            parameters=[{
                'enable_llm_safety_check': True,
                'safety_llm_model': 'mistral-7b',
                'safety_confidence_threshold': 0.7,
                'safety_check_frequency': 1.0,
                'human_proximity_threshold': 1.5,
                'collision_distance_threshold': 0.5,
                'max_linear_velocity': 1.0,
                'max_angular_velocity': 1.0,
            }]
        ),
        
        # Launch the test node
        Node(
            package='eip_safety_arbiter',
            executable='test_llm_safety_evaluation.py',
            name='safety_evaluation_tester',
            output='screen',
            prefix='python3',
            # Run with a small delay to ensure the safety monitor is ready
            arguments=[os.path.join(
                get_package_share_directory('eip_safety_arbiter'),
                'test',
                'test_llm_safety_evaluation.py'
            )]
        ),
    ])
