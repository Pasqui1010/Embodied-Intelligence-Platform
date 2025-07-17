from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='eip_advanced_learning',
            executable='learning_engine_node',
            name='learning_engine_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'learning_rate': 0.01,
                'experience_buffer_size': 100
            }]
        ),
    ])
