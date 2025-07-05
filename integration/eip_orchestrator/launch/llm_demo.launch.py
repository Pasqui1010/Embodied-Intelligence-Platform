#!/usr/bin/env python3
"""
LLM Demo Launch File

Launches the LLM interface with safety monitoring for testing
language-guided robot behavior.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for LLM demo"""
    
    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    model_name_arg = DeclareLaunchArgument(
        'model_name',
        default_value='microsoft/DialoGPT-medium',
        description='Hugging Face model name to use'
    )
    
    # Safety monitor node
    safety_monitor_node = Node(
        package='eip_safety_arbiter',
        executable='safety_monitor',
        name='safety_monitor',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'safety_check_frequency': 10.0,
            'collision_distance_threshold': 0.5,
            'human_proximity_threshold': 1.0,
            'max_linear_velocity': 1.0,
            'max_angular_velocity': 1.0,
            'enable_llm_safety_check': True,
            'safety_confidence_threshold': 0.8
        }]
    )
    
    # LLM interface node
    llm_interface_node = Node(
        package='eip_llm_interface',
        executable='llm_interface_node',
        name='llm_interface',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'model_name': LaunchConfiguration('model_name'),
            'device': 'auto',
            'max_response_time': 5.0,
            'enable_safety_embedding': True,
            'safety_confidence_threshold': 0.8
        }]
    )
    
    # Safety simulator for testing
    safety_simulator_node = Node(
        package='eip_safety_simulator',
        executable='safety_simulator_node',
        name='safety_simulator',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'simulation_frequency': 10.0,
            'enable_all_scenarios': True
        }]
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        model_name_arg,
        safety_monitor_node,
        llm_interface_node,
        safety_simulator_node
    ]) 