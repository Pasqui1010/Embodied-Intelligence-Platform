#!/usr/bin/env python3
"""
LLM Demo Launch File

Launches the LLM interface with basic SLAM demo for Week 2 integration.
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    """Generate the launch description for LLM demo"""
    
    # Launch arguments
    llm_provider_arg = DeclareLaunchArgument(
        'llm_provider',
        default_value='local_mistral',
        description='LLM provider to use (local_mistral, local_phi, openai, anthropic)'
    )
    
    enable_safety_arg = DeclareLaunchArgument(
        'enable_safety_verification',
        default_value='true',
        description='Enable safety verification for LLM plans'
    )
    
    # Include the basic SLAM demo as foundation
    slam_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('eip_slam'),
                'launch',
                'basic_slam_demo.launch.py'
            ])
        ])
    )
    
    # LLM Interface Node
    llm_interface_node = Node(
        package='eip_llm_interface',
        executable='llm_interface_node',
        name='llm_interface_node',
        output='screen',
        parameters=[{
            'llm_provider': LaunchConfiguration('llm_provider'),
            'enable_safety_verification': LaunchConfiguration('enable_safety_verification'),
            'model_path': '/opt/models/mistral-7b-instruct',
            'max_tokens': 2048,
            'temperature': 0.1,
            'safety_threshold': 0.8
        }]
    )
    
    # Natural Language Command Publisher (for testing)
    test_command_pub = Node(
        package='rostopic',
        executable='rostopic',
        name='test_command_publisher',
        arguments=['pub', '/eip/natural_language_command', 'std_msgs/String', 
                  'data: "Navigate to the red chair in the corner"'],
        output='screen'
    )
    
    return LaunchDescription([
        llm_provider_arg,
        enable_safety_arg,
        slam_demo,
        llm_interface_node,
        test_command_pub
    ]) 