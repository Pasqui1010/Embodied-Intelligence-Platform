#!/usr/bin/env python3
"""
Full System Demo Launch File

Launches the complete EIP system with all components:
- SLAM
- Safety monitoring
- LLM integration (placeholder)
- Visualization
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate the launch description for full system demo"""
    
    # Include the basic SLAM demo as the foundation
    # TODO: Add additional components as they are developed
    slam_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('eip_slam'),
                'launch',
                'basic_slam_demo.launch.py'
            ])
        ])
    )
    
    return LaunchDescription([
        slam_demo,
    ]) 