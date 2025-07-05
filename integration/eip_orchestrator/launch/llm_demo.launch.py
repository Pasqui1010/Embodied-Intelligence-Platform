#!/usr/bin/env python3
"""
LLM Demo Launch File

This is a placeholder for the LLM integration demo.
For now, it launches the basic SLAM demo as the foundation.
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate the launch description for LLM demo"""
    
    # For now, include the basic SLAM demo as foundation
    # TODO: Add LLM integration components in Week 2
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