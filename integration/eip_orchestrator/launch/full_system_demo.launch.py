#!/usr/bin/env python3
"""
Full System Demo Launch File

Launches the complete Embodied Intelligence Platform with all components
including SLAM, safety monitoring, LLM interface, and adaptive safety.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os


def generate_launch_description():
    """Generate launch description for full system demo"""
    
    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    # Get package directories
    eip_slam_dir = FindPackageShare('eip_slam')
    eip_safety_arbiter_dir = FindPackageShare('eip_safety_arbiter')
    eip_llm_interface_dir = FindPackageShare('eip_llm_interface')
    eip_adaptive_safety_dir = FindPackageShare('eip_adaptive_safety')
    eip_multimodal_safety_dir = FindPackageShare('eip_multimodal_safety')
    eip_safety_simulator_dir = FindPackageShare('eip_safety_simulator')
    
    # Launch SLAM system
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([eip_slam_dir, 'launch', 'basic_slam_demo.launch.py'])
        ]),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )
    
    # Launch safety monitor
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
    
    # Launch LLM interface
    llm_interface_node = Node(
        package='eip_llm_interface',
        executable='llm_interface_node',
        name='llm_interface',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'model_name': 'microsoft/DialoGPT-medium',
            'device': 'auto',
            'max_response_time': 5.0,
            'enable_safety_embedding': True
        }]
    )
    
    # Launch adaptive safety
    adaptive_safety_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([eip_adaptive_safety_dir, 'launch', 'adaptive_safety_demo.launch.py'])
        ]),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )
    
    # Launch multimodal safety
    multimodal_safety_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([eip_multimodal_safety_dir, 'launch', 'multimodal_safety_demo.launch.py'])
        ]),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )
    
    # Launch safety simulator
    safety_simulator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([eip_safety_simulator_dir, 'launch', 'safety_simulator_demo.launch.py'])
        ]),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )
    
    # System orchestrator node
    orchestrator_node = Node(
        package='eip_orchestrator',
        executable='orchestrator_node',
        name='system_orchestrator',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'enable_all_components': True,
            'safety_monitoring_enabled': True,
            'llm_integration_enabled': True,
            'adaptive_learning_enabled': True
        }]
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        slam_launch,
        safety_monitor_node,
        llm_interface_node,
        adaptive_safety_launch,
        multimodal_safety_launch,
        safety_simulator_launch,
        orchestrator_node
    ]) 