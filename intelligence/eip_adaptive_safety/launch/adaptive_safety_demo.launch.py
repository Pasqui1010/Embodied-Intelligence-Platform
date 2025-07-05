#!/usr/bin/env python3
"""
Launch file for Adaptive Safety Orchestration (ASO) Demo

This launch file starts the complete ASO system including:
- Adaptive Learning Engine
- Adaptive Safety Node
- Integration with existing safety infrastructure
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level for nodes'
    )
    
    # Get package share directory
    pkg_share = FindPackageShare('eip_adaptive_safety')
    
    # Adaptive Safety Node
    adaptive_safety_node = Node(
        package='eip_adaptive_safety',
        executable='adaptive_safety_node',
        name='adaptive_safety_node',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'learning_rate': 0.001,
            'batch_size': 32,
            'update_frequency': 100,
            'min_confidence_threshold': 0.7,
            'max_rules': 100
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )
    
    # Safety Arbiter Node (existing)
    safety_arbiter_node = Node(
        package='eip_safety_arbiter',
        executable='safety_monitor',
        name='safety_arbiter',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'collision_threshold': 0.5,
            'velocity_limit': 2.0,
            'human_proximity_threshold': 1.0,
            'workspace_boundary': [10.0, 10.0, 10.0]
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )
    
    # LLM Interface Node (existing)
    llm_interface_node = Node(
        package='eip_llm_interface',
        executable='llm_interface_node',
        name='llm_interface',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'model_type': 'mock',  # Use mock for demo
            'safety_validation_enabled': True,
            'max_response_time': 5.0
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )
    
    # Safety Simulator Node (for testing)
    safety_simulator_node = Node(
        package='eip_safety_simulator',
        executable='safety_simulator_node',
        name='safety_simulator',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'simulation_rate': 10.0,
            'scenario_type': 'adaptive_learning',
            'enable_safety_violations': True
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )
    
    # RViz for visualization (optional)
    rviz_config_file = PathJoinSubstitution([
        pkg_share, 'config', 'adaptive_safety.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=LaunchConfiguration('use_rviz')
    )
    
    # Declare RViz argument
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Launch RViz for visualization'
    )
    
    # Timer action to start components in sequence
    delayed_adaptive_safety = TimerAction(
        period=2.0,
        actions=[adaptive_safety_node]
    )
    
    delayed_safety_arbiter = TimerAction(
        period=1.0,
        actions=[safety_arbiter_node]
    )
    
    delayed_llm_interface = TimerAction(
        period=3.0,
        actions=[llm_interface_node]
    )
    
    delayed_simulator = TimerAction(
        period=4.0,
        actions=[safety_simulator_node]
    )
    
    # Create launch description
    return LaunchDescription([
        # Launch arguments
        use_sim_time_arg,
        log_level_arg,
        use_rviz_arg,
        
        # Start components with delays
        delayed_safety_arbiter,
        delayed_adaptive_safety,
        delayed_llm_interface,
        delayed_simulator,
        
        # Optional RViz
        rviz_node
    ]) 