#!/usr/bin/env python3
"""
Reasoning Engine Demo Launch File

This launch file demonstrates the Advanced Multi-Modal Reasoning Engine
with all its components and integration with the Embodied Intelligence Platform.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for reasoning engine demo"""
    
    # Declare launch arguments
    reasoning_mode_arg = DeclareLaunchArgument(
        'reasoning_mode',
        default_value='balanced',
        description='Reasoning mode: fast, balanced, thorough, safety_critical'
    )
    
    max_reasoning_time_arg = DeclareLaunchArgument(
        'max_reasoning_time',
        default_value='0.5',
        description='Maximum reasoning time in seconds'
    )
    
    enable_visual_reasoning_arg = DeclareLaunchArgument(
        'enable_visual_reasoning',
        default_value='true',
        description='Enable visual reasoning capabilities'
    )
    
    enable_spatial_reasoning_arg = DeclareLaunchArgument(
        'enable_spatial_reasoning',
        default_value='true',
        description='Enable spatial reasoning capabilities'
    )
    
    enable_temporal_reasoning_arg = DeclareLaunchArgument(
        'enable_temporal_reasoning',
        default_value='true',
        description='Enable temporal reasoning capabilities'
    )
    
    enable_causal_reasoning_arg = DeclareLaunchArgument(
        'enable_causal_reasoning',
        default_value='true',
        description='Enable causal reasoning capabilities'
    )
    
    enable_safety_reasoning_arg = DeclareLaunchArgument(
        'enable_safety_reasoning',
        default_value='true',
        description='Enable safety reasoning capabilities'
    )
    
    reasoning_update_rate_arg = DeclareLaunchArgument(
        'reasoning_update_rate',
        default_value='10.0',
        description='Reasoning update rate in Hz'
    )
    
    enable_performance_monitoring_arg = DeclareLaunchArgument(
        'enable_performance_monitoring',
        default_value='true',
        description='Enable performance monitoring'
    )
    
    log_reasoning_results_arg = DeclareLaunchArgument(
        'log_reasoning_results',
        default_value='true',
        description='Log detailed reasoning results'
    )
    
    # Get package share directory
    pkg_share = FindPackageShare('eip_reasoning_engine')
    
    # Configuration file path
    config_file = PathJoinSubstitution([
        pkg_share, 'config', 'reasoning_engine.yaml'
    ])
    
    # Launch reasoning engine node
    reasoning_engine_node = Node(
        package='eip_reasoning_engine',
        executable='reasoning_engine_node',
        name='reasoning_engine_node',
        output='screen',
        parameters=[
            config_file,
            {
                'reasoning_mode': LaunchConfiguration('reasoning_mode'),
                'max_reasoning_time': LaunchConfiguration('max_reasoning_time'),
                'enable_visual_reasoning': LaunchConfiguration('enable_visual_reasoning'),
                'enable_spatial_reasoning': LaunchConfiguration('enable_spatial_reasoning'),
                'enable_temporal_reasoning': LaunchConfiguration('enable_temporal_reasoning'),
                'enable_causal_reasoning': LaunchConfiguration('enable_causal_reasoning'),
                'enable_safety_reasoning': LaunchConfiguration('enable_safety_reasoning'),
                'reasoning_update_rate': LaunchConfiguration('reasoning_update_rate'),
                'enable_performance_monitoring': LaunchConfiguration('enable_performance_monitoring'),
                'log_reasoning_results': LaunchConfiguration('log_reasoning_results'),
                'collision_threshold': 0.7,
                'human_proximity_threshold': 0.8,
                'velocity_limit': 1.0,
                'workspace_boundary_x': 5.0,
                'workspace_boundary_y': 5.0,
                'workspace_boundary_z': 2.0
            }
        ],
        remappings=[
            ('/eip/vision/context', '/eip/vision/context'),
            ('/eip/language/commands', '/eip/language/commands'),
            ('/eip/slam/spatial_context', '/eip/slam/spatial_context'),
            ('/eip/safety/constraints', '/eip/safety/constraints'),
            ('/eip/robot/pose', '/eip/robot/pose'),
            ('/eip/reasoning/results', '/eip/reasoning/results'),
            ('/eip/reasoning/task_plans', '/eip/reasoning/task_plans'),
            ('/eip/reasoning/confidence', '/eip/reasoning/confidence'),
            ('/eip/reasoning/safety_score', '/eip/reasoning/safety_score'),
            ('/eip/reasoning/status', '/eip/reasoning/status'),
            ('/eip/reasoning/performance_stats', '/eip/reasoning/performance_stats')
        ]
    )
    
    # Mock data publisher for demo (if no real sensors available)
    mock_visual_context_node = Node(
        package='eip_reasoning_engine',
        executable='mock_data_publisher',
        name='mock_visual_context_publisher',
        output='screen',
        parameters=[
            {
                'topic_name': '/eip/vision/context',
                'publish_rate': 5.0,
                'data_type': 'visual_context'
            }
        ]
    )
    
    mock_spatial_context_node = Node(
        package='eip_reasoning_engine',
        executable='mock_data_publisher',
        name='mock_spatial_context_publisher',
        output='screen',
        parameters=[
            {
                'topic_name': '/eip/slam/spatial_context',
                'publish_rate': 10.0,
                'data_type': 'spatial_context'
            }
        ]
    )
    
    mock_safety_constraints_node = Node(
        package='eip_reasoning_engine',
        executable='mock_data_publisher',
        name='mock_safety_constraints_publisher',
        output='screen',
        parameters=[
            {
                'topic_name': '/eip/safety/constraints',
                'publish_rate': 2.0,
                'data_type': 'safety_constraints'
            }
        ]
    )
    
    mock_robot_pose_node = Node(
        package='eip_reasoning_engine',
        executable='mock_data_publisher',
        name='mock_robot_pose_publisher',
        output='screen',
        parameters=[
            {
                'topic_name': '/eip/robot/pose',
                'publish_rate': 20.0,
                'data_type': 'robot_pose'
            }
        ]
    )
    
    # Command generator for demo
    command_generator_node = Node(
        package='eip_reasoning_engine',
        executable='command_generator',
        name='command_generator',
        output='screen',
        parameters=[
            {
                'command_interval': 5.0,
                'commands': [
                    'Move to the red object',
                    'Pick up the blue cube',
                    'Place the object on the table',
                    'Observe the scene carefully',
                    'Move slowly to avoid obstacles'
                ]
            }
        ]
    )
    
    # Results monitor for demo
    results_monitor_node = Node(
        package='eip_reasoning_engine',
        executable='results_monitor',
        name='results_monitor',
        output='screen',
        parameters=[
            {
                'monitor_topics': [
                    '/eip/reasoning/results',
                    '/eip/reasoning/task_plans',
                    '/eip/reasoning/confidence',
                    '/eip/reasoning/safety_score'
                ],
                'log_results': True
            }
        ]
    )
    
    # Performance monitor
    performance_monitor_node = Node(
        package='eip_reasoning_engine',
        executable='performance_monitor',
        name='performance_monitor',
        output='screen',
        parameters=[
            {
                'monitor_topic': '/eip/reasoning/performance_stats',
                'update_interval': 10.0,
                'log_performance': True
            }
        ]
    )
    
    # Startup message
    startup_msg = LogInfo(
        msg="Starting Advanced Multi-Modal Reasoning Engine Demo..."
    )
    
    # Delayed startup for mock publishers
    delayed_mock_publishers = TimerAction(
        period=2.0,
        actions=[
            mock_visual_context_node,
            mock_spatial_context_node,
            mock_safety_constraints_node,
            mock_robot_pose_node
        ]
    )
    
    # Delayed startup for demo components
    delayed_demo_components = TimerAction(
        period=3.0,
        actions=[
            command_generator_node,
            results_monitor_node,
            performance_monitor_node
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        reasoning_mode_arg,
        max_reasoning_time_arg,
        enable_visual_reasoning_arg,
        enable_spatial_reasoning_arg,
        enable_temporal_reasoning_arg,
        enable_causal_reasoning_arg,
        enable_safety_reasoning_arg,
        reasoning_update_rate_arg,
        enable_performance_monitoring_arg,
        log_reasoning_results_arg,
        
        # Startup message
        startup_msg,
        
        # Main reasoning engine node
        reasoning_engine_node,
        
        # Delayed mock publishers
        delayed_mock_publishers,
        
        # Delayed demo components
        delayed_demo_components
    ]) 