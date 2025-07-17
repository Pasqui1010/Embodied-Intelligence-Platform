#!/usr/bin/env python3

"""
Launch file for Cognitive Architecture Demo

This launch file starts the cognitive architecture node along with
necessary supporting components for a complete demonstration.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for cognitive architecture demo."""
    
    # Get package directories
    cognitive_pkg_dir = get_package_share_directory('eip_cognitive_architecture')
    interfaces_pkg_dir = get_package_share_directory('eip_interfaces')
    
    # Launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='cognitive_architecture.yaml',
        description='Configuration file for cognitive architecture'
    )
    
    enable_visualization_arg = DeclareLaunchArgument(
        'enable_visualization',
        default_value='true',
        description='Enable RViz visualization'
    )
    
    enable_simulation_arg = DeclareLaunchArgument(
        'enable_simulation',
        default_value='true',
        description='Enable simulation environment'
    )
    
    # Configuration file path
    config_file = PathJoinSubstitution([
        FindPackageShare('eip_cognitive_architecture'),
        'config',
        LaunchConfiguration('config_file')
    ])
    
    # Cognitive Architecture Node
    cognitive_node = Node(
        package='eip_cognitive_architecture',
        executable='cognitive_architecture_node',
        name='cognitive_architecture_node',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('/camera/image_raw', '/camera/color/image_raw'),
            ('/camera/points', '/camera/depth/points'),
            ('/laser/scan', '/scan'),
            ('/audio/features', '/audio/features'),
            ('/task/plan', '/task/plan'),
            ('/safety/violation', '/safety/violation'),
            ('/cognitive/response', '/cognitive/response'),
            ('/cognitive/status', '/cognitive/status'),
            ('/cmd_vel', '/cmd_vel')
        ]
    )
    
    # Safety Monitor Node (if available)
    safety_monitor_node = Node(
        package='eip_safety_arbiter',
        executable='safety_monitor',
        name='safety_monitor',
        output='screen',
        parameters=[{
            'safety_check_interval': 0.1,
            'proximity_threshold': 0.5,
            'velocity_threshold': 1.0
        }],
        condition=LaunchConfiguration('enable_simulation')
    )
    
    # LLM Interface Node (if available)
    llm_node = Node(
        package='eip_llm_interface',
        executable='llm_interface_node',
        name='llm_interface_node',
        output='screen',
        parameters=[{
            'model_name': 'gpt-4',
            'max_tokens': 1000,
            'temperature': 0.7,
            'safety_enabled': True
        }],
        condition=LaunchConfiguration('enable_simulation')
    )
    
    # Reasoning Engine Node (if available)
    reasoning_node = Node(
        package='eip_reasoning_engine',
        executable='reasoning_engine_node',
        name='reasoning_engine_node',
        output='screen',
        parameters=[{
            'reasoning_mode': 'multi_modal',
            'confidence_threshold': 0.6,
            'max_reasoning_time': 1.0
        }],
        condition=LaunchConfiguration('enable_simulation')
    )
    
    # VLM Grounding Node (if available)
    vlm_node = Node(
        package='eip_vlm_grounding',
        executable='vlm_grounding_node',
        name='vlm_grounding_node',
        output='screen',
        parameters=[{
            'grounding_confidence_threshold': 0.7,
            'spatial_resolution': 0.1,
            'temporal_window': 2.0
        }],
        condition=LaunchConfiguration('enable_simulation')
    )
    
    # Adaptive Safety Node (if available)
    adaptive_safety_node = Node(
        package='eip_adaptive_safety',
        executable='adaptive_safety_node',
        name='adaptive_safety_node',
        output='screen',
        parameters=[{
            'learning_rate': 0.1,
            'adaptation_threshold': 0.8,
            'safety_margin': 0.2
        }],
        condition=LaunchConfiguration('enable_simulation')
    )
    
    # Multimodal Safety Node (if available)
    multimodal_safety_node = Node(
        package='eip_multimodal_safety',
        executable='multimodal_safety_node',
        name='multimodal_safety_node',
        output='screen',
        parameters=[{
            'fusion_method': 'weighted_average',
            'sensor_weights': [0.4, 0.3, 0.2, 0.1],
            'safety_threshold': 0.7
        }],
        condition=LaunchConfiguration('enable_simulation')
    )
    
    # RViz Node for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('eip_cognitive_architecture'),
            'config',
            'cognitive_architecture.rviz'
        ])],
        condition=LaunchConfiguration('enable_visualization')
    )
    
    # Demo task publisher (simulates task commands)
    demo_task_publisher = Node(
        package='eip_cognitive_architecture',
        executable='demo_task_publisher',
        name='demo_task_publisher',
        output='screen',
        parameters=[{
            'task_interval': 10.0,
            'task_types': ['navigation', 'interaction', 'manipulation']
        }],
        condition=LaunchConfiguration('enable_simulation')
    )
    
    # Performance monitor
    performance_monitor = Node(
        package='eip_cognitive_architecture',
        executable='performance_monitor',
        name='performance_monitor',
        output='screen',
        parameters=[{
            'monitoring_interval': 5.0,
            'log_performance': True
        }],
        condition=LaunchConfiguration('enable_simulation')
    )
    
    # Startup information
    startup_info = LogInfo(
        msg="Starting Cognitive Architecture Demo with full AI component integration"
    )
    
    return LaunchDescription([
        # Launch arguments
        config_file_arg,
        enable_visualization_arg,
        enable_simulation_arg,
        
        # Startup information
        startup_info,
        
        # Core cognitive architecture
        cognitive_node,
        
        # Supporting AI components
        safety_monitor_node,
        llm_node,
        reasoning_node,
        vlm_node,
        adaptive_safety_node,
        multimodal_safety_node,
        
        # Visualization and monitoring
        rviz_node,
        demo_task_publisher,
        performance_monitor
    ]) 