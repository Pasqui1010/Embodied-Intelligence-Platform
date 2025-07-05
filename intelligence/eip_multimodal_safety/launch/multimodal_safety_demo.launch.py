#!/usr/bin/env python3
"""
Multimodal Safety Demo Launch File

Launches the swarm safety intelligence system with multiple specialized nodes.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for multimodal safety demo"""
    
    # Package configuration
    package_name = 'eip_multimodal_safety'
    package_share = FindPackageShare(package_name)
    
    # Launch arguments
    swarm_size_arg = DeclareLaunchArgument(
        'swarm_size',
        default_value='5',
        description='Number of nodes in the safety swarm'
    )
    
    enable_vision_arg = DeclareLaunchArgument(
        'enable_vision',
        default_value='true',
        description='Enable vision-based safety node'
    )
    
    enable_audio_arg = DeclareLaunchArgument(
        'enable_audio',
        default_value='true',
        description='Enable audio-based safety node'
    )
    
    enable_tactile_arg = DeclareLaunchArgument(
        'enable_tactile',
        default_value='true',
        description='Enable tactile-based safety node'
    )
    
    enable_proprioceptive_arg = DeclareLaunchArgument(
        'enable_proprioceptive',
        default_value='true',
        description='Enable proprioceptive-based safety node'
    )
    
    enable_fusion_arg = DeclareLaunchArgument(
        'enable_fusion',
        default_value='true',
        description='Enable fusion-based safety node'
    )
    
    # Get launch configurations
    swarm_size = LaunchConfiguration('swarm_size')
    enable_vision = LaunchConfiguration('enable_vision')
    enable_audio = LaunchConfiguration('enable_audio')
    enable_tactile = LaunchConfiguration('enable_tactile')
    enable_proprioceptive = LaunchConfiguration('enable_proprioceptive')
    enable_fusion = LaunchConfiguration('enable_fusion')
    
    # Vision safety node
    vision_safety_node = Node(
        package=package_name,
        executable='multimodal_safety_node',
        name='vision_safety_node',
        parameters=[{
            'cell_type': 'vision',
            'swarm_size': swarm_size,
            'learning_rate': 0.001,
            'evolution_threshold': 0.8
        }],
        condition=LaunchConfiguration('enable_vision'),
        output='screen'
    )
    
    # Audio safety node
    audio_safety_node = Node(
        package=package_name,
        executable='multimodal_safety_node',
        name='audio_safety_node',
        parameters=[{
            'cell_type': 'audio',
            'swarm_size': swarm_size,
            'learning_rate': 0.001,
            'evolution_threshold': 0.8
        }],
        condition=LaunchConfiguration('enable_audio'),
        output='screen'
    )
    
    # Tactile safety node
    tactile_safety_node = Node(
        package=package_name,
        executable='multimodal_safety_node',
        name='tactile_safety_node',
        parameters=[{
            'cell_type': 'tactile',
            'swarm_size': swarm_size,
            'learning_rate': 0.001,
            'evolution_threshold': 0.8
        }],
        condition=LaunchConfiguration('enable_tactile'),
        output='screen'
    )
    
    # Proprioceptive safety node
    proprioceptive_safety_node = Node(
        package=package_name,
        executable='multimodal_safety_node',
        name='proprioceptive_safety_node',
        parameters=[{
            'cell_type': 'proprioceptive',
            'swarm_size': swarm_size,
            'learning_rate': 0.001,
            'evolution_threshold': 0.8
        }],
        condition=LaunchConfiguration('enable_proprioceptive'),
        output='screen'
    )
    
    # Fusion safety node (coordinator)
    fusion_safety_node = Node(
        package=package_name,
        executable='multimodal_safety_node',
        name='fusion_safety_node',
        parameters=[{
            'cell_type': 'fusion',
            'swarm_size': swarm_size,
            'learning_rate': 0.001,
            'evolution_threshold': 0.8
        }],
        condition=LaunchConfiguration('enable_fusion'),
        output='screen'
    )
    
    # Status monitoring node
    status_monitor_node = Node(
        package=package_name,
        executable='status_monitor_node',
        name='status_monitor_node',
        parameters=[{
            'monitor_interval': 2.0,
            'log_level': 'INFO'
        }],
        output='screen'
    )
    
    # Log info about the launch
    log_info = LogInfo(
        msg=['Launching Multimodal Safety Demo with swarm size: ', swarm_size]
    )
    
    # Create launch description
    return LaunchDescription([
        # Launch arguments
        swarm_size_arg,
        enable_vision_arg,
        enable_audio_arg,
        enable_tactile_arg,
        enable_proprioceptive_arg,
        enable_fusion_arg,
        
        # Log info
        log_info,
        
        # Safety nodes
        vision_safety_node,
        audio_safety_node,
        tactile_safety_node,
        proprioceptive_safety_node,
        fusion_safety_node,
        
        # Monitoring
        status_monitor_node
    ]) 