#!/usr/bin/env python3
"""
Multi-Modal Safety Demo Launch File

This launch file starts the multi-modal safety system with all sensor
modalities enabled for comprehensive safety validation.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for multi-modal safety demo"""
    
    # Declare launch arguments
    fusion_method_arg = DeclareLaunchArgument(
        'fusion_method',
        default_value='weighted_average',
        description='Sensor fusion method to use'
    )
    
    safety_update_rate_arg = DeclareLaunchArgument(
        'safety_update_rate',
        default_value='10.0',
        description='Safety monitoring update rate (Hz)'
    )
    
    enable_vision_arg = DeclareLaunchArgument(
        'enable_vision',
        default_value='true',
        description='Enable vision sensors'
    )
    
    enable_audio_arg = DeclareLaunchArgument(
        'enable_audio',
        default_value='true',
        description='Enable audio sensors'
    )
    
    enable_tactile_arg = DeclareLaunchArgument(
        'enable_tactile',
        default_value='true',
        description='Enable tactile sensors'
    )
    
    enable_proprioceptive_arg = DeclareLaunchArgument(
        'enable_proprioceptive',
        default_value='true',
        description='Enable proprioceptive sensors'
    )
    
    enable_lidar_arg = DeclareLaunchArgument(
        'enable_lidar',
        default_value='true',
        description='Enable LIDAR sensors'
    )
    
    # Multi-Modal Safety Node
    multimodal_safety_node = Node(
        package='eip_multimodal_safety',
        executable='multimodal_safety_node',
        name='multimodal_safety_node',
        output='screen',
        parameters=[{
            'fusion_method': LaunchConfiguration('fusion_method'),
            'safety_update_rate': LaunchConfiguration('safety_update_rate'),
            'enable_vision': LaunchConfiguration('enable_vision'),
            'enable_audio': LaunchConfiguration('enable_audio'),
            'enable_tactile': LaunchConfiguration('enable_tactile'),
            'enable_proprioceptive': LaunchConfiguration('enable_proprioceptive'),
            'enable_lidar': LaunchConfiguration('enable_lidar'),
            'vision_weight': 0.3,
            'audio_weight': 0.2,
            'tactile_weight': 0.25,
            'proprioceptive_weight': 0.15,
            'lidar_weight': 0.1,
            'collision_threshold': 0.7,
            'human_proximity_threshold': 0.8,
            'velocity_threshold': 0.6,
            'workspace_boundary_threshold': 0.5,
            'emergency_stop_threshold': 0.9,
            'sensor_timeout': 2.0
        }],
        remappings=[
            ('/camera/image_raw', '/camera/color/image_raw'),
            ('/camera/depth/image_raw', '/camera/depth/image_raw'),
            ('/audio/processed', '/audio/features'),
            ('/tactile/processed', '/tactile/features'),
            ('/joint_states', '/robot/joint_states'),
            ('/scan', '/lidar/scan'),
            ('/points', '/lidar/points')
        ]
    )
    
    # Safety Arbiter Node (for integration)
    safety_arbiter_node = Node(
        package='eip_safety_arbiter',
        executable='safety_monitor',
        name='safety_arbiter_node',
        output='screen',
        parameters=[{
            'collision_threshold': 0.7,
            'human_proximity_threshold': 0.8,
            'velocity_threshold': 0.6,
            'workspace_boundary_threshold': 0.5,
            'emergency_stop_threshold': 0.9
        }]
    )
    
    # LLM Interface Node (for safety-embedded LLM)
    llm_interface_node = Node(
        package='eip_llm_interface',
        executable='llm_interface_node',
        name='llm_interface_node',
        output='screen',
        parameters=[{
            'model_name': 'safety_embedded_llm',
            'safety_tokens': ['STOP', 'DANGER', 'SAFE', 'WARNING'],
            'max_response_time': 2.0,
            'enable_safety_validation': True
        }]
    )
    
    # Log info about the launch
    log_info = LogInfo(
        msg=[
            "Starting Multi-Modal Safety Demo with:",
            "  - Fusion Method: ", LaunchConfiguration('fusion_method'),
            "  - Update Rate: ", LaunchConfiguration('safety_update_rate'), " Hz",
            "  - Vision: ", LaunchConfiguration('enable_vision'),
            "  - Audio: ", LaunchConfiguration('enable_audio'),
            "  - Tactile: ", LaunchConfiguration('enable_tactile'),
            "  - Proprioceptive: ", LaunchConfiguration('enable_proprioceptive'),
            "  - LIDAR: ", LaunchConfiguration('enable_lidar')
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        fusion_method_arg,
        safety_update_rate_arg,
        enable_vision_arg,
        enable_audio_arg,
        enable_tactile_arg,
        enable_proprioceptive_arg,
        enable_lidar_arg,
        
        # Log info
        log_info,
        
        # Nodes
        multimodal_safety_node,
        safety_arbiter_node,
        llm_interface_node
    ]) 