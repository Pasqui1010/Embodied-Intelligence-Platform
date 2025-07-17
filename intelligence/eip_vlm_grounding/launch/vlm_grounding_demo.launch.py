#!/usr/bin/env python3
"""
VLM Grounding Demo Launch File

Launches the VLM grounding system with all components and parameters
for spatial reference resolution and object affordance estimation.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for VLM grounding demo"""
    
    # Package and configuration paths
    pkg_share = FindPackageShare('eip_vlm_grounding')
    config_dir = PathJoinSubstitution([pkg_share, 'config'])
    
    # Launch arguments
    spatial_resolution_rate_arg = DeclareLaunchArgument(
        'spatial_resolution_rate',
        default_value='10.0',
        description='Rate for spatial reference resolution (Hz)'
    )
    
    affordance_estimation_rate_arg = DeclareLaunchArgument(
        'affordance_estimation_rate',
        default_value='5.0',
        description='Rate for affordance estimation (Hz)'
    )
    
    scene_analysis_rate_arg = DeclareLaunchArgument(
        'scene_analysis_rate',
        default_value='2.0',
        description='Rate for scene analysis (Hz)'
    )
    
    vlm_integration_rate_arg = DeclareLaunchArgument(
        'vlm_integration_rate',
        default_value='1.0',
        description='Rate for VLM integration (Hz)'
    )
    
    enable_clip_arg = DeclareLaunchArgument(
        'enable_clip',
        default_value='true',
        description='Enable CLIP model for vision-language grounding'
    )
    
    enable_safety_validation_arg = DeclareLaunchArgument(
        'enable_safety_validation',
        default_value='true',
        description='Enable safety validation for VLM responses'
    )
    
    min_confidence_threshold_arg = DeclareLaunchArgument(
        'min_confidence_threshold',
        default_value='0.6',
        description='Minimum confidence threshold for VLM responses'
    )
    
    max_objects_per_scene_arg = DeclareLaunchArgument(
        'max_objects_per_scene',
        default_value='20',
        description='Maximum number of objects to track per scene'
    )
    
    enable_visualization_arg = DeclareLaunchArgument(
        'enable_visualization',
        default_value='true',
        description='Enable visualization markers'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to pre-trained model files'
    )
    
    safety_llm_path_arg = DeclareLaunchArgument(
        'safety_llm_path',
        default_value='',
        description='Path to Safety-Embedded LLM model'
    )
    
    # VLM Grounding Node
    vlm_grounding_node = Node(
        package='eip_vlm_grounding',
        executable='vlm_grounding_node',
        name='vlm_grounding_node',
        output='screen',
        parameters=[{
            'spatial_resolution_rate': LaunchConfiguration('spatial_resolution_rate'),
            'affordance_estimation_rate': LaunchConfiguration('affordance_estimation_rate'),
            'scene_analysis_rate': LaunchConfiguration('scene_analysis_rate'),
            'vlm_integration_rate': LaunchConfiguration('vlm_integration_rate'),
            'enable_clip': LaunchConfiguration('enable_clip'),
            'enable_safety_validation': LaunchConfiguration('enable_safety_validation'),
            'min_confidence_threshold': LaunchConfiguration('min_confidence_threshold'),
            'max_objects_per_scene': LaunchConfiguration('max_objects_per_scene'),
            'enable_visualization': LaunchConfiguration('enable_visualization'),
            'model_path': LaunchConfiguration('model_path'),
            'safety_llm_path': LaunchConfiguration('safety_llm_path'),
        }],
        remappings=[
            ('/camera/image_raw', '/camera/color/image_raw'),
            ('/camera/pointcloud', '/camera/depth/points'),
            ('/lidar/scan', '/scan'),
        ]
    )
    
    # Demo command publisher (for testing)
    demo_command_publisher = Node(
        package='rostopic',
        executable='rostopic',
        name='demo_command_publisher',
        arguments=['pub', '/vlm_grounding/spatial_query', 'std_msgs/String', 
                  'data: "move to the left of the red cup"'],
        output='screen'
    )
    
    # Demo affordance query publisher
    demo_affordance_publisher = Node(
        package='rostopic',
        executable='rostopic',
        name='demo_affordance_publisher',
        arguments=['pub', '/vlm_grounding/affordance_query', 'std_msgs/String', 
                  'data: "what can I do with the cup?"'],
        output='screen'
    )
    
    # Demo VLM query publisher
    demo_vlm_publisher = Node(
        package='rostopic',
        executable='rostopic',
        name='demo_vlm_publisher',
        arguments=['pub', '/vlm_grounding/vlm_query', 'std_msgs/String', 
                  'data: "describe the scene and identify objects"'],
        output='screen'
    )
    
    # Echo spatial reference results
    echo_spatial_reference = Node(
        package='rostopic',
        executable='rostopic',
        name='echo_spatial_reference',
        arguments=['echo', '/vlm_grounding/spatial_reference'],
        output='screen'
    )
    
    # Echo affordance results
    echo_affordance_result = Node(
        package='rostopic',
        executable='rostopic',
        name='echo_affordance_result',
        arguments=['echo', '/vlm_grounding/affordance_result'],
        output='screen'
    )
    
    # Echo VLM results
    echo_vlm_result = Node(
        package='rostopic',
        executable='rostopic',
        name='echo_vlm_result',
        arguments=['echo', '/vlm_grounding/vlm_result'],
        output='screen'
    )
    
    # Echo scene description
    echo_scene_description = Node(
        package='rostopic',
        executable='rostopic',
        name='echo_scene_description',
        arguments=['echo', '/vlm_grounding/scene_description'],
        output='screen'
    )
    
    # Log info
    log_info = LogInfo(
        msg="VLM Grounding Demo launched successfully. "
            "Use the following topics to interact with the system:\n"
            "- /vlm_grounding/spatial_query: Send spatial reference queries\n"
            "- /vlm_grounding/affordance_query: Send affordance estimation queries\n"
            "- /vlm_grounding/vlm_query: Send VLM reasoning queries\n"
            "- /vlm_grounding/spatial_reference: Receive spatial reference results\n"
            "- /vlm_grounding/affordance_result: Receive affordance estimation results\n"
            "- /vlm_grounding/vlm_result: Receive VLM reasoning results\n"
            "- /vlm_grounding/scene_description: Receive scene descriptions\n"
            "- /vlm_grounding/visualization: Visualization markers (if enabled)"
    )
    
    return LaunchDescription([
        # Launch arguments
        spatial_resolution_rate_arg,
        affordance_estimation_rate_arg,
        scene_analysis_rate_arg,
        vlm_integration_rate_arg,
        enable_clip_arg,
        enable_safety_validation_arg,
        min_confidence_threshold_arg,
        max_objects_per_scene_arg,
        enable_visualization_arg,
        model_path_arg,
        safety_llm_path_arg,
        
        # Nodes
        vlm_grounding_node,
        
        # Demo publishers (comment out if not needed)
        # demo_command_publisher,
        # demo_affordance_publisher,
        # demo_vlm_publisher,
        
        # Echo nodes (comment out if not needed)
        # echo_spatial_reference,
        # echo_affordance_result,
        # echo_vlm_result,
        # echo_scene_description,
        
        # Log info
        log_info,
    ]) 