#!/usr/bin/env python3
"""
Safety Simulator Demo Launch File

Launches the Digital Twin Safety Ecosystem with comprehensive safety validation
for the Safety-Embedded LLM system.
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    """Generate the launch description for safety simulator demo"""
    
    # Launch arguments
    scenario_arg = DeclareLaunchArgument(
        'scenario',
        default_value='collision_avoidance',
        description='Safety scenario to run (collision_avoidance, human_proximity, velocity_limits, etc.)'
    )
    
    enable_gazebo_arg = DeclareLaunchArgument(
        'enable_gazebo',
        default_value='true',
        description='Enable Gazebo simulation'
    )
    
    safety_threshold_arg = DeclareLaunchArgument(
        'safety_threshold',
        default_value='0.8',
        description='Safety threshold for validation'
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
    
    # Include LLM interface
    llm_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('eip_llm_interface'),
                'launch',
                'llm_demo.launch.py'
            ])
        ])
    )
    
    # Safety Simulator Node
    safety_simulator_node = Node(
        package='eip_safety_simulator',
        executable='safety_simulator_node',
        name='safety_simulator_node',
        output='screen',
        parameters=[{
            'simulation_config_file': 'config/default_simulation.yaml',
            'enable_gazebo': LaunchConfiguration('enable_gazebo'),
            'enable_metrics_collection': True,
            'safety_threshold': LaunchConfiguration('safety_threshold'),
            'max_simulation_time': 300.0
        }]
    )
    
    # Scenario Generator Node
    scenario_generator_node = Node(
        package='eip_safety_simulator',
        executable='scenario_generator',
        name='scenario_generator_node',
        output='screen',
        parameters=[{
            'scenario_type': LaunchConfiguration('scenario'),
            'random_seed': 42,
            'enable_visualization': True
        }]
    )
    
    # Safety Validator Node
    safety_validator_node = Node(
        package='eip_safety_simulator',
        executable='safety_validator',
        name='safety_validator_node',
        output='screen',
        parameters=[{
            'validation_mode': 'real_time',
            'safety_threshold': LaunchConfiguration('safety_threshold'),
            'enable_logging': True
        }]
    )
    
    # Test command publisher (for testing scenarios)
    test_scenario_pub = Node(
        package='rostopic',
        executable='rostopic',
        name='test_scenario_publisher',
        arguments=['pub', '/eip/safety_simulator/start', 'std_msgs/String', 
                  f'data: "{LaunchConfiguration("scenario")}"'],
        output='screen'
    )
    
    # Metrics visualization (optional)
    metrics_viz_node = Node(
        package='rqt_plot',
        executable='rqt_plot',
        name='metrics_visualization',
        arguments=['/eip/safety_simulator/metrics'],
        output='screen'
    )
    
    return LaunchDescription([
        scenario_arg,
        enable_gazebo_arg,
        safety_threshold_arg,
        slam_demo,
        llm_demo,
        safety_simulator_node,
        scenario_generator_node,
        safety_validator_node,
        test_scenario_pub,
        metrics_viz_node
    ]) 