#!/usr/bin/env python3
"""
Basic SLAM Demo Launch File

Launches a complete SLAM demonstration with:
- Gazebo simulation with TurtleBot3
- Basic SLAM node
- Safety monitor
- RViz visualization

This provides a working end-to-end demo of the EIP platform.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate the launch description for basic SLAM demo"""
    
    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time if true'
    )
    
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='turtlebot3_world',
        description='Gazebo world to load'
    )
    
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value='basic_slam.rviz',
        description='RViz configuration file'
    )
    
    robot_model_arg = DeclareLaunchArgument(
        'robot_model',
        default_value='waffle_pi',
        description='TurtleBot3 model'
    )
    
    # Launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    rviz_config = LaunchConfiguration('rviz_config')
    robot_model = LaunchConfiguration('robot_model')
    
    # Gazebo simulation launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_gazebo'),
                'launch',
                'turtlebot3_world.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'world': world,
        }.items(),
    )
    
    # TurtleBot3 robot state publisher
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_bringup'),
                'launch',
                'robot_state_publisher.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items(),
    )
    
    # Basic SLAM node
    slam_node = Node(
        package='eip_slam',
        executable='basic_slam_node.py',
        name='basic_slam_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'slam_update_rate': 10.0,
            'icp_max_distance': 0.1,
            'voxel_size': 0.05,
            'loop_closure_threshold': 2.0,
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
        }],
        remappings=[
            ('/scan', '/scan'),
            ('/odom', '/odom'),
            ('/cmd_vel_raw', '/cmd_vel'),
        ]
    )
    
    # Safety monitor node
    safety_monitor = Node(
        package='eip_safety_arbiter',
        executable='safety_monitor_node',
        name='safety_monitor',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'safety_check_frequency': 10.0,
            'collision_distance_threshold': 0.5,
            'human_proximity_threshold': 1.0,
            'max_linear_velocity': 0.5,  # Conservative for demo
            'max_angular_velocity': 1.0,
            'enable_llm_safety_check': False,  # Disabled for basic demo
        }],
        remappings=[
            ('/cmd_vel_raw', '/cmd_vel_raw'),
            ('/cmd_vel_safe', '/cmd_vel'),
        ]
    )
    
    # Teleop keyboard control
    teleop_node = Node(
        package='turtlebot3_teleop',
        executable='teleop_keyboard',
        name='teleop_keyboard',
        output='screen',
        prefix='xterm -e',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('/cmd_vel', '/cmd_vel_raw'),  # Send to safety monitor first
        ]
    )
    
    # RViz visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('eip_slam'),
            'config',
            rviz_config
        ])],
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        condition=IfCondition('true')  # Always launch RViz for demo
    )
    
    # Static transform publisher (map -> odom)
    static_transform_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_publisher',
        arguments=[
            '0', '0', '0',  # x, y, z
            '0', '0', '0', '1',  # qx, qy, qz, qw
            'map', 'odom'
        ],
        parameters=[{
            'use_sim_time': use_sim_time,
        }]
    )
    
    return LaunchDescription([
        # Launch arguments
        use_sim_time_arg,
        world_arg,
        rviz_config_arg,
        robot_model_arg,
        
        # Simulation environment
        gazebo_launch,
        robot_state_publisher,
        static_transform_publisher,
        
        # EIP components
        slam_node,
        safety_monitor,
        
        # Human interface
        teleop_node,
        rviz_node,
    ]) 