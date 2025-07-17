#!/usr/bin/env python3

"""
Launch file for Social Intelligence Demo

This launch file sets up the social intelligence system with all
necessary components and parameters for demonstration.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for social intelligence demo"""
    
    # Package and configuration paths
    package_name = 'eip_social_intelligence'
    config_file = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'social_intelligence.yaml'
    ])
    
    # Launch arguments
    cultural_context_arg = DeclareLaunchArgument(
        'cultural_context',
        default_value='western',
        description='Cultural context for social interaction (western, eastern, middle_eastern, latin_american)'
    )
    
    personality_profile_arg = DeclareLaunchArgument(
        'personality_profile',
        default_value='friendly_assistant',
        description='Personality profile for the robot (friendly_assistant, professional_expert, encouraging_coach, calm_companion)'
    )
    
    learning_enabled_arg = DeclareLaunchArgument(
        'learning_enabled',
        default_value='true',
        description='Enable social learning capabilities'
    )
    
    safety_threshold_arg = DeclareLaunchArgument(
        'safety_threshold',
        default_value='0.8',
        description='Safety threshold for social interactions'
    )
    
    response_timeout_arg = DeclareLaunchArgument(
        'response_timeout',
        default_value='2.0',
        description='Timeout for social responses in seconds'
    )
    
    # Social Intelligence Node
    social_intelligence_node = Node(
        package=package_name,
        executable='social_intelligence_node',
        name='social_intelligence_node',
        output='screen',
        parameters=[
            config_file,
            {
                'cultural_context': LaunchConfiguration('cultural_context'),
                'personality_profile': LaunchConfiguration('personality_profile'),
                'learning_enabled': LaunchConfiguration('learning_enabled'),
                'safety_threshold': LaunchConfiguration('safety_threshold'),
                'response_timeout': LaunchConfiguration('response_timeout'),
                'max_interaction_history': 1000,
                'emotion_recognition_confidence_threshold': 0.7,
                'social_behavior_confidence_threshold': 0.7,
                'cultural_adaptation_sensitivity': 0.8,
                'personality_consistency_threshold': 0.8,
                'learning_rate': 0.1,
                'pattern_confidence_threshold': 0.7,
                'max_patterns': 100,
                'max_adaptation_history': 50,
                'max_personality_history': 100,
                'max_behavior_history': 20
            }
        ],
        remappings=[
            ('sensors/facial_image', '/camera/facial_image'),
            ('sensors/voice_audio', '/microphone/audio'),
            ('sensors/body_pose', '/pose_estimation/body_poses'),
            ('speech_recognition/text', '/speech_to_text/transcript'),
            ('context/social_context', '/context_manager/social_context'),
            ('robot/state', '/robot_state_publisher/robot_state'),
            ('feedback/human_feedback', '/feedback_collector/human_feedback'),
            ('planning/task_plan', '/task_planner/current_plan'),
            ('safety/verification_request', '/safety_monitor/verification_request'),
            ('social_intelligence/verbal_response', '/speech_synthesis/verbal_response'),
            ('social_intelligence/gesture_response', '/gesture_controller/gesture_command'),
            ('social_intelligence/facial_response', '/facial_expression_controller/expression'),
            ('social_intelligence/emotion_analysis', '/emotion_analyzer/current_emotion'),
            ('social_intelligence/confidence', '/social_confidence_monitor/confidence'),
            ('social_intelligence/learning_insights', '/learning_analyzer/insights'),
            ('social_intelligence/cultural_adaptation', '/cultural_analyzer/adaptation_status'),
            ('social_intelligence/personality_state', '/personality_monitor/current_state'),
            ('social_intelligence/safety_status', '/safety_monitor/social_safety'),
            ('social_intelligence/interaction_status', '/interaction_monitor/status')
        ]
    )
    
    # Mock sensor nodes for demo (if not using real sensors)
    mock_facial_sensor = Node(
        package='eip_social_intelligence',
        executable='mock_facial_sensor.py',
        name='mock_facial_sensor',
        output='screen',
        parameters=[{
            'publish_rate': 10.0,  # 10 Hz
            'image_width': 640,
            'image_height': 480
        }],
        condition=LaunchConfiguration('use_mock_sensors')
    )
    
    mock_voice_sensor = Node(
        package='eip_social_intelligence',
        executable='mock_voice_sensor.py',
        name='mock_voice_sensor',
        output='screen',
        parameters=[{
            'publish_rate': 20.0,  # 20 Hz
            'audio_duration': 0.1  # 100ms chunks
        }],
        condition=LaunchConfiguration('use_mock_sensors')
    )
    
    mock_pose_sensor = Node(
        package='eip_social_intelligence',
        executable='mock_pose_sensor.py',
        name='mock_pose_sensor',
        output='screen',
        parameters=[{
            'publish_rate': 30.0,  # 30 Hz
            'num_poses': 17  # COCO keypoints
        }],
        condition=LaunchConfiguration('use_mock_sensors')
    )
    
    # Mock context manager
    mock_context_manager = Node(
        package='eip_social_intelligence',
        executable='mock_context_manager.py',
        name='mock_context_manager',
        output='screen',
        parameters=[{
            'update_rate': 1.0,  # 1 Hz
            'environment': 'indoor',
            'relationship': 'friendly',
            'cultural_context': LaunchConfiguration('cultural_context')
        }],
        condition=LaunchConfiguration('use_mock_sensors')
    )
    
    # Mock robot state publisher
    mock_robot_state = Node(
        package='eip_social_intelligence',
        executable='mock_robot_state.py',
        name='mock_robot_state',
        output='screen',
        parameters=[{
            'update_rate': 5.0,  # 5 Hz
            'capabilities': ['verbal', 'gestural', 'facial', 'proxemic'],
            'energy_level': 0.8,
            'social_comfort': 0.7
        }],
        condition=LaunchConfiguration('use_mock_sensors')
    )
    
    # Mock feedback collector
    mock_feedback_collector = Node(
        package='eip_social_intelligence',
        executable='mock_feedback_collector.py',
        name='mock_feedback_collector',
        output='screen',
        parameters=[{
            'feedback_rate': 0.1,  # Every 10 seconds
            'positive_feedback_probability': 0.7
        }],
        condition=LaunchConfiguration('use_mock_sensors')
    )
    
    # Mock response actuators
    mock_speech_synthesis = Node(
        package='eip_social_intelligence',
        executable='mock_speech_synthesis.py',
        name='mock_speech_synthesis',
        output='screen',
        parameters=[{
            'speech_rate': 150,  # words per minute
            'voice_type': 'friendly'
        }],
        condition=LaunchConfiguration('use_mock_sensors')
    )
    
    mock_gesture_controller = Node(
        package='eip_social_intelligence',
        executable='mock_gesture_controller.py',
        name='mock_gesture_controller',
        output='screen',
        parameters=[{
            'gesture_execution_time': 2.0,
            'smooth_transitions': True
        }],
        condition=LaunchConfiguration('use_mock_sensors')
    )
    
    mock_facial_controller = Node(
        package='eip_social_intelligence',
        executable='mock_facial_controller.py',
        name='mock_facial_controller',
        output='screen',
        parameters=[{
            'expression_transition_time': 0.5,
            'blink_rate': 0.2  # blinks per second
        }],
        condition=LaunchConfiguration('use_mock_sensors')
    )
    
    # Monitoring and visualization nodes
    social_monitor = Node(
        package='eip_social_intelligence',
        executable='social_monitor.py',
        name='social_monitor',
        output='screen',
        parameters=[{
            'monitoring_rate': 1.0,  # 1 Hz
            'log_interactions': True,
            'save_metrics': True
        }]
    )
    
    # RViz configuration for visualization
    rviz_config = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'social_intelligence.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=LaunchConfiguration('use_rviz')
    )
    
    # Launch argument for mock sensors
    use_mock_sensors_arg = DeclareLaunchArgument(
        'use_mock_sensors',
        default_value='true',
        description='Use mock sensors for demonstration'
    )
    
    # Launch argument for RViz
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz for visualization'
    )
    
    # Log messages
    startup_info = LogInfo(
        msg=['Starting Social Intelligence Demo with:',
             '  Cultural Context: ', LaunchConfiguration('cultural_context'),
             '  Personality Profile: ', LaunchConfiguration('personality_profile'),
             '  Learning Enabled: ', LaunchConfiguration('learning_enabled'),
             '  Safety Threshold: ', LaunchConfiguration('safety_threshold')]
    )
    
    # Create launch description
    return LaunchDescription([
        # Launch arguments
        cultural_context_arg,
        personality_profile_arg,
        learning_enabled_arg,
        safety_threshold_arg,
        response_timeout_arg,
        use_mock_sensors_arg,
        use_rviz_arg,
        
        # Log startup info
        startup_info,
        
        # Main social intelligence node
        social_intelligence_node,
        
        # Mock sensor nodes
        mock_facial_sensor,
        mock_voice_sensor,
        mock_pose_sensor,
        mock_context_manager,
        mock_robot_state,
        mock_feedback_collector,
        
        # Mock actuator nodes
        mock_speech_synthesis,
        mock_gesture_controller,
        mock_facial_controller,
        
        # Monitoring and visualization
        social_monitor,
        rviz_node
    ]) 