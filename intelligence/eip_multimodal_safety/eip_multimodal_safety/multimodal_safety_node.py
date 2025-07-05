#!/usr/bin/env python3
"""
Multi-Modal Safety Node

This node integrates multiple sensor modalities for comprehensive safety validation.
It subscribes to various sensor topics, performs sensor fusion, and publishes
safety validation results and violations.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import time
import logging
from typing import Dict, List, Optional
import threading

from sensor_msgs.msg import Image, LaserScan, PointCloud2, Imu
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Pose, Twist
from eip_interfaces.msg import SafetyViolation, SafetyVerificationRequest, SafetyVerificationResponse
from eip_interfaces.srv import ValidateTaskPlan

from .sensor_fusion import (
    SensorFusionEngine, SensorData, SensorType, FusionMethod,
    SafetyEvent, FusionResult
)


class MultiModalSafetyNode(Node):
    """
    Multi-Modal Safety Node for comprehensive safety validation
    
    This node integrates vision, audio, tactile, and proprioceptive sensors
    to provide robust safety monitoring with cross-modal correlation.
    """
    
    def __init__(self):
        super().__init__('multimodal_safety_node')
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('fusion_method', 'weighted_average'),
                ('safety_update_rate', 10.0),
                ('sensor_timeout', 2.0),
                ('enable_vision', True),
                ('enable_audio', True),
                ('enable_tactile', True),
                ('enable_proprioceptive', True),
                ('enable_lidar', True),
                ('vision_weight', 0.3),
                ('audio_weight', 0.2),
                ('tactile_weight', 0.25),
                ('proprioceptive_weight', 0.15),
                ('lidar_weight', 0.1),
                ('collision_threshold', 0.7),
                ('human_proximity_threshold', 0.8),
                ('velocity_threshold', 0.6),
                ('workspace_boundary_threshold', 0.5),
                ('emergency_stop_threshold', 0.9)
            ]
        )
        
        # Initialize sensor fusion engine
        fusion_method = FusionMethod(self.get_parameter('fusion_method').value)
        self.fusion_engine = SensorFusionEngine(fusion_method)
        
        # Set up sensor weights
        self._setup_sensor_weights()
        
        # Set up safety thresholds
        self._setup_safety_thresholds()
        
        # Initialize sensor data tracking
        self.last_sensor_data = {}
        self.sensor_timeouts = {}
        self.sensor_health = {}
        
        # Set up QoS profiles
        self.qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.qos_safety = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )
        
        # Set up callback groups
        self.sensor_callback_group = ReentrantCallbackGroup()
        self.safety_callback_group = ReentrantCallbackGroup()
        
        # Initialize publishers
        self._setup_publishers()
        
        # Initialize subscribers
        self._setup_subscribers()
        
        # Initialize services
        self._setup_services()
        
        # Start sensor fusion processing
        self.fusion_engine.start_processing()
        
        # Set up safety monitoring timer
        self.safety_timer = self.create_timer(
            1.0 / self.get_parameter('safety_update_rate').value,
            self._safety_monitoring_callback,
            callback_group=self.safety_callback_group
        )
        
        # Set up sensor health monitoring timer
        self.health_timer = self.create_timer(
            5.0,  # Check health every 5 seconds
            self._health_monitoring_callback,
            callback_group=self.safety_callback_group
        )
        
        self.logger.info("Multi-Modal Safety Node initialized successfully")
    
    def _setup_sensor_weights(self):
        """Set up sensor weights for fusion"""
        weights = {
            SensorType.VISION: self.get_parameter('vision_weight').value,
            SensorType.AUDIO: self.get_parameter('audio_weight').value,
            SensorType.TACTILE: self.get_parameter('tactile_weight').value,
            SensorType.PROPRIOCEPTIVE: self.get_parameter('proprioceptive_weight').value,
            SensorType.LIDAR: self.get_parameter('lidar_weight').value
        }
        
        self.fusion_engine.update_sensor_weights(weights)
        self.logger.info(f"Set sensor weights: {weights}")
    
    def _setup_safety_thresholds(self):
        """Set up safety thresholds"""
        self.safety_thresholds = {
            'collision_risk': self.get_parameter('collision_threshold').value,
            'human_proximity': self.get_parameter('human_proximity_threshold').value,
            'velocity_limit': self.get_parameter('velocity_threshold').value,
            'workspace_boundary': self.get_parameter('workspace_boundary_threshold').value,
            'emergency_stop': self.get_parameter('emergency_stop_threshold').value
        }
        
        self.logger.info(f"Set safety thresholds: {self.safety_thresholds}")
    
    def _setup_publishers(self):
        """Set up ROS publishers"""
        self.safety_violation_pub = self.create_publisher(
            SafetyViolation,
            '/eip/safety/violations',
            10,
            qos_profile=self.qos_safety
        )
        
        self.safety_score_pub = self.create_publisher(
            Float32,
            '/eip/safety/score',
            10,
            qos_profile=self.qos_safety
        )
        
        self.sensor_health_pub = self.create_publisher(
            String,
            '/eip/safety/sensor_health',
            10,
            qos_profile=self.qos_safety
        )
        
        self.fusion_result_pub = self.create_publisher(
            String,
            '/eip/safety/fusion_result',
            10,
            qos_profile=self.qos_safety
        )
    
    def _setup_subscribers(self):
        """Set up ROS subscribers for sensor data"""
        if self.get_parameter('enable_vision').value:
            self._subscribe_vision()
        if self.get_parameter('enable_audio').value:
            self._subscribe_audio()
        if self.get_parameter('enable_tactile').value:
            self._subscribe_tactile()
        if self.get_parameter('enable_proprioceptive').value:
            self._subscribe_proprioceptive()
        if self.get_parameter('enable_lidar').value:
            self._subscribe_lidar()

    def _subscribe_vision(self):
        # Vision sensors
        self.vision_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self._vision_callback,
            10,
            qos_profile=self.qos_sensor,
            callback_group=self.sensor_callback_group
        )
    def _subscribe_audio(self):
        self.audio_sub = self.create_subscription(
            String,
            '/microphone/audio',
            self._audio_callback,
            10,
            qos_profile=self.qos_sensor,
            callback_group=self.sensor_callback_group
        )
    def _subscribe_tactile(self):
        self.tactile_sub = self.create_subscription(
            String,
            '/tactile/sensor',
            self._tactile_callback,
            10,
            qos_profile=self.qos_sensor,
            callback_group=self.sensor_callback_group
        )
    def _subscribe_proprioceptive(self):
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self._imu_callback,
            10,
            qos_profile=self.qos_sensor,
            callback_group=self.sensor_callback_group
        )
        self.joint_states_sub = self.create_subscription(
            String,
            '/joint_states',
            self._joint_states_callback,
            10,
            qos_profile=self.qos_sensor,
            callback_group=self.sensor_callback_group
        )
    def _subscribe_lidar(self):
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self._lidar_callback,
            10,
            qos_profile=self.qos_sensor,
            callback_group=self.sensor_callback_group
        )
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/pointcloud',
            self._pointcloud_callback,
            10,
            qos_profile=self.qos_sensor,
            callback_group=self.sensor_callback_group
        )
    
    def _setup_services(self):
        """Set up ROS services"""
        self.safety_verification_service = self.create_service(
            SafetyVerificationRequest,
            '/eip/safety/verify',
            self._safety_verification_callback,
            callback_group=self.safety_callback_group
        )
        
        self.task_validation_service = self.create_service(
            ValidateTaskPlan,
            '/eip/safety/validate_task',
            self._task_validation_callback,
            callback_group=self.safety_callback_group
        )
    
    def _vision_callback(self, msg: Image):
        """Process vision sensor data"""
        try:
            # Extract safety-relevant features from image
            # This is a simplified implementation
            confidence = self._analyze_vision_safety(msg)
            
            sensor_data = SensorData(
                sensor_type=SensorType.VISION,
                timestamp=time.time(),
                data=msg,
                confidence=confidence,
                source_id='camera_main'
            )
            
            self.fusion_engine.add_sensor_data(sensor_data)
            self._update_sensor_timestamp('vision', time.time())
            
        except Exception as e:
            self.logger.error(f"Error processing vision data: {e}")
    
    def _depth_callback(self, msg: Image):
        """Process depth sensor data"""
        try:
            # Extract safety-relevant features from depth image
            confidence = self._analyze_depth_safety(msg)
            
            sensor_data = SensorData(
                sensor_type=SensorType.VISION,
                timestamp=time.time(),
                data=msg,
                confidence=confidence,
                source_id='camera_depth'
            )
            
            self.fusion_engine.add_sensor_data(sensor_data)
            self._update_sensor_timestamp('depth', time.time())
            
        except Exception as e:
            self.logger.error(f"Error processing depth data: {e}")
    
    def _audio_callback(self, msg: String):
        """Process audio sensor data"""
        try:
            # Extract safety-relevant features from audio
            confidence = self._analyze_audio_safety(msg)
            
            sensor_data = SensorData(
                sensor_type=SensorType.AUDIO,
                timestamp=time.time(),
                data=msg,
                confidence=confidence,
                source_id='microphone_array'
            )
            
            self.fusion_engine.add_sensor_data(sensor_data)
            self._update_sensor_timestamp('audio', time.time())
            
        except Exception as e:
            self.logger.error(f"Error processing audio data: {e}")
    
    def _tactile_callback(self, msg: String):
        """Process tactile sensor data"""
        try:
            # Extract safety-relevant features from tactile data
            confidence = self._analyze_tactile_safety(msg)
            
            sensor_data = SensorData(
                sensor_type=SensorType.TACTILE,
                timestamp=time.time(),
                data=msg,
                confidence=confidence,
                source_id='tactile_sensors'
            )
            
            self.fusion_engine.add_sensor_data(sensor_data)
            self._update_sensor_timestamp('tactile', time.time())
            
        except Exception as e:
            self.logger.error(f"Error processing tactile data: {e}")
    
    def _imu_callback(self, msg: Imu):
        """Process IMU sensor data"""
        try:
            # Extract safety-relevant features from IMU
            confidence = self._analyze_imu_safety(msg)
            
            sensor_data = SensorData(
                sensor_type=SensorType.PROPRIOCEPTIVE,
                timestamp=time.time(),
                data=msg,
                confidence=confidence,
                source_id='imu_main'
            )
            
            self.fusion_engine.add_sensor_data(sensor_data)
            self._update_sensor_timestamp('imu', time.time())
            
        except Exception as e:
            self.logger.error(f"Error processing IMU data: {e}")
    
    def _joint_states_callback(self, msg: String):
        """Process joint states data"""
        try:
            # Extract safety-relevant features from joint states
            confidence = self._analyze_joint_safety(msg)
            
            sensor_data = SensorData(
                sensor_type=SensorType.PROPRIOCEPTIVE,
                timestamp=time.time(),
                data=msg,
                confidence=confidence,
                source_id='joint_controller'
            )
            
            self.fusion_engine.add_sensor_data(sensor_data)
            self._update_sensor_timestamp('joints', time.time())
            
        except Exception as e:
            self.logger.error(f"Error processing joint states: {e}")
    
    def _lidar_callback(self, msg: LaserScan):
        """Process LIDAR sensor data"""
        try:
            # Extract safety-relevant features from LIDAR
            confidence = self._analyze_lidar_safety(msg)
            
            sensor_data = SensorData(
                sensor_type=SensorType.LIDAR,
                timestamp=time.time(),
                data=msg,
                confidence=confidence,
                source_id='lidar_main'
            )
            
            self.fusion_engine.add_sensor_data(sensor_data)
            self._update_sensor_timestamp('lidar', time.time())
            
        except Exception as e:
            self.logger.error(f"Error processing LIDAR data: {e}")
    
    def _pointcloud_callback(self, msg: PointCloud2):
        """Process point cloud data"""
        try:
            # Extract safety-relevant features from point cloud
            confidence = self._analyze_pointcloud_safety(msg)
            
            sensor_data = SensorData(
                sensor_type=SensorType.LIDAR,
                timestamp=time.time(),
                data=msg,
                confidence=confidence,
                source_id='pointcloud_processor'
            )
            
            self.fusion_engine.add_sensor_data(sensor_data)
            self._update_sensor_timestamp('pointcloud', time.time())
            
        except Exception as e:
            self.logger.error(f"Error processing point cloud data: {e}")
    
    def _analyze_vision_safety(self, msg: Image) -> float:
        """Analyze vision data for safety concerns"""
        # Simplified vision safety analysis
        # In a real implementation, this would use computer vision algorithms
        return 0.8  # Simulated confidence
    
    def _analyze_depth_safety(self, msg: Image) -> float:
        """Analyze depth data for safety concerns"""
        # Simplified depth safety analysis
        return 0.9  # Simulated confidence
    
    def _analyze_audio_safety(self, msg: String) -> float:
        """Analyze audio data for safety concerns"""
        # Simplified audio safety analysis
        return 0.7  # Simulated confidence
    
    def _analyze_tactile_safety(self, msg: String) -> float:
        """Analyze tactile data for safety concerns"""
        # Simplified tactile safety analysis
        return 0.85  # Simulated confidence
    
    def _analyze_imu_safety(self, msg: Imu) -> float:
        """Analyze IMU data for safety concerns"""
        # Simplified IMU safety analysis
        return 0.9  # Simulated confidence
    
    def _analyze_joint_safety(self, msg: String) -> float:
        """Analyze joint states for safety concerns"""
        # Simplified joint safety analysis
        return 0.8  # Simulated confidence
    
    def _analyze_lidar_safety(self, msg: LaserScan) -> float:
        """Analyze LIDAR data for safety concerns"""
        # Simplified LIDAR safety analysis
        return 0.95  # Simulated confidence
    
    def _analyze_pointcloud_safety(self, msg: PointCloud2) -> float:
        """Analyze point cloud data for safety concerns"""
        # Simplified point cloud safety analysis
        return 0.9  # Simulated confidence
    
    def _update_sensor_timestamp(self, sensor_id: str, timestamp: float):
        """Update sensor timestamp for health monitoring"""
        self.last_sensor_data[sensor_id] = timestamp
    
    def _safety_monitoring_callback(self):
        """Main safety monitoring callback"""
        try:
            # Perform sensor fusion
            fusion_result = self.fusion_engine.fuse_sensors()
            
            # Publish safety score
            safety_score_msg = Float32()
            safety_score_msg.data = fusion_result.safety_score
            self.safety_score_pub.publish(safety_score_msg)
            
            # Check for safety violations
            violations = self._check_safety_violations(fusion_result)
            
            # Publish violations
            for violation in violations:
                self.safety_violation_pub.publish(violation)
            
            # Publish fusion result summary
            self._publish_fusion_summary(fusion_result)
            
        except Exception as e:
            self.logger.error(f"Error in safety monitoring: {e}")
    
    def _check_safety_violations(self, fusion_result: FusionResult) -> List[SafetyViolation]:
        """Check for safety violations based on fusion result"""
        violations = []
        
        # Check overall safety score
        if fusion_result.safety_score < 0.5:
            violation = SafetyViolation()
            violation.violation_type = "overall_safety_low"
            violation.severity = 1.0 - fusion_result.safety_score
            violation.description = f"Overall safety score too low: {fusion_result.safety_score:.2f}"
            violation.timestamp = self.get_clock().now().to_msg()
            violations.append(violation)
        
        # Check individual safety events
        for event in fusion_result.events:
            if event.severity > 0.7:  # High severity events
                violation = SafetyViolation()
                violation.violation_type = event.event_type
                violation.severity = event.severity
                violation.description = event.description
                violation.timestamp = self.get_clock().now().to_msg()
                violations.append(violation)
        
        return violations
    
    def _publish_fusion_summary(self, fusion_result: FusionResult):
        """Publish fusion result summary"""
        summary = {
            'safety_score': fusion_result.safety_score,
            'fusion_confidence': fusion_result.fusion_confidence,
            'processing_time': fusion_result.processing_time,
            'event_count': len(fusion_result.events),
            'sensor_health': fusion_result.sensor_health
        }
        
        summary_msg = String()
        summary_msg.data = str(summary)
        self.fusion_result_pub.publish(summary_msg)
    
    def _health_monitoring_callback(self):
        """Monitor sensor health"""
        try:
            current_time = time.time()
            timeout_threshold = self.get_parameter('sensor_timeout').value
            
            # Check sensor timeouts
            for sensor_id, last_time in self.last_sensor_data.items():
                if current_time - last_time > timeout_threshold:
                    self.logger.warning(f"Sensor {sensor_id} timeout detected")
            
            # Get sensor health from fusion engine
            health_metrics = self.fusion_engine.get_performance_metrics()
            
            # Publish health metrics
            health_msg = String()
            health_msg.data = str(health_metrics)
            self.sensor_health_pub.publish(health_msg)
            
        except Exception as e:
            self.logger.error(f"Error in health monitoring: {e}")
    
    def _safety_verification_callback(self, request: SafetyVerificationRequest, response: SafetyVerificationResponse) -> SafetyVerificationResponse:
        """Handle safety verification requests"""
        try:
            # Perform real-time safety verification
            fusion_result = self.fusion_engine.fuse_sensors()
            
            response.is_safe = fusion_result.safety_score > 0.5
            response.confidence = fusion_result.fusion_confidence
            response.safety_score = fusion_result.safety_score
            response.description = f"Safety verification completed. Score: {fusion_result.safety_score:.2f}"
            
            self.logger.info(f"Safety verification: {response.is_safe} (score: {fusion_result.safety_score:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error in safety verification: {e}")
            response.is_safe = False
            response.confidence = 0.0
            response.safety_score = 0.0
            response.description = f"Safety verification failed: {str(e)}"
        
        return response
    
    def _task_validation_callback(self, request: ValidateTaskPlan, response) -> ValidateTaskPlan.Response:
        """Handle task plan validation requests"""
        try:
            # Validate task plan against current safety conditions
            fusion_result = self.fusion_engine.fuse_sensors()
            
            # Check if task is safe to execute
            is_safe = fusion_result.safety_score > 0.6
            
            response.is_valid = is_safe
            response.confidence = fusion_result.fusion_confidence
            response.reason = f"Task validation based on safety score: {fusion_result.safety_score:.2f}"
            
            self.logger.info(f"Task validation: {is_safe} (score: {fusion_result.safety_score:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error in task validation: {e}")
            response.is_valid = False
            response.confidence = 0.0
            response.reason = f"Task validation failed: {str(e)}"
        
        return response
    
    def on_shutdown(self):
        """Cleanup on shutdown"""
        self.fusion_engine.stop_processing()
        self.logger.info("Multi-Modal Safety Node shutdown complete")


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    node = MultiModalSafetyNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 