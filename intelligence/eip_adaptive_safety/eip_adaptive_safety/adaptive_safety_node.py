#!/usr/bin/env python3
"""
Adaptive Safety Node

This node implements adaptive safety learning for continuous safety improvement.
It learns from safety experiences, identifies patterns, and dynamically adjusts
safety thresholds to improve overall safety performance.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any
import threading

from std_msgs.msg import String, Float32
from eip_interfaces.msg import SafetyViolation, SafetyVerificationRequest, SafetyVerificationResponse
from eip_interfaces.srv import ValidateTaskPlan

from .adaptive_learning_engine import (
    AdaptiveSafetyLearningEngine, SafetyExperience, LearningMethod,
    SafetyPattern, AdaptiveThreshold, LearningResult
)


class AdaptiveSafetyNode(Node):
    """
    Adaptive Safety Node for continuous safety improvement
    
    This node learns from safety experiences, identifies patterns, and
    dynamically adjusts safety thresholds to improve safety performance.
    """
    
    def __init__(self):
        super().__init__('adaptive_safety_node')
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('learning_methods', ['online_learning', 'pattern_recognition', 'threshold_adjustment']),
                ('learning_update_rate', 5.0),
                ('experience_buffer_size', 10000),
                ('pattern_detection_enabled', True),
                ('threshold_optimization_enabled', True),
                ('online_learning_enabled', True),
                ('federated_learning_enabled', False),
                ('reinforcement_learning_enabled', False),
                ('learning_confidence_threshold', 0.7),
                ('pattern_confidence_threshold', 0.6),
                ('threshold_adaptation_rate', 0.1),
                ('save_learning_state', True),
                ('learning_state_file', '/tmp/adaptive_safety_state.pkl')
            ]
        )
        
        # Initialize adaptive learning engine
        learning_methods = self._parse_learning_methods()
        self.learning_engine = AdaptiveSafetyLearningEngine(learning_methods)
        
        # Initialize experience tracking
        self.current_experience = None
        self.experience_start_time = None
        
        # Set up QoS profiles
        self.qos_safety = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )
        
        # Set up callback groups
        self.safety_callback_group = ReentrantCallbackGroup()
        self.learning_callback_group = ReentrantCallbackGroup()
        
        # Initialize publishers
        self._setup_publishers()
        
        # Initialize subscribers
        self._setup_subscribers()
        
        # Initialize services
        self._setup_services()
        
        # Start adaptive learning
        self.learning_engine.start_learning()
        
        # Set up learning monitoring timer
        self.learning_timer = self.create_timer(
            1.0 / self.get_parameter('learning_update_rate').value,
            self._learning_monitoring_callback,
            callback_group=self.learning_callback_group
        )
        
        # Set up state saving timer
        if self.get_parameter('save_learning_state').value:
            self.state_save_timer = self.create_timer(
                300.0,  # Save every 5 minutes
                self._save_learning_state_callback,
                callback_group=self.learning_callback_group
            )
        
        self.logger.info("Adaptive Safety Node initialized successfully")
    
    def _parse_learning_methods(self) -> List[LearningMethod]:
        """Parse learning methods from parameters"""
        method_strings = self.get_parameter('learning_methods').value
        learning_methods = []
        
        method_mapping = {
            'online_learning': LearningMethod.ONLINE_LEARNING,
            'pattern_recognition': LearningMethod.PATTERN_RECOGNITION,
            'threshold_adjustment': LearningMethod.THRESHOLD_ADJUSTMENT,
            'federated_learning': LearningMethod.FEDERATED_LEARNING,
            'reinforcement_learning': LearningMethod.REINFORCEMENT_LEARNING
        }
        
        for method_str in method_strings:
            if method_str in method_mapping:
                learning_methods.append(method_mapping[method_str])
        
        return learning_methods
    
    def _setup_publishers(self):
        """Set up ROS publishers"""
        self.learning_result_pub = self.create_publisher(
            String,
            '/eip/adaptive_safety/learning_result',
            10,
            qos_profile=self.qos_safety
        )
        
        self.pattern_detection_pub = self.create_publisher(
            String,
            '/eip/adaptive_safety/patterns',
            10,
            qos_profile=self.qos_safety
        )
        
        self.threshold_update_pub = self.create_publisher(
            String,
            '/eip/adaptive_safety/thresholds',
            10,
            qos_profile=self.qos_safety
        )
        
        self.learning_metrics_pub = self.create_publisher(
            String,
            '/eip/adaptive_safety/metrics',
            10,
            qos_profile=self.qos_safety
        )
        
        self.recommendations_pub = self.create_publisher(
            String,
            '/eip/adaptive_safety/recommendations',
            10,
            qos_profile=self.qos_safety
        )
    
    def _setup_subscribers(self):
        """Set up ROS subscribers"""
        self._subscribe_safety_violation()
        self._subscribe_safety_score()
        self._subscribe_sensor_health()

    def _subscribe_safety_violation(self):
        self.safety_violation_sub = self.create_subscription(
            SafetyViolation,
            '/eip/safety/violations',
            self._safety_violation_callback,
            10,
            qos_profile=self.qos_safety,
            callback_group=self.safety_callback_group
        )
    def _subscribe_safety_score(self):
        self.safety_score_sub = self.create_subscription(
            Float32,
            '/eip/safety/score',
            self._safety_score_callback,
            10,
            qos_profile=self.qos_safety,
            callback_group=self.safety_callback_group
        )
    def _subscribe_sensor_health(self):
        self.sensor_health_sub = self.create_subscription(
            String,
            '/eip/safety/sensor_health',
            self._sensor_health_callback,
            10,
            qos_profile=self.qos_safety,
            callback_group=self.safety_callback_group
        )
        
        # Fusion result subscriber
        self.fusion_result_sub = self.create_subscription(
            String,
            '/eip/safety/fusion_result',
            self._fusion_result_callback,
            10,
            qos_profile=self.qos_safety,
            callback_group=self.safety_callback_group
        )
    
    def _setup_services(self):
        """Set up ROS services"""
        self.learning_analysis_service = self.create_service(
            SafetyVerificationRequest,
            '/eip/adaptive_safety/analyze',
            self._learning_analysis_callback,
            callback_group=self.safety_callback_group
        )
        
        self.pattern_query_service = self.create_service(
            ValidateTaskPlan,
            '/eip/adaptive_safety/query_patterns',
            self._pattern_query_callback,
            callback_group=self.safety_callback_group
        )
    
    def _safety_violation_callback(self, msg: SafetyViolation):
        """Process safety violations for learning"""
        try:
            # Start new experience if not already started
            if self.current_experience is None:
                self._start_new_experience()
            
            # Update current experience with violation
            if self.current_experience:
                self.current_experience.safety_events.append(msg.violation_type)
                
                # Determine outcome based on violation severity
                if msg.severity > 0.8:
                    self.current_experience.outcome = 'failure'
                elif msg.severity > 0.5:
                    self.current_experience.outcome = 'near_miss'
                else:
                    self.current_experience.outcome = 'success'
                
                # Complete experience if significant time has passed
                if time.time() - self.experience_start_time > 5.0:
                    self._complete_experience()
            
        except Exception as e:
            self.logger.error(f"Error processing safety violation: {e}")
    
    def _safety_score_callback(self, msg: Float32):
        """Process safety scores for learning"""
        try:
            # Update current experience with safety score
            if self.current_experience:
                self.current_experience.safety_score = msg.data
            
        except Exception as e:
            self.logger.error(f"Error processing safety score: {e}")
    
    def _sensor_health_callback(self, msg: String):
        """Process sensor health for learning"""
        try:
            # Parse sensor health data
            sensor_health = eval(msg.data)  # Simple parsing for demo
            
            # Update current experience with sensor data
            if self.current_experience:
                self.current_experience.sensor_data.update(sensor_health)
            
        except Exception as e:
            self.logger.error(f"Error processing sensor health: {e}")
    
    def _fusion_result_callback(self, msg: String):
        """Process fusion results for learning"""
        try:
            # Parse fusion result data
            fusion_result = eval(msg.data)  # Simple parsing for demo
            
            # Update current experience with fusion data
            if self.current_experience:
                self.current_experience.sensor_data.update(fusion_result)
            
        except Exception as e:
            self.logger.error(f"Error processing fusion result: {e}")
    
    def _start_new_experience(self):
        """Start a new safety experience"""
        self.current_experience = SafetyExperience(
            timestamp=time.time(),
            sensor_data={},
            safety_score=0.0,
            safety_events=[],
            action_taken="monitoring",
            outcome="success",
            environment_context={},
            robot_state={}
        )
        self.experience_start_time = time.time()
    
    def _complete_experience(self):
        """Complete current experience and add to learning engine"""
        try:
            if self.current_experience:
                # Add environment context
                self.current_experience.environment_context = {
                    'timestamp': time.time(),
                    'node_id': self.get_name()
                }
                
                # Add robot state (simulated)
                self.current_experience.robot_state = {
                    'status': 'operational',
                    'battery_level': 0.8,
                    'temperature': 25.0
                }
                
                # Add to learning engine
                self.learning_engine.add_experience(self.current_experience)
                
                self.logger.debug(f"Completed experience: {self.current_experience.outcome}")
                
                # Reset for next experience
                self.current_experience = None
                self.experience_start_time = None
            
        except Exception as e:
            self.logger.error(f"Error completing experience: {e}")
    
    def _learning_monitoring_callback(self):
        """Main learning monitoring callback"""
        try:
            # Get learning results
            learning_result = self.learning_engine.get_learning_result()
            
            # Publish learning results
            self._publish_learning_results(learning_result)
            
            # Check for significant patterns
            if learning_result.new_patterns:
                self._publish_pattern_detection(learning_result.new_patterns)
            
            # Check for threshold updates
            if learning_result.updated_thresholds:
                self._publish_threshold_updates(learning_result.updated_thresholds)
            
            # Publish recommendations
            if learning_result.recommendations:
                self._publish_recommendations(learning_result.recommendations)
            
        except Exception as e:
            self.logger.error(f"Error in learning monitoring: {e}")
    
    def _publish_learning_results(self, learning_result: LearningResult):
        """Publish learning results"""
        result_msg = String()
        result_msg.data = str({
            'learning_confidence': learning_result.learning_confidence,
            'new_patterns_count': len(learning_result.new_patterns),
            'updated_thresholds_count': len(learning_result.updated_thresholds),
            'recommendations_count': len(learning_result.recommendations)
        })
        self.learning_result_pub.publish(result_msg)
    
    def _publish_pattern_detection(self, patterns: List[SafetyPattern]):
        """Publish pattern detection results"""
        pattern_msg = String()
        pattern_data = []
        
        for pattern in patterns:
            pattern_data.append({
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type.value,
                'confidence': pattern.confidence,
                'frequency': pattern.frequency,
                'success_rate': pattern.success_rate
            })
        
        pattern_msg.data = str(pattern_data)
        self.pattern_detection_pub.publish(pattern_msg)
    
    def _publish_threshold_updates(self, thresholds: Dict[str, AdaptiveThreshold]):
        """Publish threshold updates"""
        threshold_msg = String()
        threshold_data = {}
        
        for name, threshold in thresholds.items():
            threshold_data[name] = {
                'current_value': threshold.current_value,
                'confidence': threshold.confidence,
                'last_updated': threshold.last_updated
            }
        
        threshold_msg.data = str(threshold_data)
        self.threshold_update_pub.publish(threshold_msg)
    
    def _publish_recommendations(self, recommendations: List[str]):
        """Publish safety recommendations"""
        recommendation_msg = String()
        recommendation_msg.data = str(recommendations)
        self.recommendations_pub.publish(recommendation_msg)
    
    def _save_learning_state_callback(self):
        """Save learning state to file"""
        try:
            filepath = self.get_parameter('learning_state_file').value
            self.learning_engine.save_learning_state(filepath)
            
        except Exception as e:
            self.logger.error(f"Error saving learning state: {e}")
    
    def _learning_analysis_callback(self, request: SafetyVerificationRequest, response: SafetyVerificationResponse) -> SafetyVerificationResponse:
        """Handle learning analysis requests"""
        try:
            # Get current learning state
            learning_result = self.learning_engine.get_learning_result()
            
            # Analyze safety based on learning
            is_safe = learning_result.learning_confidence > self.get_parameter('learning_confidence_threshold').value
            
            response.is_safe = is_safe
            response.confidence = learning_result.learning_confidence
            response.safety_score = learning_result.learning_confidence
            response.description = f"Learning-based safety analysis. Confidence: {learning_result.learning_confidence:.2f}"
            
            self.logger.info(f"Learning analysis: {is_safe} (confidence: {learning_result.learning_confidence:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error in learning analysis: {e}")
            response.is_safe = False
            response.confidence = 0.0
            response.safety_score = 0.0
            response.description = f"Learning analysis failed: {str(e)}"
        
        return response
    
    def _pattern_query_callback(self, request: ValidateTaskPlan, response) -> ValidateTaskPlan.Response:
        """Handle pattern query requests"""
        try:
            # Get patterns from learning engine
            patterns = self.learning_engine.get_patterns()
            
            # Analyze patterns for task validation
            high_confidence_patterns = [
                p for p in patterns 
                if p.confidence > self.get_parameter('pattern_confidence_threshold').value
            ]
            
            # Determine if task is valid based on patterns
            is_valid = len(high_confidence_patterns) == 0  # No high-confidence patterns = safe
            
            response.is_valid = is_valid
            response.confidence = 0.8 if is_valid else 0.3
            response.reason = f"Pattern analysis: {len(high_confidence_patterns)} high-confidence patterns detected"
            
            self.logger.info(f"Pattern query: {is_valid} ({len(high_confidence_patterns)} patterns)")
            
        except Exception as e:
            self.logger.error(f"Error in pattern query: {e}")
            response.is_valid = False
            response.confidence = 0.0
            response.reason = f"Pattern query failed: {str(e)}"
        
        return response
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get current learning metrics"""
        learning_result = self.learning_engine.get_learning_result()
        
        return {
            'learning_confidence': learning_result.learning_confidence,
            'total_experiences': self.learning_engine.learning_metrics['total_experiences'],
            'patterns_identified': self.learning_engine.learning_metrics['patterns_identified'],
            'threshold_updates': self.learning_engine.learning_metrics['threshold_updates'],
            'success_rate': self.learning_engine.learning_metrics['success_rate'],
            'learning_accuracy': self.learning_engine.learning_metrics['learning_accuracy'],
            'active_patterns': len(self.learning_engine.pattern_database),
            'active_thresholds': len(self.learning_engine.threshold_registry)
        }
    
    def on_shutdown(self):
        """Cleanup on shutdown"""
        # Complete any pending experience
        if self.current_experience:
            self._complete_experience()
        
        # Stop learning engine
        self.learning_engine.stop_learning()
        
        # Save final state
        if self.get_parameter('save_learning_state').value:
            filepath = self.get_parameter('learning_state_file').value
            self.learning_engine.save_learning_state(filepath)
        
        self.logger.info("Adaptive Safety Node shutdown complete")


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    node = AdaptiveSafetyNode()
    
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