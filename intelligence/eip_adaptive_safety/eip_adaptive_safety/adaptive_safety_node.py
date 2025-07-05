#!/usr/bin/env python3
"""
Adaptive Safety Node

Main node for adaptive safety learning system that integrates with ROS 2
and provides real-time safety assessment with continuous learning.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import numpy as np
import threading
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import uuid

# ROS 2 imports
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import Image, PointCloud2, Imu
from geometry_msgs.msg import Twist, PoseStamped
from eip_interfaces.msg import SafetyVerificationRequest, SafetyVerificationResponse, SafetyViolation
from eip_interfaces.srv import ValidateTaskPlan

# Custom imports
from .adaptive_learning_engine import AdaptiveLearningEngine, SafetyExperience


class AdaptiveSafetyNode(Node):
    """
    Adaptive Safety Node
    
    Integrates adaptive learning engine with ROS 2 for real-time
    safety assessment and continuous learning.
    """
    
    def __init__(self):
        super().__init__('adaptive_safety_node')
        
        # Node configuration
        self.node_id = str(uuid.uuid4())[:8]
        self.learning_rate = self.declare_parameter('learning_rate', 0.001).value
        self.memory_size = self.declare_parameter('memory_size', 1000).value
        self.update_rate = self.declare_parameter('update_rate', 10.0).value
        self.enable_learning = self.declare_parameter('enable_learning', True).value
        
        # Initialize adaptive learning engine
        self.learning_engine = AdaptiveLearningEngine(
            learning_rate=self.learning_rate,
            memory_size=self.memory_size
        )
        
        # State management
        self.last_sensor_data = {}
        self.safety_history = []
        self.learning_enabled = self.enable_learning
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Setup QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Setup callback groups
        self.callback_group = ReentrantCallbackGroup()
        
        # Initialize publishers and subscribers
        self._setup_communications(qos_profile)
        
        # Start background threads
        self._start_background_threads()
        
        self.get_logger().info(f"Adaptive Safety Node {self.node_id} initialized")
    
    def _setup_communications(self, qos_profile: QoSProfile):
        """Setup ROS 2 communications"""
        
        # Publishers
        self.safety_response_pub = self.create_publisher(
            SafetyVerificationResponse,
            '/adaptive_safety/response',
            10
        )
        
        self.safety_violation_pub = self.create_publisher(
            SafetyViolation,
            '/adaptive_safety/violation',
            10
        )
        
        self.learning_status_pub = self.create_publisher(
            String,
            '/adaptive_safety/learning_status',
            10
        )
        
        self.pattern_update_pub = self.create_publisher(
            String,
            '/adaptive_safety/pattern_update',
            10
        )
        
        # Subscribers
        self.safety_request_sub = self.create_subscription(
            SafetyVerificationRequest,
            '/adaptive_safety/request',
            self._handle_safety_request,
            10,
            callback_group=self.callback_group
        )
        
        # Services
        self.validate_task_service = self.create_service(
            ValidateTaskPlan,
            '/adaptive_safety/validate_task',
            self._handle_validate_task,
            callback_group=self.callback_group
        )
        
        # Timers
        self.learning_timer = self.create_timer(
            1.0 / self.update_rate,  # Update at specified rate
            self._learning_update_loop,
            callback_group=self.callback_group
        )
        
        self.status_timer = self.create_timer(
            5.0,  # Status update every 5 seconds
            self._publish_status,
            callback_group=self.callback_group
        )
    
    def _start_background_threads(self):
        """Start background processing threads"""
        
        # Learning thread
        self.learning_thread = threading.Thread(
            target=self._continuous_learning_loop,
            daemon=True
        )
        self.learning_thread.start()
        
        # Pattern analysis thread
        self.pattern_thread = threading.Thread(
            target=self._pattern_analysis_loop,
            daemon=True
        )
        self.pattern_thread.start()
    
    def _handle_safety_request(self, request: SafetyVerificationRequest):
        """Handle incoming safety verification requests"""
        try:
            with self.lock:
                # Extract sensor data from request
                sensor_data = self._extract_sensor_data(request)
                
                # Assess safety using adaptive learning engine
                safety_assessment = self.learning_engine.assess_safety(sensor_data)
                
                # Create safety experience for learning
                if self.learning_enabled:
                    experience = self._create_safety_experience(sensor_data, safety_assessment)
                    self.learning_engine.learn_from_experience(experience)
                
                # Publish response
                response = SafetyVerificationResponse()
                response.request_id = request.request_id
                response.is_safe = safety_assessment['is_safe']
                response.confidence = safety_assessment['confidence']
                response.safety_level = safety_assessment['safety_level']
                response.metadata = json.dumps(safety_assessment)
                
                self.safety_response_pub.publish(response)
                
                # Store in history
                self.safety_history.append({
                    'timestamp': time.time(),
                    'request_id': request.request_id,
                    'assessment': safety_assessment
                })
                
                # Keep only recent history
                if len(self.safety_history) > 100:
                    self.safety_history = self.safety_history[-100:]
                
                self.get_logger().debug(f"Processed safety request {request.request_id}")
                
        except Exception as e:
            self.get_logger().error(f"Error processing safety request: {e}")
            # Publish failure response
            response = SafetyVerificationResponse()
            response.request_id = request.request_id
            response.is_safe = False
            response.confidence = 0.0
            response.safety_level = 0.0
            response.metadata = json.dumps({"error": str(e)})
            self.safety_response_pub.publish(response)
    
    def _handle_validate_task(self, request, response):
        """Handle task plan validation requests"""
        try:
            with self.lock:
                # Parse task plan
                task_plan = json.loads(request.task_plan)
                
                # Validate using learned patterns and rules
                validation_result = self._validate_task_plan(task_plan)
                
                response.is_valid = validation_result['is_valid']
                response.confidence = validation_result['confidence']
                response.safety_level = validation_result['safety_level']
                response.recommendations = validation_result['recommendations']
                
                return response
                
        except Exception as e:
            self.get_logger().error(f"Error validating task plan: {e}")
            response.is_valid = False
            response.confidence = 0.0
            response.safety_level = 0.0
            response.recommendations = [f"Validation error: {str(e)}"]
            return response
    
    def _extract_sensor_data(self, request: SafetyVerificationRequest) -> Dict[str, Any]:
        """Extract sensor data from safety request"""
        sensor_data = {
            'timestamp': time.time(),
            'node_id': self.node_id
        }
        
        # Parse metadata for sensor information
        if request.metadata:
            try:
                metadata = json.loads(request.metadata)
                sensor_data.update(metadata)
            except json.JSONDecodeError:
                self.get_logger().warning("Invalid metadata format in safety request")
        
        # Store for learning
        self.last_sensor_data = sensor_data.copy()
        
        return sensor_data
    
    def _create_safety_experience(self, sensor_data: Dict[str, Any], assessment: Dict[str, Any]) -> SafetyExperience:
        """Create a safety experience from current assessment"""
        
        # Determine outcome based on assessment
        if assessment['is_safe']:
            outcome = 'safe'
        elif assessment['safety_level'] < 0.3:
            outcome = 'violation'
        elif assessment['safety_level'] < 0.5:
            outcome = 'unsafe'
        else:
            outcome = 'near_miss'
        
        # Create context
        context = {
            'assessment_confidence': assessment['confidence'],
            'neural_safety': assessment.get('neural_safety', 0.0),
            'pattern_safety': assessment.get('pattern_safety', 0.0),
            'active_patterns': assessment.get('active_patterns', 0),
            'active_rules': assessment.get('active_rules', 0)
        }
        
        experience = SafetyExperience(
            experience_id=f"exp_{int(time.time() * 1000)}",
            timestamp=time.time(),
            sensor_data=sensor_data,
            safety_level=assessment['safety_level'],
            outcome=outcome,
            context=context
        )
        
        return experience
    
    def _validate_task_plan(self, task_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a task plan using learned patterns and rules"""
        
        # Extract task features
        task_features = self._extract_task_features(task_plan)
        
        # Assess safety using learning engine
        safety_assessment = self.learning_engine.assess_safety(task_features)
        
        # Generate recommendations
        recommendations = self._generate_task_recommendations(task_plan, safety_assessment)
        
        return {
            'is_valid': safety_assessment['is_safe'],
            'confidence': safety_assessment['confidence'],
            'safety_level': safety_assessment['safety_level'],
            'recommendations': recommendations
        }
    
    def _extract_task_features(self, task_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from task plan for safety assessment"""
        
        # Create synthetic sensor data from task plan
        task_features = {
            'timestamp': time.time(),
            'node_id': self.node_id
        }
        
        # Extract motion-related features
        if 'motion' in task_plan:
            motion = task_plan['motion']
            task_features['proprioceptive'] = {
                'velocity': motion.get('velocity', 0.0),
                'acceleration': motion.get('acceleration', 0.0),
                'features': np.array([
                    motion.get('velocity', 0.0),
                    motion.get('acceleration', 0.0),
                    motion.get('angular_velocity', 0.0),
                    motion.get('force', 0.0)
                ])
            }
        
        # Extract environment features
        if 'environment' in task_plan:
            env = task_plan['environment']
            task_features['vision'] = {
                'human_proximity': env.get('human_proximity', 1.0),
                'obstacle_density': env.get('obstacle_density', 0.0),
                'features': np.array([
                    env.get('human_proximity', 1.0),
                    env.get('obstacle_density', 0.0),
                    env.get('lighting_condition', 0.5),
                    env.get('workspace_crowding', 0.0)
                ])
            }
        
        # Extract interaction features
        if 'interaction' in task_plan:
            interaction = task_plan['interaction']
            task_features['tactile'] = {
                'contact_force': interaction.get('contact_force', 0.0),
                'contact_area': interaction.get('contact_area', 0.0),
                'features': np.array([
                    interaction.get('contact_force', 0.0),
                    interaction.get('contact_area', 0.0),
                    interaction.get('friction_coefficient', 0.5),
                    interaction.get('surface_hardness', 0.5)
                ])
            }
        
        return task_features
    
    def _generate_task_recommendations(self, task_plan: Dict[str, Any], assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations for task plan"""
        
        recommendations = []
        
        # Check safety level
        if assessment['safety_level'] < 0.5:
            recommendations.append("Task has significant safety risks")
        
        if assessment['confidence'] < 0.7:
            recommendations.append("Low confidence in safety assessment - recommend human supervision")
        
        # Check specific task components
        if 'motion' in task_plan:
            motion = task_plan['motion']
            if motion.get('velocity', 0.0) > 0.8:
                recommendations.append("High velocity detected - consider reducing speed")
            if motion.get('acceleration', 0.0) > 0.7:
                recommendations.append("High acceleration detected - consider limiting acceleration")
        
        if 'environment' in task_plan:
            env = task_plan['environment']
            if env.get('human_proximity', 1.0) < 0.5:
                recommendations.append("Human proximity detected - ensure safe distance")
            if env.get('obstacle_density', 0.0) > 0.6:
                recommendations.append("High obstacle density - consider path planning")
        
        if 'interaction' in task_plan:
            interaction = task_plan['interaction']
            if interaction.get('contact_force', 0.0) > 0.8:
                recommendations.append("High contact force - consider force limiting")
        
        # Add rule-based recommendations
        rule_recommendations = assessment.get('rule_recommendations', [])
        recommendations.extend(rule_recommendations)
        
        return recommendations
    
    def _learning_update_loop(self):
        """Main learning update loop"""
        
        try:
            with self.lock:
                # Check if we have recent sensor data
                if self.last_sensor_data and self.learning_enabled:
                    # Create synthetic experience for continuous learning
                    current_time = time.time()
                    if current_time - self.last_sensor_data.get('timestamp', 0) < 1.0:
                        # Create experience from current data
                        experience = SafetyExperience(
                            experience_id=f"continuous_{int(current_time * 1000)}",
                            timestamp=current_time,
                            sensor_data=self.last_sensor_data,
                            safety_level=0.5,  # Neutral default
                            outcome='safe',  # Assume safe for continuous learning
                            context={'source': 'continuous_learning'}
                        )
                        
                        # Learn from experience
                        self.learning_engine.learn_from_experience(experience)
                
        except Exception as e:
            self.get_logger().error(f"Error in learning update loop: {e}")
    
    def _continuous_learning_loop(self):
        """Background thread for continuous learning"""
        
        while rclpy.ok():
            try:
                with self.lock:
                    # Perform background learning tasks
                    self._perform_background_learning()
                
                time.sleep(2.0)  # Learn every 2 seconds
                
            except Exception as e:
                self.get_logger().error(f"Error in continuous learning: {e}")
                time.sleep(5.0)
    
    def _pattern_analysis_loop(self):
        """Background thread for pattern analysis"""
        
        while rclpy.ok():
            try:
                with self.lock:
                    # Analyze patterns and publish updates
                    self._analyze_and_publish_patterns()
                
                time.sleep(10.0)  # Analyze every 10 seconds
                
            except Exception as e:
                self.get_logger().error(f"Error in pattern analysis: {e}")
                time.sleep(15.0)
    
    def _perform_background_learning(self):
        """Perform background learning tasks"""
        
        # Check for evolution opportunities
        if self.learning_engine._should_evolve():
            self.learning_engine._trigger_evolution()
            self.get_logger().info("Learning evolution triggered")
        
        # Clean up old data
        if len(self.safety_history) > 200:
            self.safety_history = self.safety_history[-100:]
    
    def _analyze_and_publish_patterns(self):
        """Analyze patterns and publish updates"""
        
        # Get learning engine status
        status = self.learning_engine.get_status()
        
        # Create pattern update message
        pattern_update = {
            'timestamp': time.time(),
            'node_id': self.node_id,
            'pattern_count': status['pattern_count'],
            'rule_count': status['rule_count'],
            'learning_rounds': status['learning_rounds'],
            'pattern_discoveries': status['pattern_discoveries'],
            'rule_evolutions': status['rule_evolutions']
        }
        
        # Publish pattern update
        pattern_msg = String()
        pattern_msg.data = json.dumps(pattern_update)
        self.pattern_update_pub.publish(pattern_msg)
    
    def _publish_status(self):
        """Publish learning status"""
        
        try:
            with self.lock:
                # Get learning engine status
                status = self.learning_engine.get_status()
                
                # Add node-specific status
                status.update({
                    'node_id': self.node_id,
                    'learning_enabled': self.learning_enabled,
                    'safety_history_size': len(self.safety_history),
                    'last_update': time.time()
                })
                
                # Publish status
                status_msg = String()
                status_msg.data = json.dumps(status)
                self.learning_status_pub.publish(status_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error publishing status: {e}")
    
    def toggle_learning(self, enable: bool):
        """Toggle learning on/off"""
        
        with self.lock:
            self.learning_enabled = enable
            self.get_logger().info(f"Learning {'enabled' if enable else 'disabled'}")
    
    def save_learned_model(self, filepath: str):
        """Save the learned model to file"""
        
        try:
            with self.lock:
                self.learning_engine.save_model(filepath)
                self.get_logger().info(f"Model saved to {filepath}")
        except Exception as e:
            self.get_logger().error(f"Error saving model: {e}")
    
    def load_learned_model(self, filepath: str):
        """Load the learned model from file"""
        
        try:
            with self.lock:
                self.learning_engine.load_model(filepath)
                self.get_logger().info(f"Model loaded from {filepath}")
        except Exception as e:
            self.get_logger().error(f"Error loading model: {e}")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status"""
        
        with self.lock:
            status = self.learning_engine.get_status()
            status.update({
                'node_id': self.node_id,
                'learning_enabled': self.learning_enabled,
                'safety_history_size': len(self.safety_history)
            })
            return status


def main(args=None):
    rclpy.init(args=args)
    
    # Create node
    node = AdaptiveSafetyNode()
    
    # Create executor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        # Spin the node
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 