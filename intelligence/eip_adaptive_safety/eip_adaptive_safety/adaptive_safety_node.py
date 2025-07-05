#!/usr/bin/env python3
"""
Adaptive Safety Orchestration (ASO) - Main Node

This node coordinates the adaptive learning engine with the existing safety infrastructure,
providing a unified interface for adaptive safety validation.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
import threading
import time
import json
import logging
from typing import Dict, List, Optional

# ROS 2 message imports
from std_msgs.msg import String, Bool, Float32
from eip_interfaces.msg import SafetyViolation, SafetyVerificationRequest, SafetyVerificationResponse
from eip_interfaces.srv import ValidateTaskPlan

# Local imports
from .adaptive_learning_engine import AdaptiveLearningEngine, SafetyRule

class AdaptiveSafetyNode(Node):
    """Main node for Adaptive Safety Orchestration"""
    
    def __init__(self):
        super().__init__('adaptive_safety_node')
        
        # Initialize components
        self.learning_engine = AdaptiveLearningEngine()
        self.safety_rules_cache: Dict[str, SafetyRule] = {}
        self.learning_status = {
            'active': False,
            'experience_count': 0,
            'rule_count': 0,
            'last_update': 0.0
        }
        
        # Setup ROS 2 communication
        self._setup_ros_communication()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.get_logger().info("Adaptive Safety Node initialized")
    
    def _setup_ros_communication(self):
        """Setup ROS 2 publishers and subscribers"""
        
        # QoS for real-time safety communication
        safety_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers for safety events
        self.safety_violation_sub = self.create_subscription(
            SafetyViolation,
            '/safety/violation',
            self._handle_safety_violation,
            safety_qos
        )
        
        self.learning_status_sub = self.create_subscription(
            String,
            '/safety/learning_status',
            self._handle_learning_status,
            safety_qos
        )
        
        self.safety_rules_sub = self.create_subscription(
            String,
            '/safety/adaptive_rules',
            self._handle_safety_rules,
            safety_qos
        )
        
        # Publishers for adaptive safety
        self.adaptive_safety_pub = self.create_publisher(
            String,
            '/safety/adaptive_validation',
            safety_qos
        )
        
        self.safety_metrics_pub = self.create_publisher(
            String,
            '/safety/adaptive_metrics',
            safety_qos
        )
        
        # Services
        self.validate_task_service = self.create_service(
            ValidateTaskPlan,
            '/safety/validate_task_adaptive',
            self._validate_task_adaptive
        )
        
        # Timers for periodic tasks
        self.metrics_timer = self.create_timer(5.0, self._publish_metrics)
        self.health_check_timer = self.create_timer(10.0, self._health_check)
    
    def _start_background_tasks(self):
        """Start background processing tasks"""
        # Start learning engine in separate thread
        self.learning_thread = threading.Thread(
            target=self._run_learning_engine,
            daemon=True
        )
        self.learning_thread.start()
    
    def _run_learning_engine(self):
        """Run learning engine in background"""
        try:
            # Create executor for learning engine
            executor = MultiThreadedExecutor()
            executor.add_node(self.learning_engine)
            
            # Run learning engine
            executor.spin()
            
        except Exception as e:
            self.get_logger().error(f"Error in learning engine thread: {e}")
    
    def _handle_safety_violation(self, msg: SafetyViolation):
        """Handle safety violation events"""
        try:
            self.get_logger().info(f"Safety violation detected: {msg.violation_type}")
            
            # Forward to learning engine
            self.learning_engine._handle_safety_violation(msg)
            
            # Update local metrics
            self.learning_status['last_update'] = time.time()
            
        except Exception as e:
            self.get_logger().error(f"Error handling safety violation: {e}")
    
    def _handle_learning_status(self, msg: String):
        """Handle learning status updates"""
        try:
            status_data = json.loads(msg.data)
            
            # Update local status
            self.learning_status.update({
                'active': status_data.get('learning_active', False),
                'experience_count': status_data.get('experience_count', 0),
                'rule_count': status_data.get('rule_count', 0),
                'last_update': time.time()
            })
            
        except Exception as e:
            self.get_logger().error(f"Error handling learning status: {e}")
    
    def _handle_safety_rules(self, msg: String):
        """Handle safety rules updates"""
        try:
            rules_data = json.loads(msg.data)
            
            # Update local cache
            for rule_info in rules_data.get('rules', []):
                rule = SafetyRule(
                    rule_id=rule_info['id'],
                    condition=rule_info['condition'],
                    threshold=rule_info['threshold'],
                    confidence=rule_info['confidence'],
                    priority=rule_info['priority'],
                    created_at=0.0,  # Will be updated
                    last_updated=time.time(),
                    usage_count=rule_info['usage_count'],
                    success_rate=rule_info['success_rate']
                )
                self.safety_rules_cache[rule.rule_id] = rule
            
            self.get_logger().debug(f"Updated {len(rules_data.get('rules', []))} safety rules")
            
        except Exception as e:
            self.get_logger().error(f"Error handling safety rules: {e}")
    
    def _validate_task_adaptive(self, request: ValidateTaskPlan.Request, 
                               response: ValidateTaskPlan.Response) -> ValidateTaskPlan.Response:
        """Validate task plan using adaptive safety rules"""
        try:
            self.get_logger().info(f"Validating task plan: {request.task_plan[:50]}...")
            
            # Use learning engine for validation
            response = self.learning_engine._validate_task_adaptive(request, response)
            
            # Publish validation result
            self._publish_validation_result(request.task_plan, response)
            
            return response
            
        except Exception as e:
            self.get_logger().error(f"Error validating task: {e}")
            response.is_safe = False
            response.safety_score = 0.0
            response.violations = [f"Validation error: {e}"]
            response.confidence = 0.0
            return response
    
    def _publish_validation_result(self, task_plan: str, response: ValidateTaskPlan.Response):
        """Publish validation result"""
        try:
            validation_data = {
                'timestamp': time.time(),
                'task_plan': task_plan[:100],  # Truncate for logging
                'is_safe': response.is_safe,
                'safety_score': response.safety_score,
                'violations': response.violations,
                'confidence': response.confidence
            }
            
            validation_msg = String()
            validation_msg.data = json.dumps(validation_data)
            self.adaptive_safety_pub.publish(validation_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing validation result: {e}")
    
    def _publish_metrics(self):
        """Publish adaptive safety metrics"""
        try:
            metrics_data = {
                'timestamp': time.time(),
                'learning_active': self.learning_status['active'],
                'experience_count': self.learning_status['experience_count'],
                'rule_count': self.learning_status['rule_count'],
                'cache_size': len(self.safety_rules_cache),
                'average_confidence': self._calculate_average_confidence(),
                'system_health': self._calculate_system_health()
            }
            
            metrics_msg = String()
            metrics_msg.data = json.dumps(metrics_data)
            self.safety_metrics_pub.publish(metrics_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing metrics: {e}")
    
    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence of safety rules"""
        try:
            if not self.safety_rules_cache:
                return 0.0
            
            confidences = [rule.confidence for rule in self.safety_rules_cache.values()]
            return sum(confidences) / len(confidences)
            
        except Exception as e:
            self.get_logger().error(f"Error calculating average confidence: {e}")
            return 0.0
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        try:
            health_score = 1.0
            
            # Check learning engine status
            if not self.learning_status['active']:
                health_score *= 0.5
            
            # Check rule count
            if self.learning_status['rule_count'] < 5:
                health_score *= 0.8
            
            # Check experience count
            if self.learning_status['experience_count'] < 100:
                health_score *= 0.9
            
            # Check last update time
            time_since_update = time.time() - self.learning_status['last_update']
            if time_since_update > 300:  # 5 minutes
                health_score *= 0.7
            
            return max(0.0, min(1.0, health_score))
            
        except Exception as e:
            self.get_logger().error(f"Error calculating system health: {e}")
            return 0.0
    
    def _health_check(self):
        """Periodic health check"""
        try:
            health_score = self._calculate_system_health()
            
            if health_score < 0.5:
                self.get_logger().warn(f"System health degraded: {health_score:.2f}")
            elif health_score < 0.8:
                self.get_logger().info(f"System health moderate: {health_score:.2f}")
            else:
                self.get_logger().debug(f"System health good: {health_score:.2f}")
                
        except Exception as e:
            self.get_logger().error(f"Error in health check: {e}")
    
    def shutdown(self):
        """Clean shutdown"""
        try:
            # Shutdown learning engine
            if hasattr(self, 'learning_engine'):
                self.learning_engine.shutdown()
            
            self.get_logger().info("Adaptive Safety Node shutdown complete")
            
        except Exception as e:
            self.get_logger().error(f"Error during shutdown: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    adaptive_safety_node = AdaptiveSafetyNode()
    
    try:
        rclpy.spin(adaptive_safety_node)
    except KeyboardInterrupt:
        pass
    finally:
        adaptive_safety_node.shutdown()
        adaptive_safety_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 