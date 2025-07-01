#!/usr/bin/env python3
"""
Safety Monitor Node - Implementation of SAFER Framework

This module implements the Safety-Aware Framework for Execution in Robotics (SAFER)
for LLM-guided robotic systems. It provides multi-LLM safety verification and
real-time behavior arbitration.

Key Features:
- Multi-LLM safety verification
- Real-time behavior constraint enforcement  
- LLM-as-a-Judge safety evaluation
- Emergency stop capabilities
- Safety violation logging and analysis
"""

import json
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import OccupancyGrid

from eip_interfaces.msg import (
    SafetyVerificationRequest,
    SafetyVerificationResponse, 
    TaskPlan,
    SafetyViolation,
    EmergencyStop
)
from eip_interfaces.srv import ValidateTaskPlan


class SafetyLevel(Enum):
    """Safety criticality levels"""
    SAFE = 0
    LOW_RISK = 1
    MEDIUM_RISK = 2
    HIGH_RISK = 3
    CRITICAL = 4


class ViolationType(Enum):
    """Types of safety violations"""
    COLLISION_RISK = "collision_risk"
    HUMAN_PROXIMITY = "human_proximity"
    WORKSPACE_BOUNDARY = "workspace_boundary"
    INVALID_MANIPULATION = "invalid_manipulation"
    SOCIAL_NORM_VIOLATION = "social_norm_violation"
    LLM_HALLUCINATION = "llm_hallucination"


@dataclass
class SafetyConstraint:
    """Defines a safety constraint with validation rules"""
    name: str
    description: str
    max_violation_threshold: float
    constraint_type: str
    validation_function: callable
    recovery_action: Optional[str] = None


@dataclass
class SafetyEvaluation:
    """Result of safety evaluation"""
    is_safe: bool
    safety_level: SafetyLevel
    violations: List[ViolationType]
    confidence_score: float
    explanation: str
    suggested_modifications: Optional[List[str]] = None


class SafetyMonitor(Node):
    """
    Core safety monitoring and arbitration node.
    
    Implements the SAFER framework with multi-LLM verification,
    real-time constraint checking, and emergency intervention.
    """
    
    def __init__(self):
        super().__init__('safety_monitor')
        
        # Configuration
        self.declare_parameters()
        self.load_configuration()
        
        # State management
        self.robot_state = {}
        self.environment_state = {}
        self.safety_constraints = self.initialize_safety_constraints()
        self.violation_history = []
        self.emergency_stop_active = False
        
        # Thread safety
        self.state_lock = threading.Lock()
        self.constraint_lock = threading.Lock()
        
        # ROS 2 interfaces
        self.setup_publishers()
        self.setup_subscribers()
        self.setup_services()
        
        # Safety evaluation thread
        self.safety_evaluation_thread = threading.Thread(
            target=self.continuous_safety_evaluation,
            daemon=True
        )
        self.safety_evaluation_thread.start()
        
        self.get_logger().info("Safety Monitor initialized and monitoring...")

    def declare_parameters(self):
        """Declare ROS 2 parameters with defaults"""
        self.declare_parameter('safety_check_frequency', 10.0)  # Hz
        self.declare_parameter('collision_distance_threshold', 0.5)  # meters
        self.declare_parameter('human_proximity_threshold', 1.0)  # meters
        self.declare_parameter('max_linear_velocity', 1.0)  # m/s
        self.declare_parameter('max_angular_velocity', 1.0)  # rad/s
        self.declare_parameter('enable_llm_safety_check', True)
        self.declare_parameter('safety_llm_model', 'gpt-4')
        self.declare_parameter('safety_confidence_threshold', 0.8)

    def load_configuration(self):
        """Load safety configuration from parameters"""
        self.safety_check_freq = self.get_parameter('safety_check_frequency').value
        self.collision_threshold = self.get_parameter('collision_distance_threshold').value
        self.human_proximity_threshold = self.get_parameter('human_proximity_threshold').value
        self.max_linear_vel = self.get_parameter('max_linear_velocity').value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').value
        self.enable_llm_safety = self.get_parameter('enable_llm_safety_check').value
        self.safety_llm_model = self.get_parameter('safety_llm_model').value
        self.safety_confidence_threshold = self.get_parameter('safety_confidence_threshold').value

    def setup_publishers(self):
        """Setup ROS 2 publishers"""
        # QoS for safety-critical messages
        safety_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        self.safety_status_pub = self.create_publisher(
            SafetyVerificationResponse, 
            '/safety/status', 
            safety_qos
        )
        
        self.emergency_stop_pub = self.create_publisher(
            EmergencyStop,
            '/safety/emergency_stop',
            safety_qos
        )
        
        self.safety_violation_pub = self.create_publisher(
            SafetyViolation,
            '/safety/violations',
            safety_qos
        )
        
        self.cmd_vel_override_pub = self.create_publisher(
            Twist,
            '/cmd_vel_safe',
            10
        )

    def setup_subscribers(self):
        """Setup ROS 2 subscribers"""
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel_raw',
            self.cmd_vel_callback,
            10
        )
        
        self.laser_scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_scan_callback,
            10
        )
        
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/robot_pose',
            self.pose_callback,
            10
        )
        
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        self.task_plan_sub = self.create_subscription(
            TaskPlan,
            '/llm/task_plan',
            self.task_plan_callback,
            10
        )

    def setup_services(self):
        """Setup ROS 2 services"""
        self.validate_plan_service = self.create_service(
            ValidateTaskPlan,
            '/safety/validate_task_plan',
            self.validate_task_plan_callback
        )

    def initialize_safety_constraints(self) -> List[SafetyConstraint]:
        """Initialize safety constraints with validation functions"""
        constraints = [
            SafetyConstraint(
                name="collision_avoidance",
                description="Prevent collisions with obstacles",
                max_violation_threshold=self.collision_threshold,
                constraint_type="spatial",
                validation_function=self.check_collision_risk
            ),
            SafetyConstraint(
                name="human_proximity",
                description="Maintain safe distance from humans",
                max_violation_threshold=self.human_proximity_threshold,
                constraint_type="social",
                validation_function=self.check_human_proximity
            ),
            SafetyConstraint(
                name="velocity_limits",
                description="Enforce velocity safety limits",
                max_violation_threshold=1.0,
                constraint_type="dynamic",
                validation_function=self.check_velocity_limits
            ),
            SafetyConstraint(
                name="workspace_boundary",
                description="Stay within designated workspace",
                max_violation_threshold=0.0,
                constraint_type="spatial",
                validation_function=self.check_workspace_boundary
            )
        ]
        
        return constraints

    def continuous_safety_evaluation(self):
        """Continuous safety monitoring loop"""
        rate = 1.0 / self.safety_check_freq
        
        while rclpy.ok():
            try:
                # Perform comprehensive safety check
                safety_eval = self.evaluate_current_safety()
                
                # Publish safety status
                self.publish_safety_status(safety_eval)
                
                # Handle critical violations
                if safety_eval.safety_level == SafetyLevel.CRITICAL:
                    self.trigger_emergency_stop(safety_eval)
                
                # Log violations for learning
                if safety_eval.violations:
                    self.log_safety_violations(safety_eval)
                
                time.sleep(rate)
                
            except Exception as e:
                self.get_logger().error(f"Safety evaluation error: {e}")
                # Fail-safe: trigger emergency stop on evaluation failure
                self.trigger_emergency_stop_immediate("Safety evaluation failure")

    def evaluate_current_safety(self) -> SafetyEvaluation:
        """
        Perform comprehensive safety evaluation
        
        Returns:
            SafetyEvaluation containing safety assessment
        """
        violations = []
        max_safety_level = SafetyLevel.SAFE
        explanations = []
        
        with self.constraint_lock:
            # Check each safety constraint
            for constraint in self.safety_constraints:
                try:
                    violation_result = constraint.validation_function()
                    
                    if violation_result['is_violation']:
                        violation_type = ViolationType(violation_result['type'])
                        violations.append(violation_type)
                        
                        # Update max safety level
                        violation_level = violation_result.get('safety_level', SafetyLevel.MEDIUM_RISK)
                        if violation_level.value > max_safety_level.value:
                            max_safety_level = violation_level
                        
                        explanations.append(violation_result.get('explanation', ''))
                        
                except Exception as e:
                    self.get_logger().warn(f"Constraint check failed for {constraint.name}: {e}")
        
        # Calculate overall confidence
        confidence_score = self.calculate_safety_confidence(violations, max_safety_level)
        
        return SafetyEvaluation(
            is_safe=(max_safety_level.value <= SafetyLevel.LOW_RISK.value),
            safety_level=max_safety_level,
            violations=violations,
            confidence_score=confidence_score,
            explanation="; ".join(explanations),
            suggested_modifications=self.generate_safety_modifications(violations)
        )

    def check_collision_risk(self) -> Dict[str, Any]:
        """Check for collision risks using laser scan data"""
        with self.state_lock:
            if 'laser_scan' not in self.robot_state:
                return {'is_violation': False}
            
            laser_data = self.robot_state['laser_scan']
            min_distance = min([r for r in laser_data.ranges if r > 0])
            
            if min_distance < self.collision_threshold:
                return {
                    'is_violation': True,
                    'type': ViolationType.COLLISION_RISK.value,
                    'safety_level': SafetyLevel.HIGH_RISK if min_distance < 0.2 else SafetyLevel.MEDIUM_RISK,
                    'explanation': f"Obstacle detected at {min_distance:.2f}m (threshold: {self.collision_threshold}m)",
                    'min_distance': min_distance
                }
            
            return {'is_violation': False}

    def check_human_proximity(self) -> Dict[str, Any]:
        """Check for human proximity violations"""
        # Placeholder for human detection integration
        # In real implementation, this would use computer vision
        # or dedicated human detection sensors
        
        with self.state_lock:
            # Simulated human detection logic
            if 'detected_humans' in self.robot_state:
                humans = self.robot_state['detected_humans']
                
                for human in humans:
                    distance = human.get('distance', float('inf'))
                    if distance < self.human_proximity_threshold:
                        return {
                            'is_violation': True,
                            'type': ViolationType.HUMAN_PROXIMITY.value,
                            'safety_level': SafetyLevel.HIGH_RISK,
                            'explanation': f"Human detected at {distance:.2f}m (threshold: {self.human_proximity_threshold}m)",
                            'human_distance': distance
                        }
            
            return {'is_violation': False}

    def check_velocity_limits(self) -> Dict[str, Any]:
        """Check velocity safety limits"""
        with self.state_lock:
            if 'cmd_vel' not in self.robot_state:
                return {'is_violation': False}
            
            cmd_vel = self.robot_state['cmd_vel']
            linear_vel = abs(cmd_vel.linear.x)
            angular_vel = abs(cmd_vel.angular.z)
            
            if linear_vel > self.max_linear_vel or angular_vel > self.max_angular_vel:
                return {
                    'is_violation': True,
                    'type': ViolationType.INVALID_MANIPULATION.value,
                    'safety_level': SafetyLevel.MEDIUM_RISK,
                    'explanation': f"Velocity limits exceeded: linear={linear_vel:.2f} (max={self.max_linear_vel}), angular={angular_vel:.2f} (max={self.max_angular_vel})",
                    'linear_velocity': linear_vel,
                    'angular_velocity': angular_vel
                }
            
            return {'is_violation': False}

    def check_workspace_boundary(self) -> Dict[str, Any]:
        """Check if robot is within designated workspace"""
        # Placeholder for workspace boundary checking
        # Implementation would depend on workspace definition
        return {'is_violation': False}

    def calculate_safety_confidence(self, violations: List[ViolationType], safety_level: SafetyLevel) -> float:
        """Calculate confidence score for safety evaluation"""
        base_confidence = 1.0
        
        # Reduce confidence based on number of violations
        violation_penalty = len(violations) * 0.1
        
        # Reduce confidence based on safety level
        level_penalty = safety_level.value * 0.15
        
        confidence = max(0.0, base_confidence - violation_penalty - level_penalty)
        return confidence

    def generate_safety_modifications(self, violations: List[ViolationType]) -> List[str]:
        """Generate suggested modifications to address safety violations"""
        modifications = []
        
        for violation in violations:
            if violation == ViolationType.COLLISION_RISK:
                modifications.append("Reduce velocity and plan alternative path")
            elif violation == ViolationType.HUMAN_PROXIMITY:
                modifications.append("Wait for human to move or request permission to proceed")
            elif violation == ViolationType.INVALID_MANIPULATION:
                modifications.append("Reduce velocity to within safety limits")
            elif violation == ViolationType.WORKSPACE_BOUNDARY:
                modifications.append("Return to designated workspace area")
        
        return modifications

    def trigger_emergency_stop(self, safety_eval: SafetyEvaluation):
        """Trigger emergency stop due to safety violation"""
        if self.emergency_stop_active:
            return  # Already in emergency stop
        
        self.emergency_stop_active = True
        
        # Publish emergency stop message
        emergency_msg = EmergencyStop()
        emergency_msg.timestamp = self.get_clock().now().to_msg()
        emergency_msg.reason = safety_eval.explanation
        emergency_msg.safety_level = safety_eval.safety_level.value
        emergency_msg.violations = [v.value for v in safety_eval.violations]
        
        self.emergency_stop_pub.publish(emergency_msg)
        
        # Stop robot immediately
        stop_cmd = Twist()
        self.cmd_vel_override_pub.publish(stop_cmd)
        
        self.get_logger().warn(f"EMERGENCY STOP triggered: {safety_eval.explanation}")

    def trigger_emergency_stop_immediate(self, reason: str):
        """Immediate emergency stop without full evaluation"""
        self.emergency_stop_active = True
        
        emergency_msg = EmergencyStop()
        emergency_msg.timestamp = self.get_clock().now().to_msg()
        emergency_msg.reason = reason
        emergency_msg.safety_level = SafetyLevel.CRITICAL.value
        
        self.emergency_stop_pub.publish(emergency_msg)
        
        stop_cmd = Twist()
        self.cmd_vel_override_pub.publish(stop_cmd)
        
        self.get_logger().error(f"IMMEDIATE EMERGENCY STOP: {reason}")

    def publish_safety_status(self, safety_eval: SafetyEvaluation):
        """Publish current safety status"""
        status_msg = SafetyVerificationResponse()
        status_msg.timestamp = self.get_clock().now().to_msg()
        status_msg.is_safe = safety_eval.is_safe
        status_msg.safety_level = safety_eval.safety_level.value
        status_msg.confidence_score = safety_eval.confidence_score
        status_msg.explanation = safety_eval.explanation
        status_msg.violations = [v.value for v in safety_eval.violations]
        
        if safety_eval.suggested_modifications:
            status_msg.suggested_modifications = safety_eval.suggested_modifications
        
        self.safety_status_pub.publish(status_msg)

    def log_safety_violations(self, safety_eval: SafetyEvaluation):
        """Log safety violations for analysis and learning"""
        violation_entry = {
            'timestamp': time.time(),
            'violations': [v.value for v in safety_eval.violations],
            'safety_level': safety_eval.safety_level.value,
            'confidence': safety_eval.confidence_score,
            'explanation': safety_eval.explanation,
            'robot_state': dict(self.robot_state),  # Copy current state
            'environment_state': dict(self.environment_state)
        }
        
        self.violation_history.append(violation_entry)
        
        # Publish violation for external logging/learning systems
        violation_msg = SafetyViolation()
        violation_msg.timestamp = self.get_clock().now().to_msg()
        violation_msg.violation_types = [v.value for v in safety_eval.violations]
        violation_msg.safety_level = safety_eval.safety_level.value
        violation_msg.explanation = safety_eval.explanation
        
        self.safety_violation_pub.publish(violation_msg)

    # ROS 2 Callback Methods
    
    def cmd_vel_callback(self, msg: Twist):
        """Process velocity commands with safety filtering"""
        with self.state_lock:
            self.robot_state['cmd_vel'] = msg
        
        # Apply velocity limits
        filtered_cmd = self.filter_velocity_command(msg)
        
        # Publish filtered command if not in emergency stop
        if not self.emergency_stop_active:
            self.cmd_vel_override_pub.publish(filtered_cmd)

    def filter_velocity_command(self, cmd_vel: Twist) -> Twist:
        """Apply safety filters to velocity commands"""
        filtered_cmd = Twist()
        
        # Clamp linear velocity
        filtered_cmd.linear.x = max(-self.max_linear_vel, 
                                  min(self.max_linear_vel, cmd_vel.linear.x))
        
        # Clamp angular velocity  
        filtered_cmd.angular.z = max(-self.max_angular_vel,
                                   min(self.max_angular_vel, cmd_vel.angular.z))
        
        return filtered_cmd

    def laser_scan_callback(self, msg: LaserScan):
        """Process laser scan data for collision detection"""
        with self.state_lock:
            self.robot_state['laser_scan'] = msg

    def pose_callback(self, msg: PoseStamped):
        """Process robot pose updates"""
        with self.state_lock:
            self.robot_state['pose'] = msg

    def map_callback(self, msg: OccupancyGrid):
        """Process map updates"""
        with self.state_lock:
            self.environment_state['map'] = msg

    def task_plan_callback(self, msg: TaskPlan):
        """Process incoming task plans for safety validation"""
        # Validate task plan safety in background
        threading.Thread(
            target=self.validate_task_plan_async,
            args=(msg,),
            daemon=True
        ).start()

    def validate_task_plan_async(self, task_plan: TaskPlan):
        """Asynchronously validate task plan safety"""
        try:
            # Simulate LLM safety evaluation
            safety_evaluation = self.evaluate_task_plan_safety(task_plan)
            
            if not safety_evaluation.is_safe:
                self.get_logger().warn(f"Task plan safety violation detected: {safety_evaluation.explanation}")
                
                # Publish safety concern
                violation_msg = SafetyViolation()
                violation_msg.timestamp = self.get_clock().now().to_msg()
                violation_msg.violation_types = [v.value for v in safety_evaluation.violations]
                violation_msg.explanation = f"Task plan validation: {safety_evaluation.explanation}"
                
                self.safety_violation_pub.publish(violation_msg)
                
        except Exception as e:
            self.get_logger().error(f"Task plan validation error: {e}")

    def evaluate_task_plan_safety(self, task_plan: TaskPlan) -> SafetyEvaluation:
        """Evaluate the safety of a proposed task plan"""
        # Placeholder for LLM-based task plan safety evaluation
        # This would implement the SAFER framework's LLM-as-a-Judge approach
        
        violations = []
        safety_level = SafetyLevel.SAFE
        explanation = "Task plan appears safe"
        
        # Example safety checks on task plan
        # In real implementation, this would use LLM evaluation
        
        return SafetyEvaluation(
            is_safe=True,
            safety_level=safety_level,
            violations=violations,
            confidence_score=0.9,
            explanation=explanation
        )

    def validate_task_plan_callback(self, request, response):
        """Service callback for task plan validation"""
        try:
            task_plan = request.task_plan
            safety_eval = self.evaluate_task_plan_safety(task_plan)
            
            response.is_safe = safety_eval.is_safe
            response.safety_level = safety_eval.safety_level.value
            response.confidence_score = safety_eval.confidence_score
            response.explanation = safety_eval.explanation
            response.violations = [v.value for v in safety_eval.violations]
            
            if safety_eval.suggested_modifications:
                response.suggested_modifications = safety_eval.suggested_modifications
            
        except Exception as e:
            self.get_logger().error(f"Task plan validation service error: {e}")
            response.is_safe = False
            response.explanation = f"Validation error: {e}"
        
        return response


def main(args=None):
    """Main entry point for safety monitor node"""
    rclpy.init(args=args)
    
    try:
        safety_monitor = SafetyMonitor()
        rclpy.spin(safety_monitor)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Safety monitor error: {e}")
    finally:
        if 'safety_monitor' in locals():
            safety_monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 