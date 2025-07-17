#!/usr/bin/env python3
"""
Reasoning Engine Node

This node provides the main interface for the Advanced Multi-Modal Reasoning Engine,
integrating with ROS and coordinating reasoning requests from other system components.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup
import time
import logging
from typing import Dict, List, Optional
import threading
import json

from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Twist
from eip_interfaces.msg import TaskPlan, TaskStep, SafetyViolation
from eip_interfaces.srv import ValidateTaskPlan

from .reasoning_orchestrator import (
    ReasoningOrchestrator, ReasoningMode, ReasoningRequest, ReasoningResponse
)
from .multi_modal_reasoner import (
    VisualContext, SpatialContext, SafetyConstraints
)


class ReasoningEngineNode(Node):
    """
    Main reasoning engine node for the Embodied Intelligence Platform
    
    This node provides advanced multi-modal reasoning capabilities by integrating
    visual perception, natural language understanding, spatial awareness, and
    safety constraints for autonomous robotic systems.
    """
    
    def __init__(self):
        super().__init__('reasoning_engine_node')
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('reasoning_mode', 'balanced'),
                ('max_reasoning_time', 0.5),
                ('enable_visual_reasoning', True),
                ('enable_spatial_reasoning', True),
                ('enable_temporal_reasoning', True),
                ('enable_causal_reasoning', True),
                ('enable_safety_reasoning', True),
                ('reasoning_update_rate', 10.0),
                ('enable_performance_monitoring', True),
                ('log_reasoning_results', True),
                ('collision_threshold', 0.7),
                ('human_proximity_threshold', 0.8),
                ('velocity_limit', 1.0),
                ('workspace_boundary_x', 5.0),
                ('workspace_boundary_y', 5.0),
                ('workspace_boundary_z', 2.0)
            ]
        )
        
        # Initialize reasoning orchestrator
        self.orchestrator = ReasoningOrchestrator()
        
        # Set up QoS profiles
        self.qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.qos_reasoning = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )
        
        # Set up callback groups
        self.sensor_callback_group = ReentrantCallbackGroup()
        self.reasoning_callback_group = ReentrantCallbackGroup()
        self.service_callback_group = ReentrantCallbackGroup()
        
        # Initialize publishers
        self._setup_publishers()
        
        # Initialize subscribers
        self._setup_subscribers()
        
        # Initialize services
        self._setup_services()
        
        # Initialize state tracking
        self._initialize_state_tracking()
        
        # Set up reasoning monitoring timer
        self.reasoning_timer = self.create_timer(
            1.0 / self.get_parameter('reasoning_update_rate').value,
            self._reasoning_monitoring_callback,
            callback_group=self.reasoning_callback_group
        )
        
        # Set up performance monitoring timer
        if self.get_parameter('enable_performance_monitoring').value:
            self.performance_timer = self.create_timer(
                10.0,  # Every 10 seconds
                self._performance_monitoring_callback,
                callback_group=self.reasoning_callback_group
            )
        
        self.logger.info("Reasoning Engine Node initialized successfully")
    
    def _setup_publishers(self):
        """Set up ROS publishers"""
        self.reasoning_result_pub = self.create_publisher(
            String,
            '/eip/reasoning/results',
            10,
            qos_profile=self.qos_reasoning
        )
        
        self.task_plan_pub = self.create_publisher(
            TaskPlan,
            '/eip/reasoning/task_plans',
            10,
            qos_profile=self.qos_reasoning
        )
        
        self.reasoning_confidence_pub = self.create_publisher(
            Float32,
            '/eip/reasoning/confidence',
            10,
            qos_profile=self.qos_reasoning
        )
        
        self.reasoning_safety_pub = self.create_publisher(
            Float32,
            '/eip/reasoning/safety_score',
            10,
            qos_profile=self.qos_reasoning
        )
        
        self.reasoning_status_pub = self.create_publisher(
            String,
            '/eip/reasoning/status',
            10,
            qos_profile=self.qos_reasoning
        )
        
        self.performance_stats_pub = self.create_publisher(
            String,
            '/eip/reasoning/performance_stats',
            10,
            qos_profile=self.qos_reasoning
        )
    
    def _setup_subscribers(self):
        """Set up ROS subscribers"""
        # Visual context subscription
        if self.get_parameter('enable_visual_reasoning').value:
            self.visual_context_sub = self.create_subscription(
                String,
                '/eip/vision/context',
                self._visual_context_callback,
                10,
                callback_group=self.sensor_callback_group,
                qos_profile=self.qos_sensor
            )
        
        # Language command subscription
        self.language_command_sub = self.create_subscription(
            String,
            '/eip/language/commands',
            self._language_command_callback,
            10,
            callback_group=self.sensor_callback_group,
            qos_profile=self.qos_sensor
        )
        
        # Spatial context subscription
        if self.get_parameter('enable_spatial_reasoning').value:
            self.spatial_context_sub = self.create_subscription(
                String,
                '/eip/slam/spatial_context',
                self._spatial_context_callback,
                10,
                callback_group=self.sensor_callback_group,
                qos_profile=self.qos_sensor
            )
        
        # Safety constraints subscription
        if self.get_parameter('enable_safety_reasoning').value:
            self.safety_constraints_sub = self.create_subscription(
                String,
                '/eip/safety/constraints',
                self._safety_constraints_callback,
                10,
                callback_group=self.sensor_callback_group,
                qos_profile=self.qos_sensor
            )
        
        # Robot pose subscription
        self.robot_pose_sub = self.create_subscription(
            Pose,
            '/eip/robot/pose',
            self._robot_pose_callback,
            10,
            callback_group=self.sensor_callback_group,
            qos_profile=self.qos_sensor
        )
    
    def _setup_services(self):
        """Set up ROS services"""
        # Task validation service
        self.task_validation_srv = self.create_service(
            ValidateTaskPlan,
            '/eip/reasoning/validate_task',
            self._task_validation_callback,
            callback_group=self.service_callback_group
        )
    
    def _initialize_state_tracking(self):
        """Initialize state tracking variables"""
        self.current_visual_context = None
        self.current_spatial_context = None
        self.current_safety_constraints = None
        self.current_robot_pose = None
        self.last_language_command = None
        self.last_reasoning_result = None
        
        # Performance tracking
        self.reasoning_requests = 0
        self.reasoning_successes = 0
        self.avg_reasoning_time = 0.0
        
        # State timestamps
        self.last_visual_update = 0.0
        self.last_spatial_update = 0.0
        self.last_safety_update = 0.0
        self.last_pose_update = 0.0
    
    def _visual_context_callback(self, msg: String):
        """Handle visual context updates"""
        try:
            visual_data = json.loads(msg.data)
            
            # Create VisualContext object
            self.current_visual_context = VisualContext(
                objects=visual_data.get('objects', []),
                scene_description=visual_data.get('scene_description', ''),
                spatial_relationships=visual_data.get('spatial_relationships', {}),
                affordances=visual_data.get('affordances', {}),
                confidence=visual_data.get('confidence', 0.5)
            )
            
            self.last_visual_update = time.time()
            self.logger.debug("Visual context updated")
            
        except Exception as e:
            self.logger.error(f"Error processing visual context: {e}")
    
    def _language_command_callback(self, msg: String):
        """Handle language command updates"""
        try:
            command_data = json.loads(msg.data)
            command = command_data.get('command', '')
            
            if command and command != self.last_language_command:
                self.last_language_command = command
                self.logger.info(f"Received language command: {command}")
                
                # Trigger reasoning if we have sufficient context
                self._trigger_reasoning_if_ready()
            
        except Exception as e:
            self.logger.error(f"Error processing language command: {e}")
    
    def _spatial_context_callback(self, msg: String):
        """Handle spatial context updates"""
        try:
            spatial_data = json.loads(msg.data)
            
            # Create SpatialContext object
            self.current_spatial_context = SpatialContext(
                robot_pose=spatial_data.get('robot_pose', {}),
                object_positions=spatial_data.get('object_positions', {}),
                workspace_boundaries=spatial_data.get('workspace_boundaries', {}),
                navigation_graph=spatial_data.get('navigation_graph', {}),
                occupancy_grid=None  # Would be populated from actual grid data
            )
            
            self.last_spatial_update = time.time()
            self.logger.debug("Spatial context updated")
            
        except Exception as e:
            self.logger.error(f"Error processing spatial context: {e}")
    
    def _safety_constraints_callback(self, msg: String):
        """Handle safety constraints updates"""
        try:
            safety_data = json.loads(msg.data)
            
            # Create SafetyConstraints object
            self.current_safety_constraints = SafetyConstraints(
                collision_threshold=safety_data.get('collision_threshold', 0.7),
                human_proximity_threshold=safety_data.get('human_proximity_threshold', 0.8),
                velocity_limits=safety_data.get('velocity_limits', {}),
                workspace_boundaries=safety_data.get('workspace_boundaries', {}),
                emergency_stop_conditions=safety_data.get('emergency_stop_conditions', [])
            )
            
            self.last_safety_update = time.time()
            self.logger.debug("Safety constraints updated")
            
        except Exception as e:
            self.logger.error(f"Error processing safety constraints: {e}")
    
    def _robot_pose_callback(self, msg: Pose):
        """Handle robot pose updates"""
        try:
            # Convert Pose message to dictionary
            pose_dict = {
                'x': msg.position.x,
                'y': msg.position.y,
                'z': msg.position.z,
                'orientation_x': msg.orientation.x,
                'orientation_y': msg.orientation.y,
                'orientation_z': msg.orientation.z,
                'orientation_w': msg.orientation.w
            }
            
            # Update spatial context if available
            if self.current_spatial_context:
                self.current_spatial_context.robot_pose = pose_dict
            
            self.current_robot_pose = pose_dict
            self.last_pose_update = time.time()
            
        except Exception as e:
            self.logger.error(f"Error processing robot pose: {e}")
    
    def _trigger_reasoning_if_ready(self):
        """Trigger reasoning if sufficient context is available"""
        current_time = time.time()
        context_timeout = 2.0  # seconds
        
        # Check if we have recent context data
        has_recent_visual = (current_time - self.last_visual_update) < context_timeout
        has_recent_spatial = (current_time - self.last_spatial_update) < context_timeout
        has_recent_safety = (current_time - self.last_safety_update) < context_timeout
        has_recent_pose = (current_time - self.last_pose_update) < context_timeout
        
        # Check if we have all required context
        if (has_recent_visual and has_recent_spatial and 
            has_recent_safety and has_recent_pose and 
            self.last_language_command):
            
            # Determine reasoning mode based on command urgency
            mode = self._determine_reasoning_mode(self.last_language_command)
            
            # Perform reasoning
            self._perform_reasoning(mode)
    
    def _determine_reasoning_mode(self, command: str) -> ReasoningMode:
        """Determine appropriate reasoning mode based on command"""
        command_lower = command.lower()
        
        # Safety-critical commands
        if any(word in command_lower for word in ['stop', 'emergency', 'danger', 'halt']):
            return ReasoningMode.SAFETY_CRITICAL
        
        # Complex commands requiring thorough analysis
        if any(word in command_lower for word in ['complex', 'careful', 'precise', 'detailed']):
            return ReasoningMode.THOROUGH
        
        # Simple commands for fast response
        if any(word in command_lower for word in ['quick', 'fast', 'immediate', 'now']):
            return ReasoningMode.FAST
        
        # Default to balanced mode
        return ReasoningMode.BALANCED
    
    def _perform_reasoning(self, mode: ReasoningMode):
        """Perform reasoning with the specified mode"""
        try:
            self.reasoning_requests += 1
            
            # Create reasoning request
            response = self.orchestrator.reason_about_scene(
                visual_context=self.current_visual_context,
                language_command=self.last_language_command,
                spatial_context=self.current_spatial_context,
                safety_constraints=self.current_safety_constraints,
                mode=mode,
                priority=5,
                timeout=self.get_parameter('max_reasoning_time').value
            )
            
            if response.success:
                self.reasoning_successes += 1
                self.last_reasoning_result = response.result
                
                # Publish results
                self._publish_reasoning_results(response)
                
                # Log results if enabled
                if self.get_parameter('log_reasoning_results').value:
                    self._log_reasoning_results(response)
                
                self.logger.info(f"Reasoning completed successfully in {response.execution_time:.3f}s")
            else:
                self.logger.warning(f"Reasoning failed: {response.error_message}")
            
        except Exception as e:
            self.logger.error(f"Error performing reasoning: {e}")
    
    def _publish_reasoning_results(self, response: ReasoningResponse):
        """Publish reasoning results to ROS topics"""
        try:
            # Publish reasoning result summary
            result_summary = {
                'mode': response.mode_used.value,
                'execution_time': response.execution_time,
                'success': response.success,
                'component_times': response.component_times,
                'reasoning_steps': response.result.reasoning_steps
            }
            
            result_msg = String()
            result_msg.data = json.dumps(result_summary)
            self.reasoning_result_pub.publish(result_msg)
            
            # Publish task plan
            self.task_plan_pub.publish(response.result.plan)
            
            # Publish confidence
            confidence_msg = Float32()
            confidence_msg.data = response.result.confidence
            self.reasoning_confidence_pub.publish(confidence_msg)
            
            # Publish safety score
            safety_msg = Float32()
            safety_msg.data = response.result.safety_score
            self.reasoning_safety_pub.publish(safety_msg)
            
            # Publish status
            status_msg = String()
            status_data = {
                'status': 'active' if response.success else 'error',
                'mode': response.mode_used.value,
                'last_command': self.last_language_command
            }
            status_msg.data = json.dumps(status_data)
            self.reasoning_status_pub.publish(status_msg)
            
        except Exception as e:
            self.logger.error(f"Error publishing reasoning results: {e}")
    
    def _log_reasoning_results(self, response: ReasoningResponse):
        """Log detailed reasoning results"""
        log_data = {
            'timestamp': time.time(),
            'command': self.last_language_command,
            'mode': response.mode_used.value,
            'execution_time': response.execution_time,
            'confidence': response.result.confidence,
            'safety_score': response.result.safety_score,
            'reasoning_steps': response.result.reasoning_steps,
            'component_times': response.component_times
        }
        
        self.logger.info(f"Reasoning log: {json.dumps(log_data, indent=2)}")
    
    def _task_validation_callback(self, request: ValidateTaskPlan, response) -> ValidateTaskPlan.Response:
        """Handle task validation service requests"""
        try:
            # Use reasoning orchestrator to validate task
            validation_result = self.orchestrator.reason_about_scene(
                visual_context=self.current_visual_context or self._create_default_visual_context(),
                language_command=request.task_plan.goal_description,
                spatial_context=self.current_spatial_context or self._create_default_spatial_context(),
                safety_constraints=self.current_safety_constraints or self._create_default_safety_constraints(),
                mode=ReasoningMode.SAFETY_CRITICAL,
                priority=8,
                timeout=1.0
            )
            
            # Set response based on reasoning result
            response.is_valid = validation_result.success and validation_result.result.safety_score > 0.7
            response.confidence = validation_result.result.confidence
            response.safety_score = validation_result.result.safety_score
            response.reasoning_steps = validation_result.result.reasoning_steps
            
            self.logger.info(f"Task validation completed: valid={response.is_valid}, safety={response.safety_score}")
            
        except Exception as e:
            self.logger.error(f"Error in task validation: {e}")
            response.is_valid = False
            response.confidence = 0.0
            response.safety_score = 0.0
            response.reasoning_steps = [f"Validation error: {str(e)}"]
        
        return response
    
    def _create_default_visual_context(self) -> VisualContext:
        """Create default visual context when none is available"""
        return VisualContext(
            objects=[],
            scene_description="Default scene",
            spatial_relationships={},
            affordances={},
            confidence=0.1
        )
    
    def _create_default_spatial_context(self) -> SpatialContext:
        """Create default spatial context when none is available"""
        return SpatialContext(
            robot_pose={'x': 0.0, 'y': 0.0, 'z': 0.0},
            object_positions={},
            workspace_boundaries={
                'min_x': -self.get_parameter('workspace_boundary_x').value,
                'max_x': self.get_parameter('workspace_boundary_x').value,
                'min_y': -self.get_parameter('workspace_boundary_y').value,
                'max_y': self.get_parameter('workspace_boundary_y').value,
                'min_z': 0.0,
                'max_z': self.get_parameter('workspace_boundary_z').value
            },
            navigation_graph={},
            occupancy_grid=None
        )
    
    def _create_default_safety_constraints(self) -> SafetyConstraints:
        """Create default safety constraints when none are available"""
        return SafetyConstraints(
            collision_threshold=self.get_parameter('collision_threshold').value,
            human_proximity_threshold=self.get_parameter('human_proximity_threshold').value,
            velocity_limits={'linear': self.get_parameter('velocity_limit').value, 'angular': 1.0},
            workspace_boundaries={
                'min_x': -self.get_parameter('workspace_boundary_x').value,
                'max_x': self.get_parameter('workspace_boundary_x').value,
                'min_y': -self.get_parameter('workspace_boundary_y').value,
                'max_y': self.get_parameter('workspace_boundary_y').value
            },
            emergency_stop_conditions=['collision_detected', 'human_proximity', 'boundary_violation']
        )
    
    def _reasoning_monitoring_callback(self):
        """Monitor reasoning system status"""
        try:
            current_time = time.time()
            
            # Check system health
            is_healthy = True
            health_issues = []
            
            # Check context freshness
            if (current_time - self.last_visual_update) > 5.0:
                is_healthy = False
                health_issues.append("Stale visual context")
            
            if (current_time - self.last_spatial_update) > 5.0:
                is_healthy = False
                health_issues.append("Stale spatial context")
            
            # Check reasoning performance
            if self.reasoning_requests > 0:
                success_rate = self.reasoning_successes / self.reasoning_requests
                if success_rate < 0.8:
                    is_healthy = False
                    health_issues.append(f"Low success rate: {success_rate:.2f}")
            
            # Publish health status
            health_msg = String()
            health_data = {
                'healthy': is_healthy,
                'issues': health_issues,
                'requests': self.reasoning_requests,
                'successes': self.reasoning_successes,
                'success_rate': self.reasoning_successes / self.reasoning_requests if self.reasoning_requests > 0 else 0.0
            }
            health_msg.data = json.dumps(health_data)
            self.reasoning_status_pub.publish(health_msg)
            
        except Exception as e:
            self.logger.error(f"Error in reasoning monitoring: {e}")
    
    def _performance_monitoring_callback(self):
        """Monitor and publish performance statistics"""
        try:
            # Get performance stats from orchestrator
            stats = self.orchestrator.get_performance_stats()
            
            # Publish performance stats
            stats_msg = String()
            stats_msg.data = json.dumps(stats, indent=2)
            self.performance_stats_pub.publish(stats_msg)
            
            self.logger.info(f"Performance stats: {json.dumps(stats, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Error in performance monitoring: {e}")
    
    def on_shutdown(self):
        """Cleanup on shutdown"""
        try:
            self.orchestrator.shutdown()
            self.logger.info("Reasoning Engine Node shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        node = ReasoningEngineNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Error in reasoning engine node: {e}")
    finally:
        if 'node' in locals():
            node.on_shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 