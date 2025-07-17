"""
Cognitive Architecture Node for Embodied Intelligence Platform

This module implements the main cognitive architecture node that orchestrates
all AI components (perception, reasoning, planning, execution) to create a
unified intelligent system capable of complex autonomous behavior while
maintaining safety and social awareness.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import time
import threading
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np

# Import cognitive components
from .attention_mechanism import AttentionMechanism, MultiModalSensorData, AttentionFocus
from .working_memory import WorkingMemory, MemoryType
from .long_term_memory import LongTermMemory, MemoryCategory
from .executive_control import ExecutiveControl, ExecutiveDecision, TaskPriority
from .learning_engine import LearningEngine, LearningEvent
from .social_intelligence import SocialIntelligence, SocialCue, SocialBehavior, SocialAdjustment

# Import ROS messages
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from nav_msgs.msg import OccupancyGrid, Path
from eip_interfaces.msg import TaskPlan, TaskStep, SafetyViolation
from eip_interfaces.srv import ValidateTaskPlan


@dataclass
class CognitiveResponse:
    """Represents the response from cognitive architecture"""
    planned_actions: List[Dict[str, Any]]
    reasoning: str
    confidence: float  # 0.0 to 1.0
    social_context: Dict[str, Any]
    safety_status: str
    attention_summary: Dict[str, Any]
    memory_summary: Dict[str, Any]
    learning_progress: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class Context:
    """Represents current context for cognitive processing"""
    task_context: Optional[Dict[str, Any]] = None
    environmental_context: Optional[Dict[str, Any]] = None
    social_context: Optional[Dict[str, Any]] = None
    safety_context: Optional[Dict[str, Any]] = None
    temporal_context: Optional[Dict[str, Any]] = None
    spatial_context: Optional[Dict[str, Any]] = None


class CognitiveArchitectureNode(Node):
    """
    Main cognitive architecture node that orchestrates all AI components.
    
    This node provides:
    - Unified interface for cognitive processing
    - Integration of all AI components
    - Real-time cognitive decision making
    - Safety and social awareness
    - Learning and adaptation
    """
    
    def __init__(self):
        """Initialize the cognitive architecture node."""
        super().__init__('cognitive_architecture_node')
        
        # Initialize cognitive components
        self.attention = AttentionMechanism()
        self.working_memory = WorkingMemory()
        self.long_term_memory = LongTermMemory()
        self.executive_control = ExecutiveControl()
        self.learning_engine = LearningEngine()
        self.social_intelligence = SocialIntelligence()
        
        # Performance monitoring
        self.performance_metrics = {
            'response_time': 0.0,
            'decision_confidence': 0.0,
            'safety_violations': 0,
            'learning_rate': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Setup ROS interfaces
        self._setup_ros_interfaces()
        
        # Setup processing timer
        self.processing_timer = self.create_timer(0.1, self._cognitive_processing_loop)  # 10 Hz
        
        # Setup performance monitoring timer
        self.monitoring_timer = self.create_timer(1.0, self._performance_monitoring_loop)  # 1 Hz
        
        self.get_logger().info("Cognitive Architecture Node initialized successfully")
    
    def _setup_ros_interfaces(self):
        """Setup ROS publishers and subscribers."""
        # QoS profile for real-time communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers for sensor data
        self.sensor_subscribers = {
            'camera': self.create_subscription(
                Image, '/camera/image_raw', self._camera_callback, qos_profile
            ),
            'laser': self.create_subscription(
                LaserScan, '/laser/scan', self._laser_callback, qos_profile
            ),
            'pointcloud': self.create_subscription(
                PointCloud2, '/camera/points', self._pointcloud_callback, qos_profile
            ),
            'audio': self.create_subscription(
                Float32MultiArray, '/audio/features', self._audio_callback, qos_profile
            )
        }
        
        # Subscribers for task and safety information
        self.task_subscriber = self.create_subscription(
            TaskPlan, '/task/plan', self._task_callback, qos_profile
        )
        
        self.safety_subscriber = self.create_subscription(
            SafetyViolation, '/safety/violation', self._safety_callback, qos_profile
        )
        
        # Publishers for cognitive output
        self.cognitive_publisher = self.create_publisher(
            String, '/cognitive/response', qos_profile
        )
        
        self.action_publisher = self.create_publisher(
            Twist, '/cmd_vel', qos_profile
        )
        
        self.status_publisher = self.create_publisher(
            String, '/cognitive/status', qos_profile
        )
        
        # Service for task validation
        self.task_validation_service = self.create_service(
            ValidateTaskPlan, '/cognitive/validate_task', self._validate_task_callback
        )
    
    def process_input(self, 
                     sensor_data: MultiModalSensorData,
                     user_input: Optional[str] = None,
                     current_context: Context = None) -> CognitiveResponse:
        """
        Process multi-modal input through cognitive architecture.
        
        Args:
            sensor_data: Current sensor readings
            user_input: Optional user command or interaction
            current_context: Current task and environmental context
            
        Returns:
            CognitiveResponse with planned actions and reasoning
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # 1. Attention mechanism focuses on relevant stimuli
                focused_attention = self.attention.focus_attention(
                    sensor_data, user_input, 
                    current_context.task_context if current_context else None
                )
                
                # 2. Working memory updates with current context
                self.working_memory.update_context(focused_attention, 
                    current_context.task_context if current_context else None)
                
                # 3. Get relevant patterns from long-term memory
                relevant_patterns = self.long_term_memory.get_relevant_patterns(
                    current_context.task_context if current_context else None,
                    focused_attention
                )
                
                # 4. Executive control makes high-level decisions
                working_memory_state = self.working_memory.get_current_state()
                executive_decision = self.executive_control.make_decision(
                    working_memory_state, relevant_patterns
                )
                
                # 5. Social intelligence adjusts behavior
                social_adjustment = self.social_intelligence.adjust_behavior(
                    executive_decision, focused_attention
                )
                
                # 6. Learning engine updates patterns
                updated_patterns = self.learning_engine.update_patterns(
                    focused_attention, executive_decision, social_adjustment
                )
                
                # 7. Generate cognitive response
                response = self._generate_cognitive_response(
                    focused_attention, executive_decision, social_adjustment,
                    working_memory_state, relevant_patterns
                )
                
                # Update performance metrics
                self.performance_metrics['response_time'] = time.time() - start_time
                self.performance_metrics['decision_confidence'] = executive_decision.confidence
                
                return response
                
            except Exception as e:
                self.get_logger().error(f"Error in cognitive processing: {e}")
                return self._generate_error_response(str(e))
    
    def _generate_cognitive_response(self,
                                   focused_attention: List[AttentionFocus],
                                   executive_decision: ExecutiveDecision,
                                   social_adjustment: SocialAdjustment,
                                   working_memory_state: Dict[str, Any],
                                   relevant_patterns: List[Any]) -> CognitiveResponse:
        """Generate comprehensive cognitive response."""
        
        # Extract planned actions from executive decision and social adjustment
        planned_actions = []
        
        # Add executive decision actions
        if executive_decision and executive_decision.actions:
            planned_actions.extend(executive_decision.actions)
        
        # Add social adjustment actions
        if social_adjustment and social_adjustment.actions:
            planned_actions.extend(social_adjustment.actions)
        
        # Get summaries from components
        attention_summary = self.attention.get_attention_summary()
        memory_summary = self.working_memory.get_memory_summary()
        learning_summary = self.learning_engine.get_learning_summary()
        social_summary = self.social_intelligence.get_social_summary()
        
        # Determine safety status
        safety_state = self.working_memory.get_safety_state()
        safety_status = safety_state.safety_level if safety_state else "safe"
        
        # Create social context
        social_context = {
            'context': social_summary['current_context'],
            'cultural_context': social_summary['cultural_context'],
            'tracked_humans': social_summary['tracked_humans'],
            'recent_behaviors': social_summary['recent_behaviors']
        }
        
        return CognitiveResponse(
            planned_actions=planned_actions,
            reasoning=executive_decision.reasoning if executive_decision else "No decision made",
            confidence=executive_decision.confidence if executive_decision else 0.0,
            social_context=social_context,
            safety_status=safety_status,
            attention_summary=attention_summary,
            memory_summary=memory_summary,
            learning_progress=learning_summary,
            timestamp=time.time()
        )
    
    def _generate_error_response(self, error_message: str) -> CognitiveResponse:
        """Generate error response when cognitive processing fails."""
        return CognitiveResponse(
            planned_actions=[{
                'action_type': 'error_recovery',
                'description': 'Cognitive processing error',
                'parameters': {'error': error_message}
            }],
            reasoning=f"Error in cognitive processing: {error_message}",
            confidence=0.0,
            social_context={'context': 'error', 'error': error_message},
            safety_status='caution',
            attention_summary={'error': error_message},
            memory_summary={'error': error_message},
            learning_progress={'error': error_message},
            timestamp=time.time()
        )
    
    def _cognitive_processing_loop(self):
        """Main cognitive processing loop."""
        try:
            # Get current sensor data (simplified - in practice, this would come from actual sensors)
            sensor_data = self._get_current_sensor_data()
            
            # Get current context
            current_context = self._get_current_context()
            
            # Process through cognitive architecture
            response = self.process_input(sensor_data, None, current_context)
            
            # Publish cognitive response
            self._publish_cognitive_response(response)
            
            # Execute planned actions
            self._execute_planned_actions(response.planned_actions)
            
        except Exception as e:
            self.get_logger().error(f"Error in cognitive processing loop: {e}")
    
    def _get_current_sensor_data(self) -> MultiModalSensorData:
        """Get current sensor data (placeholder implementation)."""
        # In practice, this would integrate with actual sensor data
        return MultiModalSensorData(
            visual_data={
                'objects': [],
                'faces': [],
                'motion': [],
                'gestures': []
            },
            audio_data={
                'speech': [],
                'sounds': [],
                'voice_characteristics': {}
            },
            tactile_data={
                'contacts': [],
                'forces': []
            },
            proprioceptive_data={
                'position': [0.0, 0.0, 0.0],
                'orientation': [0.0, 0.0, 0.0, 1.0],
                'velocity': [0.0, 0.0, 0.0]
            },
            timestamp=time.time()
        )
    
    def _get_current_context(self) -> Context:
        """Get current context information."""
        return Context(
            task_context=self.working_memory.get_task_context().__dict__ if self.working_memory.get_task_context() else None,
            environmental_context={
                'location': 'unknown',
                'time': time.time(),
                'weather': 'unknown'
            },
            social_context=self.social_intelligence.get_social_summary(),
            safety_context=self.working_memory.get_safety_state().__dict__ if self.working_memory.get_safety_state() else None,
            temporal_context={
                'time_of_day': time.localtime().tm_hour,
                'day_of_week': time.localtime().tm_wday
            },
            spatial_context={
                'position': [0.0, 0.0, 0.0],
                'orientation': [0.0, 0.0, 0.0, 1.0]
            }
        )
    
    def _publish_cognitive_response(self, response: CognitiveResponse):
        """Publish cognitive response to ROS topics."""
        # Publish main cognitive response
        response_msg = String()
        response_msg.data = f"Actions: {len(response.planned_actions)}, Confidence: {response.confidence:.2f}, Safety: {response.safety_status}"
        self.cognitive_publisher.publish(response_msg)
        
        # Publish status
        status_msg = String()
        status_msg.data = f"Active|Confidence:{response.confidence:.2f}|Safety:{response.safety_status}"
        self.status_publisher.publish(status_msg)
    
    def _execute_planned_actions(self, planned_actions: List[Dict[str, Any]]):
        """Execute planned actions."""
        for action in planned_actions:
            action_type = action.get('action_type', 'unknown')
            
            if action_type == 'movement':
                self._execute_movement_action(action)
            elif action_type == 'speak':
                self._execute_speech_action(action)
            elif action_type == 'facial_expression':
                self._execute_expression_action(action)
            elif action_type == 'gesture':
                self._execute_gesture_action(action)
            elif action_type == 'safety_override':
                self._execute_safety_action(action)
            else:
                self.get_logger().warn(f"Unknown action type: {action_type}")
    
    def _execute_movement_action(self, action: Dict[str, Any]):
        """Execute movement action."""
        try:
            # Create Twist message for movement
            twist_msg = Twist()
            
            # Extract movement parameters
            direction = action.get('direction', 'stop')
            speed = action.get('speed', 0.0)
            distance = action.get('distance', 0.0)
            
            if direction == 'forward':
                twist_msg.linear.x = speed
            elif direction == 'backward':
                twist_msg.linear.x = -speed
            elif direction == 'left':
                twist_msg.angular.z = speed
            elif direction == 'right':
                twist_msg.angular.z = -speed
            else:
                # Stop
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
            
            self.action_publisher.publish(twist_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error executing movement action: {e}")
    
    def _execute_speech_action(self, action: Dict[str, Any]):
        """Execute speech action."""
        content = action.get('content', '')
        tone = action.get('tone', 'neutral')
        volume = action.get('volume', 'normal')
        
        # In practice, this would integrate with a speech synthesis system
        self.get_logger().info(f"Speech: {content} (tone: {tone}, volume: {volume})")
    
    def _execute_expression_action(self, action: Dict[str, Any]):
        """Execute facial expression action."""
        expression = action.get('expression', 'neutral')
        intensity = action.get('intensity', 0.5)
        
        # In practice, this would control a robot's facial display
        self.get_logger().info(f"Expression: {expression} (intensity: {intensity})")
    
    def _execute_gesture_action(self, action: Dict[str, Any]):
        """Execute gesture action."""
        gesture_type = action.get('gesture_type', 'unknown')
        intensity = action.get('intensity', 0.5)
        
        # In practice, this would control a robot's arms/body
        self.get_logger().info(f"Gesture: {gesture_type} (intensity: {intensity})")
    
    def _execute_safety_action(self, action: Dict[str, Any]):
        """Execute safety action."""
        safety_level = action.get('safety_level', 'caution')
        override_all_tasks = action.get('override_all_tasks', False)
        
        # In practice, this would trigger safety protocols
        self.get_logger().warn(f"Safety override: {safety_level} (override_all: {override_all_tasks})")
    
    def _performance_monitoring_loop(self):
        """Performance monitoring loop."""
        try:
            # Collect performance metrics from all components
            attention_metrics = self.attention.get_performance_metrics()
            memory_metrics = self.working_memory.get_performance_metrics()
            learning_metrics = self.learning_engine.get_performance_metrics()
            social_metrics = self.social_intelligence.get_performance_metrics()
            
            # Update overall performance metrics
            self.performance_metrics['learning_rate'] = learning_metrics.get('learning_rate', 0.0)
            
            # Log performance summary
            self.get_logger().debug(
                f"Performance - Response: {self.performance_metrics['response_time']:.3f}s, "
                f"Confidence: {self.performance_metrics['decision_confidence']:.2f}, "
                f"Learning: {self.performance_metrics['learning_rate']:.3f}"
            )
            
        except Exception as e:
            self.get_logger().error(f"Error in performance monitoring: {e}")
    
    # ROS callback methods
    def _camera_callback(self, msg: Image):
        """Handle camera image data."""
        # In practice, this would process image data and update sensor data
        pass
    
    def _laser_callback(self, msg: LaserScan):
        """Handle laser scan data."""
        # In practice, this would process laser data and update sensor data
        pass
    
    def _pointcloud_callback(self, msg: PointCloud2):
        """Handle point cloud data."""
        # In practice, this would process point cloud data and update sensor data
        pass
    
    def _audio_callback(self, msg: Float32MultiArray):
        """Handle audio feature data."""
        # In practice, this would process audio data and update sensor data
        pass
    
    def _task_callback(self, msg: TaskPlan):
        """Handle task plan updates."""
        # Update working memory with new task
        task_context = {
            'id': msg.task_id,
            'type': msg.task_type,
            'description': msg.description,
            'status': 'active',
            'current_step': 0,
            'total_steps': len(msg.steps),
            'start_time': time.time()
        }
        
        self.working_memory.store_memory(
            MemoryType.TASK_CONTEXT,
            task_context,
            0.8
        )
    
    def _safety_callback(self, msg: SafetyViolation):
        """Handle safety violation updates."""
        # Update safety state
        safety_context = {
            'violation_type': msg.violation_type,
            'severity': msg.severity,
            'location': [msg.location.x, msg.location.y, msg.location.z],
            'timestamp': time.time()
        }
        
        self.working_memory.store_memory(
            MemoryType.SAFETY_STATE,
            safety_context,
            0.9
        )
    
    def _validate_task_callback(self, request, response):
        """Handle task validation requests."""
        try:
            # Validate task plan using cognitive components
            task_context = {
                'id': request.task_plan.task_id,
                'type': request.task_plan.task_type,
                'description': request.task_plan.description,
                'steps': [step.step_description for step in request.task_plan.steps]
            }
            
            # Check against safety constraints
            safety_state = self.working_memory.get_safety_state()
            if safety_state and safety_state.safety_level == 'critical':
                response.is_valid = False
                response.reason = "Safety critical state - task not allowed"
                return response
            
            # Check against social context
            social_context = self.social_intelligence.get_social_context()
            if social_context.value == 'interacting':
                # Modify task for social appropriateness
                response.is_valid = True
                response.reason = "Task validated with social considerations"
            else:
                response.is_valid = True
                response.reason = "Task validated successfully"
            
            return response
            
        except Exception as e:
            response.is_valid = False
            response.reason = f"Validation error: {e}"
            return response
    
    def get_cognitive_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of cognitive architecture state."""
        with self._lock:
            return {
                'performance_metrics': self.performance_metrics,
                'attention_summary': self.attention.get_attention_summary(),
                'memory_summary': self.working_memory.get_memory_summary(),
                'learning_summary': self.learning_engine.get_learning_summary(),
                'social_summary': self.social_intelligence.get_social_summary(),
                'executive_summary': self.executive_control.get_decision_summary(),
                'long_term_memory_stats': self.long_term_memory.get_memory_statistics()
            }


def main(args=None):
    """Main function to run the cognitive architecture node."""
    rclpy.init(args=args)
    
    try:
        node = CognitiveArchitectureNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in cognitive architecture node: {e}")
    finally:
        # Cleanup
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 