"""
Social Intelligence Node

This module provides the main ROS2 node for social intelligence and
human-robot interaction, orchestrating all social intelligence components.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import Image, AudioData
from geometry_msgs.msg import PoseArray, Pose
from eip_interfaces.msg import TaskPlan, SafetyVerificationRequest
from eip_interfaces.srv import ValidateTaskPlan

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import json
import time
from threading import Lock

from .emotion_recognizer import EmotionRecognizer, HumanInput, EmotionAnalysis
from .social_behavior_engine import SocialBehaviorEngine, SocialBehavior, SocialContext, RobotState
from .cultural_adaptation import CulturalAdaptationEngine, CulturalAdaptation
from .personality_engine import PersonalityEngine, PersonalityAdaptation
from .social_learning import SocialLearning, SocialLearningResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SocialIntelligenceNode(Node):
    """
    Main social intelligence node for human-robot interaction
    
    This node orchestrates all social intelligence components including
    emotion recognition, social behavior generation, cultural adaptation,
    personality management, and social learning.
    """
    
    def __init__(self):
        """Initialize the social intelligence node"""
        super().__init__('social_intelligence_node')
        
        # Initialize components
        self.emotion_recognizer = EmotionRecognizer()
        self.social_behavior_engine = SocialBehaviorEngine()
        self.cultural_adaptation = CulturalAdaptationEngine()
        self.personality_engine = PersonalityEngine()
        self.social_learning = SocialLearning()
        
        # Initialize state
        self.current_social_context = self._initialize_social_context()
        self.current_robot_state = self._initialize_robot_state()
        self.interaction_lock = Lock()
        
        # Setup QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Setup publishers
        self.setup_publishers(qos_profile)
        
        # Setup subscribers
        self.setup_subscribers(qos_profile)
        
        # Setup services
        self.setup_services()
        
        # Setup timers
        self.setup_timers()
        
        # Initialize parameters
        self.declare_parameters()
        
        logger.info("Social Intelligence Node initialized successfully")
    
    def declare_parameters(self):
        """Declare ROS2 parameters"""
        self.declare_parameter('cultural_context', 'western')
        self.declare_parameter('personality_profile', 'friendly_assistant')
        self.declare_parameter('learning_enabled', True)
        self.declare_parameter('safety_threshold', 0.8)
        self.declare_parameter('response_timeout', 2.0)
        self.declare_parameter('max_interaction_history', 1000)
    
    def setup_publishers(self, qos_profile: QoSProfile):
        """Setup ROS2 publishers"""
        # Social response publishers
        self.verbal_response_pub = self.create_publisher(
            String, 'social_intelligence/verbal_response', qos_profile
        )
        self.gesture_response_pub = self.create_publisher(
            String, 'social_intelligence/gesture_response', qos_profile
        )
        self.facial_response_pub = self.create_publisher(
            String, 'social_intelligence/facial_response', qos_profile
        )
        
        # Emotion analysis publishers
        self.emotion_analysis_pub = self.create_publisher(
            String, 'social_intelligence/emotion_analysis', qos_profile
        )
        self.social_confidence_pub = self.create_publisher(
            Float32, 'social_intelligence/confidence', qos_profile
        )
        
        # Learning and adaptation publishers
        self.learning_insights_pub = self.create_publisher(
            String, 'social_intelligence/learning_insights', qos_profile
        )
        self.cultural_adaptation_pub = self.create_publisher(
            String, 'social_intelligence/cultural_adaptation', qos_profile
        )
        self.personality_state_pub = self.create_publisher(
            String, 'social_intelligence/personality_state', qos_profile
        )
        
        # Safety and monitoring publishers
        self.safety_status_pub = self.create_publisher(
            Bool, 'social_intelligence/safety_status', qos_profile
        )
        self.interaction_status_pub = self.create_publisher(
            String, 'social_intelligence/interaction_status', qos_profile
        )
    
    def setup_subscribers(self, qos_profile: QoSProfile):
        """Setup ROS2 subscribers"""
        # Human input subscribers
        self.facial_image_sub = self.create_subscription(
            Image, 'sensors/facial_image', self.facial_image_callback, qos_profile
        )
        self.voice_audio_sub = self.create_subscription(
            AudioData, 'sensors/voice_audio', self.voice_audio_callback, qos_profile
        )
        self.body_pose_sub = self.create_subscription(
            PoseArray, 'sensors/body_pose', self.body_pose_callback, qos_profile
        )
        self.speech_text_sub = self.create_subscription(
            String, 'speech_recognition/text', self.speech_text_callback, qos_profile
        )
        
        # Context and state subscribers
        self.social_context_sub = self.create_subscription(
            String, 'context/social_context', self.social_context_callback, qos_profile
        )
        self.robot_state_sub = self.create_subscription(
            String, 'robot/state', self.robot_state_callback, qos_profile
        )
        self.human_feedback_sub = self.create_subscription(
            String, 'feedback/human_feedback', self.human_feedback_callback, qos_profile
        )
        
        # Task and safety subscribers
        self.task_plan_sub = self.create_subscription(
            TaskPlan, 'planning/task_plan', self.task_plan_callback, qos_profile
        )
        self.safety_request_sub = self.create_subscription(
            SafetyVerificationRequest, 'safety/verification_request', 
            self.safety_request_callback, qos_profile
        )
    
    def setup_services(self):
        """Setup ROS2 services"""
        self.validate_social_behavior_srv = self.create_service(
            ValidateTaskPlan, 'social_intelligence/validate_behavior',
            self.validate_social_behavior_callback
        )
    
    def setup_timers(self):
        """Setup ROS2 timers"""
        # Periodic status updates
        self.status_timer = self.create_timer(1.0, self.publish_status)
        
        # Periodic learning updates
        self.learning_timer = self.create_timer(5.0, self.publish_learning_insights)
        
        # Periodic safety checks
        self.safety_timer = self.create_timer(0.5, self.publish_safety_status)
    
    def _initialize_social_context(self) -> SocialContext:
        """Initialize default social context"""
        return SocialContext(
            environment='indoor',
            relationship='neutral',
            cultural_context=self.get_parameter('cultural_context').value,
            social_norms={},
            interaction_history=[],
            current_task=None,
            time_of_day='day',
            privacy_level='public'
        )
    
    def _initialize_robot_state(self) -> RobotState:
        """Initialize default robot state"""
        return RobotState(
            capabilities=['verbal', 'gestural', 'facial', 'proxemic'],
            current_emotion='neutral',
            energy_level=0.8,
            task_engagement=0.5,
            social_comfort=0.7,
            safety_status='safe'
        )
    
    def facial_image_callback(self, msg: Image):
        """Handle facial image input"""
        try:
            # Convert ROS Image to numpy array
            # This is a placeholder - in real implementation, you'd convert the image
            facial_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder
            
            # Update human input
            with self.interaction_lock:
                if not hasattr(self, 'current_human_input'):
                    self.current_human_input = HumanInput()
                self.current_human_input.facial_image = facial_image
                self.current_human_input.timestamp = time.time()
            
            # Process social interaction
            self.process_social_interaction()
            
        except Exception as e:
            logger.error(f"Error processing facial image: {e}")
    
    def voice_audio_callback(self, msg: AudioData):
        """Handle voice audio input"""
        try:
            # Convert ROS AudioData to numpy array
            # This is a placeholder - in real implementation, you'd convert the audio
            voice_audio = np.array(msg.data, dtype=np.float32)  # Placeholder
            
            # Update human input
            with self.interaction_lock:
                if not hasattr(self, 'current_human_input'):
                    self.current_human_input = HumanInput()
                self.current_human_input.voice_audio = voice_audio
                self.current_human_input.timestamp = time.time()
            
            # Process social interaction
            self.process_social_interaction()
            
        except Exception as e:
            logger.error(f"Error processing voice audio: {e}")
    
    def body_pose_callback(self, msg: PoseArray):
        """Handle body pose input"""
        try:
            # Convert ROS PoseArray to numpy array
            # This is a placeholder - in real implementation, you'd convert the poses
            body_pose = np.array([[pose.position.x, pose.position.y, pose.position.z] 
                                for pose in msg.poses])
            
            # Update human input
            with self.interaction_lock:
                if not hasattr(self, 'current_human_input'):
                    self.current_human_input = HumanInput()
                self.current_human_input.body_pose = body_pose
                self.current_human_input.timestamp = time.time()
            
            # Process social interaction
            self.process_social_interaction()
            
        except Exception as e:
            logger.error(f"Error processing body pose: {e}")
    
    def speech_text_callback(self, msg: String):
        """Handle speech text input"""
        try:
            # Update human input
            with self.interaction_lock:
                if not hasattr(self, 'current_human_input'):
                    self.current_human_input = HumanInput()
                self.current_human_input.speech_text = msg.data
                self.current_human_input.timestamp = time.time()
            
            # Process social interaction
            self.process_social_interaction()
            
        except Exception as e:
            logger.error(f"Error processing speech text: {e}")
    
    def social_context_callback(self, msg: String):
        """Handle social context updates"""
        try:
            context_data = json.loads(msg.data)
            
            with self.interaction_lock:
                self.current_social_context.environment = context_data.get('environment', 'indoor')
                self.current_social_context.relationship = context_data.get('relationship', 'neutral')
                self.current_social_context.cultural_context = context_data.get('cultural_context', 'western')
                self.current_social_context.current_task = context_data.get('current_task')
                self.current_social_context.time_of_day = context_data.get('time_of_day', 'day')
                self.current_social_context.privacy_level = context_data.get('privacy_level', 'public')
                
                # Update interaction history
                if 'interaction_history' in context_data:
                    self.current_social_context.interaction_history = context_data['interaction_history']
            
        except Exception as e:
            logger.error(f"Error processing social context: {e}")
    
    def robot_state_callback(self, msg: String):
        """Handle robot state updates"""
        try:
            state_data = json.loads(msg.data)
            
            with self.interaction_lock:
                self.current_robot_state.capabilities = state_data.get('capabilities', ['verbal'])
                self.current_robot_state.current_emotion = state_data.get('current_emotion', 'neutral')
                self.current_robot_state.energy_level = state_data.get('energy_level', 0.8)
                self.current_robot_state.task_engagement = state_data.get('task_engagement', 0.5)
                self.current_robot_state.social_comfort = state_data.get('social_comfort', 0.7)
                self.current_robot_state.safety_status = state_data.get('safety_status', 'safe')
            
        except Exception as e:
            logger.error(f"Error processing robot state: {e}")
    
    def human_feedback_callback(self, msg: String):
        """Handle human feedback"""
        try:
            feedback_data = json.loads(msg.data)
            
            # Store feedback for learning
            with self.interaction_lock:
                if hasattr(self, 'current_human_feedback'):
                    self.current_human_feedback = feedback_data
                else:
                    self.current_human_feedback = feedback_data
            
            # Trigger learning from feedback
            self.learn_from_feedback(feedback_data)
            
        except Exception as e:
            logger.error(f"Error processing human feedback: {e}")
    
    def task_plan_callback(self, msg: TaskPlan):
        """Handle task plan updates"""
        try:
            with self.interaction_lock:
                self.current_social_context.current_task = msg.task_name
            
        except Exception as e:
            logger.error(f"Error processing task plan: {e}")
    
    def safety_request_callback(self, msg: SafetyVerificationRequest):
        """Handle safety verification requests"""
        try:
            # Validate social behavior for safety
            safety_status = self.validate_social_safety(msg.behavior_description)
            
            # Publish safety status
            safety_msg = Bool()
            safety_msg.data = safety_status
            self.safety_status_pub.publish(safety_msg)
            
        except Exception as e:
            logger.error(f"Error processing safety request: {e}")
    
    def validate_social_behavior_callback(self, request, response):
        """Handle social behavior validation service"""
        try:
            # Validate the proposed behavior
            validation_result = self.validate_social_behavior(request.task_plan)
            
            response.is_valid = validation_result['is_valid']
            response.validation_message = validation_result['message']
            response.confidence = validation_result['confidence']
            
            return response
            
        except Exception as e:
            logger.error(f"Error validating social behavior: {e}")
            response.is_valid = False
            response.validation_message = f"Validation error: {str(e)}"
            response.confidence = 0.0
            return response
    
    def process_social_interaction(self):
        """Process social interaction with all components"""
        try:
            with self.interaction_lock:
                if not hasattr(self, 'current_human_input') or self.current_human_input is None:
                    return
                
                human_input = self.current_human_input
                social_context = self.current_social_context
                robot_state = self.current_robot_state
            
            # Step 1: Recognize emotions
            emotion_analysis = self.emotion_recognizer.analyze_emotions(
                human_input, social_context.__dict__
            )
            
            # Step 2: Generate social behavior
            social_behavior = self.social_behavior_engine.generate_behavior(
                emotion_analysis, social_context, robot_state
            )
            
            # Step 3: Apply cultural adaptation
            cultural_adaptation = self.cultural_adaptation.adapt_behavior(
                social_behavior.content, social_context.cultural_context, social_context.__dict__
            )
            
            # Step 4: Apply personality
            personality_adaptation = self.personality_engine.apply_personality(
                cultural_adaptation.adapted_behavior, social_context.__dict__, robot_state.__dict__
            )
            
            # Step 5: Generate final response
            final_response = self.generate_final_response(personality_adaptation.adapted_behavior)
            
            # Step 6: Publish responses
            self.publish_social_responses(final_response, emotion_analysis, social_behavior)
            
            # Step 7: Learn from interaction
            if self.get_parameter('learning_enabled').value:
                self.learn_from_interaction(human_input, final_response, social_context.__dict__)
            
            # Clear current input
            with self.interaction_lock:
                self.current_human_input = None
            
        except Exception as e:
            logger.error(f"Error processing social interaction: {e}")
    
    def generate_final_response(self, adapted_behavior: Dict[str, any]) -> Dict[str, any]:
        """Generate final social response from adapted behavior"""
        final_response = {}
        
        # Extract verbal response
        if 'verbal_response' in adapted_behavior:
            final_response['verbal'] = adapted_behavior['verbal_response']
        
        # Extract gesture response
        if 'gesture_type' in adapted_behavior:
            final_response['gesture'] = adapted_behavior['gesture_type']
        
        # Extract facial response
        if 'facial_expression' in adapted_behavior:
            final_response['facial'] = adapted_behavior['facial_expression']
        
        # Extract proxemic response
        if 'proxemic_behavior' in adapted_behavior:
            final_response['proxemic'] = adapted_behavior['proxemic_behavior']
        
        return final_response
    
    def publish_social_responses(self, 
                               final_response: Dict[str, any],
                               emotion_analysis: EmotionAnalysis,
                               social_behavior: SocialBehavior):
        """Publish social responses to ROS topics"""
        try:
            # Publish verbal response
            if 'verbal' in final_response:
                verbal_msg = String()
                verbal_msg.data = final_response['verbal']
                self.verbal_response_pub.publish(verbal_msg)
            
            # Publish gesture response
            if 'gesture' in final_response:
                gesture_msg = String()
                gesture_msg.data = final_response['gesture']
                self.gesture_response_pub.publish(gesture_msg)
            
            # Publish facial response
            if 'facial' in final_response:
                facial_msg = String()
                facial_msg.data = final_response['facial']
                self.facial_response_pub.publish(facial_msg)
            
            # Publish emotion analysis
            emotion_msg = String()
            emotion_data = {
                'primary_emotion': emotion_analysis.primary_emotion.value,
                'confidence': emotion_analysis.confidence,
                'intensity': emotion_analysis.intensity,
                'overall_emotional_state': emotion_analysis.overall_emotional_state
            }
            emotion_msg.data = json.dumps(emotion_data)
            self.emotion_analysis_pub.publish(emotion_msg)
            
            # Publish social confidence
            confidence_msg = Float32()
            confidence_msg.data = social_behavior.confidence
            self.social_confidence_pub.publish(confidence_msg)
            
        except Exception as e:
            logger.error(f"Error publishing social responses: {e}")
    
    def learn_from_interaction(self,
                             human_input: HumanInput,
                             robot_response: Dict[str, any],
                             social_context: Dict):
        """Learn from social interaction"""
        try:
            # Get human feedback if available
            human_feedback = None
            with self.interaction_lock:
                if hasattr(self, 'current_human_feedback'):
                    human_feedback = self.current_human_feedback
                    self.current_human_feedback = None
            
            # Learn from interaction
            learning_result = self.social_learning.learn_from_interaction(
                human_input.__dict__, robot_response, human_feedback, social_context
            )
            
            # Store learning result for periodic publishing
            with self.interaction_lock:
                self.current_learning_result = learning_result
            
        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")
    
    def learn_from_feedback(self, feedback_data: Dict):
        """Learn from human feedback"""
        try:
            # This would integrate with the social learning system
            # to learn from explicit feedback
            logger.info(f"Learning from feedback: {feedback_data}")
            
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
    
    def validate_social_safety(self, behavior_description: str) -> bool:
        """Validate social behavior for safety"""
        try:
            # Placeholder for safety validation
            # In real implementation, this would check for safety concerns
            safety_threshold = self.get_parameter('safety_threshold').value
            
            # Simple safety check
            unsafe_keywords = ['aggressive', 'threatening', 'inappropriate']
            is_safe = not any(keyword in behavior_description.lower() 
                            for keyword in unsafe_keywords)
            
            return is_safe
            
        except Exception as e:
            logger.error(f"Error validating social safety: {e}")
            return False
    
    def validate_social_behavior(self, task_plan) -> Dict[str, any]:
        """Validate social behavior for appropriateness"""
        try:
            # Placeholder for behavior validation
            # In real implementation, this would validate the behavior
            
            validation_result = {
                'is_valid': True,
                'message': 'Behavior validated successfully',
                'confidence': 0.8
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating social behavior: {e}")
            return {
                'is_valid': False,
                'message': f'Validation error: {str(e)}',
                'confidence': 0.0
            }
    
    def publish_status(self):
        """Publish periodic status updates"""
        try:
            # Publish interaction status
            status_msg = String()
            status_data = {
                'node_status': 'active',
                'components_ready': True,
                'last_interaction_time': time.time(),
                'interaction_count': len(self.social_learning.get_interaction_history())
            }
            status_msg.data = json.dumps(status_data)
            self.interaction_status_pub.publish(status_msg)
            
            # Publish personality state
            personality_state = self.personality_engine.get_personality_state()
            personality_msg = String()
            personality_data = {
                'current_emotion': personality_state.emotional_state,
                'energy_level': personality_state.energy_level,
                'social_comfort': personality_state.social_comfort,
                'interaction_style': personality_state.interaction_style,
                'consistency_score': personality_state.consistency_score
            }
            personality_msg.data = json.dumps(personality_data)
            self.personality_state_pub.publish(personality_msg)
            
        except Exception as e:
            logger.error(f"Error publishing status: {e}")
    
    def publish_learning_insights(self):
        """Publish periodic learning insights"""
        try:
            with self.interaction_lock:
                if hasattr(self, 'current_learning_result'):
                    learning_result = self.current_learning_result
                    
                    # Publish learning insights
                    insights_msg = String()
                    insights_data = {
                        'insights': learning_result.learning_insights,
                        'confidence_improvement': learning_result.confidence_improvement,
                        'new_patterns_count': len(learning_result.new_patterns),
                        'updated_patterns_count': len(learning_result.updated_patterns)
                    }
                    insights_msg.data = json.dumps(insights_data)
                    self.learning_insights_pub.publish(insights_msg)
                    
                    # Clear current learning result
                    self.current_learning_result = None
                    
        except Exception as e:
            logger.error(f"Error publishing learning insights: {e}")
    
    def publish_safety_status(self):
        """Publish periodic safety status"""
        try:
            # Check current safety status
            safety_status = self.current_robot_state.safety_status == 'safe'
            
            # Publish safety status
            safety_msg = Bool()
            safety_msg.data = safety_status
            self.safety_status_pub.publish(safety_msg)
            
        except Exception as e:
            logger.error(f"Error publishing safety status: {e}")


def main(args=None):
    """Main function to run the social intelligence node"""
    rclpy.init(args=args)
    
    try:
        node = SocialIntelligenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("Social Intelligence Node interrupted by user")
    except Exception as e:
        logger.error(f"Error in Social Intelligence Node: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main() 