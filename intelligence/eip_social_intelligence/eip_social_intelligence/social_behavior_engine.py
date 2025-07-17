"""
Social Behavior Engine Module

This module generates appropriate social responses and behaviors for
human-robot interaction based on emotion analysis, social context,
and robot capabilities.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehaviorType(Enum):
    """Enumeration of social behavior types"""
    GREETING = "greeting"
    CONVERSATION = "conversation"
    ASSISTANCE = "assistance"
    COMFORT = "comfort"
    EXPLANATION = "explanation"
    APOLOGY = "apology"
    CELEBRATION = "celebration"
    CALMING = "calming"
    ENCOURAGEMENT = "encouragement"
    NEUTRAL = "neutral"


class ResponseModality(Enum):
    """Enumeration of response modalities"""
    VERBAL = "verbal"
    GESTURAL = "gestural"
    FACIAL = "facial"
    PROXEMIC = "proxemic"
    MULTIMODAL = "multimodal"


@dataclass
class SocialBehavior:
    """Data class for social behavior specification"""
    behavior_type: BehaviorType
    modality: ResponseModality
    content: Dict[str, any]
    confidence: float
    appropriateness_score: float
    safety_score: float
    duration: float
    priority: int


@dataclass
class SocialContext:
    """Data class for social context information"""
    environment: str
    relationship: str
    cultural_context: str
    social_norms: Dict[str, any]
    interaction_history: List[Dict]
    current_task: Optional[str]
    time_of_day: str
    privacy_level: str


@dataclass
class RobotState:
    """Data class for robot state information"""
    capabilities: List[str]
    current_emotion: str
    energy_level: float
    task_engagement: float
    social_comfort: float
    safety_status: str


class SocialBehaviorEngine:
    """
    Advanced social behavior generation engine for human-robot interaction
    
    This class generates appropriate social responses and behaviors
    based on emotion analysis, social context, and robot capabilities.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the social behavior engine
        
        Args:
            config: Configuration dictionary for behavior generation
        """
        self.config = config or self._get_default_config()
        self.behavior_templates = self._load_behavior_templates()
        self.response_generator = self._initialize_response_generator()
        self.safety_validator = self._initialize_safety_validator()
        self.context_analyzer = self._initialize_context_analyzer()
        self.behavior_history = []
        
        logger.info("Social behavior engine initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for behavior generation"""
        return {
            'max_response_time': 2.0,
            'safety_threshold': 0.8,
            'appropriateness_threshold': 0.7,
            'max_behavior_history': 20,
            'cultural_sensitivity_weight': 0.3,
            'emotional_appropriateness_weight': 0.4,
            'context_relevance_weight': 0.3
        }
    
    def _load_behavior_templates(self) -> Dict:
        """Load behavior templates for different scenarios"""
        return {
            BehaviorType.GREETING: {
                'verbal': [
                    "Hello! How are you today?",
                    "Good to see you!",
                    "Hi there! I'm here to help.",
                    "Greetings! How can I assist you?"
                ],
                'gestural': ['wave', 'nod', 'open_arms'],
                'facial': ['smile', 'friendly_expression'],
                'proxemic': ['approach_politely', 'maintain_distance']
            },
            BehaviorType.CONVERSATION: {
                'verbal': [
                    "I understand what you're saying.",
                    "That's interesting, tell me more.",
                    "I see your point.",
                    "Let me think about that."
                ],
                'gestural': ['attentive_posture', 'nodding', 'hand_gestures'],
                'facial': ['attentive_expression', 'eye_contact'],
                'proxemic': ['conversational_distance', 'face_toward']
            },
            BehaviorType.ASSISTANCE: {
                'verbal': [
                    "I'd be happy to help you with that.",
                    "Let me assist you with this task.",
                    "I can help you accomplish that.",
                    "Allow me to support you."
                ],
                'gestural': ['helping_gesture', 'pointing', 'demonstrating'],
                'facial': ['helpful_expression', 'encouraging_smile'],
                'proxemic': ['approach_helpfully', 'position_for_assistance']
            },
            BehaviorType.COMFORT: {
                'verbal': [
                    "I understand this might be difficult.",
                    "It's okay to feel that way.",
                    "I'm here to support you.",
                    "Take your time, I'm patient."
                ],
                'gestural': ['gentle_gestures', 'calming_movements'],
                'facial': ['empathetic_expression', 'gentle_smile'],
                'proxemic': ['comfortable_distance', 'non_threatening_position']
            },
            BehaviorType.EXPLANATION: {
                'verbal': [
                    "Let me explain this step by step.",
                    "Here's how this works...",
                    "I'll break this down for you.",
                    "Let me clarify this for you."
                ],
                'gestural': ['explanatory_gestures', 'pointing', 'demonstrating'],
                'facial': ['focused_expression', 'clear_communication'],
                'proxemic': ['optimal_explanation_distance', 'face_toward']
            },
            BehaviorType.APOLOGY: {
                'verbal': [
                    "I apologize for that mistake.",
                    "I'm sorry for the confusion.",
                    "That was my error, let me fix it.",
                    "I apologize for any inconvenience."
                ],
                'gestural': ['apologetic_gesture', 'lowered_posture'],
                'facial': ['apologetic_expression', 'concerned_look'],
                'proxemic': ['respectful_distance', 'non_dominant_position']
            },
            BehaviorType.CELEBRATION: {
                'verbal': [
                    "Excellent work!",
                    "That's fantastic!",
                    "Great job!",
                    "You did it!"
                ],
                'gestural': ['celebration_gesture', 'clapping', 'thumbs_up'],
                'facial': ['excited_expression', 'joyful_smile'],
                'proxemic': ['closer_distance', 'enthusiastic_position']
            },
            BehaviorType.CALMING: {
                'verbal': [
                    "Let's take a deep breath together.",
                    "Everything will be okay.",
                    "Stay calm, I'm here to help.",
                    "Let's work through this calmly."
                ],
                'gestural': ['calming_gestures', 'slow_movements'],
                'facial': ['calm_expression', 'reassuring_smile'],
                'proxemic': ['comfortable_distance', 'stable_position']
            },
            BehaviorType.ENCOURAGEMENT: {
                'verbal': [
                    "You're doing great!",
                    "Keep going, you've got this!",
                    "I believe in you!",
                    "You're making excellent progress!"
                ],
                'gestural': ['encouraging_gestures', 'thumbs_up', 'clapping'],
                'facial': ['encouraging_expression', 'supportive_smile'],
                'proxemic': ['supportive_distance', 'encouraging_position']
            },
            BehaviorType.NEUTRAL: {
                'verbal': [
                    "I understand.",
                    "I see.",
                    "Okay.",
                    "Understood."
                ],
                'gestural': ['neutral_posture', 'minimal_movement'],
                'facial': ['neutral_expression', 'attentive_look'],
                'proxemic': ['standard_distance', 'neutral_position']
            }
        }
    
    def _initialize_response_generator(self):
        """Initialize response generation components"""
        return {
            'verbal_generator': self._initialize_verbal_generator(),
            'gestural_generator': self._initialize_gestural_generator(),
            'facial_generator': self._initialize_facial_generator(),
            'proxemic_generator': self._initialize_proxemic_generator()
        }
    
    def _initialize_safety_validator(self):
        """Initialize safety validation components"""
        return {
            'behavior_safety_checker': self._initialize_safety_checker(),
            'appropriateness_validator': self._initialize_appropriateness_validator(),
            'cultural_sensitivity_checker': self._initialize_cultural_checker()
        }
    
    def _initialize_context_analyzer(self):
        """Initialize context analysis components"""
        return {
            'situation_analyzer': self._initialize_situation_analyzer(),
            'relationship_analyzer': self._initialize_relationship_analyzer(),
            'cultural_analyzer': self._initialize_cultural_analyzer()
        }
    
    def _initialize_verbal_generator(self):
        """Initialize verbal response generator"""
        return "verbal_generator_placeholder"
    
    def _initialize_gestural_generator(self):
        """Initialize gestural response generator"""
        return "gestural_generator_placeholder"
    
    def _initialize_facial_generator(self):
        """Initialize facial expression generator"""
        return "facial_generator_placeholder"
    
    def _initialize_proxemic_generator(self):
        """Initialize proxemic behavior generator"""
        return "proxemic_generator_placeholder"
    
    def _initialize_safety_checker(self):
        """Initialize behavior safety checker"""
        return "safety_checker_placeholder"
    
    def _initialize_appropriateness_validator(self):
        """Initialize appropriateness validator"""
        return "appropriateness_validator_placeholder"
    
    def _initialize_cultural_checker(self):
        """Initialize cultural sensitivity checker"""
        return "cultural_checker_placeholder"
    
    def _initialize_situation_analyzer(self):
        """Initialize situation analyzer"""
        return "situation_analyzer_placeholder"
    
    def _initialize_relationship_analyzer(self):
        """Initialize relationship analyzer"""
        return "relationship_analyzer_placeholder"
    
    def _initialize_cultural_analyzer(self):
        """Initialize cultural context analyzer"""
        return "cultural_analyzer_placeholder"
    
    def generate_behavior(self,
                         emotion_analysis: 'EmotionAnalysis',
                         social_context: SocialContext,
                         robot_state: RobotState) -> SocialBehavior:
        """
        Generate appropriate social behavior based on context
        
        Args:
            emotion_analysis: Analysis of human emotions
            social_context: Current social context
            robot_state: Current robot state
            
        Returns:
            SocialBehavior with appropriate response specification
        """
        try:
            # Analyze context and determine behavior type
            behavior_type = self._determine_behavior_type(
                emotion_analysis, social_context, robot_state
            )
            
            # Select appropriate modality
            modality = self._select_response_modality(
                behavior_type, emotion_analysis, social_context, robot_state
            )
            
            # Generate behavior content
            content = self._generate_behavior_content(
                behavior_type, modality, emotion_analysis, social_context, robot_state
            )
            
            # Validate behavior for safety and appropriateness
            safety_score = self._validate_behavior_safety(content, social_context)
            appropriateness_score = self._validate_behavior_appropriateness(
                content, emotion_analysis, social_context
            )
            
            # Calculate confidence and priority
            confidence = self._calculate_behavior_confidence(
                behavior_type, modality, content, emotion_analysis, social_context
            )
            priority = self._calculate_behavior_priority(
                behavior_type, emotion_analysis, social_context
            )
            
            # Create behavior specification
            behavior = SocialBehavior(
                behavior_type=behavior_type,
                modality=modality,
                content=content,
                confidence=confidence,
                appropriateness_score=appropriateness_score,
                safety_score=safety_score,
                duration=self._estimate_behavior_duration(content, modality),
                priority=priority
            )
            
            # Update behavior history
            self._update_behavior_history(behavior)
            
            logger.debug(f"Generated behavior: {behavior_type.value} with confidence {confidence:.2f}")
            return behavior
            
        except Exception as e:
            logger.error(f"Error generating behavior: {e}")
            return self._get_default_behavior()
    
    def _determine_behavior_type(self,
                                emotion_analysis: 'EmotionAnalysis',
                                social_context: SocialContext,
                                robot_state: RobotState) -> BehaviorType:
        """Determine appropriate behavior type based on context"""
        primary_emotion = emotion_analysis.primary_emotion
        intensity = emotion_analysis.intensity
        relationship = social_context.relationship
        current_task = social_context.current_task
        
        # Rule-based behavior selection
        if relationship == "first_meeting":
            return BehaviorType.GREETING
        
        elif primary_emotion.value in ["sad", "fear", "angry"] and intensity > 0.7:
            return BehaviorType.COMFORT
        
        elif primary_emotion.value == "happy" and intensity > 0.8:
            return BehaviorType.CELEBRATION
        
        elif primary_emotion.value == "confused" and intensity > 0.6:
            return BehaviorType.EXPLANATION
        
        elif current_task and "assist" in current_task.lower():
            return BehaviorType.ASSISTANCE
        
        elif robot_state.safety_status != "safe":
            return BehaviorType.APOLOGY
        
        elif primary_emotion.value == "neutral" and intensity < 0.3:
            return BehaviorType.CONVERSATION
        
        else:
            return BehaviorType.NEUTRAL
    
    def _select_response_modality(self,
                                 behavior_type: BehaviorType,
                                 emotion_analysis: 'EmotionAnalysis',
                                 social_context: SocialContext,
                                 robot_state: RobotState) -> ResponseModality:
        """Select appropriate response modality"""
        # Consider robot capabilities
        available_modalities = []
        if "verbal" in robot_state.capabilities:
            available_modalities.append(ResponseModality.VERBAL)
        if "gestural" in robot_state.capabilities:
            available_modalities.append(ResponseModality.GESTURAL)
        if "facial" in robot_state.capabilities:
            available_modalities.append(ResponseModality.FACIAL)
        if "proxemic" in robot_state.capabilities:
            available_modalities.append(ResponseModality.PROXEMIC)
        
        # Select based on behavior type and context
        if behavior_type in [BehaviorType.GREETING, BehaviorType.CONVERSATION]:
            if ResponseModality.VERBAL in available_modalities:
                return ResponseModality.VERBAL
            elif ResponseModality.GESTURAL in available_modalities:
                return ResponseModality.GESTURAL
        
        elif behavior_type in [BehaviorType.COMFORT, BehaviorType.CALMING]:
            if ResponseModality.MULTIMODAL in available_modalities:
                return ResponseModality.MULTIMODAL
            elif ResponseModality.VERBAL in available_modalities:
                return ResponseModality.VERBAL
        
        elif behavior_type in [BehaviorType.CELEBRATION, BehaviorType.ENCOURAGEMENT]:
            if ResponseModality.GESTURAL in available_modalities:
                return ResponseModality.GESTURAL
            elif ResponseModality.FACIAL in available_modalities:
                return ResponseModality.FACIAL
        
        # Default to first available modality
        return available_modalities[0] if available_modalities else ResponseModality.VERBAL
    
    def _generate_behavior_content(self,
                                  behavior_type: BehaviorType,
                                  modality: ResponseModality,
                                  emotion_analysis: 'EmotionAnalysis',
                                  social_context: SocialContext,
                                  robot_state: RobotState) -> Dict[str, any]:
        """Generate specific behavior content"""
        templates = self.behavior_templates.get(behavior_type, {})
        
        if modality == ResponseModality.VERBAL:
            verbal_templates = templates.get('verbal', ["I understand."])
            selected_template = random.choice(verbal_templates)
            
            # Personalize based on context
            personalized_content = self._personalize_verbal_content(
                selected_template, emotion_analysis, social_context
            )
            
            return {
                'verbal_response': personalized_content,
                'tone': self._determine_verbal_tone(emotion_analysis),
                'speech_rate': self._determine_speech_rate(emotion_analysis),
                'volume': self._determine_volume(emotion_analysis, social_context)
            }
        
        elif modality == ResponseModality.GESTURAL:
            gestural_templates = templates.get('gestural', ['neutral_posture'])
            selected_gesture = random.choice(gestural_templates)
            
            return {
                'gesture_type': selected_gesture,
                'gesture_intensity': self._determine_gesture_intensity(emotion_analysis),
                'gesture_duration': self._determine_gesture_duration(emotion_analysis),
                'gesture_speed': self._determine_gesture_speed(emotion_analysis)
            }
        
        elif modality == ResponseModality.FACIAL:
            facial_templates = templates.get('facial', ['neutral_expression'])
            selected_expression = random.choice(facial_templates)
            
            return {
                'facial_expression': selected_expression,
                'expression_intensity': self._determine_expression_intensity(emotion_analysis),
                'eye_contact': self._determine_eye_contact(emotion_analysis, social_context),
                'expression_duration': self._determine_expression_duration(emotion_analysis)
            }
        
        elif modality == ResponseModality.PROXEMIC:
            proxemic_templates = templates.get('proxemic', ['standard_distance'])
            selected_proxemic = random.choice(proxemic_templates)
            
            return {
                'proxemic_behavior': selected_proxemic,
                'distance': self._determine_appropriate_distance(emotion_analysis, social_context),
                'orientation': self._determine_orientation(emotion_analysis, social_context),
                'movement_speed': self._determine_movement_speed(emotion_analysis)
            }
        
        elif modality == ResponseModality.MULTIMODAL:
            # Combine multiple modalities
            return {
                'verbal': self._generate_behavior_content(
                    behavior_type, ResponseModality.VERBAL, 
                    emotion_analysis, social_context, robot_state
                ),
                'gestural': self._generate_behavior_content(
                    behavior_type, ResponseModality.GESTURAL,
                    emotion_analysis, social_context, robot_state
                ),
                'facial': self._generate_behavior_content(
                    behavior_type, ResponseModality.FACIAL,
                    emotion_analysis, social_context, robot_state
                )
            }
        
        else:
            return {'default_response': 'I understand.'}
    
    def _personalize_verbal_content(self,
                                   template: str,
                                   emotion_analysis: 'EmotionAnalysis',
                                   social_context: SocialContext) -> str:
        """Personalize verbal content based on context"""
        # Simple personalization based on emotion and relationship
        if emotion_analysis.primary_emotion.value == "sad":
            template = template.replace("!", ".")
        
        if social_context.relationship == "formal":
            template = template.replace("Hi", "Hello")
            template = template.replace("!", ".")
        
        return template
    
    def _determine_verbal_tone(self, emotion_analysis: 'EmotionAnalysis') -> str:
        """Determine appropriate verbal tone"""
        emotion = emotion_analysis.primary_emotion.value
        intensity = emotion_analysis.intensity
        
        if emotion == "sad" or emotion == "fear":
            return "gentle"
        elif emotion == "angry":
            return "calm"
        elif emotion == "happy":
            return "enthusiastic"
        elif emotion == "excited":
            return "excited"
        else:
            return "neutral"
    
    def _determine_speech_rate(self, emotion_analysis: 'EmotionAnalysis') -> float:
        """Determine appropriate speech rate"""
        emotion = emotion_analysis.primary_emotion.value
        intensity = emotion_analysis.intensity
        
        base_rate = 1.0
        if emotion == "excited":
            return base_rate * 1.2
        elif emotion == "sad":
            return base_rate * 0.8
        elif emotion == "angry":
            return base_rate * 1.1
        else:
            return base_rate
    
    def _determine_volume(self, 
                         emotion_analysis: 'EmotionAnalysis',
                         social_context: SocialContext) -> float:
        """Determine appropriate volume"""
        emotion = emotion_analysis.primary_emotion.value
        privacy = social_context.privacy_level
        
        base_volume = 0.7
        if privacy == "private":
            return base_volume * 0.8
        elif emotion == "excited":
            return base_volume * 1.1
        elif emotion == "sad":
            return base_volume * 0.9
        else:
            return base_volume
    
    def _determine_gesture_intensity(self, emotion_analysis: 'EmotionAnalysis') -> float:
        """Determine gesture intensity"""
        intensity = emotion_analysis.intensity
        return min(1.0, intensity * 1.2)
    
    def _determine_gesture_duration(self, emotion_analysis: 'EmotionAnalysis') -> float:
        """Determine gesture duration"""
        emotion = emotion_analysis.primary_emotion.value
        if emotion == "sad":
            return 2.0
        elif emotion == "excited":
            return 1.0
        else:
            return 1.5
    
    def _determine_gesture_speed(self, emotion_analysis: 'EmotionAnalysis') -> float:
        """Determine gesture speed"""
        emotion = emotion_analysis.primary_emotion.value
        if emotion == "excited":
            return 1.2
        elif emotion == "sad":
            return 0.8
        else:
            return 1.0
    
    def _determine_expression_intensity(self, emotion_analysis: 'EmotionAnalysis') -> float:
        """Determine facial expression intensity"""
        return emotion_analysis.intensity
    
    def _determine_eye_contact(self, 
                              emotion_analysis: 'EmotionAnalysis',
                              social_context: SocialContext) -> float:
        """Determine appropriate eye contact level"""
        emotion = emotion_analysis.primary_emotion.value
        relationship = social_context.relationship
        
        if relationship == "formal":
            return 0.6
        elif emotion == "sad":
            return 0.8
        elif emotion == "angry":
            return 0.4
        else:
            return 0.7
    
    def _determine_expression_duration(self, emotion_analysis: 'EmotionAnalysis') -> float:
        """Determine facial expression duration"""
        return 2.0
    
    def _determine_appropriate_distance(self,
                                      emotion_analysis: 'EmotionAnalysis',
                                      social_context: SocialContext) -> float:
        """Determine appropriate interpersonal distance"""
        emotion = emotion_analysis.primary_emotion.value
        relationship = social_context.relationship
        
        if relationship == "intimate":
            return 0.5
        elif relationship == "personal":
            return 1.0
        elif relationship == "social":
            return 1.5
        elif relationship == "public":
            return 2.0
        else:
            return 1.2
    
    def _determine_orientation(self,
                             emotion_analysis: 'EmotionAnalysis',
                             social_context: SocialContext) -> str:
        """Determine appropriate body orientation"""
        emotion = emotion_analysis.primary_emotion.value
        if emotion == "angry":
            return "slightly_away"
        else:
            return "toward"
    
    def _determine_movement_speed(self, emotion_analysis: 'EmotionAnalysis') -> float:
        """Determine appropriate movement speed"""
        emotion = emotion_analysis.primary_emotion.value
        if emotion == "excited":
            return 1.2
        elif emotion == "sad":
            return 0.8
        else:
            return 1.0
    
    def _validate_behavior_safety(self, content: Dict, social_context: SocialContext) -> float:
        """Validate behavior for safety"""
        # Placeholder for safety validation
        # In real implementation, this would check for potential safety issues
        return 0.9
    
    def _validate_behavior_appropriateness(self,
                                         content: Dict,
                                         emotion_analysis: 'EmotionAnalysis',
                                         social_context: SocialContext) -> float:
        """Validate behavior for appropriateness"""
        # Placeholder for appropriateness validation
        # In real implementation, this would check cultural and social appropriateness
        return 0.8
    
    def _calculate_behavior_confidence(self,
                                     behavior_type: BehaviorType,
                                     modality: ResponseModality,
                                     content: Dict,
                                     emotion_analysis: 'EmotionAnalysis',
                                     social_context: SocialContext) -> float:
        """Calculate confidence in the generated behavior"""
        # Base confidence on emotion recognition confidence
        base_confidence = emotion_analysis.confidence
        
        # Adjust based on behavior type familiarity
        behavior_familiarity = self._get_behavior_familiarity(behavior_type)
        
        # Adjust based on context clarity
        context_clarity = self._get_context_clarity(social_context)
        
        confidence = (base_confidence + behavior_familiarity + context_clarity) / 3
        return max(0.0, min(1.0, confidence))
    
    def _calculate_behavior_priority(self,
                                   behavior_type: BehaviorType,
                                   emotion_analysis: 'EmotionAnalysis',
                                   social_context: SocialContext) -> int:
        """Calculate behavior priority"""
        # Higher priority for safety-related behaviors
        if behavior_type == BehaviorType.APOLOGY:
            return 1
        elif behavior_type == BehaviorType.COMFORT:
            return 2
        elif behavior_type == BehaviorType.CALMING:
            return 3
        elif behavior_type == BehaviorType.GREETING:
            return 4
        else:
            return 5
    
    def _estimate_behavior_duration(self, content: Dict, modality: ResponseModality) -> float:
        """Estimate behavior duration"""
        if modality == ResponseModality.VERBAL:
            verbal_response = content.get('verbal_response', '')
            words = len(verbal_response.split())
            return words * 0.5  # Assume 0.5 seconds per word
        
        elif modality == ResponseModality.GESTURAL:
            return content.get('gesture_duration', 1.5)
        
        elif modality == ResponseModality.FACIAL:
            return content.get('expression_duration', 2.0)
        
        elif modality == ResponseModality.PROXEMIC:
            return 1.0
        
        else:
            return 2.0
    
    def _get_behavior_familiarity(self, behavior_type: BehaviorType) -> float:
        """Get familiarity score for behavior type"""
        # Placeholder for behavior familiarity calculation
        return 0.8
    
    def _get_context_clarity(self, social_context: SocialContext) -> float:
        """Get context clarity score"""
        # Placeholder for context clarity calculation
        return 0.7
    
    def _update_behavior_history(self, behavior: SocialBehavior):
        """Update behavior history"""
        self.behavior_history.append(behavior)
        
        # Keep only recent history
        max_history = self.config.get('max_behavior_history', 20)
        if len(self.behavior_history) > max_history:
            self.behavior_history.pop(0)
    
    def _get_default_behavior(self) -> SocialBehavior:
        """Get default behavior when generation fails"""
        return SocialBehavior(
            behavior_type=BehaviorType.NEUTRAL,
            modality=ResponseModality.VERBAL,
            content={'verbal_response': 'I understand.'},
            confidence=0.5,
            appropriateness_score=0.7,
            safety_score=0.9,
            duration=1.0,
            priority=5
        ) 