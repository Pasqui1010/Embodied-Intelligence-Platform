"""
Personality Engine Module

This module provides personality management capabilities for human-robot
interaction, maintaining consistent personality traits across different
interactions and contexts.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum
import random
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonalityTrait(Enum):
    """Enumeration of personality traits"""
    EXTROVERSION = "extroversion"
    INTROVERSION = "introversion"
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"
    EMPATHY = "empathy"
    ASSERTIVENESS = "assertiveness"
    HUMOR = "humor"
    PROFESSIONALISM = "professionalism"


class PersonalityStyle(Enum):
    """Enumeration of personality styles"""
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    PLAYFUL = "playful"
    SERIOUS = "serious"
    ENCOURAGING = "encouraging"
    CALM = "calm"
    ENTHUSIASTIC = "enthusiastic"
    RESERVED = "reserved"


@dataclass
class PersonalityProfile:
    """Data class for personality profile information"""
    personality_id: str
    name: str
    description: str
    base_traits: Dict[PersonalityTrait, float]
    personality_style: PersonalityStyle
    behavioral_patterns: Dict[str, any]
    communication_preferences: Dict[str, any]
    emotional_tendencies: Dict[str, any]
    adaptation_rules: Dict[str, any]


@dataclass
class PersonalityState:
    """Data class for current personality state"""
    current_traits: Dict[PersonalityTrait, float]
    emotional_state: str
    energy_level: float
    social_comfort: float
    interaction_style: str
    consistency_score: float


@dataclass
class PersonalityAdaptation:
    """Data class for personality adaptation results"""
    adapted_behavior: Dict[str, any]
    personality_consistency: float
    trait_expression: Dict[PersonalityTrait, float]
    adaptation_confidence: float
    personality_notes: List[str]


class PersonalityEngine:
    """
    Personality engine for maintaining consistent robot personality
    
    This class manages personality traits and ensures consistent
    personality expression across different interactions and contexts.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the personality engine
        
        Args:
            config: Configuration dictionary for personality management
        """
        self.config = config or self._get_default_config()
        self.personality_profiles = self._load_personality_profiles()
        self.current_profile = self._get_default_personality_profile()
        self.personality_state = self._initialize_personality_state()
        self.trait_manager = self._initialize_trait_manager()
        self.consistency_monitor = self._initialize_consistency_monitor()
        self.personality_history = []
        
        logger.info("Personality engine initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for personality management"""
        return {
            'consistency_threshold': 0.8,
            'trait_stability_factor': 0.9,
            'adaptation_sensitivity': 0.3,
            'max_personality_history': 100,
            'trait_expression_weight': 0.4,
            'consistency_weight': 0.3,
            'context_adaptation_weight': 0.3
        }
    
    def _load_personality_profiles(self) -> Dict[str, PersonalityProfile]:
        """Load predefined personality profiles"""
        return {
            'friendly_assistant': PersonalityProfile(
                personality_id='friendly_assistant',
                name='Friendly Assistant',
                description='A warm, approachable, and helpful personality',
                base_traits={
                    PersonalityTrait.EXTROVERSION: 0.7,
                    PersonalityTrait.INTROVERSION: 0.3,
                    PersonalityTrait.OPENNESS: 0.6,
                    PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
                    PersonalityTrait.AGREEABLENESS: 0.9,
                    PersonalityTrait.NEUROTICISM: 0.2,
                    PersonalityTrait.EMPATHY: 0.8,
                    PersonalityTrait.ASSERTIVENESS: 0.5,
                    PersonalityTrait.HUMOR: 0.6,
                    PersonalityTrait.PROFESSIONALISM: 0.7
                },
                personality_style=PersonalityStyle.FRIENDLY,
                behavioral_patterns={
                    'greeting_style': 'warm_and_welcoming',
                    'conversation_style': 'engaging_and_supportive',
                    'problem_solving': 'collaborative',
                    'feedback_style': 'constructive_and_encouraging'
                },
                communication_preferences={
                    'tone': 'friendly',
                    'formality': 'moderate',
                    'pace': 'comfortable',
                    'volume': 'moderate'
                },
                emotional_tendencies={
                    'baseline_mood': 'positive',
                    'stress_response': 'calm_and_supportive',
                    'excitement_expression': 'moderate_enthusiasm',
                    'empathy_expression': 'high'
                },
                adaptation_rules={
                    'context_adaptation': 'moderate',
                    'trait_flexibility': 'high',
                    'consistency_priority': 'high'
                }
            ),
            'professional_expert': PersonalityProfile(
                personality_id='professional_expert',
                name='Professional Expert',
                description='A knowledgeable, reliable, and efficient personality',
                base_traits={
                    PersonalityTrait.EXTROVERSION: 0.4,
                    PersonalityTrait.INTROVERSION: 0.6,
                    PersonalityTrait.OPENNESS: 0.7,
                    PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
                    PersonalityTrait.AGREEABLENESS: 0.6,
                    PersonalityTrait.NEUROTICISM: 0.1,
                    PersonalityTrait.EMPATHY: 0.5,
                    PersonalityTrait.ASSERTIVENESS: 0.8,
                    PersonalityTrait.HUMOR: 0.3,
                    PersonalityTrait.PROFESSIONALISM: 0.9
                },
                personality_style=PersonalityStyle.PROFESSIONAL,
                behavioral_patterns={
                    'greeting_style': 'polite_and_efficient',
                    'conversation_style': 'focused_and_informative',
                    'problem_solving': 'systematic',
                    'feedback_style': 'direct_and_constructive'
                },
                communication_preferences={
                    'tone': 'professional',
                    'formality': 'high',
                    'pace': 'efficient',
                    'volume': 'moderate'
                },
                emotional_tendencies={
                    'baseline_mood': 'neutral',
                    'stress_response': 'focused_and_methodical',
                    'excitement_expression': 'controlled_enthusiasm',
                    'empathy_expression': 'moderate'
                },
                adaptation_rules={
                    'context_adaptation': 'low',
                    'trait_flexibility': 'moderate',
                    'consistency_priority': 'very_high'
                }
            ),
            'encouraging_coach': PersonalityProfile(
                personality_id='encouraging_coach',
                name='Encouraging Coach',
                description='A motivating, supportive, and inspiring personality',
                base_traits={
                    PersonalityTrait.EXTROVERSION: 0.8,
                    PersonalityTrait.INTROVERSION: 0.2,
                    PersonalityTrait.OPENNESS: 0.7,
                    PersonalityTrait.CONSCIENTIOUSNESS: 0.7,
                    PersonalityTrait.AGREEABLENESS: 0.8,
                    PersonalityTrait.NEUROTICISM: 0.2,
                    PersonalityTrait.EMPATHY: 0.9,
                    PersonalityTrait.ASSERTIVENESS: 0.7,
                    PersonalityTrait.HUMOR: 0.7,
                    PersonalityTrait.PROFESSIONALISM: 0.6
                },
                personality_style=PersonalityStyle.ENCOURAGING,
                behavioral_patterns={
                    'greeting_style': 'enthusiastic_and_encouraging',
                    'conversation_style': 'motivational_and_supportive',
                    'problem_solving': 'empowering',
                    'feedback_style': 'positive_and_constructive'
                },
                communication_preferences={
                    'tone': 'encouraging',
                    'formality': 'moderate',
                    'pace': 'energetic',
                    'volume': 'moderate_to_high'
                },
                emotional_tendencies={
                    'baseline_mood': 'positive_and_energetic',
                    'stress_response': 'calm_and_reassuring',
                    'excitement_expression': 'high_enthusiasm',
                    'empathy_expression': 'very_high'
                },
                adaptation_rules={
                    'context_adaptation': 'high',
                    'trait_flexibility': 'high',
                    'consistency_priority': 'moderate'
                }
            ),
            'calm_companion': PersonalityProfile(
                personality_id='calm_companion',
                name='Calm Companion',
                description='A peaceful, patient, and understanding personality',
                base_traits={
                    PersonalityTrait.EXTROVERSION: 0.3,
                    PersonalityTrait.INTROVERSION: 0.7,
                    PersonalityTrait.OPENNESS: 0.6,
                    PersonalityTrait.CONSCIENTIOUSNESS: 0.6,
                    PersonalityTrait.AGREEABLENESS: 0.9,
                    PersonalityTrait.NEUROTICISM: 0.1,
                    PersonalityTrait.EMPATHY: 0.8,
                    PersonalityTrait.ASSERTIVENESS: 0.3,
                    PersonalityTrait.HUMOR: 0.4,
                    PersonalityTrait.PROFESSIONALISM: 0.5
                },
                personality_style=PersonalityStyle.CALM,
                behavioral_patterns={
                    'greeting_style': 'gentle_and_peaceful',
                    'conversation_style': 'patient_and_understanding',
                    'problem_solving': 'thoughtful',
                    'feedback_style': 'gentle_and_supportive'
                },
                communication_preferences={
                    'tone': 'calm',
                    'formality': 'low',
                    'pace': 'slow_and_steady',
                    'volume': 'soft'
                },
                emotional_tendencies={
                    'baseline_mood': 'peaceful',
                    'stress_response': 'very_calm_and_reassuring',
                    'excitement_expression': 'gentle_enthusiasm',
                    'empathy_expression': 'high'
                },
                adaptation_rules={
                    'context_adaptation': 'high',
                    'trait_flexibility': 'very_high',
                    'consistency_priority': 'moderate'
                }
            )
        }
    
    def _get_default_personality_profile(self) -> PersonalityProfile:
        """Get default personality profile"""
        return self.personality_profiles['friendly_assistant']
    
    def _initialize_personality_state(self) -> PersonalityState:
        """Initialize current personality state"""
        return PersonalityState(
            current_traits=self.current_profile.base_traits.copy(),
            emotional_state='neutral',
            energy_level=0.7,
            social_comfort=0.8,
            interaction_style='friendly',
            consistency_score=1.0
        )
    
    def _initialize_trait_manager(self):
        """Initialize trait management components"""
        return {
            'trait_calculator': self._initialize_trait_calculator(),
            'trait_stabilizer': self._initialize_trait_stabilizer(),
            'trait_adapter': self._initialize_trait_adapter()
        }
    
    def _initialize_consistency_monitor(self):
        """Initialize consistency monitoring components"""
        return {
            'consistency_checker': self._initialize_consistency_checker(),
            'personality_validator': self._initialize_personality_validator(),
            'trait_tracker': self._initialize_trait_tracker()
        }
    
    def _initialize_trait_calculator(self):
        """Initialize trait calculation component"""
        return "trait_calculator_placeholder"
    
    def _initialize_trait_stabilizer(self):
        """Initialize trait stabilization component"""
        return "trait_stabilizer_placeholder"
    
    def _initialize_trait_adapter(self):
        """Initialize trait adaptation component"""
        return "trait_adapter_placeholder"
    
    def _initialize_consistency_checker(self):
        """Initialize consistency checking component"""
        return "consistency_checker_placeholder"
    
    def _initialize_personality_validator(self):
        """Initialize personality validation component"""
        return "personality_validator_placeholder"
    
    def _initialize_trait_tracker(self):
        """Initialize trait tracking component"""
        return "trait_tracker_placeholder"
    
    def apply_personality(self,
                         behavior: Dict[str, any],
                         social_context: Dict,
                         robot_state: Dict) -> PersonalityAdaptation:
        """
        Apply personality traits to behavior
        
        Args:
            behavior: Original behavior specification
            social_context: Social context information
            robot_state: Current robot state
            
        Returns:
            PersonalityAdaptation with personality-adjusted behavior
        """
        try:
            # Update personality state based on context
            self._update_personality_state(social_context, robot_state)
            
            # Calculate trait expression for current context
            trait_expression = self._calculate_trait_expression(social_context)
            
            # Apply personality to behavior
            adapted_behavior = self._apply_personality_to_behavior(
                behavior, trait_expression, social_context
            )
            
            # Validate personality consistency
            consistency_score = self._validate_personality_consistency(
                adapted_behavior, trait_expression
            )
            
            # Calculate adaptation confidence
            adaptation_confidence = self._calculate_adaptation_confidence(
                adapted_behavior, trait_expression, consistency_score
            )
            
            # Generate personality notes
            personality_notes = self._generate_personality_notes(
                adapted_behavior, trait_expression, social_context
            )
            
            # Create adaptation result
            adaptation = PersonalityAdaptation(
                adapted_behavior=adapted_behavior,
                personality_consistency=consistency_score,
                trait_expression=trait_expression,
                adaptation_confidence=adaptation_confidence,
                personality_notes=personality_notes
            )
            
            # Update personality history
            self._update_personality_history(adaptation)
            
            logger.debug(f"Personality applied with consistency {consistency_score:.2f}")
            return adaptation
            
        except Exception as e:
            logger.error(f"Error applying personality: {e}")
            return self._get_default_personality_adaptation(behavior)
    
    def _update_personality_state(self, social_context: Dict, robot_state: Dict):
        """Update current personality state based on context"""
        # Update emotional state
        if 'emotional_context' in social_context:
            self.personality_state.emotional_state = social_context['emotional_context']
        
        # Update energy level
        if 'energy_level' in robot_state:
            self.personality_state.energy_level = robot_state['energy_level']
        
        # Update social comfort
        if 'social_comfort' in robot_state:
            self.personality_state.social_comfort = robot_state['social_comfort']
        
        # Update interaction style based on context
        self.personality_state.interaction_style = self._determine_interaction_style(
            social_context, robot_state
        )
    
    def _determine_interaction_style(self, social_context: Dict, robot_state: Dict) -> str:
        """Determine appropriate interaction style for current context"""
        relationship = social_context.get('relationship', 'neutral')
        formality = social_context.get('formality_level', 0.5)
        
        if relationship == 'formal' or formality > 0.7:
            return 'professional'
        elif relationship == 'friendly':
            return 'friendly'
        elif relationship == 'supportive':
            return 'encouraging'
        else:
            return self.current_profile.personality_style.value
    
    def _calculate_trait_expression(self, social_context: Dict) -> Dict[PersonalityTrait, float]:
        """Calculate trait expression for current context"""
        trait_expression = {}
        
        for trait, base_value in self.current_profile.base_traits.items():
            # Apply context-based adjustments
            adjusted_value = self._adjust_trait_for_context(trait, base_value, social_context)
            
            # Apply personality state adjustments
            adjusted_value = self._adjust_trait_for_state(trait, adjusted_value)
            
            # Ensure trait stability
            adjusted_value = self._stabilize_trait(trait, adjusted_value)
            
            trait_expression[trait] = max(0.0, min(1.0, adjusted_value))
        
        return trait_expression
    
    def _adjust_trait_for_context(self,
                                 trait: PersonalityTrait,
                                 base_value: float,
                                 social_context: Dict) -> float:
        """Adjust trait expression based on social context"""
        relationship = social_context.get('relationship', 'neutral')
        formality = social_context.get('formality_level', 0.5)
        privacy = social_context.get('privacy_level', 'public')
        
        adjustment = 0.0
        
        # Adjust based on relationship
        if relationship == 'formal':
            if trait == PersonalityTrait.PROFESSIONALISM:
                adjustment += 0.2
            elif trait == PersonalityTrait.HUMOR:
                adjustment -= 0.2
        elif relationship == 'friendly':
            if trait == PersonalityTrait.EXTROVERSION:
                adjustment += 0.1
            elif trait == PersonalityTrait.HUMOR:
                adjustment += 0.1
        
        # Adjust based on formality
        if formality > 0.7:
            if trait == PersonalityTrait.PROFESSIONALISM:
                adjustment += 0.1
            elif trait == PersonalityTrait.INTROVERSION:
                adjustment += 0.1
        elif formality < 0.3:
            if trait == PersonalityTrait.EXTROVERSION:
                adjustment += 0.1
            elif trait == PersonalityTrait.HUMOR:
                adjustment += 0.1
        
        # Adjust based on privacy
        if privacy == 'private':
            if trait == PersonalityTrait.EMPATHY:
                adjustment += 0.1
            elif trait == PersonalityTrait.INTROVERSION:
                adjustment += 0.1
        
        return base_value + adjustment
    
    def _adjust_trait_for_state(self,
                               trait: PersonalityTrait,
                               current_value: float) -> float:
        """Adjust trait expression based on personality state"""
        emotional_state = self.personality_state.emotional_state
        energy_level = self.personality_state.energy_level
        social_comfort = self.personality_state.social_comfort
        
        adjustment = 0.0
        
        # Adjust based on emotional state
        if emotional_state == 'excited':
            if trait == PersonalityTrait.EXTROVERSION:
                adjustment += 0.1
            elif trait == PersonalityTrait.ENTHUSIASM:
                adjustment += 0.2
        elif emotional_state == 'stressed':
            if trait == PersonalityTrait.CALM:
                adjustment += 0.2
            elif trait == PersonalityTrait.EMPATHY:
                adjustment += 0.1
        
        # Adjust based on energy level
        if energy_level > 0.8:
            if trait == PersonalityTrait.EXTROVERSION:
                adjustment += 0.1
        elif energy_level < 0.3:
            if trait == PersonalityTrait.INTROVERSION:
                adjustment += 0.1
        
        # Adjust based on social comfort
        if social_comfort < 0.5:
            if trait == PersonalityTrait.INTROVERSION:
                adjustment += 0.1
            elif trait == PersonalityTrait.EMPATHY:
                adjustment += 0.1
        
        return current_value + adjustment
    
    def _stabilize_trait(self, trait: PersonalityTrait, current_value: float) -> float:
        """Stabilize trait to maintain personality consistency"""
        base_value = self.current_profile.base_traits.get(trait, 0.5)
        stability_factor = self.config.get('trait_stability_factor', 0.9)
        
        # Apply stability factor to prevent rapid changes
        stabilized_value = (current_value * (1 - stability_factor) + 
                          base_value * stability_factor)
        
        return stabilized_value
    
    def _apply_personality_to_behavior(self,
                                     behavior: Dict[str, any],
                                     trait_expression: Dict[PersonalityTrait, float],
                                     social_context: Dict) -> Dict[str, any]:
        """Apply personality traits to behavior"""
        adapted_behavior = behavior.copy()
        
        # Apply personality to verbal communication
        if 'verbal_response' in adapted_behavior:
            adapted_behavior['verbal_response'] = self._apply_personality_to_verbal(
                adapted_behavior['verbal_response'], trait_expression
            )
        
        # Apply personality to tone
        if 'tone' in adapted_behavior:
            adapted_behavior['tone'] = self._apply_personality_to_tone(
                adapted_behavior['tone'], trait_expression
            )
        
        # Apply personality to speech rate
        if 'speech_rate' in adapted_behavior:
            adapted_behavior['speech_rate'] = self._apply_personality_to_speech_rate(
                adapted_behavior['speech_rate'], trait_expression
            )
        
        # Apply personality to volume
        if 'volume' in adapted_behavior:
            adapted_behavior['volume'] = self._apply_personality_to_volume(
                adapted_behavior['volume'], trait_expression
            )
        
        # Apply personality to gestures
        if 'gesture_type' in adapted_behavior:
            adapted_behavior['gesture_type'] = self._apply_personality_to_gestures(
                adapted_behavior['gesture_type'], trait_expression
            )
        
        # Apply personality to facial expressions
        if 'facial_expression' in adapted_behavior:
            adapted_behavior['facial_expression'] = self._apply_personality_to_facial(
                adapted_behavior['facial_expression'], trait_expression
            )
        
        return adapted_behavior
    
    def _apply_personality_to_verbal(self,
                                   verbal_response: str,
                                   trait_expression: Dict[PersonalityTrait, float]) -> str:
        """Apply personality to verbal communication"""
        # Apply extroversion
        if trait_expression.get(PersonalityTrait.EXTROVERSION, 0.5) > 0.7:
            if not verbal_response.endswith('!'):
                verbal_response += '!'
        
        # Apply humor
        if trait_expression.get(PersonalityTrait.HUMOR, 0.5) > 0.6:
            # Add light humor (placeholder)
            pass
        
        # Apply empathy
        if trait_expression.get(PersonalityTrait.EMPATHY, 0.5) > 0.7:
            if 'I understand' not in verbal_response:
                verbal_response = f"I understand. {verbal_response}"
        
        return verbal_response
    
    def _apply_personality_to_tone(self,
                                  tone: str,
                                  trait_expression: Dict[PersonalityTrait, float]) -> str:
        """Apply personality to tone"""
        if trait_expression.get(PersonalityTrait.EXTROVERSION, 0.5) > 0.7:
            return 'enthusiastic'
        elif trait_expression.get(PersonalityTrait.INTROVERSION, 0.5) > 0.7:
            return 'calm'
        elif trait_expression.get(PersonalityTrait.PROFESSIONALISM, 0.5) > 0.7:
            return 'professional'
        else:
            return tone
    
    def _apply_personality_to_speech_rate(self,
                                         speech_rate: float,
                                         trait_expression: Dict[PersonalityTrait, float]) -> float:
        """Apply personality to speech rate"""
        if trait_expression.get(PersonalityTrait.EXTROVERSION, 0.5) > 0.7:
            return speech_rate * 1.1
        elif trait_expression.get(PersonalityTrait.INTROVERSION, 0.5) > 0.7:
            return speech_rate * 0.9
        else:
            return speech_rate
    
    def _apply_personality_to_volume(self,
                                   volume: float,
                                   trait_expression: Dict[PersonalityTrait, float]) -> float:
        """Apply personality to volume"""
        if trait_expression.get(PersonalityTrait.EXTROVERSION, 0.5) > 0.7:
            return min(1.0, volume * 1.1)
        elif trait_expression.get(PersonalityTrait.INTROVERSION, 0.5) > 0.7:
            return max(0.3, volume * 0.9)
        else:
            return volume
    
    def _apply_personality_to_gestures(self,
                                     gesture_type: str,
                                     trait_expression: Dict[PersonalityTrait, float]) -> str:
        """Apply personality to gestures"""
        if trait_expression.get(PersonalityTrait.EXTROVERSION, 0.5) > 0.7:
            if gesture_type == 'neutral_gesture':
                return 'enthusiastic_gesture'
        elif trait_expression.get(PersonalityTrait.INTROVERSION, 0.5) > 0.7:
            if gesture_type == 'enthusiastic_gesture':
                return 'subtle_gesture'
        
        return gesture_type
    
    def _apply_personality_to_facial(self,
                                   facial_expression: str,
                                   trait_expression: Dict[PersonalityTrait, float]) -> str:
        """Apply personality to facial expressions"""
        if trait_expression.get(PersonalityTrait.EXTROVERSION, 0.5) > 0.7:
            if facial_expression == 'neutral_expression':
                return 'friendly_expression'
        elif trait_expression.get(PersonalityTrait.INTROVERSION, 0.5) > 0.7:
            if facial_expression == 'enthusiastic_expression':
                return 'calm_expression'
        
        return facial_expression
    
    def _validate_personality_consistency(self,
                                        adapted_behavior: Dict[str, any],
                                        trait_expression: Dict[PersonalityTrait, float]) -> float:
        """Validate personality consistency"""
        consistency_score = 1.0
        
        # Check trait consistency with base profile
        for trait, expression in trait_expression.items():
            base_value = self.current_profile.base_traits.get(trait, 0.5)
            deviation = abs(expression - base_value)
            
            if deviation > 0.3:
                consistency_score -= 0.1
        
        # Check behavioral consistency
        if self._is_behavior_consistent_with_personality(adapted_behavior, trait_expression):
            consistency_score += 0.1
        
        return max(0.0, min(1.0, consistency_score))
    
    def _is_behavior_consistent_with_personality(self,
                                               behavior: Dict[str, any],
                                               trait_expression: Dict[PersonalityTrait, float]) -> bool:
        """Check if behavior is consistent with personality traits"""
        # Placeholder for behavior-personality consistency check
        return True
    
    def _calculate_adaptation_confidence(self,
                                       adapted_behavior: Dict[str, any],
                                       trait_expression: Dict[PersonalityTrait, float],
                                       consistency_score: float) -> float:
        """Calculate confidence in personality adaptation"""
        confidence = 0.7
        
        # Adjust based on consistency
        confidence += consistency_score * 0.2
        
        # Adjust based on trait expression clarity
        trait_clarity = self._calculate_trait_clarity(trait_expression)
        confidence += trait_clarity * 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_trait_clarity(self, trait_expression: Dict[PersonalityTrait, float]) -> float:
        """Calculate clarity of trait expression"""
        # Calculate variance in trait expression
        values = list(trait_expression.values())
        variance = np.var(values) if len(values) > 1 else 0.0
        
        # Higher variance indicates clearer trait expression
        return min(1.0, variance * 5.0)
    
    def _generate_personality_notes(self,
                                  adapted_behavior: Dict[str, any],
                                  trait_expression: Dict[PersonalityTrait, float],
                                  social_context: Dict) -> List[str]:
        """Generate notes about personality application"""
        notes = []
        
        # Note dominant traits
        dominant_traits = [trait for trait, value in trait_expression.items() if value > 0.7]
        if dominant_traits:
            notes.append(f"Dominant traits: {', '.join([trait.value for trait in dominant_traits])}")
        
        # Note personality style
        notes.append(f"Personality style: {self.current_profile.personality_style.value}")
        
        # Note context adaptation
        if social_context.get('relationship') != 'neutral':
            notes.append(f"Adapted for {social_context['relationship']} relationship")
        
        return notes
    
    def _update_personality_history(self, adaptation: PersonalityAdaptation):
        """Update personality history"""
        self.personality_history.append(adaptation)
        
        # Keep only recent history
        max_history = self.config.get('max_personality_history', 100)
        if len(self.personality_history) > max_history:
            self.personality_history.pop(0)
    
    def _get_default_personality_adaptation(self, behavior: Dict[str, any]) -> PersonalityAdaptation:
        """Get default personality adaptation when adaptation fails"""
        return PersonalityAdaptation(
            adapted_behavior=behavior,
            personality_consistency=0.7,
            trait_expression=self.current_profile.base_traits,
            adaptation_confidence=0.5,
            personality_notes=["Default personality applied"]
        )
    
    def set_personality_profile(self, profile_id: str):
        """Set personality profile"""
        if profile_id in self.personality_profiles:
            self.current_profile = self.personality_profiles[profile_id]
            self.personality_state.current_traits = self.current_profile.base_traits.copy()
            logger.info(f"Personality profile set to: {profile_id}")
        else:
            logger.warning(f"Personality profile not found: {profile_id}")
    
    def get_personality_profile(self) -> PersonalityProfile:
        """Get current personality profile"""
        return self.current_profile
    
    def get_personality_state(self) -> PersonalityState:
        """Get current personality state"""
        return self.personality_state 