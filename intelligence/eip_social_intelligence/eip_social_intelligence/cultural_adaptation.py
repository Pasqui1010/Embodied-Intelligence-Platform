"""
Cultural Adaptation Module

This module provides cultural adaptation capabilities for human-robot
interaction, enabling the robot to adapt its behavior to different
cultural contexts while maintaining sensitivity and appropriateness.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CulturalDimension(Enum):
    """Enumeration of cultural dimensions"""
    POWER_DISTANCE = "power_distance"
    INDIVIDUALISM_COLLECTIVISM = "individualism_collectivism"
    MASCULINITY_FEMININITY = "masculinity_femininity"
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance"
    LONG_TERM_ORIENTATION = "long_term_orientation"
    INDULGENCE_RESTRAINT = "indulgence_restraint"


class CommunicationStyle(Enum):
    """Enumeration of communication styles"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    FORMAL = "formal"
    INFORMAL = "informal"
    HIGH_CONTEXT = "high_context"
    LOW_CONTEXT = "low_context"


@dataclass
class CulturalProfile:
    """Data class for cultural profile information"""
    culture_name: str
    region: str
    language: str
    cultural_dimensions: Dict[CulturalDimension, float]
    communication_style: CommunicationStyle
    social_norms: Dict[str, any]
    taboos: List[str]
    preferred_interaction_distance: float
    eye_contact_preference: float
    gesture_sensitivity: float
    formality_level: float


@dataclass
class CulturalAdaptation:
    """Data class for cultural adaptation results"""
    adapted_behavior: Dict[str, any]
    cultural_sensitivity_score: float
    appropriateness_score: float
    adaptation_confidence: float
    cultural_notes: List[str]
    risk_factors: List[str]


class CulturalAdaptationEngine:
    """
    Cultural adaptation engine for human-robot interaction
    
    This class provides comprehensive cultural adaptation capabilities
    to ensure appropriate behavior across different cultural contexts.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the cultural adaptation engine
        
        Args:
            config: Configuration dictionary for cultural adaptation
        """
        self.config = config or self._get_default_config()
        self.cultural_database = self._load_cultural_database()
        self.adaptation_rules = self._load_adaptation_rules()
        self.sensitivity_checker = self._initialize_sensitivity_checker()
        self.cultural_analyzer = self._initialize_cultural_analyzer()
        self.adaptation_history = []
        
        logger.info("Cultural adaptation engine initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for cultural adaptation"""
        return {
            'sensitivity_threshold': 0.8,
            'adaptation_confidence_threshold': 0.7,
            'max_adaptation_history': 50,
            'cultural_learning_rate': 0.1,
            'stereotype_avoidance_weight': 0.3,
            'cultural_sensitivity_weight': 0.4,
            'appropriateness_weight': 0.3
        }
    
    def _load_cultural_database(self) -> Dict[str, CulturalProfile]:
        """Load cultural database with cultural profiles"""
        return {
            'western': CulturalProfile(
                culture_name='Western',
                region='North America, Europe, Australia',
                language='English',
                cultural_dimensions={
                    CulturalDimension.POWER_DISTANCE: 0.3,
                    CulturalDimension.INDIVIDUALISM_COLLECTIVISM: 0.8,
                    CulturalDimension.MASCULINITY_FEMININITY: 0.6,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.4,
                    CulturalDimension.LONG_TERM_ORIENTATION: 0.3,
                    CulturalDimension.INDULGENCE_RESTRAINT: 0.7
                },
                communication_style=CommunicationStyle.DIRECT,
                social_norms={
                    'personal_space': 1.2,
                    'eye_contact': 0.8,
                    'formality': 0.4,
                    'gesture_acceptance': 0.7
                },
                taboos=['personal_questions', 'age_questions'],
                preferred_interaction_distance=1.2,
                eye_contact_preference=0.8,
                gesture_sensitivity=0.3,
                formality_level=0.4
            ),
            'eastern': CulturalProfile(
                culture_name='Eastern',
                region='Asia',
                language='Various',
                cultural_dimensions={
                    CulturalDimension.POWER_DISTANCE: 0.7,
                    CulturalDimension.INDIVIDUALISM_COLLECTIVISM: 0.3,
                    CulturalDimension.MASCULINITY_FEMININITY: 0.5,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.6,
                    CulturalDimension.LONG_TERM_ORIENTATION: 0.8,
                    CulturalDimension.INDULGENCE_RESTRAINT: 0.4
                },
                communication_style=CommunicationStyle.INDIRECT,
                social_norms={
                    'personal_space': 0.8,
                    'eye_contact': 0.4,
                    'formality': 0.8,
                    'gesture_acceptance': 0.4
                },
                taboos=['direct_refusal', 'pointing_feet', 'touching_head'],
                preferred_interaction_distance=0.8,
                eye_contact_preference=0.4,
                gesture_sensitivity=0.7,
                formality_level=0.8
            ),
            'middle_eastern': CulturalProfile(
                culture_name='Middle Eastern',
                region='Middle East, North Africa',
                language='Arabic',
                cultural_dimensions={
                    CulturalDimension.POWER_DISTANCE: 0.8,
                    CulturalDimension.INDIVIDUALISM_COLLECTIVISM: 0.4,
                    CulturalDimension.MASCULINITY_FEMININITY: 0.7,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.7,
                    CulturalDimension.LONG_TERM_ORIENTATION: 0.5,
                    CulturalDimension.INDULGENCE_RESTRAINT: 0.3
                },
                communication_style=CommunicationStyle.HIGH_CONTEXT,
                social_norms={
                    'personal_space': 0.6,
                    'eye_contact': 0.6,
                    'formality': 0.7,
                    'gesture_acceptance': 0.5
                },
                taboos=['left_hand_use', 'showing_soles', 'direct_criticism'],
                preferred_interaction_distance=0.6,
                eye_contact_preference=0.6,
                gesture_sensitivity=0.6,
                formality_level=0.7
            ),
            'latin_american': CulturalProfile(
                culture_name='Latin American',
                region='Latin America',
                language='Spanish, Portuguese',
                cultural_dimensions={
                    CulturalDimension.POWER_DISTANCE: 0.6,
                    CulturalDimension.INDIVIDUALISM_COLLECTIVISM: 0.4,
                    CulturalDimension.MASCULINITY_FEMININITY: 0.6,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.8,
                    CulturalDimension.LONG_TERM_ORIENTATION: 0.4,
                    CulturalDimension.INDULGENCE_RESTRAINT: 0.6
                },
                communication_style=CommunicationStyle.HIGH_CONTEXT,
                social_norms={
                    'personal_space': 0.7,
                    'eye_contact': 0.7,
                    'formality': 0.5,
                    'gesture_acceptance': 0.8
                },
                taboos=['personal_space_violation', 'formal_address'],
                preferred_interaction_distance=0.7,
                eye_contact_preference=0.7,
                gesture_sensitivity=0.2,
                formality_level=0.5
            )
        }
    
    def _load_adaptation_rules(self) -> Dict[str, Dict]:
        """Load adaptation rules for different cultural contexts"""
        return {
            'communication_style': {
                'direct': {
                    'western': 1.0,
                    'eastern': 0.3,
                    'middle_eastern': 0.4,
                    'latin_american': 0.6
                },
                'indirect': {
                    'western': 0.3,
                    'eastern': 1.0,
                    'middle_eastern': 0.8,
                    'latin_american': 0.7
                },
                'formal': {
                    'western': 0.4,
                    'eastern': 0.8,
                    'middle_eastern': 0.7,
                    'latin_american': 0.5
                },
                'informal': {
                    'western': 0.8,
                    'eastern': 0.3,
                    'middle_eastern': 0.4,
                    'latin_american': 0.6
                }
            },
            'interaction_distance': {
                'western': 1.2,
                'eastern': 0.8,
                'middle_eastern': 0.6,
                'latin_american': 0.7
            },
            'eye_contact': {
                'western': 0.8,
                'eastern': 0.4,
                'middle_eastern': 0.6,
                'latin_american': 0.7
            },
            'gesture_sensitivity': {
                'western': 0.3,
                'eastern': 0.7,
                'middle_eastern': 0.6,
                'latin_american': 0.2
            },
            'formality': {
                'western': 0.4,
                'eastern': 0.8,
                'middle_eastern': 0.7,
                'latin_american': 0.5
            }
        }
    
    def _initialize_sensitivity_checker(self):
        """Initialize cultural sensitivity checker"""
        return {
            'stereotype_detector': self._initialize_stereotype_detector(),
            'appropriateness_validator': self._initialize_appropriateness_validator(),
            'cultural_risk_assessor': self._initialize_risk_assessor()
        }
    
    def _initialize_cultural_analyzer(self):
        """Initialize cultural context analyzer"""
        return {
            'context_analyzer': self._initialize_context_analyzer(),
            'norm_validator': self._initialize_norm_validator(),
            'cultural_learner': self._initialize_cultural_learner()
        }
    
    def _initialize_stereotype_detector(self):
        """Initialize stereotype detection"""
        return "stereotype_detector_placeholder"
    
    def _initialize_appropriateness_validator(self):
        """Initialize appropriateness validation"""
        return "appropriateness_validator_placeholder"
    
    def _initialize_risk_assessor(self):
        """Initialize cultural risk assessment"""
        return "risk_assessor_placeholder"
    
    def _initialize_context_analyzer(self):
        """Initialize cultural context analysis"""
        return "context_analyzer_placeholder"
    
    def _initialize_norm_validator(self):
        """Initialize social norm validation"""
        return "norm_validator_placeholder"
    
    def _initialize_cultural_learner(self):
        """Initialize cultural learning component"""
        return "cultural_learner_placeholder"
    
    def adapt_behavior(self,
                      behavior: Dict[str, any],
                      cultural_context: str,
                      social_context: Dict) -> CulturalAdaptation:
        """
        Adapt behavior to cultural context
        
        Args:
            behavior: Original behavior specification
            cultural_context: Target cultural context
            social_context: Social context information
            
        Returns:
            CulturalAdaptation with adapted behavior and validation
        """
        try:
            # Get cultural profile
            cultural_profile = self._get_cultural_profile(cultural_context)
            
            # Analyze cultural requirements
            cultural_requirements = self._analyze_cultural_requirements(
                behavior, cultural_profile, social_context
            )
            
            # Generate cultural adaptations
            adapted_behavior = self._generate_cultural_adaptations(
                behavior, cultural_requirements, cultural_profile
            )
            
            # Validate cultural sensitivity
            sensitivity_score = self._validate_cultural_sensitivity(
                adapted_behavior, cultural_profile
            )
            
            # Check appropriateness
            appropriateness_score = self._validate_cultural_appropriateness(
                adapted_behavior, cultural_profile, social_context
            )
            
            # Calculate adaptation confidence
            adaptation_confidence = self._calculate_adaptation_confidence(
                adapted_behavior, cultural_profile, cultural_requirements
            )
            
            # Generate cultural notes and risk factors
            cultural_notes = self._generate_cultural_notes(
                adapted_behavior, cultural_profile
            )
            risk_factors = self._identify_risk_factors(
                adapted_behavior, cultural_profile
            )
            
            # Create adaptation result
            adaptation = CulturalAdaptation(
                adapted_behavior=adapted_behavior,
                cultural_sensitivity_score=sensitivity_score,
                appropriateness_score=appropriateness_score,
                adaptation_confidence=adaptation_confidence,
                cultural_notes=cultural_notes,
                risk_factors=risk_factors
            )
            
            # Update adaptation history
            self._update_adaptation_history(adaptation)
            
            logger.debug(f"Cultural adaptation completed for {cultural_context}")
            return adaptation
            
        except Exception as e:
            logger.error(f"Error in cultural adaptation: {e}")
            return self._get_default_adaptation(behavior)
    
    def _get_cultural_profile(self, cultural_context: str) -> CulturalProfile:
        """Get cultural profile for given context"""
        # Map cultural context to profile
        context_mapping = {
            'western': 'western',
            'eastern': 'eastern',
            'asian': 'eastern',
            'middle_eastern': 'middle_eastern',
            'arabic': 'middle_eastern',
            'latin_american': 'latin_american',
            'hispanic': 'latin_american'
        }
        
        profile_key = context_mapping.get(cultural_context.lower(), 'western')
        return self.cultural_database.get(profile_key, self.cultural_database['western'])
    
    def _analyze_cultural_requirements(self,
                                     behavior: Dict[str, any],
                                     cultural_profile: CulturalProfile,
                                     social_context: Dict) -> Dict[str, any]:
        """Analyze cultural requirements for behavior adaptation"""
        requirements = {
            'communication_style': self._determine_communication_style(cultural_profile),
            'interaction_distance': cultural_profile.preferred_interaction_distance,
            'eye_contact_level': cultural_profile.eye_contact_preference,
            'gesture_sensitivity': cultural_profile.gesture_sensitivity,
            'formality_level': cultural_profile.formality_level,
            'taboo_avoidance': cultural_profile.taboos,
            'social_norms': cultural_profile.social_norms
        }
        
        # Adjust based on social context
        if social_context.get('relationship') == 'formal':
            requirements['formality_level'] = min(1.0, requirements['formality_level'] + 0.2)
        
        if social_context.get('privacy_level') == 'public':
            requirements['interaction_distance'] = max(1.0, requirements['interaction_distance'])
        
        return requirements
    
    def _determine_communication_style(self, cultural_profile: CulturalProfile) -> CommunicationStyle:
        """Determine appropriate communication style for culture"""
        individualism = cultural_profile.cultural_dimensions.get(
            CulturalDimension.INDIVIDUALISM_COLLECTIVISM, 0.5
        )
        
        if individualism > 0.6:
            return CommunicationStyle.DIRECT
        else:
            return CommunicationStyle.INDIRECT
    
    def _generate_cultural_adaptations(self,
                                     behavior: Dict[str, any],
                                     requirements: Dict[str, any],
                                     cultural_profile: CulturalProfile) -> Dict[str, any]:
        """Generate culturally adapted behavior"""
        adapted_behavior = behavior.copy()
        
        # Adapt communication style
        if 'verbal_response' in adapted_behavior:
            adapted_behavior['verbal_response'] = self._adapt_verbal_communication(
                adapted_behavior['verbal_response'], requirements, cultural_profile
            )
        
        # Adapt interaction distance
        if 'proxemic_behavior' in adapted_behavior:
            adapted_behavior['proxemic_behavior'] = self._adapt_proxemic_behavior(
                adapted_behavior['proxemic_behavior'], requirements, cultural_profile
            )
        
        # Adapt eye contact
        if 'eye_contact' in adapted_behavior:
            adapted_behavior['eye_contact'] = self._adapt_eye_contact(
                adapted_behavior['eye_contact'], requirements, cultural_profile
            )
        
        # Adapt gestures
        if 'gesture_type' in adapted_behavior:
            adapted_behavior['gesture_type'] = self._adapt_gestures(
                adapted_behavior['gesture_type'], requirements, cultural_profile
            )
        
        # Adapt formality
        adapted_behavior['formality_level'] = requirements['formality_level']
        
        return adapted_behavior
    
    def _adapt_verbal_communication(self,
                                   verbal_response: str,
                                   requirements: Dict[str, any],
                                   cultural_profile: CulturalProfile) -> str:
        """Adapt verbal communication to cultural context"""
        communication_style = requirements['communication_style']
        formality_level = requirements['formality_level']
        
        # Adapt formality
        if formality_level > 0.7:
            # Make more formal
            if "Hi" in verbal_response:
                verbal_response = verbal_response.replace("Hi", "Hello")
            if "!" in verbal_response:
                verbal_response = verbal_response.replace("!", ".")
        
        # Adapt directness
        if communication_style == CommunicationStyle.INDIRECT:
            # Make more indirect
            if "I want" in verbal_response:
                verbal_response = verbal_response.replace("I want", "I would like")
            if "You should" in verbal_response:
                verbal_response = verbal_response.replace("You should", "Perhaps you might consider")
        
        # Avoid cultural taboos
        for taboo in cultural_profile.taboos:
            if taboo in verbal_response.lower():
                verbal_response = self._replace_taboo_content(verbal_response, taboo)
        
        return verbal_response
    
    def _adapt_proxemic_behavior(self,
                                proxemic_behavior: str,
                                requirements: Dict[str, any],
                                cultural_profile: CulturalProfile) -> str:
        """Adapt proxemic behavior to cultural context"""
        preferred_distance = requirements['interaction_distance']
        
        if preferred_distance < 0.8:
            # Cultures with closer interaction distance
            if 'maintain_distance' in proxemic_behavior:
                proxemic_behavior = proxemic_behavior.replace('maintain_distance', 'approach_closer')
            elif 'standard_distance' in proxemic_behavior:
                proxemic_behavior = proxemic_behavior.replace('standard_distance', 'closer_distance')
        else:
            # Cultures with larger interaction distance
            if 'approach_closer' in proxemic_behavior:
                proxemic_behavior = proxemic_behavior.replace('approach_closer', 'maintain_distance')
            elif 'closer_distance' in proxemic_behavior:
                proxemic_behavior = proxemic_behavior.replace('closer_distance', 'standard_distance')
        
        return proxemic_behavior
    
    def _adapt_eye_contact(self,
                          eye_contact: float,
                          requirements: Dict[str, any],
                          cultural_profile: CulturalProfile) -> float:
        """Adapt eye contact level to cultural context"""
        preferred_eye_contact = requirements['eye_contact_level']
        
        # Adjust eye contact based on cultural preference
        if preferred_eye_contact < 0.5:
            # Cultures with lower eye contact preference
            return max(0.1, eye_contact * 0.7)
        else:
            # Cultures with higher eye contact preference
            return min(1.0, eye_contact * 1.2)
    
    def _adapt_gestures(self,
                       gesture_type: str,
                       requirements: Dict[str, any],
                       cultural_profile: CulturalProfile) -> str:
        """Adapt gestures to cultural context"""
        gesture_sensitivity = requirements['gesture_sensitivity']
        
        # Avoid sensitive gestures in high-sensitivity cultures
        if gesture_sensitivity > 0.6:
            sensitive_gestures = ['pointing', 'thumbs_up', 'ok_sign']
            if gesture_type in sensitive_gestures:
                return 'neutral_gesture'
        
        return gesture_type
    
    def _replace_taboo_content(self, text: str, taboo: str) -> str:
        """Replace taboo content with appropriate alternatives"""
        taboo_replacements = {
            'personal_questions': 'general_questions',
            'age_questions': 'experience_questions',
            'direct_refusal': 'polite_decline',
            'pointing_feet': 'pointing_hand',
            'touching_head': 'respectful_gesture',
            'left_hand_use': 'right_hand_use',
            'showing_soles': 'respectful_posture',
            'direct_criticism': 'constructive_feedback'
        }
        
        replacement = taboo_replacements.get(taboo, 'appropriate_content')
        return text.replace(taboo, replacement)
    
    def _validate_cultural_sensitivity(self,
                                     adapted_behavior: Dict[str, any],
                                     cultural_profile: CulturalProfile) -> float:
        """Validate cultural sensitivity of adapted behavior"""
        sensitivity_score = 1.0
        
        # Check for taboo violations
        for taboo in cultural_profile.taboos:
            if taboo in str(adapted_behavior).lower():
                sensitivity_score -= 0.2
        
        # Check for stereotype reinforcement
        if self._detect_stereotypes(adapted_behavior, cultural_profile):
            sensitivity_score -= 0.3
        
        # Check for cultural appropriation
        if self._detect_cultural_appropriation(adapted_behavior, cultural_profile):
            sensitivity_score -= 0.2
        
        return max(0.0, min(1.0, sensitivity_score))
    
    def _validate_cultural_appropriateness(self,
                                         adapted_behavior: Dict[str, any],
                                         cultural_profile: CulturalProfile,
                                         social_context: Dict) -> float:
        """Validate cultural appropriateness of adapted behavior"""
        appropriateness_score = 0.8
        
        # Check formality appropriateness
        expected_formality = cultural_profile.formality_level
        actual_formality = adapted_behavior.get('formality_level', 0.5)
        formality_diff = abs(expected_formality - actual_formality)
        appropriateness_score -= formality_diff * 0.3
        
        # Check communication style appropriateness
        if self._is_communication_style_appropriate(adapted_behavior, cultural_profile):
            appropriateness_score += 0.1
        
        # Check social norm compliance
        if self._complies_with_social_norms(adapted_behavior, cultural_profile):
            appropriateness_score += 0.1
        
        return max(0.0, min(1.0, appropriateness_score))
    
    def _detect_stereotypes(self,
                           adapted_behavior: Dict[str, any],
                           cultural_profile: CulturalProfile) -> bool:
        """Detect stereotype reinforcement in adapted behavior"""
        # Placeholder for stereotype detection
        # In real implementation, this would use more sophisticated detection
        return False
    
    def _detect_cultural_appropriation(self,
                                     adapted_behavior: Dict[str, any],
                                     cultural_profile: CulturalProfile) -> bool:
        """Detect cultural appropriation in adapted behavior"""
        # Placeholder for cultural appropriation detection
        # In real implementation, this would use more sophisticated detection
        return False
    
    def _is_communication_style_appropriate(self,
                                          adapted_behavior: Dict[str, any],
                                          cultural_profile: CulturalProfile) -> bool:
        """Check if communication style is appropriate for culture"""
        # Placeholder for communication style validation
        return True
    
    def _complies_with_social_norms(self,
                                   adapted_behavior: Dict[str, any],
                                   cultural_profile: CulturalProfile) -> bool:
        """Check if behavior complies with social norms"""
        # Placeholder for social norm validation
        return True
    
    def _calculate_adaptation_confidence(self,
                                       adapted_behavior: Dict[str, any],
                                       cultural_profile: CulturalProfile,
                                       requirements: Dict[str, any]) -> float:
        """Calculate confidence in cultural adaptation"""
        confidence = 0.7
        
        # Adjust based on cultural profile completeness
        if cultural_profile.cultural_dimensions:
            confidence += 0.1
        
        # Adjust based on requirements clarity
        if requirements:
            confidence += 0.1
        
        # Adjust based on adaptation quality
        if self._is_adaptation_high_quality(adapted_behavior, requirements):
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _is_adaptation_high_quality(self,
                                   adapted_behavior: Dict[str, any],
                                   requirements: Dict[str, any]) -> bool:
        """Check if adaptation is high quality"""
        # Placeholder for adaptation quality assessment
        return True
    
    def _generate_cultural_notes(self,
                                adapted_behavior: Dict[str, any],
                                cultural_profile: CulturalProfile) -> List[str]:
        """Generate cultural notes about adaptation"""
        notes = []
        
        # Add notes about communication style
        if cultural_profile.communication_style == CommunicationStyle.INDIRECT:
            notes.append("Adapted to indirect communication style")
        
        # Add notes about formality
        if cultural_profile.formality_level > 0.7:
            notes.append("Increased formality for cultural appropriateness")
        
        # Add notes about interaction distance
        if cultural_profile.preferred_interaction_distance < 1.0:
            notes.append("Adjusted for closer interaction distance preference")
        
        return notes
    
    def _identify_risk_factors(self,
                              adapted_behavior: Dict[str, any],
                              cultural_profile: CulturalProfile) -> List[str]:
        """Identify potential cultural risk factors"""
        risk_factors = []
        
        # Check for high gesture sensitivity
        if cultural_profile.gesture_sensitivity > 0.6:
            risk_factors.append("High gesture sensitivity - monitor for inappropriate gestures")
        
        # Check for taboo content
        for taboo in cultural_profile.taboos:
            if taboo in str(adapted_behavior).lower():
                risk_factors.append(f"Potential taboo violation: {taboo}")
        
        # Check for formality mismatch
        if adapted_behavior.get('formality_level', 0.5) < 0.6 and cultural_profile.formality_level > 0.7:
            risk_factors.append("Formality level may be too low for cultural context")
        
        return risk_factors
    
    def _update_adaptation_history(self, adaptation: CulturalAdaptation):
        """Update adaptation history"""
        self.adaptation_history.append(adaptation)
        
        # Keep only recent history
        max_history = self.config.get('max_adaptation_history', 50)
        if len(self.adaptation_history) > max_history:
            self.adaptation_history.pop(0)
    
    def _get_default_adaptation(self, behavior: Dict[str, any]) -> CulturalAdaptation:
        """Get default adaptation when adaptation fails"""
        return CulturalAdaptation(
            adapted_behavior=behavior,
            cultural_sensitivity_score=0.7,
            appropriateness_score=0.6,
            adaptation_confidence=0.5,
            cultural_notes=["Default adaptation applied"],
            risk_factors=["Low confidence in cultural adaptation"]
        ) 