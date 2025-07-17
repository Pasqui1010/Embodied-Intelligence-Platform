"""
Social Intelligence for Cognitive Architecture

This module implements a social intelligence system that understands and responds
to social cues, manages human-robot interactions, and maintains appropriate
social behavior.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np


class SocialContext(Enum):
    """Types of social contexts"""
    ALONE = "alone"
    INTERACTING = "interacting"
    OBSERVING = "observing"
    GROUP = "group"
    FORMAL = "formal"
    INFORMAL = "informal"


class EmotionalState(Enum):
    """Human emotional states"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    NEUTRAL = "neutral"
    STRESSED = "stressed"
    CALM = "calm"
    EXCITED = "excited"


class InteractionType(Enum):
    """Types of social interactions"""
    VERBAL = "verbal"
    NONVERBAL = "nonverbal"
    GESTURAL = "gestural"
    PROXIMITY = "proximity"
    TOUCH = "touch"
    EYE_CONTACT = "eye_contact"


@dataclass
class SocialCue:
    """Represents a social cue from the environment"""
    cue_id: str
    cue_type: InteractionType
    source: str  # Person ID or 'environment'
    intensity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    location: Optional[Tuple[float, float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanState:
    """Represents the state of a human in the environment"""
    person_id: str
    name: Optional[str] = None
    emotional_state: EmotionalState = EmotionalState.NEUTRAL
    attention_focus: Optional[str] = None
    proximity: float = float('inf')  # Distance to robot
    trust_level: float = 0.5  # 0.0 to 1.0
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    last_interaction: Optional[float] = None
    cultural_context: Optional[str] = None


@dataclass
class SocialBehavior:
    """Represents a social behavior response"""
    behavior_id: str
    behavior_type: str  # 'greeting', 'assistance', 'explanation', 'apology', etc.
    target_person: Optional[str] = None
    intensity: float  # 0.0 to 1.0
    appropriateness: float  # 0.0 to 1.0
    cultural_sensitivity: float  # 0.0 to 1.0
    actions: List[Dict[str, Any]]
    reasoning: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class SocialRule:
    """Represents a social rule or norm"""
    rule_id: str
    rule_name: str
    context: SocialContext
    trigger_conditions: Dict[str, Any]
    appropriate_behaviors: List[str]
    inappropriate_behaviors: List[str]
    cultural_variations: Dict[str, List[str]]
    priority: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0


@dataclass
class SocialAdjustment:
    """Represents social behavior adjustments"""
    adjustment_id: str
    original_behavior: Dict[str, Any]
    adjusted_behavior: Dict[str, Any]
    reasoning: str
    confidence: float  # 0.0 to 1.0
    social_context: SocialContext
    cultural_considerations: List[str]
    timestamp: float = field(default_factory=time.time)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class SocialIntelligence:
    """
    Social intelligence system for understanding and responding to social cues.
    
    This component manages:
    - Social cue detection and interpretation
    - Human state tracking and modeling
    - Social behavior generation and adaptation
    - Cultural sensitivity and appropriateness
    - Trust and relationship management
    - Social rule learning and application
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the social intelligence system.
        
        Args:
            config: Configuration dictionary for social intelligence parameters
        """
        self.config = config or self._get_default_config()
        
        # Tracked humans
        self.humans: Dict[str, HumanState] = {}
        
        # Social rules database
        self.social_rules: Dict[str, SocialRule] = {}
        
        # Social behavior history
        self.behavior_history: deque = deque(maxlen=100)
        
        # Current social context
        self.current_context = SocialContext.ALONE
        
        # Cultural context
        self.cultural_context = "western"  # Default cultural context
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self.performance_metrics = {
            'social_cue_accuracy': 0.0,
            'behavior_appropriateness': 0.0,
            'trust_building_rate': 0.0,
            'cultural_sensitivity': 0.0
        }
        
        # Initialize with basic social rules
        self._initialize_social_rules()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for social intelligence."""
        return {
            'max_tracked_humans': 10,
            'proximity_threshold': 2.0,  # meters
            'trust_building_rate': 0.1,
            'trust_decay_rate': 0.01,
            'cultural_sensitivity_threshold': 0.7,
            'behavior_appropriateness_threshold': 0.6,
            'social_cue_confidence_threshold': 0.5,
            'interaction_timeout': 300,  # 5 minutes
            'max_interaction_history': 50
        }
    
    def _initialize_social_rules(self):
        """Initialize with basic social rules."""
        basic_rules = [
            SocialRule(
                rule_id="greeting_rule",
                rule_name="Appropriate Greeting",
                context=SocialContext.INTERACTING,
                trigger_conditions={
                    'proximity': 1.5,
                    'attention_focus': 'robot',
                    'interaction_type': 'first_contact'
                },
                appropriate_behaviors=['verbal_greeting', 'nonverbal_greeting'],
                inappropriate_behaviors=['ignoring', 'aggressive_approach'],
                cultural_variations={
                    'western': ['handshake', 'wave'],
                    'eastern': ['bow', 'nod'],
                    'middle_eastern': ['handshake', 'cheek_kiss']
                },
                priority=0.8,
                confidence=0.9
            ),
            SocialRule(
                rule_id="personal_space_rule",
                rule_name="Respect Personal Space",
                context=SocialContext.INTERACTING,
                trigger_conditions={
                    'proximity': 0.5,
                    'interaction_type': 'close_contact'
                },
                appropriate_behaviors=['maintain_distance', 'step_back'],
                inappropriate_behaviors=['invade_space', 'touch_without_consent'],
                cultural_variations={
                    'western': ['1.5m_distance'],
                    'eastern': ['1.2m_distance'],
                    'middle_eastern': ['1.0m_distance']
                },
                priority=0.9,
                confidence=0.95
            ),
            SocialRule(
                rule_id="assistance_rule",
                rule_name="Offer Appropriate Assistance",
                context=SocialContext.INTERACTING,
                trigger_conditions={
                    'help_requested': True,
                    'capability': 'available'
                },
                appropriate_behaviors=['offer_help', 'explain_capabilities'],
                inappropriate_behaviors=['force_assistance', 'ignore_request'],
                cultural_variations={
                    'western': ['direct_offer'],
                    'eastern': ['polite_offer'],
                    'middle_eastern': ['respectful_offer']
                },
                priority=0.7,
                confidence=0.8
            )
        ]
        
        for rule in basic_rules:
            self.social_rules[rule.rule_id] = rule
    
    def process_social_cues(self, 
                           social_cues: List[SocialCue],
                           current_context: Optional[Dict[str, Any]] = None) -> List[SocialBehavior]:
        """
        Process social cues and generate appropriate behaviors.
        
        Args:
            social_cues: List of detected social cues
            current_context: Current context information
            
        Returns:
            List of appropriate social behaviors
        """
        with self._lock:
            behaviors = []
            
            # Update human states based on cues
            self._update_human_states(social_cues)
            
            # Update social context
            self._update_social_context(social_cues, current_context)
            
            # Generate behaviors for each cue
            for cue in social_cues:
                if cue.confidence > self.config['social_cue_confidence_threshold']:
                    behavior = self._generate_behavior_for_cue(cue, current_context)
                    if behavior:
                        behaviors.append(behavior)
            
            # Generate context-appropriate behaviors
            context_behaviors = self._generate_context_behaviors(current_context)
            behaviors.extend(context_behaviors)
            
            # Store behaviors in history
            for behavior in behaviors:
                self.behavior_history.append(behavior)
            
            return behaviors
    
    def _update_human_states(self, social_cues: List[SocialCue]):
        """Update human states based on social cues."""
        for cue in social_cues:
            if cue.source != 'environment':
                # Update or create human state
                if cue.source not in self.humans:
                    self.humans[cue.source] = HumanState(person_id=cue.source)
                
                human = self.humans[cue.source]
                
                # Update based on cue type
                if cue.cue_type == InteractionType.VERBAL:
                    human.last_interaction = time.time()
                    # Extract emotional state from verbal cues
                    if 'emotion' in cue.metadata:
                        human.emotional_state = EmotionalState(cue.metadata['emotion'])
                
                elif cue.cue_type == InteractionType.NONVERBAL:
                    human.last_interaction = time.time()
                    # Extract emotional state from nonverbal cues
                    if 'emotion' in cue.metadata:
                        human.emotional_state = EmotionalState(cue.metadata['emotion'])
                
                elif cue.cue_type == InteractionType.PROXIMITY:
                    human.proximity = cue.metadata.get('distance', float('inf'))
                
                # Update interaction history
                interaction_record = {
                    'cue_type': cue.cue_type.value,
                    'timestamp': cue.timestamp,
                    'intensity': cue.intensity,
                    'metadata': cue.metadata
                }
                human.interaction_history.append(interaction_record)
                
                # Limit history size
                if len(human.interaction_history) > self.config['max_interaction_history']:
                    human.interaction_history.pop(0)
    
    def _update_social_context(self, 
                              social_cues: List[SocialCue],
                              current_context: Optional[Dict[str, Any]]):
        """Update current social context."""
        # Count nearby humans
        nearby_humans = sum(
            1 for human in self.humans.values()
            if human.proximity < self.config['proximity_threshold']
        )
        
        if nearby_humans == 0:
            self.current_context = SocialContext.ALONE
        elif nearby_humans == 1:
            self.current_context = SocialContext.INTERACTING
        else:
            self.current_context = SocialContext.GROUP
        
        # Check for formal context indicators
        if current_context:
            if current_context.get('formal_setting', False):
                self.current_context = SocialContext.FORMAL
            elif current_context.get('casual_setting', False):
                self.current_context = SocialContext.INFORMAL
    
    def _generate_behavior_for_cue(self, 
                                  cue: SocialCue,
                                  current_context: Optional[Dict[str, Any]]) -> Optional[SocialBehavior]:
        """Generate appropriate behavior for a specific social cue."""
        behavior_id = f"behavior_{cue.cue_id}"
        
        if cue.cue_type == InteractionType.VERBAL:
            return self._generate_verbal_response(cue, current_context)
        
        elif cue.cue_type == InteractionType.NONVERBAL:
            return self._generate_nonverbal_response(cue, current_context)
        
        elif cue.cue_type == InteractionType.PROXIMITY:
            return self._generate_proximity_response(cue, current_context)
        
        elif cue.cue_type == InteractionType.GESTURAL:
            return self._generate_gestural_response(cue, current_context)
        
        return None
    
    def _generate_verbal_response(self, 
                                 cue: SocialCue,
                                 current_context: Optional[Dict[str, Any]]) -> SocialBehavior:
        """Generate verbal response to verbal cue."""
        content = cue.metadata.get('content', '')
        emotion = cue.metadata.get('emotion', 'neutral')
        
        # Determine appropriate response based on content and emotion
        if 'hello' in content.lower() or 'hi' in content.lower():
            response_type = 'greeting'
            response_content = self._get_culturally_appropriate_greeting()
        elif 'help' in content.lower() or 'assist' in content.lower():
            response_type = 'assistance'
            response_content = "I'd be happy to help you. What do you need assistance with?"
        elif 'thank' in content.lower():
            response_type = 'acknowledgment'
            response_content = "You're welcome! I'm glad I could help."
        else:
            response_type = 'general_response'
            response_content = "I understand. How can I assist you further?"
        
        return SocialBehavior(
            behavior_id=f"verbal_{cue.cue_id}",
            behavior_type=response_type,
            target_person=cue.source,
            intensity=cue.intensity,
            appropriateness=self._calculate_appropriateness(response_type, emotion),
            cultural_sensitivity=self._calculate_cultural_sensitivity(response_type),
            actions=[
                {
                    'action_type': 'speak',
                    'content': response_content,
                    'tone': self._get_appropriate_tone(emotion),
                    'volume': self._get_appropriate_volume(cue.intensity)
                }
            ],
            reasoning=f"Responding to verbal cue: {content}"
        )
    
    def _generate_nonverbal_response(self, 
                                    cue: SocialCue,
                                    current_context: Optional[Dict[str, Any]]) -> SocialBehavior:
        """Generate nonverbal response to nonverbal cue."""
        emotion = cue.metadata.get('emotion', 'neutral')
        
        if emotion == EmotionalState.HAPPY:
            response_type = 'positive_acknowledgment'
            actions = [
                {
                    'action_type': 'facial_expression',
                    'expression': 'smile',
                    'intensity': min(cue.intensity, 0.8)
                }
            ]
        elif emotion == EmotionalState.SAD:
            response_type = 'empathic_response'
            actions = [
                {
                    'action_type': 'facial_expression',
                    'expression': 'concerned',
                    'intensity': 0.6
                },
                {
                    'action_type': 'speak',
                    'content': "I notice you seem upset. Is there anything I can do to help?",
                    'tone': 'gentle'
                }
            ]
        else:
            response_type = 'neutral_acknowledgment'
            actions = [
                {
                    'action_type': 'facial_expression',
                    'expression': 'neutral',
                    'intensity': 0.5
                }
            ]
        
        return SocialBehavior(
            behavior_id=f"nonverbal_{cue.cue_id}",
            behavior_type=response_type,
            target_person=cue.source,
            intensity=cue.intensity,
            appropriateness=self._calculate_appropriateness(response_type, emotion),
            cultural_sensitivity=self._calculate_cultural_sensitivity(response_type),
            actions=actions,
            reasoning=f"Responding to nonverbal cue with emotion: {emotion.value}"
        )
    
    def _generate_proximity_response(self, 
                                    cue: SocialCue,
                                    current_context: Optional[Dict[str, Any]]) -> SocialBehavior:
        """Generate response to proximity cue."""
        distance = cue.metadata.get('distance', float('inf'))
        
        if distance < 0.5:  # Too close
            response_type = 'personal_space_maintenance'
            actions = [
                {
                    'action_type': 'movement',
                    'direction': 'backward',
                    'distance': 0.5,
                    'speed': 'slow'
                },
                {
                    'action_type': 'speak',
                    'content': "I'll give you some space.",
                    'tone': 'polite'
                }
            ]
        elif distance < 1.0:  # Close interaction
            response_type = 'close_interaction'
            actions = [
                {
                    'action_type': 'facial_expression',
                    'expression': 'attentive',
                    'intensity': 0.7
                }
            ]
        else:
            response_type = 'normal_interaction'
            actions = [
                {
                    'action_type': 'facial_expression',
                    'expression': 'neutral',
                    'intensity': 0.5
                }
            ]
        
        return SocialBehavior(
            behavior_id=f"proximity_{cue.cue_id}",
            behavior_type=response_type,
            target_person=cue.source,
            intensity=cue.intensity,
            appropriateness=self._calculate_appropriateness(response_type, 'neutral'),
            cultural_sensitivity=self._calculate_cultural_sensitivity(response_type),
            actions=actions,
            reasoning=f"Responding to proximity cue: distance = {distance}m"
        )
    
    def _generate_gestural_response(self, 
                                   cue: SocialCue,
                                   current_context: Optional[Dict[str, Any]]) -> SocialBehavior:
        """Generate response to gestural cue."""
        gesture_type = cue.metadata.get('gesture_type', 'unknown')
        
        if gesture_type == 'wave':
            response_type = 'greeting'
            actions = [
                {
                    'action_type': 'gesture',
                    'gesture_type': 'wave',
                    'intensity': cue.intensity
                }
            ]
        elif gesture_type == 'point':
            response_type = 'attention_focus'
            actions = [
                {
                    'action_type': 'look_at',
                    'target': cue.metadata.get('pointed_at', 'unknown'),
                    'duration': 2.0
                }
            ]
        else:
            response_type = 'acknowledgment'
            actions = [
                {
                    'action_type': 'facial_expression',
                    'expression': 'neutral',
                    'intensity': 0.5
                }
            ]
        
        return SocialBehavior(
            behavior_id=f"gestural_{cue.cue_id}",
            behavior_type=response_type,
            target_person=cue.source,
            intensity=cue.intensity,
            appropriateness=self._calculate_appropriateness(response_type, 'neutral'),
            cultural_sensitivity=self._calculate_cultural_sensitivity(response_type),
            actions=actions,
            reasoning=f"Responding to gestural cue: {gesture_type}"
        )
    
    def _generate_context_behaviors(self, 
                                   current_context: Optional[Dict[str, Any]]) -> List[SocialBehavior]:
        """Generate context-appropriate behaviors."""
        behaviors = []
        
        # Check applicable social rules
        for rule in self.social_rules.values():
            if rule.context == self.current_context:
                if self._rule_conditions_met(rule, current_context):
                    behavior = self._generate_rule_based_behavior(rule, current_context)
                    if behavior:
                        behaviors.append(behavior)
        
        return behaviors
    
    def _rule_conditions_met(self, 
                            rule: SocialRule,
                            current_context: Optional[Dict[str, Any]]) -> bool:
        """Check if social rule conditions are met."""
        for condition_key, condition_value in rule.trigger_conditions.items():
            if condition_key == 'proximity':
                # Check if any human is within proximity
                nearby_humans = [
                    human for human in self.humans.values()
                    if human.proximity <= condition_value
                ]
                if not nearby_humans:
                    return False
            
            elif condition_key in current_context:
                if current_context[condition_key] != condition_value:
                    return False
            else:
                # Required condition not found
                return False
        
        return True
    
    def _generate_rule_based_behavior(self, 
                                     rule: SocialRule,
                                     current_context: Optional[Dict[str, Any]]) -> Optional[SocialBehavior]:
        """Generate behavior based on social rule."""
        # Select appropriate behavior for current cultural context
        cultural_behaviors = rule.cultural_variations.get(self.cultural_context, rule.appropriate_behaviors)
        
        if not cultural_behaviors:
            return None
        
        # Select first appropriate behavior
        behavior_type = cultural_behaviors[0]
        
        return SocialBehavior(
            behavior_id=f"rule_{rule.rule_id}",
            behavior_type=behavior_type,
            intensity=0.6,
            appropriateness=rule.confidence,
            cultural_sensitivity=0.9,
            actions=[
                {
                    'action_type': 'social_rule_application',
                    'rule_id': rule.rule_id,
                    'behavior_type': behavior_type
                }
            ],
            reasoning=f"Applying social rule: {rule.rule_name}"
        )
    
    def adjust_behavior(self, 
                       executive_decision: Any,
                       focused_attention: List[Any]) -> SocialAdjustment:
        """
        Adjust behavior based on executive decision and attention focus.
        
        Args:
            executive_decision: Executive decision made
            focused_attention: Current attention foci
            
        Returns:
            Social adjustment with modified behavior
        """
        with self._lock:
            # Create base adjustment
            adjustment = SocialAdjustment(
                adjustment_id=f"adjustment_{int(time.time() * 1000)}",
                original_behavior={
                    'decision_type': executive_decision.decision_type.value if executive_decision else None,
                    'actions': executive_decision.actions if executive_decision else []
                },
                adjusted_behavior={},
                reasoning="Social intelligence adjustment",
                confidence=0.7,
                social_context=self.current_context,
                cultural_considerations=[self.cultural_context]
            )
            
            # Apply social adjustments based on context
            if self.current_context == SocialContext.INTERACTING:
                adjustment = self._apply_interaction_adjustments(adjustment, focused_attention)
            elif self.current_context == SocialContext.GROUP:
                adjustment = self._apply_group_adjustments(adjustment, focused_attention)
            elif self.current_context == SocialContext.FORMAL:
                adjustment = self._apply_formal_adjustments(adjustment, focused_attention)
            
            # Add cultural considerations
            adjustment.cultural_considerations.extend(self._get_cultural_considerations())
            
            return adjustment
    
    def _apply_interaction_adjustments(self, 
                                      adjustment: SocialAdjustment,
                                      focused_attention: List[Any]) -> SocialAdjustment:
        """Apply adjustments for one-on-one interactions."""
        # Check for social attention foci
        social_foci = [
            focus for focus in focused_attention
            if hasattr(focus, 'attention_type') and focus.attention_type.value == 'social'
        ]
        
        if social_foci:
            # Prioritize social interaction
            adjustment.adjusted_behavior['priority_boost'] = 1.5
            adjustment.adjusted_behavior['social_focus'] = True
            
            # Add social actions
            adjustment.actions.append({
                'action_type': 'social_engagement',
                'parameters': {
                    'engagement_level': 'high',
                    'response_time': 'immediate'
                }
            })
        
        return adjustment
    
    def _apply_group_adjustments(self, 
                                adjustment: SocialAdjustment,
                                focused_attention: List[Any]) -> SocialAdjustment:
        """Apply adjustments for group interactions."""
        # Reduce individual attention to share across group
        adjustment.adjusted_behavior['attention_distribution'] = 'group_shared'
        adjustment.adjusted_behavior['interaction_style'] = 'inclusive'
        
        # Add group-appropriate actions
        adjustment.actions.append({
            'action_type': 'group_awareness',
            'parameters': {
                'scan_frequency': 'high',
                'inclusive_behavior': True
            }
        })
        
        return adjustment
    
    def _apply_formal_adjustments(self, 
                                 adjustment: SocialAdjustment,
                                 focused_attention: List[Any]) -> SocialAdjustment:
        """Apply adjustments for formal contexts."""
        # Increase formality and politeness
        adjustment.adjusted_behavior['formality_level'] = 'high'
        adjustment.adjusted_behavior['politeness_boost'] = 1.3
        
        # Add formal behavior actions
        adjustment.actions.append({
            'action_type': 'formal_behavior',
            'parameters': {
                'speech_style': 'formal',
                'gesture_restraint': True,
                'respect_level': 'high'
            }
        })
        
        return adjustment
    
    def _get_cultural_considerations(self) -> List[str]:
        """Get cultural considerations for current context."""
        considerations = []
        
        if self.cultural_context == "western":
            considerations.extend(['personal_space', 'direct_communication', 'individual_focus'])
        elif self.cultural_context == "eastern":
            considerations.extend(['hierarchy_respect', 'indirect_communication', 'group_harmony'])
        elif self.cultural_context == "middle_eastern":
            considerations.extend(['hospitality', 'respect_for_elders', 'modest_behavior'])
        
        return considerations
    
    def _get_culturally_appropriate_greeting(self) -> str:
        """Get culturally appropriate greeting."""
        if self.cultural_context == "western":
            return "Hello! How can I help you today?"
        elif self.cultural_context == "eastern":
            return "Greetings. I am here to assist you."
        elif self.cultural_context == "middle_eastern":
            return "Peace be upon you. How may I serve you?"
        else:
            return "Hello! How can I help you today?"
    
    def _calculate_appropriateness(self, behavior_type: str, emotion: str) -> float:
        """Calculate appropriateness of behavior."""
        base_appropriateness = 0.7
        
        # Adjust based on behavior type
        if behavior_type in ['greeting', 'acknowledgment']:
            base_appropriateness += 0.2
        elif behavior_type in ['assistance', 'empathic_response']:
            base_appropriateness += 0.1
        
        # Adjust based on emotion
        if emotion == 'neutral':
            base_appropriateness += 0.1
        elif emotion in ['happy', 'calm']:
            base_appropriateness += 0.05
        
        return min(1.0, base_appropriateness)
    
    def _calculate_cultural_sensitivity(self, behavior_type: str) -> float:
        """Calculate cultural sensitivity of behavior."""
        base_sensitivity = 0.8
        
        # Adjust based on behavior type
        if behavior_type in ['greeting', 'acknowledgment']:
            base_sensitivity += 0.1
        elif behavior_type in ['personal_space_maintenance']:
            base_sensitivity += 0.05
        
        return min(1.0, base_sensitivity)
    
    def _get_appropriate_tone(self, emotion: str) -> str:
        """Get appropriate tone for emotional state."""
        tone_mapping = {
            'happy': 'cheerful',
            'sad': 'gentle',
            'angry': 'calm',
            'stressed': 'reassuring',
            'neutral': 'neutral',
            'excited': 'enthusiastic'
        }
        return tone_mapping.get(emotion, 'neutral')
    
    def _get_appropriate_volume(self, intensity: float) -> str:
        """Get appropriate volume based on intensity."""
        if intensity > 0.8:
            return 'loud'
        elif intensity > 0.5:
            return 'normal'
        else:
            return 'quiet'
    
    def get_social_context(self) -> SocialContext:
        """Get current social context."""
        return self.current_context
    
    def get_human_states(self) -> Dict[str, HumanState]:
        """Get current human states."""
        return self.humans.copy()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def get_social_summary(self) -> Dict[str, Any]:
        """Get a summary of social intelligence state."""
        with self._lock:
            summary = {
                'current_context': self.current_context.value,
                'cultural_context': self.cultural_context,
                'tracked_humans': len(self.humans),
                'social_rules': len(self.social_rules),
                'behavior_history': len(self.behavior_history),
                'performance_metrics': self.performance_metrics,
                'recent_behaviors': []
            }
            
            # Get recent behaviors
            recent_behaviors = list(self.behavior_history)[-5:]  # Last 5 behaviors
            summary['recent_behaviors'] = [
                {
                    'type': behavior.behavior_type,
                    'target': behavior.target_person,
                    'appropriateness': behavior.appropriateness,
                    'timestamp': behavior.timestamp
                }
                for behavior in recent_behaviors
            ]
            
            return summary 