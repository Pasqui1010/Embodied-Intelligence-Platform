"""
Social Learning Module

This module provides social learning capabilities for human-robot
interaction, enabling the robot to learn from social interactions,
feedback, and experiences to improve future interactions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearningType(Enum):
    """Enumeration of learning types"""
    REINFORCEMENT = "reinforcement"
    OBSERVATIONAL = "observational"
    FEEDBACK_BASED = "feedback_based"
    PATTERN_RECOGNITION = "pattern_recognition"
    ADAPTIVE_BEHAVIOR = "adaptive_behavior"


class LearningOutcome(Enum):
    """Enumeration of learning outcomes"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNCLEAR = "unclear"


@dataclass
class SocialInteraction:
    """Data class for social interaction data"""
    interaction_id: str
    timestamp: float
    human_input: Dict[str, any]
    robot_response: Dict[str, any]
    human_feedback: Optional[Dict[str, any]]
    social_context: Dict[str, any]
    outcome: LearningOutcome
    duration: float


@dataclass
class LearningPattern:
    """Data class for learned patterns"""
    pattern_id: str
    pattern_type: str
    trigger_conditions: Dict[str, any]
    successful_responses: List[Dict[str, any]]
    success_rate: float
    confidence: float
    last_used: float
    usage_count: int


@dataclass
class SocialLearningResult:
    """Data class for social learning results"""
    new_patterns: List[LearningPattern]
    updated_patterns: List[LearningPattern]
    learning_insights: List[str]
    confidence_improvement: float
    behavior_recommendations: List[Dict[str, any]]
    learning_metrics: Dict[str, float]


class SocialLearning:
    """
    Social learning system for human-robot interaction
    
    This class enables the robot to learn from social interactions,
    feedback, and experiences to improve future interactions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the social learning system
        
        Args:
            config: Configuration dictionary for social learning
        """
        self.config = config or self._get_default_config()
        self.learning_patterns = {}
        self.interaction_history = []
        self.feedback_analyzer = self._initialize_feedback_analyzer()
        self.pattern_recognizer = self._initialize_pattern_recognizer()
        self.learning_optimizer = self._initialize_learning_optimizer()
        self.knowledge_base = self._initialize_knowledge_base()
        
        logger.info("Social learning system initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for social learning"""
        return {
            'learning_rate': 0.1,
            'pattern_confidence_threshold': 0.7,
            'max_patterns': 100,
            'max_interaction_history': 1000,
            'feedback_weight': 0.4,
            'outcome_weight': 0.3,
            'pattern_weight': 0.3,
            'forgetting_factor': 0.95,
            'pattern_decay_rate': 0.01
        }
    
    def _initialize_feedback_analyzer(self):
        """Initialize feedback analysis components"""
        return {
            'sentiment_analyzer': self._initialize_sentiment_analyzer(),
            'feedback_classifier': self._initialize_feedback_classifier(),
            'improvement_detector': self._initialize_improvement_detector()
        }
    
    def _initialize_pattern_recognizer(self):
        """Initialize pattern recognition components"""
        return {
            'pattern_extractor': self._initialize_pattern_extractor(),
            'pattern_validator': self._initialize_pattern_validator(),
            'pattern_optimizer': self._initialize_pattern_optimizer()
        }
    
    def _initialize_learning_optimizer(self):
        """Initialize learning optimization components"""
        return {
            'learning_rate_adapter': self._initialize_learning_rate_adapter(),
            'knowledge_integrator': self._initialize_knowledge_integrator(),
            'performance_monitor': self._initialize_performance_monitor()
        }
    
    def _initialize_knowledge_base(self):
        """Initialize knowledge base"""
        return {
            'interaction_patterns': {},
            'successful_behaviors': {},
            'failed_behaviors': {},
            'context_preferences': {},
            'user_preferences': {}
        }
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analysis"""
        return "sentiment_analyzer_placeholder"
    
    def _initialize_feedback_classifier(self):
        """Initialize feedback classification"""
        return "feedback_classifier_placeholder"
    
    def _initialize_improvement_detector(self):
        """Initialize improvement detection"""
        return "improvement_detector_placeholder"
    
    def _initialize_pattern_extractor(self):
        """Initialize pattern extraction"""
        return "pattern_extractor_placeholder"
    
    def _initialize_pattern_validator(self):
        """Initialize pattern validation"""
        return "pattern_validator_placeholder"
    
    def _initialize_pattern_optimizer(self):
        """Initialize pattern optimization"""
        return "pattern_optimizer_placeholder"
    
    def _initialize_learning_rate_adapter(self):
        """Initialize learning rate adaptation"""
        return "learning_rate_adapter_placeholder"
    
    def _initialize_knowledge_integrator(self):
        """Initialize knowledge integration"""
        return "knowledge_integrator_placeholder"
    
    def _initialize_performance_monitor(self):
        """Initialize performance monitoring"""
        return "performance_monitor_placeholder"
    
    def learn_from_interaction(self,
                              human_input: Dict[str, any],
                              robot_response: Dict[str, any],
                              human_feedback: Optional[Dict[str, any]],
                              social_context: Dict[str, any]) -> SocialLearningResult:
        """
        Learn from a social interaction
        
        Args:
            human_input: Human input during interaction
            robot_response: Robot response during interaction
            human_feedback: Human feedback after interaction
            social_context: Social context of interaction
            
        Returns:
            SocialLearningResult with learning outcomes and insights
        """
        try:
            # Create interaction record
            interaction = self._create_interaction_record(
                human_input, robot_response, human_feedback, social_context
            )
            
            # Analyze interaction outcome
            outcome = self._analyze_interaction_outcome(
                human_input, robot_response, human_feedback, social_context
            )
            interaction.outcome = outcome
            
            # Store interaction
            self._store_interaction(interaction)
            
            # Extract learning patterns
            new_patterns = self._extract_learning_patterns(interaction)
            
            # Update existing patterns
            updated_patterns = self._update_existing_patterns(interaction)
            
            # Generate learning insights
            insights = self._generate_learning_insights(
                interaction, new_patterns, updated_patterns
            )
            
            # Calculate confidence improvement
            confidence_improvement = self._calculate_confidence_improvement(
                new_patterns, updated_patterns
            )
            
            # Generate behavior recommendations
            recommendations = self._generate_behavior_recommendations(
                interaction, new_patterns, updated_patterns
            )
            
            # Calculate learning metrics
            metrics = self._calculate_learning_metrics(
                interaction, new_patterns, updated_patterns
            )
            
            # Create learning result
            result = SocialLearningResult(
                new_patterns=new_patterns,
                updated_patterns=updated_patterns,
                learning_insights=insights,
                confidence_improvement=confidence_improvement,
                behavior_recommendations=recommendations,
                learning_metrics=metrics
            )
            
            # Update knowledge base
            self._update_knowledge_base(result)
            
            logger.debug(f"Learning completed with {len(new_patterns)} new patterns")
            return result
            
        except Exception as e:
            logger.error(f"Error in social learning: {e}")
            return self._get_default_learning_result()
    
    def _create_interaction_record(self,
                                 human_input: Dict[str, any],
                                 robot_response: Dict[str, any],
                                 human_feedback: Optional[Dict[str, any]],
                                 social_context: Dict[str, any]) -> SocialInteraction:
        """Create interaction record"""
        return SocialInteraction(
            interaction_id=f"interaction_{int(time.time())}",
            timestamp=time.time(),
            human_input=human_input,
            robot_response=robot_response,
            human_feedback=human_feedback,
            social_context=social_context,
            outcome=LearningOutcome.NEUTRAL,  # Will be updated later
            duration=self._calculate_interaction_duration(human_input, robot_response)
        )
    
    def _calculate_interaction_duration(self,
                                      human_input: Dict[str, any],
                                      robot_response: Dict[str, any]) -> float:
        """Calculate interaction duration"""
        # Placeholder for duration calculation
        return 5.0  # Default 5 seconds
    
    def _analyze_interaction_outcome(self,
                                   human_input: Dict[str, any],
                                   robot_response: Dict[str, any],
                                   human_feedback: Optional[Dict[str, any]],
                                   social_context: Dict[str, any]) -> LearningOutcome:
        """Analyze interaction outcome"""
        # Analyze human feedback
        if human_feedback:
            feedback_outcome = self._analyze_feedback_outcome(human_feedback)
            if feedback_outcome != LearningOutcome.UNCLEAR:
                return feedback_outcome
        
        # Analyze interaction patterns
        pattern_outcome = self._analyze_pattern_outcome(
            human_input, robot_response, social_context
        )
        
        # Analyze emotional response
        emotional_outcome = self._analyze_emotional_outcome(
            human_input, robot_response, social_context
        )
        
        # Combine outcomes
        return self._combine_outcomes([pattern_outcome, emotional_outcome])
    
    def _analyze_feedback_outcome(self, human_feedback: Dict[str, any]) -> LearningOutcome:
        """Analyze human feedback for outcome"""
        # Extract feedback sentiment
        sentiment = human_feedback.get('sentiment', 'neutral')
        rating = human_feedback.get('rating', 0.5)
        comments = human_feedback.get('comments', '')
        
        # Determine outcome based on feedback
        if sentiment == 'positive' or rating > 0.7:
            return LearningOutcome.POSITIVE
        elif sentiment == 'negative' or rating < 0.3:
            return LearningOutcome.NEGATIVE
        elif sentiment == 'neutral' or 0.3 <= rating <= 0.7:
            return LearningOutcome.NEUTRAL
        else:
            return LearningOutcome.UNCLEAR
    
    def _analyze_pattern_outcome(self,
                               human_input: Dict[str, any],
                               robot_response: Dict[str, any],
                               social_context: Dict[str, any]) -> LearningOutcome:
        """Analyze interaction patterns for outcome"""
        # Check for successful interaction patterns
        if self._is_successful_pattern(human_input, robot_response, social_context):
            return LearningOutcome.POSITIVE
        elif self._is_failed_pattern(human_input, robot_response, social_context):
            return LearningOutcome.NEGATIVE
        else:
            return LearningOutcome.NEUTRAL
    
    def _analyze_emotional_outcome(self,
                                 human_input: Dict[str, any],
                                 robot_response: Dict[str, any],
                                 social_context: Dict[str, any]) -> LearningOutcome:
        """Analyze emotional response for outcome"""
        # Extract emotional indicators
        human_emotion = human_input.get('emotion', 'neutral')
        emotional_intensity = human_input.get('emotional_intensity', 0.5)
        
        # Determine outcome based on emotions
        if human_emotion in ['happy', 'excited', 'satisfied']:
            return LearningOutcome.POSITIVE
        elif human_emotion in ['angry', 'frustrated', 'disappointed']:
            return LearningOutcome.NEGATIVE
        else:
            return LearningOutcome.NEUTRAL
    
    def _is_successful_pattern(self,
                             human_input: Dict[str, any],
                             robot_response: Dict[str, any],
                             social_context: Dict[str, any]) -> bool:
        """Check if interaction follows successful pattern"""
        # Placeholder for successful pattern detection
        return False
    
    def _is_failed_pattern(self,
                          human_input: Dict[str, any],
                          robot_response: Dict[str, any],
                          social_context: Dict[str, any]) -> bool:
        """Check if interaction follows failed pattern"""
        # Placeholder for failed pattern detection
        return False
    
    def _combine_outcomes(self, outcomes: List[LearningOutcome]) -> LearningOutcome:
        """Combine multiple outcomes into single outcome"""
        if not outcomes:
            return LearningOutcome.NEUTRAL
        
        # Count outcomes
        positive_count = outcomes.count(LearningOutcome.POSITIVE)
        negative_count = outcomes.count(LearningOutcome.NEGATIVE)
        neutral_count = outcomes.count(LearningOutcome.NEUTRAL)
        
        # Determine dominant outcome
        if positive_count > negative_count and positive_count > neutral_count:
            return LearningOutcome.POSITIVE
        elif negative_count > positive_count and negative_count > neutral_count:
            return LearningOutcome.NEGATIVE
        else:
            return LearningOutcome.NEUTRAL
    
    def _store_interaction(self, interaction: SocialInteraction):
        """Store interaction in history"""
        self.interaction_history.append(interaction)
        
        # Keep only recent history
        max_history = self.config.get('max_interaction_history', 1000)
        if len(self.interaction_history) > max_history:
            self.interaction_history.pop(0)
    
    def _extract_learning_patterns(self, interaction: SocialInteraction) -> List[LearningPattern]:
        """Extract new learning patterns from interaction"""
        patterns = []
        
        # Extract behavioral patterns
        behavioral_patterns = self._extract_behavioral_patterns(interaction)
        patterns.extend(behavioral_patterns)
        
        # Extract contextual patterns
        contextual_patterns = self._extract_contextual_patterns(interaction)
        patterns.extend(contextual_patterns)
        
        # Extract feedback patterns
        feedback_patterns = self._extract_feedback_patterns(interaction)
        patterns.extend(feedback_patterns)
        
        return patterns
    
    def _extract_behavioral_patterns(self, interaction: SocialInteraction) -> List[LearningPattern]:
        """Extract behavioral patterns from interaction"""
        patterns = []
        
        # Extract successful response patterns
        if interaction.outcome == LearningOutcome.POSITIVE:
            pattern = LearningPattern(
                pattern_id=f"behavioral_{int(time.time())}",
                pattern_type="successful_response",
                trigger_conditions={
                    'human_emotion': interaction.human_input.get('emotion', 'neutral'),
                    'interaction_type': interaction.social_context.get('interaction_type', 'general'),
                    'relationship': interaction.social_context.get('relationship', 'neutral')
                },
                successful_responses=[interaction.robot_response],
                success_rate=1.0,
                confidence=0.8,
                last_used=interaction.timestamp,
                usage_count=1
            )
            patterns.append(pattern)
        
        return patterns
    
    def _extract_contextual_patterns(self, interaction: SocialInteraction) -> List[LearningPattern]:
        """Extract contextual patterns from interaction"""
        patterns = []
        
        # Extract context-specific patterns
        context_key = f"{interaction.social_context.get('environment', 'unknown')}_{interaction.social_context.get('relationship', 'neutral')}"
        
        pattern = LearningPattern(
            pattern_id=f"contextual_{int(time.time())}",
            pattern_type="context_preference",
            trigger_conditions={
                'environment': interaction.social_context.get('environment', 'unknown'),
                'relationship': interaction.social_context.get('relationship', 'neutral'),
                'time_of_day': interaction.social_context.get('time_of_day', 'unknown')
            },
            successful_responses=[interaction.robot_response],
            success_rate=1.0 if interaction.outcome == LearningOutcome.POSITIVE else 0.0,
            confidence=0.7,
            last_used=interaction.timestamp,
            usage_count=1
        )
        patterns.append(pattern)
        
        return patterns
    
    def _extract_feedback_patterns(self, interaction: SocialInteraction) -> List[LearningPattern]:
        """Extract feedback patterns from interaction"""
        patterns = []
        
        if interaction.human_feedback:
            pattern = LearningPattern(
                pattern_id=f"feedback_{int(time.time())}",
                pattern_type="feedback_response",
                trigger_conditions={
                    'feedback_sentiment': interaction.human_feedback.get('sentiment', 'neutral'),
                    'feedback_rating': interaction.human_feedback.get('rating', 0.5)
                },
                successful_responses=[interaction.robot_response],
                success_rate=1.0 if interaction.outcome == LearningOutcome.POSITIVE else 0.0,
                confidence=0.6,
                last_used=interaction.timestamp,
                usage_count=1
            )
            patterns.append(pattern)
        
        return patterns
    
    def _update_existing_patterns(self, interaction: SocialInteraction) -> List[LearningPattern]:
        """Update existing patterns based on interaction"""
        updated_patterns = []
        
        for pattern_id, pattern in self.learning_patterns.items():
            # Check if pattern matches current interaction
            if self._pattern_matches_interaction(pattern, interaction):
                # Update pattern
                updated_pattern = self._update_pattern(pattern, interaction)
                updated_patterns.append(updated_pattern)
                self.learning_patterns[pattern_id] = updated_pattern
        
        return updated_patterns
    
    def _pattern_matches_interaction(self,
                                   pattern: LearningPattern,
                                   interaction: SocialInteraction) -> bool:
        """Check if pattern matches current interaction"""
        # Check trigger conditions
        for condition_key, condition_value in pattern.trigger_conditions.items():
            if condition_key in interaction.human_input:
                if interaction.human_input[condition_key] != condition_value:
                    return False
            elif condition_key in interaction.social_context:
                if interaction.social_context[condition_key] != condition_value:
                    return False
            else:
                return False
        
        return True
    
    def _update_pattern(self,
                       pattern: LearningPattern,
                       interaction: SocialInteraction) -> LearningPattern:
        """Update pattern based on interaction"""
        # Update success rate
        current_success = 1.0 if interaction.outcome == LearningOutcome.POSITIVE else 0.0
        new_success_rate = (pattern.success_rate * pattern.usage_count + current_success) / (pattern.usage_count + 1)
        
        # Update usage count
        new_usage_count = pattern.usage_count + 1
        
        # Update confidence
        new_confidence = min(1.0, pattern.confidence + self.config.get('learning_rate', 0.1))
        
        # Update last used
        new_last_used = interaction.timestamp
        
        # Add response if successful
        new_responses = pattern.successful_responses.copy()
        if interaction.outcome == LearningOutcome.POSITIVE:
            new_responses.append(interaction.robot_response)
        
        return LearningPattern(
            pattern_id=pattern.pattern_id,
            pattern_type=pattern.pattern_type,
            trigger_conditions=pattern.trigger_conditions,
            successful_responses=new_responses,
            success_rate=new_success_rate,
            confidence=new_confidence,
            last_used=new_last_used,
            usage_count=new_usage_count
        )
    
    def _generate_learning_insights(self,
                                  interaction: SocialInteraction,
                                  new_patterns: List[LearningPattern],
                                  updated_patterns: List[LearningPattern]) -> List[str]:
        """Generate learning insights from interaction"""
        insights = []
        
        # Insight about interaction outcome
        if interaction.outcome == LearningOutcome.POSITIVE:
            insights.append("Interaction was successful - reinforcing positive behaviors")
        elif interaction.outcome == LearningOutcome.NEGATIVE:
            insights.append("Interaction needs improvement - identifying areas for adjustment")
        
        # Insight about new patterns
        if new_patterns:
            insights.append(f"Discovered {len(new_patterns)} new interaction patterns")
        
        # Insight about pattern updates
        if updated_patterns:
            insights.append(f"Updated {len(updated_patterns)} existing patterns")
        
        # Insight about feedback
        if interaction.human_feedback:
            insights.append("Received valuable human feedback for learning")
        
        return insights
    
    def _calculate_confidence_improvement(self,
                                        new_patterns: List[LearningPattern],
                                        updated_patterns: List[LearningPattern]) -> float:
        """Calculate confidence improvement from learning"""
        improvement = 0.0
        
        # Improvement from new patterns
        for pattern in new_patterns:
            improvement += pattern.confidence * 0.1
        
        # Improvement from updated patterns
        for pattern in updated_patterns:
            improvement += (pattern.confidence - 0.7) * 0.05  # Assuming base confidence of 0.7
        
        return min(1.0, improvement)
    
    def _generate_behavior_recommendations(self,
                                         interaction: SocialInteraction,
                                         new_patterns: List[LearningPattern],
                                         updated_patterns: List[LearningPattern]) -> List[Dict[str, any]]:
        """Generate behavior recommendations based on learning"""
        recommendations = []
        
        # Recommendations based on successful patterns
        for pattern in new_patterns + updated_patterns:
            if pattern.success_rate > 0.8 and pattern.confidence > 0.7:
                recommendation = {
                    'type': 'pattern_based',
                    'pattern_id': pattern.pattern_id,
                    'recommendation': f"Use {pattern.pattern_type} pattern in similar contexts",
                    'confidence': pattern.confidence,
                    'success_rate': pattern.success_rate
                }
                recommendations.append(recommendation)
        
        # Recommendations based on feedback
        if interaction.human_feedback:
            feedback_rec = {
                'type': 'feedback_based',
                'recommendation': 'Consider user feedback for future interactions',
                'confidence': 0.8,
                'feedback': interaction.human_feedback
            }
            recommendations.append(feedback_rec)
        
        return recommendations
    
    def _calculate_learning_metrics(self,
                                  interaction: SocialInteraction,
                                  new_patterns: List[LearningPattern],
                                  updated_patterns: List[LearningPattern]) -> Dict[str, float]:
        """Calculate learning metrics"""
        return {
            'total_patterns': len(self.learning_patterns),
            'new_patterns_created': len(new_patterns),
            'patterns_updated': len(updated_patterns),
            'average_pattern_confidence': self._calculate_average_pattern_confidence(),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'interaction_success_rate': self._calculate_interaction_success_rate()
        }
    
    def _calculate_average_pattern_confidence(self) -> float:
        """Calculate average confidence of all patterns"""
        if not self.learning_patterns:
            return 0.0
        
        total_confidence = sum(pattern.confidence for pattern in self.learning_patterns.values())
        return total_confidence / len(self.learning_patterns)
    
    def _calculate_interaction_success_rate(self) -> float:
        """Calculate success rate of recent interactions"""
        recent_interactions = self.interaction_history[-50:]  # Last 50 interactions
        
        if not recent_interactions:
            return 0.0
        
        successful_count = sum(1 for interaction in recent_interactions 
                             if interaction.outcome == LearningOutcome.POSITIVE)
        return successful_count / len(recent_interactions)
    
    def _update_knowledge_base(self, result: SocialLearningResult):
        """Update knowledge base with learning results"""
        # Update interaction patterns
        for pattern in result.new_patterns:
            self.learning_patterns[pattern.pattern_id] = pattern
        
        # Update successful behaviors
        for pattern in result.new_patterns:
            if pattern.success_rate > 0.7:
                self.knowledge_base['successful_behaviors'][pattern.pattern_id] = pattern
        
        # Update failed behaviors
        for pattern in result.new_patterns:
            if pattern.success_rate < 0.3:
                self.knowledge_base['failed_behaviors'][pattern.pattern_id] = pattern
    
    def _get_default_learning_result(self) -> SocialLearningResult:
        """Get default learning result when learning fails"""
        return SocialLearningResult(
            new_patterns=[],
            updated_patterns=[],
            learning_insights=["Learning system encountered an error"],
            confidence_improvement=0.0,
            behavior_recommendations=[],
            learning_metrics={
                'total_patterns': len(self.learning_patterns),
                'new_patterns_created': 0,
                'patterns_updated': 0,
                'average_pattern_confidence': 0.0,
                'learning_rate': 0.0,
                'interaction_success_rate': 0.0
            }
        )
    
    def get_learning_patterns(self) -> Dict[str, LearningPattern]:
        """Get all learning patterns"""
        return self.learning_patterns.copy()
    
    def get_interaction_history(self) -> List[SocialInteraction]:
        """Get interaction history"""
        return self.interaction_history.copy()
    
    def get_knowledge_base(self) -> Dict[str, any]:
        """Get knowledge base"""
        return self.knowledge_base.copy()
    
    def clear_learning_data(self):
        """Clear all learning data"""
        self.learning_patterns.clear()
        self.interaction_history.clear()
        self.knowledge_base = self._initialize_knowledge_base()
        logger.info("Learning data cleared") 