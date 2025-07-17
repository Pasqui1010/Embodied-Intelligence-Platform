"""
Learning Engine for Cognitive Architecture

This module implements a learning engine that handles continuous adaptation,
skill acquisition, and pattern recognition to improve cognitive performance
over time.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import json


class LearningType(Enum):
    """Types of learning mechanisms"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"
    META_LEARNING = "meta_learning"


class SkillLevel(Enum):
    """Skill proficiency levels"""
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class LearningEvent:
    """Represents a learning event or experience"""
    event_id: str
    learning_type: LearningType
    skill_domain: str
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    performance_metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Skill:
    """Represents a learned skill"""
    skill_id: str
    skill_name: str
    domain: str
    description: str
    level: SkillLevel
    proficiency: float  # 0.0 to 1.0
    practice_count: int = 0
    success_rate: float = 0.0
    last_practiced: Optional[float] = None
    created_time: float = field(default_factory=time.time)
    prerequisites: List[str] = field(default_factory=list)
    related_skills: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningPattern:
    """Represents a learned pattern or rule"""
    pattern_id: str
    pattern_type: str  # 'if-then', 'sequence', 'association', 'classification'
    description: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    confidence: float  # 0.0 to 1.0
    usage_count: int = 0
    success_count: int = 0
    created_time: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationRule:
    """Represents an adaptation rule for behavior modification"""
    rule_id: str
    rule_name: str
    trigger_conditions: Dict[str, Any]
    adaptation_actions: List[Dict[str, Any]]
    priority: float  # 0.0 to 1.0
    effectiveness: float  # 0.0 to 1.0
    activation_count: int = 0
    last_activated: Optional[float] = None
    created_time: float = field(default_factory=time.time)
    domain: str = "general"


@dataclass
class LearningProgress:
    """Represents learning progress in a domain"""
    domain: str
    overall_proficiency: float  # 0.0 to 1.0
    skill_count: int
    recent_improvements: List[Dict[str, Any]]
    learning_rate: float  # Rate of improvement
    plateau_detected: bool = False
    last_assessment: float = field(default_factory=time.time)


class LearningEngine:
    """
    Learning engine for continuous adaptation and skill acquisition.
    
    This component manages:
    - Skill acquisition and improvement
    - Pattern recognition and learning
    - Adaptation rules for behavior modification
    - Transfer learning between domains
    - Meta-learning for learning optimization
    - Performance tracking and assessment
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the learning engine.
        
        Args:
            config: Configuration dictionary for learning parameters
        """
        self.config = config or self._get_default_config()
        
        # Skills database
        self.skills: Dict[str, Skill] = {}
        
        # Learning patterns
        self.learning_patterns: Dict[str, LearningPattern] = {}
        
        # Adaptation rules
        self.adaptation_rules: Dict[str, AdaptationRule] = {}
        
        # Learning events history
        self.learning_events: deque = deque(maxlen=1000)
        
        # Learning progress by domain
        self.learning_progress: Dict[str, LearningProgress] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self.performance_metrics = {
            'learning_rate': 0.0,
            'skill_improvement': 0.0,
            'pattern_recognition': 0.0,
            'adaptation_effectiveness': 0.0
        }
        
        # Initialize with basic skills
        self._initialize_basic_skills()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for learning engine."""
        return {
            'learning_rate': 0.1,
            'forgetting_rate': 0.01,
            'skill_decay_rate': 0.005,
            'pattern_confidence_threshold': 0.6,
            'adaptation_effectiveness_threshold': 0.5,
            'transfer_learning_threshold': 0.7,
            'meta_learning_interval': 3600,  # 1 hour
            'assessment_interval': 300,  # 5 minutes
            'max_skills_per_domain': 50,
            'max_patterns_per_domain': 100
        }
    
    def _initialize_basic_skills(self):
        """Initialize with basic cognitive skills."""
        basic_skills = [
            Skill(
                skill_id="attention_management",
                skill_name="Attention Management",
                domain="cognitive",
                description="Ability to manage and focus attention",
                level=SkillLevel.BEGINNER,
                proficiency=0.3
            ),
            Skill(
                skill_id="memory_consolidation",
                skill_name="Memory Consolidation",
                domain="cognitive",
                description="Ability to consolidate information in memory",
                level=SkillLevel.BEGINNER,
                proficiency=0.3
            ),
            Skill(
                skill_id="decision_making",
                skill_name="Decision Making",
                domain="cognitive",
                description="Ability to make effective decisions",
                level=SkillLevel.BEGINNER,
                proficiency=0.3
            ),
            Skill(
                skill_id="safety_monitoring",
                skill_name="Safety Monitoring",
                domain="safety",
                description="Ability to monitor and respond to safety concerns",
                level=SkillLevel.INTERMEDIATE,
                proficiency=0.6
            ),
            Skill(
                skill_id="social_interaction",
                skill_name="Social Interaction",
                domain="social",
                description="Ability to interact appropriately with humans",
                level=SkillLevel.NOVICE,
                proficiency=0.2
            )
        ]
        
        for skill in basic_skills:
            self.skills[skill.skill_id] = skill
            self._update_learning_progress(skill.domain)
    
    def update_patterns(self, 
                       focused_attention: List[Any],
                       executive_decision: Any,
                       social_adjustment: Any) -> List[str]:
        """
        Update learning patterns based on cognitive processing.
        
        Args:
            focused_attention: Current attention foci
            executive_decision: Executive decision made
            social_adjustment: Social behavior adjustments
            
        Returns:
            List of pattern IDs that were updated or created
        """
        with self._lock:
            updated_patterns = []
            
            # Extract patterns from attention focus
            attention_patterns = self._extract_attention_patterns(focused_attention)
            updated_patterns.extend(attention_patterns)
            
            # Extract patterns from executive decisions
            decision_patterns = self._extract_decision_patterns(executive_decision)
            updated_patterns.extend(decision_patterns)
            
            # Extract patterns from social adjustments
            social_patterns = self._extract_social_patterns(social_adjustment)
            updated_patterns.extend(social_patterns)
            
            # Create learning event
            learning_event = LearningEvent(
                event_id=f"event_{int(time.time() * 1000)}",
                learning_type=LearningType.UNSUPERVISED,
                skill_domain="cognitive",
                description="Pattern learning from cognitive processing",
                input_data={
                    'attention_foci': len(focused_attention),
                    'decision_type': executive_decision.decision_type.value if executive_decision else None,
                    'social_context': social_adjustment is not None
                },
                output_data={
                    'patterns_updated': len(updated_patterns),
                    'skills_improved': 0
                },
                success=len(updated_patterns) > 0,
                performance_metrics={
                    'pattern_recognition_rate': len(updated_patterns) / max(1, len(focused_attention)),
                    'learning_efficiency': 0.5
                }
            )
            
            self.learning_events.append(learning_event)
            
            return updated_patterns
    
    def _extract_attention_patterns(self, focused_attention: List[Any]) -> List[str]:
        """Extract learning patterns from attention focus."""
        patterns = []
        
        for focus in focused_attention:
            if hasattr(focus, 'attention_type') and hasattr(focus, 'metadata'):
                # Create attention pattern
                pattern_id = f"attention_{focus.attention_type.value}_{int(time.time() * 1000)}"
                
                pattern = LearningPattern(
                    pattern_id=pattern_id,
                    pattern_type="if-then",
                    description=f"Attention pattern for {focus.attention_type.value}",
                    conditions={
                        'attention_type': focus.attention_type.value,
                        'priority_threshold': focus.priority,
                        'context_tags': list(focus.metadata.keys()) if focus.metadata else []
                    },
                    actions=[
                        {
                            'action_type': 'focus_attention',
                            'parameters': {
                                'attention_type': focus.attention_type.value,
                                'priority': focus.priority,
                                'confidence': focus.confidence
                            }
                        }
                    ],
                    confidence=focus.confidence,
                    domain="attention"
                )
                
                self.learning_patterns[pattern_id] = pattern
                patterns.append(pattern_id)
        
        return patterns
    
    def _extract_decision_patterns(self, executive_decision: Any) -> List[str]:
        """Extract learning patterns from executive decisions."""
        patterns = []
        
        if executive_decision and hasattr(executive_decision, 'decision_type'):
            # Create decision pattern
            pattern_id = f"decision_{executive_decision.decision_type.value}_{int(time.time() * 1000)}"
            
            pattern = LearningPattern(
                pattern_id=pattern_id,
                pattern_type="if-then",
                description=f"Decision pattern for {executive_decision.decision_type.value}",
                conditions={
                    'decision_type': executive_decision.decision_type.value,
                    'confidence_threshold': executive_decision.confidence,
                    'priority': executive_decision.priority.value
                },
                actions=executive_decision.actions,
                confidence=executive_decision.confidence,
                domain="decision_making"
            )
            
            self.learning_patterns[pattern_id] = pattern
            patterns.append(pattern_id)
        
        return patterns
    
    def _extract_social_patterns(self, social_adjustment: Any) -> List[str]:
        """Extract learning patterns from social adjustments."""
        patterns = []
        
        if social_adjustment and hasattr(social_adjustment, 'actions'):
            # Create social pattern
            pattern_id = f"social_{int(time.time() * 1000)}"
            
            pattern = LearningPattern(
                pattern_id=pattern_id,
                pattern_type="sequence",
                description="Social interaction pattern",
                conditions={
                    'social_context': True,
                    'interaction_type': 'adjustment'
                },
                actions=social_adjustment.actions,
                confidence=0.6,
                domain="social"
            )
            
            self.learning_patterns[pattern_id] = pattern
            patterns.append(pattern_id)
        
        return patterns
    
    def learn_from_experience(self, 
                             experience_data: Dict[str, Any],
                             outcome: str,
                             performance_metrics: Dict[str, float]) -> str:
        """
        Learn from a specific experience.
        
        Args:
            experience_data: Data about the experience
            outcome: Outcome of the experience ('success', 'failure', 'partial')
            performance_metrics: Performance metrics from the experience
            
        Returns:
            Learning event ID
        """
        with self._lock:
            # Create learning event
            learning_event = LearningEvent(
                event_id=f"experience_{int(time.time() * 1000)}",
                learning_type=LearningType.REINFORCEMENT,
                skill_domain=experience_data.get('domain', 'general'),
                description=experience_data.get('description', 'Experience learning'),
                input_data=experience_data,
                output_data={'outcome': outcome, 'performance': performance_metrics},
                success=outcome == 'success',
                performance_metrics=performance_metrics
            )
            
            self.learning_events.append(learning_event)
            
            # Update relevant skills
            self._update_skills_from_experience(learning_event)
            
            # Create adaptation rules if needed
            self._create_adaptation_rules(learning_event)
            
            return learning_event.event_id
    
    def _update_skills_from_experience(self, learning_event: LearningEvent):
        """Update skills based on learning event."""
        domain = learning_event.skill_domain
        
        # Find relevant skills for the domain
        relevant_skills = [
            skill for skill in self.skills.values()
            if skill.domain == domain
        ]
        
        for skill in relevant_skills:
            # Calculate improvement based on performance
            if learning_event.success:
                improvement = self.config['learning_rate'] * learning_event.performance_metrics.get('efficiency', 0.5)
                skill.proficiency = min(1.0, skill.proficiency + improvement)
            else:
                # Small improvement even from failures (learning from mistakes)
                improvement = self.config['learning_rate'] * 0.1
                skill.proficiency = min(1.0, skill.proficiency + improvement)
            
            skill.practice_count += 1
            skill.last_practiced = time.time()
            
            # Update skill level
            skill.level = self._calculate_skill_level(skill.proficiency)
            
            # Update success rate
            if hasattr(skill, 'success_count'):
                if learning_event.success:
                    skill.success_count += 1
                skill.success_rate = skill.success_count / skill.practice_count
            
            # Update learning progress
            self._update_learning_progress(domain)
    
    def _calculate_skill_level(self, proficiency: float) -> SkillLevel:
        """Calculate skill level based on proficiency."""
        if proficiency >= 0.9:
            return SkillLevel.EXPERT
        elif proficiency >= 0.7:
            return SkillLevel.ADVANCED
        elif proficiency >= 0.5:
            return SkillLevel.INTERMEDIATE
        elif proficiency >= 0.3:
            return SkillLevel.BEGINNER
        else:
            return SkillLevel.NOVICE
    
    def _create_adaptation_rules(self, learning_event: LearningEvent):
        """Create adaptation rules based on learning event."""
        if learning_event.success and learning_event.performance_metrics.get('efficiency', 0) > 0.7:
            # Create positive adaptation rule
            rule_id = f"adaptation_{learning_event.event_id}"
            
            rule = AdaptationRule(
                rule_id=rule_id,
                rule_name=f"Successful {learning_event.skill_domain} adaptation",
                trigger_conditions={
                    'domain': learning_event.skill_domain,
                    'context_similarity': 0.8,
                    'performance_threshold': 0.7
                },
                adaptation_actions=[
                    {
                        'action_type': 'apply_successful_pattern',
                        'parameters': {
                            'pattern_id': learning_event.event_id,
                            'confidence_boost': 0.1
                        }
                    }
                ],
                priority=0.7,
                effectiveness=learning_event.performance_metrics.get('efficiency', 0.5),
                domain=learning_event.skill_domain
            )
            
            self.adaptation_rules[rule_id] = rule
    
    def _update_learning_progress(self, domain: str):
        """Update learning progress for a domain."""
        if domain not in self.learning_progress:
            self.learning_progress[domain] = LearningProgress(
                domain=domain,
                overall_proficiency=0.0,
                skill_count=0,
                recent_improvements=[],
                learning_rate=0.0
            )
        
        progress = self.learning_progress[domain]
        
        # Calculate overall proficiency
        domain_skills = [skill for skill in self.skills.values() if skill.domain == domain]
        if domain_skills:
            progress.overall_proficiency = sum(skill.proficiency for skill in domain_skills) / len(domain_skills)
            progress.skill_count = len(domain_skills)
        
        # Calculate learning rate
        recent_events = [
            event for event in self.learning_events
            if event.skill_domain == domain and 
            time.time() - event.timestamp < 3600  # Last hour
        ]
        
        if recent_events:
            progress.learning_rate = len(recent_events) / 3600  # Events per second
        
        progress.last_assessment = time.time()
    
    def get_relevant_patterns(self, 
                             context: Dict[str, Any],
                             domain: Optional[str] = None) -> List[LearningPattern]:
        """
        Get learning patterns relevant to current context.
        
        Args:
            context: Current context for pattern matching
            domain: Optional domain filter
            
        Returns:
            List of relevant learning patterns
        """
        with self._lock:
            relevant_patterns = []
            
            for pattern in self.learning_patterns.values():
                if domain and pattern.domain != domain:
                    continue
                
                # Check if pattern conditions match context
                if self._pattern_matches_context(pattern, context):
                    relevant_patterns.append(pattern)
            
            # Sort by confidence and usage count
            relevant_patterns.sort(
                key=lambda x: x.confidence * (1 + x.usage_count * 0.1),
                reverse=True
            )
            
            return relevant_patterns
    
    def _pattern_matches_context(self, pattern: LearningPattern, context: Dict[str, Any]) -> bool:
        """Check if a pattern matches the current context."""
        for condition_key, condition_value in pattern.conditions.items():
            if condition_key in context:
                context_value = context[condition_key]
                
                # Simple matching - in practice, this would be more sophisticated
                if isinstance(condition_value, (int, float)):
                    if isinstance(context_value, (int, float)):
                        if context_value < condition_value:
                            return False
                elif condition_value != context_value:
                    return False
            else:
                # Required condition not found in context
                return False
        
        return True
    
    def apply_adaptation_rules(self, 
                              current_context: Dict[str, Any],
                              current_behavior: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply adaptation rules to modify current behavior.
        
        Args:
            current_context: Current context
            current_behavior: Current behavior to adapt
            
        Returns:
            List of adaptation actions to apply
        """
        with self._lock:
            adaptation_actions = []
            
            for rule in self.adaptation_rules.values():
                if self._rule_should_activate(rule, current_context):
                    # Apply adaptation actions
                    adaptation_actions.extend(rule.adaptation_actions)
                    
                    # Update rule statistics
                    rule.activation_count += 1
                    rule.last_activated = time.time()
            
            return adaptation_actions
    
    def _rule_should_activate(self, rule: AdaptationRule, context: Dict[str, Any]) -> bool:
        """Check if an adaptation rule should activate."""
        for condition_key, condition_value in rule.trigger_conditions.items():
            if condition_key in context:
                context_value = context[condition_key]
                
                # Check if condition is met
                if isinstance(condition_value, (int, float)):
                    if isinstance(context_value, (int, float)):
                        if context_value < condition_value:
                            return False
                elif condition_value != context_value:
                    return False
            else:
                # Required condition not found
                return False
        
        return True
    
    def get_skill_proficiency(self, skill_id: str) -> Optional[float]:
        """Get proficiency level for a specific skill."""
        with self._lock:
            if skill_id in self.skills:
                return self.skills[skill_id].proficiency
            return None
    
    def get_learning_progress(self, domain: str) -> Optional[LearningProgress]:
        """Get learning progress for a specific domain."""
        with self._lock:
            return self.learning_progress.get(domain)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of learning state."""
        with self._lock:
            summary = {
                'total_skills': len(self.skills),
                'total_patterns': len(self.learning_patterns),
                'total_adaptation_rules': len(self.adaptation_rules),
                'total_learning_events': len(self.learning_events),
                'skills_by_domain': defaultdict(int),
                'patterns_by_domain': defaultdict(int),
                'recent_improvements': []
            }
            
            # Count skills by domain
            for skill in self.skills.values():
                summary['skills_by_domain'][skill.domain] += 1
            
            # Count patterns by domain
            for pattern in self.learning_patterns.values():
                summary['patterns_by_domain'][pattern.domain] += 1
            
            # Get recent improvements
            recent_events = list(self.learning_events)[-10:]  # Last 10 events
            summary['recent_improvements'] = [
                {
                    'event_id': event.event_id,
                    'domain': event.skill_domain,
                    'success': event.success,
                    'timestamp': event.timestamp
                }
                for event in recent_events
            ]
            
            return summary 