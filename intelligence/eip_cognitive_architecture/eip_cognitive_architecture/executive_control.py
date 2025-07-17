"""
Executive Control for Cognitive Architecture

This module implements an executive control system that handles high-level decision
making, task coordination, and cognitive resource management.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np


class DecisionType(Enum):
    """Types of executive decisions"""
    TASK_SELECTION = "task_selection"
    RESOURCE_ALLOCATION = "resource_allocation"
    SAFETY_OVERRIDE = "safety_override"
    SOCIAL_ADJUSTMENT = "social_adjustment"
    LEARNING_PRIORITY = "learning_priority"
    ATTENTION_MANAGEMENT = "attention_management"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class ExecutiveDecision:
    """Represents an executive decision"""
    decision_type: DecisionType
    decision_id: str
    reasoning: str
    confidence: float  # 0.0 to 1.0
    priority: TaskPriority
    actions: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    expected_outcome: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlan:
    """Represents a task execution plan"""
    task_id: str
    task_type: str
    description: str
    priority: TaskPriority
    steps: List[Dict[str, Any]]
    estimated_duration: float
    dependencies: List[str]
    success_criteria: List[Dict[str, Any]]
    safety_constraints: List[Dict[str, Any]]
    resource_requirements: Dict[str, Any]
    created_time: float = field(default_factory=time.time)
    status: str = "pending"  # pending, active, completed, failed, suspended


@dataclass
class ResourceAllocation:
    """Represents resource allocation decision"""
    resource_type: str  # 'attention', 'memory', 'processing', 'safety'
    allocation_percentage: float  # 0.0 to 1.0
    duration: float
    priority: TaskPriority
    justification: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class CognitiveState:
    """Represents current cognitive state"""
    attention_load: float  # 0.0 to 1.0
    memory_load: float  # 0.0 to 1.0
    processing_load: float  # 0.0 to 1.0
    safety_status: str  # 'safe', 'caution', 'warning', 'critical'
    social_context: str  # 'alone', 'interacting', 'observing'
    emotional_state: str  # 'neutral', 'focused', 'stressed', 'calm'
    energy_level: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)


class ExecutiveControl:
    """
    Executive control system for high-level decision making and task coordination.
    
    This component manages:
    - Task selection and prioritization
    - Resource allocation across cognitive components
    - Safety monitoring and override decisions
    - Social interaction coordination
    - Learning priority management
    - Attention management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the executive control system.
        
        Args:
            config: Configuration dictionary for executive control parameters
        """
        self.config = config or self._get_default_config()
        
        # Active task plans
        self.active_tasks: Dict[str, TaskPlan] = {}
        
        # Task queue
        self.task_queue: deque = deque()
        
        # Resource allocations
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        
        # Decision history
        self.decision_history: deque = deque(maxlen=100)
        
        # Current cognitive state
        self.cognitive_state = CognitiveState()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self.performance_metrics = {
            'decision_speed': 0.0,
            'task_completion_rate': 0.0,
            'resource_efficiency': 0.0,
            'safety_violations': 0
        }
        
        # Initialize default resource allocations
        self._initialize_default_allocations()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for executive control."""
        return {
            'max_concurrent_tasks': 3,
            'decision_timeout': 1.0,  # seconds
            'safety_priority_boost': 2.0,
            'social_priority_boost': 1.5,
            'resource_reallocation_threshold': 0.8,
            'task_switching_cost': 0.1,
            'learning_opportunity_threshold': 0.6
        }
    
    def _initialize_default_allocations(self):
        """Initialize default resource allocations."""
        default_allocations = [
            ResourceAllocation(
                resource_type="attention",
                allocation_percentage=0.3,
                duration=float('inf'),
                priority=TaskPriority.MEDIUM,
                justification="Default attention allocation"
            ),
            ResourceAllocation(
                resource_type="memory",
                allocation_percentage=0.4,
                duration=float('inf'),
                priority=TaskPriority.MEDIUM,
                justification="Default memory allocation"
            ),
            ResourceAllocation(
                resource_type="processing",
                allocation_percentage=0.3,
                duration=float('inf'),
                priority=TaskPriority.MEDIUM,
                justification="Default processing allocation"
            ),
            ResourceAllocation(
                resource_type="safety",
                allocation_percentage=0.2,
                duration=float('inf'),
                priority=TaskPriority.CRITICAL,
                justification="Continuous safety monitoring"
            )
        ]
        
        for allocation in default_allocations:
            self.resource_allocations[allocation.resource_type] = allocation
    
    def make_decision(self, 
                     working_memory_state: Dict[str, Any],
                     relevant_patterns: List[Any]) -> ExecutiveDecision:
        """
        Make high-level executive decisions based on current state and patterns.
        
        Args:
            working_memory_state: Current state of working memory
            relevant_patterns: Relevant patterns from long-term memory
            
        Returns:
            Executive decision with actions and reasoning
        """
        start_time = time.time()
        
        with self._lock:
            # Update cognitive state
            self._update_cognitive_state(working_memory_state)
            
            # Analyze current situation
            situation_analysis = self._analyze_situation(working_memory_state, relevant_patterns)
            
            # Determine decision type
            decision_type = self._determine_decision_type(situation_analysis)
            
            # Generate decision
            decision = self._generate_decision(decision_type, situation_analysis)
            
            # Store decision in history
            self.decision_history.append(decision)
            
            # Update performance metrics
            self.performance_metrics['decision_speed'] = time.time() - start_time
            
            return decision
    
    def _update_cognitive_state(self, working_memory_state: Dict[str, Any]):
        """Update current cognitive state based on working memory."""
        # Calculate attention load
        attention_focus = working_memory_state.get('attention_focus')
        if attention_focus and attention_focus.get('content'):
            foci = attention_focus['content'].get('foci', [])
            self.cognitive_state.attention_load = min(1.0, len(foci) * 0.2)
        else:
            self.cognitive_state.attention_load = 0.1
        
        # Calculate memory load
        total_memory_items = sum(
            1 for item in working_memory_state.values() 
            if item is not None
        )
        self.cognitive_state.memory_load = min(1.0, total_memory_items * 0.1)
        
        # Update safety status
        safety_state = working_memory_state.get('safety_state')
        if safety_state and safety_state.get('content'):
            self.cognitive_state.safety_status = safety_state['content'].safety_level
        
        # Update social context
        social_context = working_memory_state.get('social_context')
        if social_context and social_context.get('content'):
            nearby_humans = social_context['content'].nearby_humans
            if nearby_humans:
                self.cognitive_state.social_context = 'interacting'
            else:
                self.cognitive_state.social_context = 'alone'
        
        # Update emotional state based on safety and social factors
        if self.cognitive_state.safety_status == 'critical':
            self.cognitive_state.emotional_state = 'stressed'
        elif self.cognitive_state.social_context == 'interacting':
            self.cognitive_state.emotional_state = 'focused'
        else:
            self.cognitive_state.emotional_state = 'neutral'
        
        # Update energy level (simplified model)
        self.cognitive_state.energy_level = max(0.0, 1.0 - 
            (self.cognitive_state.attention_load + 
             self.cognitive_state.memory_load + 
             self.cognitive_state.processing_load) / 3.0)
        
        self.cognitive_state.timestamp = time.time()
    
    def _analyze_situation(self, 
                          working_memory_state: Dict[str, Any],
                          relevant_patterns: List[Any]) -> Dict[str, Any]:
        """Analyze current situation for decision making."""
        analysis = {
            'safety_critical': False,
            'social_interaction': False,
            'task_urgent': False,
            'learning_opportunity': False,
            'resource_constrained': False,
            'attention_overloaded': False
        }
        
        # Check safety status
        safety_state = working_memory_state.get('safety_state')
        if safety_state and safety_state.get('content'):
            if safety_state['content'].safety_level in ['warning', 'critical']:
                analysis['safety_critical'] = True
        
        # Check social interaction
        social_context = working_memory_state.get('social_context')
        if social_context and social_context.get('content'):
            if social_context['content'].nearby_humans:
                analysis['social_interaction'] = True
        
        # Check task urgency
        task_context = working_memory_state.get('task_context')
        if task_context and task_context.get('content'):
            task = task_context['content']
            if task.status == 'active' and task.current_step < task.total_steps:
                analysis['task_urgent'] = True
        
        # Check learning opportunities
        if relevant_patterns and len(relevant_patterns) > 0:
            # Check if there are new patterns to learn
            new_patterns = [p for p in relevant_patterns if p.strength < 0.5]
            if new_patterns:
                analysis['learning_opportunity'] = True
        
        # Check resource constraints
        total_load = (self.cognitive_state.attention_load + 
                     self.cognitive_state.memory_load + 
                     self.cognitive_state.processing_load)
        if total_load > self.config['resource_reallocation_threshold']:
            analysis['resource_constrained'] = True
        
        # Check attention overload
        if self.cognitive_state.attention_load > 0.8:
            analysis['attention_overloaded'] = True
        
        return analysis
    
    def _determine_decision_type(self, situation_analysis: Dict[str, Any]) -> DecisionType:
        """Determine the type of decision needed based on situation analysis."""
        if situation_analysis['safety_critical']:
            return DecisionType.SAFETY_OVERRIDE
        elif situation_analysis['social_interaction']:
            return DecisionType.SOCIAL_ADJUSTMENT
        elif situation_analysis['resource_constrained']:
            return DecisionType.RESOURCE_ALLOCATION
        elif situation_analysis['learning_opportunity']:
            return DecisionType.LEARNING_PRIORITY
        elif situation_analysis['attention_overloaded']:
            return DecisionType.ATTENTION_MANAGEMENT
        else:
            return DecisionType.TASK_SELECTION
    
    def _generate_decision(self, 
                          decision_type: DecisionType,
                          situation_analysis: Dict[str, Any]) -> ExecutiveDecision:
        """Generate a specific decision based on type and situation."""
        decision_id = f"decision_{int(time.time() * 1000)}"
        
        if decision_type == DecisionType.SAFETY_OVERRIDE:
            return self._generate_safety_decision(decision_id, situation_analysis)
        elif decision_type == DecisionType.SOCIAL_ADJUSTMENT:
            return self._generate_social_decision(decision_id, situation_analysis)
        elif decision_type == DecisionType.RESOURCE_ALLOCATION:
            return self._generate_resource_decision(decision_id, situation_analysis)
        elif decision_type == DecisionType.LEARNING_PRIORITY:
            return self._generate_learning_decision(decision_id, situation_analysis)
        elif decision_type == DecisionType.ATTENTION_MANAGEMENT:
            return self._generate_attention_decision(decision_id, situation_analysis)
        else:
            return self._generate_task_decision(decision_id, situation_analysis)
    
    def _generate_safety_decision(self, 
                                 decision_id: str,
                                 situation_analysis: Dict[str, Any]) -> ExecutiveDecision:
        """Generate safety override decision."""
        actions = [
            {
                'action_type': 'safety_override',
                'priority': 'critical',
                'description': 'Activate emergency safety protocols',
                'parameters': {
                    'safety_level': self.cognitive_state.safety_status,
                    'override_all_tasks': True
                }
            },
            {
                'action_type': 'resource_reallocation',
                'priority': 'high',
                'description': 'Allocate maximum resources to safety monitoring',
                'parameters': {
                    'safety_allocation': 0.8,
                    'other_allocation': 0.2
                }
            }
        ]
        
        return ExecutiveDecision(
            decision_type=DecisionType.SAFETY_OVERRIDE,
            decision_id=decision_id,
            reasoning="Safety critical situation detected - overriding normal operations",
            confidence=0.95,
            priority=TaskPriority.CRITICAL,
            actions=actions,
            constraints=[{'type': 'safety_first', 'description': 'Safety takes precedence over all other operations'}],
            expected_outcome="System enters safe state with maximum safety monitoring"
        )
    
    def _generate_social_decision(self, 
                                 decision_id: str,
                                 situation_analysis: Dict[str, Any]) -> ExecutiveDecision:
        """Generate social adjustment decision."""
        actions = [
            {
                'action_type': 'social_priority',
                'priority': 'high',
                'description': 'Prioritize social interaction tasks',
                'parameters': {
                    'social_boost': self.config['social_priority_boost']
                }
            },
            {
                'action_type': 'attention_focus',
                'priority': 'medium',
                'description': 'Focus attention on social cues',
                'parameters': {
                    'social_attention_weight': 0.6
                }
            }
        ]
        
        return ExecutiveDecision(
            decision_type=DecisionType.SOCIAL_ADJUSTMENT,
            decision_id=decision_id,
            reasoning="Social interaction detected - adjusting behavior for appropriate interaction",
            confidence=0.8,
            priority=TaskPriority.HIGH,
            actions=actions,
            constraints=[{'type': 'social_appropriate', 'description': 'Maintain appropriate social behavior'}],
            expected_outcome="Enhanced social interaction with appropriate behavior"
        )
    
    def _generate_resource_decision(self, 
                                   decision_id: str,
                                   situation_analysis: Dict[str, Any]) -> ExecutiveDecision:
        """Generate resource allocation decision."""
        # Calculate optimal resource allocation
        attention_allocation = max(0.2, 1.0 - self.cognitive_state.memory_load - self.cognitive_state.processing_load)
        memory_allocation = max(0.2, 1.0 - self.cognitive_state.attention_load - self.cognitive_state.processing_load)
        processing_allocation = max(0.2, 1.0 - self.cognitive_state.attention_load - self.cognitive_state.memory_load)
        
        actions = [
            {
                'action_type': 'resource_reallocation',
                'priority': 'high',
                'description': 'Reallocate cognitive resources for optimal performance',
                'parameters': {
                    'attention_allocation': attention_allocation,
                    'memory_allocation': memory_allocation,
                    'processing_allocation': processing_allocation
                }
            }
        ]
        
        return ExecutiveDecision(
            decision_type=DecisionType.RESOURCE_ALLOCATION,
            decision_id=decision_id,
            reasoning="Resource constraints detected - optimizing allocation for better performance",
            confidence=0.7,
            priority=TaskPriority.HIGH,
            actions=actions,
            constraints=[{'type': 'resource_balance', 'description': 'Maintain minimum resource levels for all systems'}],
            expected_outcome="Improved cognitive performance through optimized resource allocation"
        )
    
    def _generate_learning_decision(self, 
                                   decision_id: str,
                                   situation_analysis: Dict[str, Any]) -> ExecutiveDecision:
        """Generate learning priority decision."""
        actions = [
            {
                'action_type': 'learning_priority',
                'priority': 'medium',
                'description': 'Prioritize learning from current experience',
                'parameters': {
                    'learning_threshold': self.config['learning_opportunity_threshold']
                }
            },
            {
                'action_type': 'memory_consolidation',
                'priority': 'low',
                'description': 'Consolidate new patterns in long-term memory',
                'parameters': {
                    'consolidation_priority': 0.6
                }
            }
        ]
        
        return ExecutiveDecision(
            decision_type=DecisionType.LEARNING_PRIORITY,
            decision_id=decision_id,
            reasoning="Learning opportunity detected - prioritizing knowledge acquisition",
            confidence=0.6,
            priority=TaskPriority.MEDIUM,
            actions=actions,
            constraints=[{'type': 'learning_safe', 'description': 'Ensure learning doesn\'t compromise safety'}],
            expected_outcome="Enhanced knowledge and improved future performance"
        )
    
    def _generate_attention_decision(self, 
                                    decision_id: str,
                                    situation_analysis: Dict[str, Any]) -> ExecutiveDecision:
        """Generate attention management decision."""
        actions = [
            {
                'action_type': 'attention_filtering',
                'priority': 'high',
                'description': 'Filter attention to reduce overload',
                'parameters': {
                    'max_attention_foci': 3,
                    'priority_threshold': 0.5
                }
            },
            {
                'action_type': 'task_suspension',
                'priority': 'medium',
                'description': 'Suspend low-priority tasks',
                'parameters': {
                    'suspend_threshold': TaskPriority.LOW
                }
            }
        ]
        
        return ExecutiveDecision(
            decision_type=DecisionType.ATTENTION_MANAGEMENT,
            decision_id=decision_id,
            reasoning="Attention overload detected - managing focus to improve performance",
            confidence=0.75,
            priority=TaskPriority.HIGH,
            actions=actions,
            constraints=[{'type': 'attention_safe', 'description': 'Maintain attention on safety-critical information'}],
            expected_outcome="Reduced cognitive load and improved focus"
        )
    
    def _generate_task_decision(self, 
                               decision_id: str,
                               situation_analysis: Dict[str, Any]) -> ExecutiveDecision:
        """Generate task selection decision."""
        # Select next task from queue or create idle task
        if self.task_queue:
            next_task = self.task_queue[0]
            actions = [
                {
                    'action_type': 'task_execution',
                    'priority': 'medium',
                    'description': f'Execute task: {next_task.description}',
                    'parameters': {
                        'task_id': next_task.task_id,
                        'priority': next_task.priority.value
                    }
                }
            ]
            reasoning = f"Selecting next task: {next_task.description}"
        else:
            actions = [
                {
                    'action_type': 'idle_behavior',
                    'priority': 'low',
                    'description': 'Perform idle behavior and monitoring',
                    'parameters': {
                        'monitoring_mode': True,
                        'energy_conservation': True
                    }
                }
            ]
            reasoning = "No urgent tasks - entering idle monitoring mode"
        
        return ExecutiveDecision(
            decision_type=DecisionType.TASK_SELECTION,
            decision_id=decision_id,
            reasoning=reasoning,
            confidence=0.8,
            priority=TaskPriority.MEDIUM,
            actions=actions,
            constraints=[{'type': 'task_safe', 'description': 'Ensure task execution maintains safety'}],
            expected_outcome="Successful task execution or appropriate idle behavior"
        )
    
    def add_task(self, task_plan: TaskPlan) -> str:
        """Add a task to the execution queue."""
        with self._lock:
            self.task_queue.append(task_plan)
            return task_plan.task_id
    
    def get_current_tasks(self) -> List[TaskPlan]:
        """Get list of currently active tasks."""
        with self._lock:
            return list(self.active_tasks.values())
    
    def get_cognitive_state(self) -> CognitiveState:
        """Get current cognitive state."""
        return self.cognitive_state
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """Get a summary of recent decisions."""
        with self._lock:
            recent_decisions = list(self.decision_history)[-10:]  # Last 10 decisions
            
            summary = {
                'total_decisions': len(self.decision_history),
                'recent_decisions': [
                    {
                        'type': decision.decision_type.value,
                        'priority': decision.priority.value,
                        'confidence': decision.confidence,
                        'timestamp': decision.timestamp
                    }
                    for decision in recent_decisions
                ],
                'decision_types': defaultdict(int),
                'average_confidence': 0.0
            }
            
            # Count decision types
            for decision in self.decision_history:
                summary['decision_types'][decision.decision_type.value] += 1
            
            # Calculate average confidence
            if self.decision_history:
                summary['average_confidence'] = sum(
                    d.confidence for d in self.decision_history
                ) / len(self.decision_history)
            
            return summary 