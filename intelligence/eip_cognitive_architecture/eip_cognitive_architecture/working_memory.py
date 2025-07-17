"""
Working Memory for Cognitive Architecture

This module implements a working memory system that provides short-term storage
for current task context, active information, and temporary cognitive state.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np


class MemoryType(Enum):
    """Types of working memory content"""
    TASK_CONTEXT = "task_context"
    SENSORY_BUFFER = "sensory_buffer"
    ATTENTION_FOCUS = "attention_focus"
    PLANNING_STATE = "planning_state"
    EXECUTION_STATE = "execution_state"
    SOCIAL_CONTEXT = "social_context"
    SAFETY_STATE = "safety_state"


@dataclass
class MemoryItem:
    """Represents an item in working memory"""
    memory_type: MemoryType
    content: Any
    priority: float  # 0.0 to 1.0
    timestamp: float
    decay_rate: float  # Rate at which memory decays
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskContext:
    """Represents current task context"""
    task_id: str
    task_type: str
    description: str
    current_step: int
    total_steps: int
    status: str  # 'active', 'paused', 'completed', 'failed'
    start_time: float
    estimated_duration: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensoryBuffer:
    """Represents sensory information buffer"""
    modality: str  # 'visual', 'audio', 'tactile', 'proprioceptive'
    data: Any
    timestamp: float
    confidence: float
    processed: bool = False


@dataclass
class PlanningState:
    """Represents current planning state"""
    current_plan: Optional[Dict[str, Any]] = None
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    current_step_index: int = 0
    plan_confidence: float = 0.0
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExecutionState:
    """Represents current execution state"""
    current_action: Optional[str] = None
    action_progress: float = 0.0  # 0.0 to 1.0
    action_start_time: Optional[float] = None
    expected_completion: Optional[float] = None
    success_metrics: Dict[str, float] = field(default_factory=dict)
    error_state: Optional[Dict[str, Any]] = None


@dataclass
class SocialContext:
    """Represents social interaction context"""
    nearby_humans: List[Dict[str, Any]] = field(default_factory=list)
    current_interaction: Optional[Dict[str, Any]] = None
    social_rules: List[str] = field(default_factory=list)
    cultural_context: Optional[str] = None
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SafetyState:
    """Represents current safety state"""
    safety_level: str  # 'safe', 'caution', 'warning', 'critical'
    active_violations: List[Dict[str, Any]] = field(default_factory=list)
    safety_zones: List[Dict[str, Any]] = field(default_factory=list)
    emergency_procedures: List[str] = field(default_factory=list)
    last_safety_check: float = field(default_factory=time.time)


class WorkingMemory:
    """
    Working memory system for short-term storage of active cognitive information.
    
    This component manages:
    - Task context and current goals
    - Sensory information buffer
    - Attention focus information
    - Planning and execution states
    - Social interaction context
    - Safety monitoring state
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the working memory system.
        
        Args:
            config: Configuration dictionary for memory parameters
        """
        self.config = config or self._get_default_config()
        
        # Memory storage organized by type
        self.memory_stores: Dict[MemoryType, deque] = defaultdict(
            lambda: deque(maxlen=self.config['max_items_per_type'])
        )
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self.performance_metrics = {
            'memory_utilization': 0.0,
            'access_speed': 0.0,
            'decay_rate': 0.0,
            'capacity_used': 0
        }
        
        # Memory access patterns for optimization
        self.access_patterns = defaultdict(int)
        
        # Initialize with default states
        self._initialize_default_states()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for working memory."""
        return {
            'max_items_per_type': 50,
            'default_decay_rate': 0.1,
            'high_priority_decay_rate': 0.05,
            'low_priority_decay_rate': 0.2,
            'cleanup_interval_seconds': 1.0,
            'max_total_items': 200,
            'memory_consolidation_threshold': 0.8
        }
    
    def _initialize_default_states(self):
        """Initialize working memory with default states."""
        current_time = time.time()
        
        # Initialize task context
        default_task = TaskContext(
            task_id="idle",
            task_type="idle",
            description="System idle state",
            current_step=0,
            total_steps=1,
            status="active",
            start_time=current_time
        )
        self.store_memory(MemoryType.TASK_CONTEXT, default_task, 0.5)
        
        # Initialize planning state
        default_planning = PlanningState()
        self.store_memory(MemoryType.PLANNING_STATE, default_planning, 0.3)
        
        # Initialize execution state
        default_execution = ExecutionState()
        self.store_memory(MemoryType.EXECUTION_STATE, default_execution, 0.3)
        
        # Initialize social context
        default_social = SocialContext()
        self.store_memory(MemoryType.SOCIAL_CONTEXT, default_social, 0.4)
        
        # Initialize safety state
        default_safety = SafetyState()
        self.store_memory(MemoryType.SAFETY_STATE, default_safety, 0.8)
    
    def store_memory(self, 
                     memory_type: MemoryType,
                     content: Any,
                     priority: float,
                     decay_rate: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store an item in working memory.
        
        Args:
            memory_type: Type of memory to store
            content: Content to store
            priority: Priority of the memory item (0.0 to 1.0)
            decay_rate: Custom decay rate (optional)
            metadata: Additional metadata
            
        Returns:
            Memory item ID
        """
        with self._lock:
            # Generate unique ID
            memory_id = f"{memory_type.value}_{int(time.time() * 1000)}"
            
            # Use default decay rate if not specified
            if decay_rate is None:
                if priority > 0.7:
                    decay_rate = self.config['high_priority_decay_rate']
                elif priority < 0.3:
                    decay_rate = self.config['low_priority_decay_rate']
                else:
                    decay_rate = self.config['default_decay_rate']
            
            # Create memory item
            memory_item = MemoryItem(
                memory_type=memory_type,
                content=content,
                priority=priority,
                timestamp=time.time(),
                decay_rate=decay_rate,
                metadata=metadata or {}
            )
            
            # Store in appropriate memory store
            self.memory_stores[memory_type].append(memory_item)
            
            # Update access patterns
            self.access_patterns[memory_type] += 1
            
            # Check capacity and cleanup if needed
            self._check_capacity_and_cleanup()
            
            return memory_id
    
    def retrieve_memory(self, 
                       memory_type: MemoryType,
                       filter_func: Optional[callable] = None,
                       limit: Optional[int] = None) -> List[MemoryItem]:
        """
        Retrieve items from working memory.
        
        Args:
            memory_type: Type of memory to retrieve
            filter_func: Optional filter function to apply
            limit: Maximum number of items to return
            
        Returns:
            List of memory items
        """
        with self._lock:
            if memory_type not in self.memory_stores:
                return []
            
            # Get items from memory store
            items = list(self.memory_stores[memory_type])
            
            # Apply decay
            current_time = time.time()
            for item in items:
                age = current_time - item.timestamp
                decay = item.decay_rate * age
                item.priority = max(0.0, item.priority - decay)
                item.last_accessed = current_time
                item.access_count += 1
            
            # Apply filter if provided
            if filter_func:
                items = [item for item in items if filter_func(item)]
            
            # Sort by priority (highest first)
            items.sort(key=lambda x: x.priority, reverse=True)
            
            # Apply limit
            if limit:
                items = items[:limit]
            
            return items
    
    def update_context(self, 
                      focused_attention: List[Any],
                      current_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Update working memory context based on attention focus and current context.
        
        Args:
            focused_attention: Current attention foci from attention mechanism
            current_context: Current task and environmental context
        """
        with self._lock:
            current_time = time.time()
            
            # Update attention focus memory
            attention_memory = {
                'foci': focused_attention,
                'timestamp': current_time,
                'num_foci': len(focused_attention)
            }
            self.store_memory(MemoryType.ATTENTION_FOCUS, attention_memory, 0.6)
            
            # Update task context if provided
            if current_context and 'current_task' in current_context:
                task_context = current_context['current_task']
                if isinstance(task_context, dict):
                    # Convert dict to TaskContext if needed
                    task_obj = TaskContext(
                        task_id=task_context.get('id', 'unknown'),
                        task_type=task_context.get('type', 'unknown'),
                        description=task_context.get('description', ''),
                        current_step=task_context.get('current_step', 0),
                        total_steps=task_context.get('total_steps', 1),
                        status=task_context.get('status', 'active'),
                        start_time=task_context.get('start_time', current_time),
                        estimated_duration=task_context.get('estimated_duration'),
                        dependencies=task_context.get('dependencies', []),
                        parameters=task_context.get('parameters', {})
                    )
                    self.store_memory(MemoryType.TASK_CONTEXT, task_obj, 0.8)
            
            # Update social context if attention includes social elements
            social_foci = [f for f in focused_attention if hasattr(f, 'attention_type') 
                          and f.attention_type.value == 'social']
            if social_foci:
                self._update_social_context(social_foci, current_context)
            
            # Update safety context if attention includes safety elements
            safety_foci = [f for f in focused_attention if hasattr(f, 'attention_type') 
                          and f.attention_type.value == 'safety']
            if safety_foci:
                self._update_safety_context(safety_foci, current_context)
    
    def _update_social_context(self, 
                              social_foci: List[Any],
                              current_context: Optional[Dict[str, Any]]) -> None:
        """Update social context based on social attention foci."""
        # Get current social context
        social_items = self.retrieve_memory(MemoryType.SOCIAL_CONTEXT, limit=1)
        current_social = social_items[0].content if social_items else SocialContext()
        
        # Update based on social foci
        for focus in social_foci:
            if hasattr(focus, 'metadata'):
                metadata = focus.metadata
                
                # Update nearby humans
                if 'face_id' in metadata:
                    human_info = {
                        'id': metadata['face_id'],
                        'expression': metadata.get('expression', 'neutral'),
                        'location': focus.location,
                        'timestamp': time.time()
                    }
                    current_social.nearby_humans.append(human_info)
                
                # Update current interaction
                if 'speaker_id' in metadata:
                    interaction = {
                        'speaker_id': metadata['speaker_id'],
                        'content': metadata.get('content', ''),
                        'timestamp': time.time(),
                        'type': 'speech'
                    }
                    current_social.current_interaction = interaction
                    current_social.interaction_history.append(interaction)
        
        # Store updated social context
        self.store_memory(MemoryType.SOCIAL_CONTEXT, current_social, 0.7)
    
    def _update_safety_context(self, 
                              safety_foci: List[Any],
                              current_context: Optional[Dict[str, Any]]) -> None:
        """Update safety context based on safety attention foci."""
        # Get current safety state
        safety_items = self.retrieve_memory(MemoryType.SAFETY_STATE, limit=1)
        current_safety = safety_items[0].content if safety_items else SafetyState()
        
        # Update based on safety foci
        for focus in safety_foci:
            if hasattr(focus, 'metadata'):
                metadata = focus.metadata
                
                # Update safety violations
                if 'safety_type' in metadata:
                    violation = {
                        'type': metadata['safety_type'],
                        'severity': metadata.get('severity', 'medium'),
                        'location': focus.location,
                        'timestamp': time.time()
                    }
                    current_safety.active_violations.append(violation)
                    
                    # Update safety level
                    if metadata.get('severity') == 'critical':
                        current_safety.safety_level = 'critical'
                    elif metadata.get('severity') == 'high':
                        current_safety.safety_level = 'warning'
                    elif current_safety.safety_level == 'safe':
                        current_safety.safety_level = 'caution'
        
        current_safety.last_safety_check = time.time()
        
        # Store updated safety state
        self.store_memory(MemoryType.SAFETY_STATE, current_safety, 0.9)
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current state of working memory.
        
        Returns:
            Dictionary containing current state of all memory types
        """
        with self._lock:
            state = {}
            
            for memory_type in MemoryType:
                items = self.retrieve_memory(memory_type, limit=5)
                if items:
                    # Get the highest priority item
                    state[memory_type.value] = {
                        'content': items[0].content,
                        'priority': items[0].priority,
                        'timestamp': items[0].timestamp
                    }
                else:
                    state[memory_type.value] = None
            
            return state
    
    def get_task_context(self) -> Optional[TaskContext]:
        """Get current task context."""
        items = self.retrieve_memory(MemoryType.TASK_CONTEXT, limit=1)
        return items[0].content if items else None
    
    def get_planning_state(self) -> Optional[PlanningState]:
        """Get current planning state."""
        items = self.retrieve_memory(MemoryType.PLANNING_STATE, limit=1)
        return items[0].content if items else None
    
    def get_execution_state(self) -> Optional[ExecutionState]:
        """Get current execution state."""
        items = self.retrieve_memory(MemoryType.EXECUTION_STATE, limit=1)
        return items[0].content if items else None
    
    def get_social_context(self) -> Optional[SocialContext]:
        """Get current social context."""
        items = self.retrieve_memory(MemoryType.SOCIAL_CONTEXT, limit=1)
        return items[0].content if items else None
    
    def get_safety_state(self) -> Optional[SafetyState]:
        """Get current safety state."""
        items = self.retrieve_memory(MemoryType.SAFETY_STATE, limit=1)
        return items[0].content if items else None
    
    def update_planning_state(self, planning_state: PlanningState) -> None:
        """Update planning state in working memory."""
        self.store_memory(MemoryType.PLANNING_STATE, planning_state, 0.8)
    
    def update_execution_state(self, execution_state: ExecutionState) -> None:
        """Update execution state in working memory."""
        self.store_memory(MemoryType.EXECUTION_STATE, execution_state, 0.8)
    
    def _check_capacity_and_cleanup(self) -> None:
        """Check memory capacity and perform cleanup if needed."""
        total_items = sum(len(store) for store in self.memory_stores.values())
        
        if total_items > self.config['max_total_items']:
            # Remove lowest priority items
            all_items = []
            for memory_type, store in self.memory_stores.items():
                for item in store:
                    all_items.append((memory_type, item))
            
            # Sort by priority (lowest first)
            all_items.sort(key=lambda x: x[1].priority)
            
            # Remove items until under capacity
            items_to_remove = total_items - self.config['max_total_items']
            for i in range(items_to_remove):
                if i < len(all_items):
                    memory_type, item = all_items[i]
                    self.memory_stores[memory_type].remove(item)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        with self._lock:
            total_items = sum(len(store) for store in self.memory_stores.values())
            max_capacity = len(MemoryType) * self.config['max_items_per_type']
            
            self.performance_metrics['memory_utilization'] = total_items / max_capacity
            self.performance_metrics['capacity_used'] = total_items
            
            return self.performance_metrics.copy()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of working memory state."""
        with self._lock:
            summary = {
                'total_items': sum(len(store) for store in self.memory_stores.values()),
                'items_by_type': {memory_type.value: len(store) 
                                 for memory_type, store in self.memory_stores.items()},
                'performance_metrics': self.get_performance_metrics(),
                'access_patterns': dict(self.access_patterns)
            }
            
            return summary 