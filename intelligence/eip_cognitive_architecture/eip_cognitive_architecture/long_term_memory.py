"""
Long-term Memory for Cognitive Architecture

This module implements a long-term memory system that provides persistent storage
for learned patterns, experiences, and knowledge that can be retrieved and applied
to new situations.
"""

import time
import threading
import json
import pickle
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import os
from pathlib import Path


class MemoryCategory(Enum):
    """Categories of long-term memory"""
    EPISODIC = "episodic"  # Specific events and experiences
    SEMANTIC = "semantic"  # General knowledge and facts
    PROCEDURAL = "procedural"  # Skills and procedures
    EMOTIONAL = "emotional"  # Emotional associations
    SPATIAL = "spatial"  # Spatial knowledge and maps
    SOCIAL = "social"  # Social knowledge and relationships


@dataclass
class MemoryPattern:
    """Represents a learned pattern in long-term memory"""
    pattern_id: str
    category: MemoryCategory
    content: Any
    strength: float  # 0.0 to 1.0, represents how well learned
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    associations: List[str] = field(default_factory=list)  # IDs of related patterns
    context_tags: List[str] = field(default_factory=list)
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodicMemory:
    """Represents an episodic memory (specific event)"""
    event_id: str
    description: str
    timestamp: float
    location: Optional[Tuple[float, float, float]] = None
    participants: List[str] = field(default_factory=list)
    emotions: List[str] = field(default_factory=list)
    outcome: str = "unknown"
    lessons_learned: List[str] = field(default_factory=list)
    sensory_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticKnowledge:
    """Represents semantic knowledge (general facts)"""
    concept_id: str
    concept_name: str
    definition: str
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    confidence: float = 0.5
    source: str = "learned"


@dataclass
class ProceduralSkill:
    """Represents a procedural skill or procedure"""
    skill_id: str
    skill_name: str
    description: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    practice_count: int = 0
    last_practiced: Optional[float] = None
    difficulty: float = 0.5  # 0.0 to 1.0
    domain: str = "general"


@dataclass
class EmotionalAssociation:
    """Represents an emotional association"""
    stimulus_id: str
    stimulus_type: str  # 'object', 'person', 'situation', 'location'
    emotion: str
    intensity: float  # 0.0 to 1.0
    valence: float  # -1.0 to 1.0 (negative to positive)
    context: Dict[str, Any] = field(default_factory=dict)
    learned_time: float = field(default_factory=time.time)


@dataclass
class SpatialKnowledge:
    """Represents spatial knowledge and mapping"""
    location_id: str
    location_name: str
    coordinates: Tuple[float, float, float]
    landmarks: List[Dict[str, Any]] = field(default_factory=list)
    paths: List[Dict[str, Any]] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    last_visited: Optional[float] = None
    visit_count: int = 0


@dataclass
class SocialKnowledge:
    """Represents social knowledge and relationships"""
    person_id: str
    name: str
    relationship_type: str  # 'friend', 'family', 'colleague', 'stranger'
    trust_level: float  # 0.0 to 1.0
    preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    communication_style: Dict[str, Any] = field(default_factory=dict)
    last_interaction: Optional[float] = None


class LongTermMemory:
    """
    Long-term memory system for persistent storage of learned patterns and experiences.
    
    This component manages:
    - Episodic memories of specific events
    - Semantic knowledge and facts
    - Procedural skills and procedures
    - Emotional associations
    - Spatial knowledge and mapping
    - Social knowledge and relationships
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, storage_path: Optional[str] = None):
        """
        Initialize the long-term memory system.
        
        Args:
            config: Configuration dictionary for memory parameters
            storage_path: Path for persistent storage
        """
        self.config = config or self._get_default_config()
        self.storage_path = storage_path or "long_term_memory"
        
        # Create storage directory
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Memory storage organized by category
        self.memory_stores: Dict[MemoryCategory, Dict[str, MemoryPattern]] = defaultdict(dict)
        
        # Index for fast retrieval
        self.content_index: Dict[str, List[str]] = defaultdict(list)
        self.context_index: Dict[str, List[str]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self.performance_metrics = {
            'total_patterns': 0,
            'retrieval_speed': 0.0,
            'storage_utilization': 0.0,
            'consolidation_rate': 0.0
        }
        
        # Load existing memories
        self._load_memories()
        
        # Initialize with default knowledge
        self._initialize_default_knowledge()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for long-term memory."""
        return {
            'max_patterns_per_category': 1000,
            'min_strength_threshold': 0.1,
            'consolidation_threshold': 0.7,
            'forgetting_rate': 0.01,
            'association_strength_decay': 0.05,
            'retrieval_limit': 50,
            'auto_save_interval': 300,  # 5 minutes
            'backup_interval': 3600,  # 1 hour
        }
    
    def _load_memories(self):
        """Load existing memories from persistent storage."""
        try:
            for category in MemoryCategory:
                file_path = os.path.join(self.storage_path, f"{category.value}_memories.pkl")
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        patterns = pickle.load(f)
                        self.memory_stores[category] = patterns
                        
                        # Rebuild indices
                        for pattern_id, pattern in patterns.items():
                            self._add_to_indices(pattern_id, pattern)
        except Exception as e:
            print(f"Warning: Could not load existing memories: {e}")
    
    def _save_memories(self):
        """Save memories to persistent storage."""
        try:
            for category, patterns in self.memory_stores.items():
                file_path = os.path.join(self.storage_path, f"{category.value}_memories.pkl")
                with open(file_path, 'wb') as f:
                    pickle.dump(patterns, f)
        except Exception as e:
            print(f"Warning: Could not save memories: {e}")
    
    def _initialize_default_knowledge(self):
        """Initialize with default knowledge patterns."""
        # Basic safety knowledge
        safety_pattern = MemoryPattern(
            pattern_id="safety_basic",
            category=MemoryCategory.SEMANTIC,
            content=SemanticKnowledge(
                concept_id="safety_basic",
                concept_name="Basic Safety",
                definition="Fundamental safety principles for robot operation",
                properties={
                    "priority": "highest",
                    "domain": "safety"
                }
            ),
            strength=1.0,
            context_tags=["safety", "fundamental"],
            confidence=1.0
        )
        self.store_pattern(safety_pattern)
        
        # Basic social knowledge
        social_pattern = MemoryPattern(
            pattern_id="social_basic",
            category=MemoryCategory.SOCIAL,
            content=SemanticKnowledge(
                concept_id="social_basic",
                concept_name="Social Interaction",
                definition="Basic principles of human-robot interaction",
                properties={
                    "domain": "social",
                    "importance": "high"
                }
            ),
            strength=0.8,
            context_tags=["social", "interaction"],
            confidence=0.8
        )
        self.store_pattern(social_pattern)
    
    def store_pattern(self, pattern: MemoryPattern) -> str:
        """
        Store a pattern in long-term memory.
        
        Args:
            pattern: Memory pattern to store
            
        Returns:
            Pattern ID
        """
        with self._lock:
            # Generate ID if not provided
            if not pattern.pattern_id:
                content_hash = hashlib.md5(str(pattern.content).encode()).hexdigest()
                pattern.pattern_id = f"{pattern.category.value}_{content_hash[:8]}"
            
            # Store in appropriate category
            self.memory_stores[pattern.category][pattern.pattern_id] = pattern
            
            # Add to indices
            self._add_to_indices(pattern.pattern_id, pattern)
            
            # Update performance metrics
            self.performance_metrics['total_patterns'] = sum(
                len(patterns) for patterns in self.memory_stores.values()
            )
            
            return pattern.pattern_id
    
    def _add_to_indices(self, pattern_id: str, pattern: MemoryPattern):
        """Add pattern to search indices."""
        # Content index
        content_str = str(pattern.content).lower()
        words = content_str.split()
        for word in words:
            if len(word) > 2:  # Only index meaningful words
                self.content_index[word].append(pattern_id)
        
        # Context index
        for tag in pattern.context_tags:
            self.context_index[tag].append(pattern_id)
    
    def retrieve_patterns(self, 
                         category: Optional[MemoryCategory] = None,
                         query: Optional[str] = None,
                         context_tags: Optional[List[str]] = None,
                         min_strength: Optional[float] = None,
                         limit: Optional[int] = None) -> List[MemoryPattern]:
        """
        Retrieve patterns from long-term memory.
        
        Args:
            category: Specific memory category to search
            query: Text query for content-based search
            context_tags: Tags to match against
            min_strength: Minimum strength threshold
            limit: Maximum number of patterns to return
            
        Returns:
            List of matching memory patterns
        """
        with self._lock:
            start_time = time.time()
            
            # Get candidate patterns
            candidates = []
            
            if category:
                # Search specific category
                if category in self.memory_stores:
                    candidates.extend(self.memory_stores[category].values())
            else:
                # Search all categories
                for patterns in self.memory_stores.values():
                    candidates.extend(patterns.values())
            
            # Apply filters
            filtered_candidates = []
            
            for pattern in candidates:
                # Apply strength filter
                if min_strength and pattern.strength < min_strength:
                    continue
                
                # Apply context tag filter
                if context_tags:
                    if not any(tag in pattern.context_tags for tag in context_tags):
                        continue
                
                # Apply query filter
                if query:
                    query_words = query.lower().split()
                    content_str = str(pattern.content).lower()
                    if not any(word in content_str for word in query_words):
                        continue
                
                filtered_candidates.append(pattern)
            
            # Sort by relevance (strength * confidence)
            filtered_candidates.sort(
                key=lambda x: x.strength * x.confidence, 
                reverse=True
            )
            
            # Apply limit
            if limit:
                filtered_candidates = filtered_candidates[:limit]
            
            # Update access statistics
            for pattern in filtered_candidates:
                pattern.access_count += 1
                pattern.last_accessed = time.time()
            
            # Update performance metrics
            self.performance_metrics['retrieval_speed'] = time.time() - start_time
            
            return filtered_candidates
    
    def get_relevant_patterns(self, 
                             current_context: Dict[str, Any],
                             attention_foci: List[Any]) -> List[MemoryPattern]:
        """
        Get patterns relevant to current context and attention foci.
        
        Args:
            current_context: Current task and environmental context
            attention_foci: Current attention foci
            
        Returns:
            List of relevant memory patterns
        """
        # Extract context tags
        context_tags = []
        
        # Add task-related tags
        if 'current_task' in current_context:
            task = current_context['current_task']
            context_tags.extend([
                f"task_{task.get('type', 'unknown')}",
                f"status_{task.get('status', 'unknown')}"
            ])
        
        # Add attention-related tags
        for focus in attention_foci:
            if hasattr(focus, 'attention_type'):
                context_tags.append(f"attention_{focus.attention_type.value}")
            if hasattr(focus, 'metadata'):
                metadata = focus.metadata
                for key, value in metadata.items():
                    if isinstance(value, str):
                        context_tags.append(f"{key}_{value}")
        
        # Add safety tags if safety concerns detected
        if any(hasattr(f, 'attention_type') and f.attention_type.value == 'safety' 
               for f in attention_foci):
            context_tags.append("safety_critical")
        
        # Add social tags if social interactions detected
        if any(hasattr(f, 'attention_type') and f.attention_type.value == 'social' 
               for f in attention_foci):
            context_tags.append("social_interaction")
        
        # Retrieve relevant patterns
        relevant_patterns = self.retrieve_patterns(
            context_tags=context_tags,
            min_strength=self.config['min_strength_threshold'],
            limit=self.config['retrieval_limit']
        )
        
        return relevant_patterns
    
    def learn_from_experience(self, 
                             experience_data: Dict[str, Any],
                             outcome: str,
                             lessons_learned: List[str]) -> str:
        """
        Learn from an experience and store it in episodic memory.
        
        Args:
            experience_data: Data about the experience
            outcome: Outcome of the experience ('success', 'failure', 'partial')
            lessons_learned: Lessons learned from the experience
            
        Returns:
            Memory pattern ID
        """
        # Create episodic memory
        episodic_memory = EpisodicMemory(
            event_id=f"event_{int(time.time() * 1000)}",
            description=experience_data.get('description', 'Unknown event'),
            timestamp=time.time(),
            location=experience_data.get('location'),
            participants=experience_data.get('participants', []),
            emotions=experience_data.get('emotions', []),
            outcome=outcome,
            lessons_learned=lessons_learned,
            sensory_details=experience_data.get('sensory_details', {})
        )
        
        # Create memory pattern
        pattern = MemoryPattern(
            pattern_id=f"episodic_{episodic_memory.event_id}",
            category=MemoryCategory.EPISODIC,
            content=episodic_memory,
            strength=0.5,  # Initial strength
            context_tags=[
                f"outcome_{outcome}",
                f"domain_{experience_data.get('domain', 'general')}"
            ] + lessons_learned,
            confidence=0.7
        )
        
        # Store the pattern
        pattern_id = self.store_pattern(pattern)
        
        # Extract and store semantic knowledge from lessons
        for lesson in lessons_learned:
            self._extract_semantic_knowledge(lesson, experience_data)
        
        return pattern_id
    
    def _extract_semantic_knowledge(self, lesson: str, context: Dict[str, Any]):
        """Extract semantic knowledge from lessons learned."""
        # Simple extraction - in practice, this would use NLP
        if "safety" in lesson.lower():
            safety_knowledge = SemanticKnowledge(
                concept_id=f"safety_{int(time.time() * 1000)}",
                concept_name="Safety Lesson",
                definition=lesson,
                properties={"source": "experience", "domain": "safety"},
                confidence=0.6
            )
            
            pattern = MemoryPattern(
                pattern_id=f"semantic_{safety_knowledge.concept_id}",
                category=MemoryCategory.SEMANTIC,
                content=safety_knowledge,
                strength=0.4,
                context_tags=["safety", "lesson_learned"],
                confidence=0.6
            )
            
            self.store_pattern(pattern)
    
    def update_pattern_strength(self, pattern_id: str, new_strength: float):
        """Update the strength of a memory pattern."""
        with self._lock:
            for patterns in self.memory_stores.values():
                if pattern_id in patterns:
                    patterns[pattern_id].strength = max(0.0, min(1.0, new_strength))
                    break
    
    def consolidate_memories(self):
        """Consolidate and strengthen frequently accessed memories."""
        with self._lock:
            for category, patterns in self.memory_stores.items():
                for pattern_id, pattern in patterns.items():
                    # Strengthen frequently accessed patterns
                    if pattern.access_count > 10:
                        strength_boost = min(0.1, pattern.access_count * 0.01)
                        pattern.strength = min(1.0, pattern.strength + strength_boost)
                    
                    # Weaken rarely accessed patterns
                    age = time.time() - pattern.last_accessed
                    if age > 86400:  # 24 hours
                        decay = self.config['forgetting_rate'] * (age / 86400)
                        pattern.strength = max(0.0, pattern.strength - decay)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        with self._lock:
            stats = {
                'total_patterns': sum(len(patterns) for patterns in self.memory_stores.values()),
                'patterns_by_category': {
                    category.value: len(patterns) 
                    for category, patterns in self.memory_stores.items()
                },
                'average_strength': 0.0,
                'most_accessed': [],
                'recent_patterns': []
            }
            
            # Calculate average strength
            all_patterns = []
            for patterns in self.memory_stores.values():
                all_patterns.extend(patterns.values())
            
            if all_patterns:
                stats['average_strength'] = sum(p.strength for p in all_patterns) / len(all_patterns)
                
                # Most accessed patterns
                most_accessed = sorted(all_patterns, key=lambda x: x.access_count, reverse=True)[:5]
                stats['most_accessed'] = [
                    {'id': p.pattern_id, 'access_count': p.access_count, 'strength': p.strength}
                    for p in most_accessed
                ]
                
                # Recent patterns
                recent_patterns = sorted(all_patterns, key=lambda x: x.last_accessed, reverse=True)[:5]
                stats['recent_patterns'] = [
                    {'id': p.pattern_id, 'last_accessed': p.last_accessed, 'strength': p.strength}
                    for p in recent_patterns
                ]
            
            return stats
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def cleanup(self):
        """Clean up the memory system and save to persistent storage."""
        with self._lock:
            # Consolidate memories
            self.consolidate_memories()
            
            # Remove very weak patterns
            for category, patterns in list(self.memory_stores.items()):
                weak_patterns = [
                    pattern_id for pattern_id, pattern in patterns.items()
                    if pattern.strength < self.config['min_strength_threshold']
                ]
                for pattern_id in weak_patterns:
                    del patterns[pattern_id]
            
            # Save to persistent storage
            self._save_memories() 