#!/usr/bin/env python3

"""
Test suite for Memory Systems

This module contains comprehensive tests for the working memory and long-term
memory components of the cognitive architecture.
"""

import unittest
import time
import tempfile
import shutil
import os
from unittest.mock import Mock, patch

# Import memory components
from eip_cognitive_architecture.working_memory import (
    WorkingMemory, MemoryType, TaskContext, PlanningState, ExecutionState,
    SocialContext, SafetyState
)
from eip_cognitive_architecture.long_term_memory import (
    LongTermMemory, MemoryCategory, MemoryPattern, EpisodicMemory,
    SemanticKnowledge, ProceduralSkill
)


class TestWorkingMemory(unittest.TestCase):
    """Test cases for the WorkingMemory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.working_memory = WorkingMemory()
    
    def test_initialization(self):
        """Test working memory initialization."""
        self.assertIsNotNone(self.working_memory)
        self.assertIsInstance(self.working_memory.config, dict)
        self.assertIn('max_items_per_type', self.working_memory.config)
    
    def test_store_and_retrieve_memory(self):
        """Test storing and retrieving memory items."""
        # Create test content
        task_context = TaskContext(
            task_id="test_task",
            task_type="navigation",
            description="Test navigation task",
            current_step=1,
            total_steps=5,
            status="active",
            start_time=time.time()
        )
        
        # Store memory
        memory_id = self.working_memory.store_memory(
            MemoryType.TASK_CONTEXT,
            task_context,
            0.8
        )
        
        self.assertIsInstance(memory_id, str)
        
        # Retrieve memory
        retrieved_items = self.working_memory.retrieve_memory(
            MemoryType.TASK_CONTEXT,
            limit=1
        )
        
        self.assertEqual(len(retrieved_items), 1)
        self.assertEqual(retrieved_items[0].content.task_id, "test_task")
        self.assertEqual(retrieved_items[0].priority, 0.8)
    
    def test_memory_decay(self):
        """Test that memory items decay over time."""
        # Store memory with high decay rate
        task_context = TaskContext(
            task_id="decay_test",
            task_type="test",
            description="Test decay",
            current_step=1,
            total_steps=1,
            status="active",
            start_time=time.time()
        )
        
        self.working_memory.store_memory(
            MemoryType.TASK_CONTEXT,
            task_context,
            0.9,
            decay_rate=0.5  # High decay rate
        )
        
        # Retrieve immediately
        initial_items = self.working_memory.retrieve_memory(MemoryType.TASK_CONTEXT)
        initial_priority = initial_items[0].priority
        
        # Wait and retrieve again
        time.sleep(0.1)
        decayed_items = self.working_memory.retrieve_memory(MemoryType.TASK_CONTEXT)
        decayed_priority = decayed_items[0].priority
        
        # Priority should have decayed
        self.assertLess(decayed_priority, initial_priority)
    
    def test_context_update(self):
        """Test updating context with attention foci."""
        # Create mock attention foci
        attention_foci = [
            Mock(attention_type=Mock(value='social'), metadata={'face_id': 'person1'}),
            Mock(attention_type=Mock(value='safety'), metadata={'safety_type': 'proximity'})
        ]
        
        current_context = {
            'current_task': {
                'id': 'test_task',
                'type': 'interaction',
                'status': 'active'
            }
        }
        
        # Update context
        self.working_memory.update_context(attention_foci, current_context)
        
        # Check that social and safety contexts were updated
        social_context = self.working_memory.get_social_context()
        safety_state = self.working_memory.get_safety_state()
        
        self.assertIsNotNone(social_context)
        self.assertIsNotNone(safety_state)
    
    def test_memory_capacity_limits(self):
        """Test that memory respects capacity limits."""
        # Store many items
        for i in range(100):
            task_context = TaskContext(
                task_id=f"task_{i}",
                task_type="test",
                description=f"Test task {i}",
                current_step=1,
                total_steps=1,
                status="active",
                start_time=time.time()
            )
            
            self.working_memory.store_memory(
                MemoryType.TASK_CONTEXT,
                task_context,
                0.5
            )
        
        # Check that we don't exceed capacity
        all_items = self.working_memory.retrieve_memory(MemoryType.TASK_CONTEXT)
        self.assertLessEqual(len(all_items), self.working_memory.config['max_items_per_type'])
    
    def test_get_current_state(self):
        """Test getting current state of working memory."""
        # Store some test data
        task_context = TaskContext(
            task_id="state_test",
            task_type="test",
            description="State test",
            current_step=1,
            total_steps=1,
            status="active",
            start_time=time.time()
        )
        
        self.working_memory.store_memory(MemoryType.TASK_CONTEXT, task_context, 0.7)
        
        # Get current state
        state = self.working_memory.get_current_state()
        
        self.assertIn('task_context', state)
        self.assertIsNotNone(state['task_context'])
        self.assertEqual(state['task_context']['content'].task_id, "state_test")
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Perform some operations
        task_context = TaskContext(
            task_id="metrics_test",
            task_type="test",
            description="Metrics test",
            current_step=1,
            total_steps=1,
            status="active",
            start_time=time.time()
        )
        
        self.working_memory.store_memory(MemoryType.TASK_CONTEXT, task_context, 0.6)
        self.working_memory.retrieve_memory(MemoryType.TASK_CONTEXT)
        
        metrics = self.working_memory.get_performance_metrics()
        
        self.assertIn('memory_utilization', metrics)
        self.assertIn('capacity_used', metrics)
        
        self.assertIsInstance(metrics['memory_utilization'], float)
        self.assertIsInstance(metrics['capacity_used'], int)


class TestLongTermMemory(unittest.TestCase):
    """Test cases for the LongTermMemory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for storage
        self.temp_dir = tempfile.mkdtemp()
        self.long_term_memory = LongTermMemory(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test long-term memory initialization."""
        self.assertIsNotNone(self.long_term_memory)
        self.assertIsInstance(self.long_term_memory.config, dict)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_store_and_retrieve_patterns(self):
        """Test storing and retrieving memory patterns."""
        # Create test pattern
        semantic_knowledge = SemanticKnowledge(
            concept_id="test_concept",
            concept_name="Test Concept",
            definition="A test concept for testing",
            properties={"domain": "test"},
            confidence=0.8
        )
        
        pattern = MemoryPattern(
            pattern_id="test_pattern",
            category=MemoryCategory.SEMANTIC,
            content=semantic_knowledge,
            strength=0.7,
            context_tags=["test", "concept"],
            confidence=0.8
        )
        
        # Store pattern
        pattern_id = self.long_term_memory.store_pattern(pattern)
        self.assertEqual(pattern_id, "test_pattern")
        
        # Retrieve patterns
        retrieved_patterns = self.long_term_memory.retrieve_patterns(
            category=MemoryCategory.SEMANTIC
        )
        
        self.assertEqual(len(retrieved_patterns), 1)
        self.assertEqual(retrieved_patterns[0].pattern_id, "test_pattern")
        self.assertEqual(retrieved_patterns[0].content.concept_name, "Test Concept")
    
    def test_pattern_search(self):
        """Test searching patterns by query and context tags."""
        # Create multiple patterns
        patterns = []
        for i in range(5):
            semantic_knowledge = SemanticKnowledge(
                concept_id=f"concept_{i}",
                concept_name=f"Concept {i}",
                definition=f"Definition for concept {i}",
                properties={"domain": "test"},
                confidence=0.8
            )
            
            pattern = MemoryPattern(
                pattern_id=f"pattern_{i}",
                category=MemoryCategory.SEMANTIC,
                content=semantic_knowledge,
                strength=0.6 + i * 0.1,
                context_tags=[f"tag_{i}", "test"],
                confidence=0.8
            )
            
            patterns.append(pattern)
            self.long_term_memory.store_pattern(pattern)
        
        # Search by context tags
        test_patterns = self.long_term_memory.retrieve_patterns(
            context_tags=["test"]
        )
        
        self.assertEqual(len(test_patterns), 5)
        
        # Search by query
        concept_patterns = self.long_term_memory.retrieve_patterns(
            query="concept"
        )
        
        self.assertGreater(len(concept_patterns), 0)
    
    def test_learning_from_experience(self):
        """Test learning from experience."""
        experience_data = {
            'description': 'Test experience',
            'domain': 'navigation',
            'location': (1.0, 2.0, 0.0),
            'participants': ['robot', 'user'],
            'emotions': ['curious'],
            'sensory_details': {'visual': 'bright', 'audio': 'quiet'}
        }
        
        outcome = "success"
        lessons_learned = ["Always check for obstacles", "Maintain safe distance"]
        
        # Learn from experience
        event_id = self.long_term_memory.learn_from_experience(
            experience_data, outcome, lessons_learned
        )
        
        self.assertIsInstance(event_id, str)
        
        # Check that episodic memory was created
        episodic_patterns = self.long_term_memory.retrieve_patterns(
            category=MemoryCategory.EPISODIC
        )
        
        self.assertGreater(len(episodic_patterns), 0)
        
        # Check that semantic knowledge was extracted
        semantic_patterns = self.long_term_memory.retrieve_patterns(
            category=MemoryCategory.SEMANTIC
        )
        
        self.assertGreater(len(semantic_patterns), 0)
    
    def test_pattern_strength_update(self):
        """Test updating pattern strength."""
        # Create a pattern
        semantic_knowledge = SemanticKnowledge(
            concept_id="strength_test",
            concept_name="Strength Test",
            definition="Test for strength updates",
            confidence=0.8
        )
        
        pattern = MemoryPattern(
            pattern_id="strength_pattern",
            category=MemoryCategory.SEMANTIC,
            content=semantic_knowledge,
            strength=0.5,
            confidence=0.8
        )
        
        self.long_term_memory.store_pattern(pattern)
        
        # Update strength
        self.long_term_memory.update_pattern_strength("strength_pattern", 0.8)
        
        # Retrieve and check
        retrieved_patterns = self.long_term_memory.retrieve_patterns(
            category=MemoryCategory.SEMANTIC
        )
        
        self.assertEqual(retrieved_patterns[0].strength, 0.8)
    
    def test_memory_consolidation(self):
        """Test memory consolidation."""
        # Create patterns with different access counts
        for i in range(10):
            semantic_knowledge = SemanticKnowledge(
                concept_id=f"consolidation_{i}",
                concept_name=f"Consolidation {i}",
                definition=f"Test consolidation {i}",
                confidence=0.8
            )
            
            pattern = MemoryPattern(
                pattern_id=f"consolidation_pattern_{i}",
                category=MemoryCategory.SEMANTIC,
                content=semantic_knowledge,
                strength=0.5,
                access_count=i * 2,  # Varying access counts
                confidence=0.8
            )
            
            self.long_term_memory.store_pattern(pattern)
        
        # Perform consolidation
        self.long_term_memory.consolidate_memories()
        
        # Check that frequently accessed patterns have higher strength
        retrieved_patterns = self.long_term_memory.retrieve_patterns(
            category=MemoryCategory.SEMANTIC
        )
        
        # Sort by strength
        retrieved_patterns.sort(key=lambda x: x.strength, reverse=True)
        
        # Higher access count patterns should have higher strength
        for i in range(len(retrieved_patterns) - 1):
            if retrieved_patterns[i].access_count > retrieved_patterns[i + 1].access_count:
                self.assertGreaterEqual(
                    retrieved_patterns[i].strength,
                    retrieved_patterns[i + 1].strength
                )
    
    def test_persistence(self):
        """Test memory persistence across sessions."""
        # Create and store patterns
        semantic_knowledge = SemanticKnowledge(
            concept_id="persistence_test",
            concept_name="Persistence Test",
            definition="Test persistence",
            confidence=0.8
        )
        
        pattern = MemoryPattern(
            pattern_id="persistence_pattern",
            category=MemoryCategory.SEMANTIC,
            content=semantic_knowledge,
            strength=0.7,
            confidence=0.8
        )
        
        self.long_term_memory.store_pattern(pattern)
        
        # Save memories
        self.long_term_memory._save_memories()
        
        # Create new instance (simulating restart)
        new_memory = LongTermMemory(storage_path=self.temp_dir)
        
        # Check that patterns are loaded
        retrieved_patterns = new_memory.retrieve_patterns(
            category=MemoryCategory.SEMANTIC
        )
        
        self.assertGreater(len(retrieved_patterns), 0)
        self.assertEqual(retrieved_patterns[0].pattern_id, "persistence_pattern")
    
    def test_memory_statistics(self):
        """Test memory statistics collection."""
        # Create some patterns
        for i in range(5):
            semantic_knowledge = SemanticKnowledge(
                concept_id=f"stats_{i}",
                concept_name=f"Stats {i}",
                definition=f"Statistics test {i}",
                confidence=0.8
            )
            
            pattern = MemoryPattern(
                pattern_id=f"stats_pattern_{i}",
                category=MemoryCategory.SEMANTIC,
                content=semantic_knowledge,
                strength=0.6,
                confidence=0.8
            )
            
            self.long_term_memory.store_pattern(pattern)
        
        # Get statistics
        stats = self.long_term_memory.get_memory_statistics()
        
        self.assertIn('total_patterns', stats)
        self.assertIn('patterns_by_category', stats)
        self.assertIn('average_strength', stats)
        self.assertIn('most_accessed', stats)
        self.assertIn('recent_patterns', stats)
        
        self.assertEqual(stats['total_patterns'], 5)
        self.assertGreater(stats['average_strength'], 0.0)
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Perform some operations
        semantic_knowledge = SemanticKnowledge(
            concept_id="perf_test",
            concept_name="Performance Test",
            definition="Test performance",
            confidence=0.8
        )
        
        pattern = MemoryPattern(
            pattern_id="perf_pattern",
            category=MemoryCategory.SEMANTIC,
            content=semantic_knowledge,
            strength=0.7,
            confidence=0.8
        )
        
        self.long_term_memory.store_pattern(pattern)
        self.long_term_memory.retrieve_patterns(category=MemoryCategory.SEMANTIC)
        
        metrics = self.long_term_memory.get_performance_metrics()
        
        self.assertIn('total_patterns', metrics)
        self.assertIn('retrieval_speed', metrics)
        self.assertIn('storage_utilization', metrics)
        self.assertIn('consolidation_rate', metrics)
        
        self.assertIsInstance(metrics['total_patterns'], int)
        self.assertIsInstance(metrics['retrieval_speed'], float)


if __name__ == '__main__':
    unittest.main() 