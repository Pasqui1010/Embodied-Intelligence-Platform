#!/usr/bin/env python3

"""
Test suite for Executive Control

This module contains comprehensive tests for the executive control component
of the cognitive architecture.
"""

import unittest
import time
from unittest.mock import Mock, patch

# Import executive control components
from eip_cognitive_architecture.executive_control import (
    ExecutiveControl, DecisionType, TaskPriority, ExecutiveDecision,
    TaskPlan, ResourceAllocation, CognitiveState
)


class TestExecutiveControl(unittest.TestCase):
    """Test cases for the ExecutiveControl class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.executive_control = ExecutiveControl()
    
    def test_initialization(self):
        """Test executive control initialization."""
        self.assertIsNotNone(self.executive_control)
        self.assertIsInstance(self.executive_control.config, dict)
        self.assertIn('max_concurrent_tasks', self.executive_control.config)
    
    def test_make_decision_basic(self):
        """Test basic decision making functionality."""
        # Create mock working memory state
        working_memory_state = {
            'task_context': {
                'content': Mock(task_id='test_task', status='active'),
                'priority': 0.7,
                'timestamp': time.time()
            },
            'safety_state': {
                'content': Mock(safety_level='safe'),
                'priority': 0.8,
                'timestamp': time.time()
            },
            'social_context': {
                'content': Mock(nearby_humans=[]),
                'priority': 0.4,
                'timestamp': time.time()
            }
        }
        
        # Create mock relevant patterns
        relevant_patterns = [
            Mock(strength=0.7, confidence=0.8, category='semantic'),
            Mock(strength=0.6, confidence=0.7, category='episodic')
        ]
        
        # Make decision
        decision = self.executive_control.make_decision(
            working_memory_state, relevant_patterns
        )
        
        self.assertIsInstance(decision, ExecutiveDecision)
        self.assertIsInstance(decision.decision_type, DecisionType)
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
        self.assertIsInstance(decision.priority, TaskPriority)
        self.assertIsInstance(decision.actions, list)
    
    def test_safety_override_decision(self):
        """Test safety override decision making."""
        # Create safety-critical working memory state
        working_memory_state = {
            'safety_state': {
                'content': Mock(safety_level='critical'),
                'priority': 0.9,
                'timestamp': time.time()
            },
            'task_context': {
                'content': Mock(task_id='test_task', status='active'),
                'priority': 0.5,
                'timestamp': time.time()
            }
        }
        
        relevant_patterns = []
        
        # Make decision
        decision = self.executive_control.make_decision(
            working_memory_state, relevant_patterns
        )
        
        # Should be safety override decision
        self.assertEqual(decision.decision_type, DecisionType.SAFETY_OVERRIDE)
        self.assertEqual(decision.priority, TaskPriority.CRITICAL)
        self.assertGreater(decision.confidence, 0.8)
        
        # Should have safety actions
        safety_actions = [a for a in decision.actions if 'safety' in a.get('action_type', '')]
        self.assertGreater(len(safety_actions), 0)
    
    def test_social_adjustment_decision(self):
        """Test social adjustment decision making."""
        # Create social interaction state
        working_memory_state = {
            'social_context': {
                'content': Mock(nearby_humans=[{'id': 'person1'}]),
                'priority': 0.8,
                'timestamp': time.time()
            },
            'safety_state': {
                'content': Mock(safety_level='safe'),
                'priority': 0.6,
                'timestamp': time.time()
            }
        }
        
        relevant_patterns = []
        
        # Make decision
        decision = self.executive_control.make_decision(
            working_memory_state, relevant_patterns
        )
        
        # Should be social adjustment decision
        self.assertEqual(decision.decision_type, DecisionType.SOCIAL_ADJUSTMENT)
        self.assertEqual(decision.priority, TaskPriority.HIGH)
        
        # Should have social actions
        social_actions = [a for a in decision.actions if 'social' in a.get('action_type', '')]
        self.assertGreater(len(social_actions), 0)
    
    def test_resource_allocation_decision(self):
        """Test resource allocation decision making."""
        # Create resource-constrained state
        working_memory_state = {
            'attention_focus': {
                'content': Mock(foci=[Mock() for _ in range(10)]),  # Many attention foci
                'priority': 0.9,
                'timestamp': time.time()
            },
            'safety_state': {
                'content': Mock(safety_level='safe'),
                'priority': 0.6,
                'timestamp': time.time()
            }
        }
        
        relevant_patterns = []
        
        # Make decision
        decision = self.executive_control.make_decision(
            working_memory_state, relevant_patterns
        )
        
        # Should be resource allocation decision
        self.assertEqual(decision.decision_type, DecisionType.RESOURCE_ALLOCATION)
        
        # Should have resource allocation actions
        resource_actions = [a for a in decision.actions if 'resource' in a.get('action_type', '')]
        self.assertGreater(len(resource_actions), 0)
    
    def test_task_selection_decision(self):
        """Test task selection decision making."""
        # Create normal state with no special conditions
        working_memory_state = {
            'task_context': {
                'content': Mock(task_id='idle', status='active'),
                'priority': 0.3,
                'timestamp': time.time()
            },
            'safety_state': {
                'content': Mock(safety_level='safe'),
                'priority': 0.5,
                'timestamp': time.time()
            },
            'social_context': {
                'content': Mock(nearby_humans=[]),
                'priority': 0.2,
                'timestamp': time.time()
            }
        }
        
        relevant_patterns = []
        
        # Make decision
        decision = self.executive_control.make_decision(
            working_memory_state, relevant_patterns
        )
        
        # Should be task selection decision
        self.assertEqual(decision.decision_type, DecisionType.TASK_SELECTION)
        self.assertEqual(decision.priority, TaskPriority.MEDIUM)
    
    def test_add_task(self):
        """Test adding tasks to the execution queue."""
        # Create task plan
        task_plan = TaskPlan(
            task_id="test_task",
            task_type="navigation",
            description="Test navigation task",
            priority=TaskPriority.HIGH,
            steps=[
                {'step_id': 1, 'description': 'Move forward', 'duration': 5.0},
                {'step_id': 2, 'description': 'Turn right', 'duration': 3.0}
            ],
            estimated_duration=8.0,
            dependencies=[],
            success_criteria=[{'criterion': 'reached_goal', 'threshold': 0.5}],
            safety_constraints=[{'constraint': 'max_velocity', 'value': 1.0}],
            resource_requirements={'processing': 0.5, 'memory': 0.3}
        )
        
        # Add task
        task_id = self.executive_control.add_task(task_plan)
        
        self.assertEqual(task_id, "test_task")
        
        # Check that task is in queue
        current_tasks = self.executive_control.get_current_tasks()
        self.assertGreater(len(current_tasks), 0)
    
    def test_cognitive_state_tracking(self):
        """Test cognitive state tracking."""
        # Create working memory state that would affect cognitive state
        working_memory_state = {
            'attention_focus': {
                'content': Mock(foci=[Mock() for _ in range(5)]),
                'priority': 0.8,
                'timestamp': time.time()
            },
            'safety_state': {
                'content': Mock(safety_level='warning'),
                'priority': 0.7,
                'timestamp': time.time()
            },
            'social_context': {
                'content': Mock(nearby_humans=[{'id': 'person1'}]),
                'priority': 0.6,
                'timestamp': time.time()
            }
        }
        
        relevant_patterns = []
        
        # Make decision to update cognitive state
        self.executive_control.make_decision(working_memory_state, relevant_patterns)
        
        # Get cognitive state
        cognitive_state = self.executive_control.get_cognitive_state()
        
        self.assertIsInstance(cognitive_state, CognitiveState)
        self.assertGreater(cognitive_state.attention_load, 0.0)
        self.assertEqual(cognitive_state.safety_status, 'warning')
        self.assertEqual(cognitive_state.social_context, 'interacting')
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Make some decisions
        working_memory_state = {
            'task_context': {
                'content': Mock(task_id='test', status='active'),
                'priority': 0.5,
                'timestamp': time.time()
            },
            'safety_state': {
                'content': Mock(safety_level='safe'),
                'priority': 0.6,
                'timestamp': time.time()
            }
        }
        
        relevant_patterns = []
        
        for _ in range(3):
            self.executive_control.make_decision(working_memory_state, relevant_patterns)
        
        # Get performance metrics
        metrics = self.executive_control.get_performance_metrics()
        
        self.assertIn('decision_speed', metrics)
        self.assertIn('task_completion_rate', metrics)
        self.assertIn('resource_efficiency', metrics)
        self.assertIn('safety_violations', metrics)
        
        self.assertIsInstance(metrics['decision_speed'], float)
        self.assertIsInstance(metrics['safety_violations'], int)
    
    def test_decision_summary(self):
        """Test decision summary generation."""
        # Make some decisions
        working_memory_state = {
            'task_context': {
                'content': Mock(task_id='test', status='active'),
                'priority': 0.5,
                'timestamp': time.time()
            },
            'safety_state': {
                'content': Mock(safety_level='safe'),
                'priority': 0.6,
                'timestamp': time.time()
            }
        }
        
        relevant_patterns = []
        
        for _ in range(5):
            self.executive_control.make_decision(working_memory_state, relevant_patterns)
        
        # Get decision summary
        summary = self.executive_control.get_decision_summary()
        
        self.assertIn('total_decisions', summary)
        self.assertIn('recent_decisions', summary)
        self.assertIn('decision_types', summary)
        self.assertIn('average_confidence', summary)
        
        self.assertEqual(summary['total_decisions'], 5)
        self.assertIsInstance(summary['average_confidence'], float)
    
    def test_configuration_override(self):
        """Test custom configuration override."""
        custom_config = {
            'max_concurrent_tasks': 5,
            'decision_timeout': 2.0,
            'safety_priority_boost': 3.0,
            'social_priority_boost': 2.0
        }
        
        custom_executive = ExecutiveControl(config=custom_config)
        
        self.assertEqual(custom_executive.config['max_concurrent_tasks'], 5)
        self.assertEqual(custom_executive.config['decision_timeout'], 2.0)
        self.assertEqual(custom_executive.config['safety_priority_boost'], 3.0)
        self.assertEqual(custom_executive.config['social_priority_boost'], 2.0)
    
    def test_thread_safety(self):
        """Test thread safety of executive control."""
        import threading
        
        def decision_worker():
            working_memory_state = {
                'task_context': {
                    'content': Mock(task_id='test', status='active'),
                    'priority': 0.5,
                    'timestamp': time.time()
                },
                'safety_state': {
                    'content': Mock(safety_level='safe'),
                    'priority': 0.6,
                    'timestamp': time.time()
                }
            }
            
            relevant_patterns = []
            
            for _ in range(10):
                self.executive_control.make_decision(working_memory_state, relevant_patterns)
                time.sleep(0.01)
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=decision_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should not raise any exceptions
        self.assertTrue(True)
    
    def test_decision_history(self):
        """Test decision history tracking."""
        # Make decisions
        working_memory_state = {
            'task_context': {
                'content': Mock(task_id='test', status='active'),
                'priority': 0.5,
                'timestamp': time.time()
            },
            'safety_state': {
                'content': Mock(safety_level='safe'),
                'priority': 0.6,
                'timestamp': time.time()
            }
        }
        
        relevant_patterns = []
        
        for i in range(10):
            decision = self.executive_control.make_decision(working_memory_state, relevant_patterns)
            self.assertIsInstance(decision, ExecutiveDecision)
        
        # Check decision history
        summary = self.executive_control.get_decision_summary()
        self.assertEqual(summary['total_decisions'], 10)
    
    def test_resource_allocation_defaults(self):
        """Test default resource allocation initialization."""
        # Check that default allocations are set
        self.assertIn('attention', self.executive_control.resource_allocations)
        self.assertIn('memory', self.executive_control.resource_allocations)
        self.assertIn('processing', self.executive_control.resource_allocations)
        self.assertIn('safety', self.executive_control.resource_allocations)
        
        # Check allocation values
        attention_allocation = self.executive_control.resource_allocations['attention']
        self.assertIsInstance(attention_allocation, ResourceAllocation)
        self.assertEqual(attention_allocation.resource_type, 'attention')
        self.assertGreater(attention_allocation.allocation_percentage, 0.0)
        self.assertLessEqual(attention_allocation.allocation_percentage, 1.0)


if __name__ == '__main__':
    unittest.main() 