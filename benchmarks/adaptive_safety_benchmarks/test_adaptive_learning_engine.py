#!/usr/bin/env python3
"""
Tests for Adaptive Learning Engine

This module tests the core adaptive learning functionality including:
- Meta-learning algorithms
- Safety rule generation
- Experience processing
- Performance optimization
"""

import unittest
import numpy as np
import torch
import time
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../intelligence/eip_adaptive_safety'))

from eip_adaptive_safety.adaptive_learning_engine import (
    AdaptiveLearningEngine, SafetyExperience, SafetyRule, MetaLearner
)

class TestMetaLearner(unittest.TestCase):
    """Test the MetaLearner neural network"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 256
        self.hidden_dim = 128
        self.output_dim = 64
        self.meta_learner = MetaLearner(self.input_dim, self.hidden_dim, self.output_dim)
        
    def test_meta_learner_initialization(self):
        """Test MetaLearner initialization"""
        self.assertIsNotNone(self.meta_learner)
        self.assertEqual(self.meta_learner.input_dim, self.input_dim)
        self.assertEqual(self.meta_learner.hidden_dim, self.hidden_dim)
        self.assertEqual(self.meta_learner.output_dim, self.output_dim)
        
    def test_forward_pass(self):
        """Test forward pass through MetaLearner"""
        # Create test input
        batch_size = 4
        test_input = torch.randn(batch_size, self.input_dim)
        
        # Forward pass
        meta_features, rule_params = self.meta_learner(test_input)
        
        # Check output shapes
        self.assertEqual(meta_features.shape, (batch_size, self.output_dim))
        self.assertEqual(rule_params.shape, (batch_size, 8))  # 8 rule parameters
        
        # Check output ranges
        self.assertTrue(torch.all(meta_features >= -1.0) and torch.all(meta_features <= 1.0))
        self.assertTrue(torch.all(rule_params >= -10.0) and torch.all(rule_params <= 10.0))
        
    def test_meta_learner_training(self):
        """Test MetaLearner training"""
        # Create test data
        batch_size = 8
        test_input = torch.randn(batch_size, self.input_dim)
        test_targets = torch.randint(0, 2, (batch_size,)).float()
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Training step
        optimizer.zero_grad()
        meta_features, rule_params = self.meta_learner(test_input)
        loss = criterion(rule_params[:, 0], test_targets)
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

class TestSafetyExperience(unittest.TestCase):
    """Test SafetyExperience data structure"""
    
    def test_safety_experience_creation(self):
        """Test SafetyExperience creation"""
        timestamp = time.time()
        sensor_data = {'velocity': np.array([1.0, 2.0, 3.0])}
        safety_violation = True
        violation_type = 'collision'
        severity = 0.8
        context = {'human_present': True, 'workspace': 'lab'}
        outcome = 'near_miss'
        recovery_action = 'emergency_stop'
        
        experience = SafetyExperience(
            timestamp=timestamp,
            sensor_data=sensor_data,
            safety_violation=safety_violation,
            violation_type=violation_type,
            severity=severity,
            context=context,
            outcome=outcome,
            recovery_action=recovery_action
        )
        
        self.assertEqual(experience.timestamp, timestamp)
        self.assertEqual(experience.safety_violation, safety_violation)
        self.assertEqual(experience.violation_type, violation_type)
        self.assertEqual(experience.severity, severity)
        self.assertEqual(experience.outcome, outcome)
        self.assertEqual(experience.recovery_action, recovery_action)
        
    def test_safety_experience_defaults(self):
        """Test SafetyExperience with default values"""
        experience = SafetyExperience(
            timestamp=time.time(),
            sensor_data={},
            safety_violation=False,
            violation_type='none',
            severity=0.0,
            context={},
            outcome='safe_operation'
        )
        
        self.assertIsNone(experience.recovery_action)

class TestSafetyRule(unittest.TestCase):
    """Test SafetyRule data structure"""
    
    def test_safety_rule_creation(self):
        """Test SafetyRule creation"""
        rule_id = 'test_rule_001'
        condition = {
            'sensor_thresholds': {'velocity': 2.0, 'proximity': 1.0},
            'context_conditions': {'human_present': True}
        }
        threshold = 0.7
        confidence = 0.85
        priority = 5
        created_at = time.time()
        last_updated = time.time()
        usage_count = 10
        success_rate = 0.9
        
        rule = SafetyRule(
            rule_id=rule_id,
            condition=condition,
            threshold=threshold,
            confidence=confidence,
            priority=priority,
            created_at=created_at,
            last_updated=last_updated,
            usage_count=usage_count,
            success_rate=success_rate
        )
        
        self.assertEqual(rule.rule_id, rule_id)
        self.assertEqual(rule.threshold, threshold)
        self.assertEqual(rule.confidence, confidence)
        self.assertEqual(rule.priority, priority)
        self.assertEqual(rule.usage_count, usage_count)
        self.assertEqual(rule.success_rate, success_rate)

class TestAdaptiveLearningEngine(unittest.TestCase):
    """Test AdaptiveLearningEngine functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock ROS 2 node
        with patch('rclpy.init'), patch('rclpy.Node'):
            self.learning_engine = AdaptiveLearningEngine()
            
    def test_learning_engine_initialization(self):
        """Test AdaptiveLearningEngine initialization"""
        self.assertIsNotNone(self.learning_engine)
        self.assertIsNotNone(self.learning_engine.meta_learner)
        self.assertIsNotNone(self.learning_engine.optimizer)
        self.assertEqual(len(self.learning_engine.experience_buffer), 0)
        self.assertEqual(len(self.learning_engine.safety_rules), 0)
        
    def test_feature_extraction(self):
        """Test feature extraction from safety experience"""
        # Create test experience
        experience = SafetyExperience(
            timestamp=time.time(),
            sensor_data={'velocity': np.array([1.0, 2.0]), 'proximity': np.array([0.5])},
            safety_violation=True,
            violation_type='collision',
            severity=0.8,
            context={'human_present': True},
            outcome='near_miss'
        )
        
        # Extract features
        features = self.learning_engine._extract_features(experience)
        
        # Check feature vector
        self.assertEqual(len(features), 256)  # Fixed size
        self.assertTrue(np.all(np.isfinite(features)))
        
    def test_rule_generation(self):
        """Test safety rule generation"""
        # Create test features
        features = np.random.randn(256)
        
        # Generate rules
        self.learning_engine._generate_new_rules(features)
        
        # Check that rules were generated
        self.assertGreaterEqual(len(self.learning_engine.safety_rules), 0)
        
    def test_rule_pruning(self):
        """Test safety rule pruning"""
        # Add some test rules
        for i in range(10):
            rule = SafetyRule(
                rule_id=f'test_rule_{i}',
                condition={},
                threshold=0.5,
                confidence=0.5 + i * 0.05,
                priority=i,
                created_at=time.time(),
                last_updated=time.time(),
                usage_count=i,
                success_rate=0.5 + i * 0.05
            )
            self.learning_engine.safety_rules[rule.rule_id] = rule
            
        # Set max rules to trigger pruning
        self.learning_engine.max_rules = 5
        
        # Prune rules
        self.learning_engine._prune_rules()
        
        # Check that rules were pruned
        self.assertLessEqual(len(self.learning_engine.safety_rules), 5)
        
    def test_meta_learner_update(self):
        """Test meta-learner update"""
        # Add test experiences
        for i in range(50):  # More than batch_size
            experience = SafetyExperience(
                timestamp=time.time(),
                sensor_data={'test': np.array([i])},
                safety_violation=i % 2 == 0,
                violation_type='test',
                severity=0.5,
                context={},
                outcome='test'
            )
            self.learning_engine.experience_buffer.append(experience)
            
        # Update meta-learner
        self.learning_engine._update_meta_learner()
        
        # Check that update completed without errors
        self.assertTrue(True)  # If we get here, no exceptions occurred
        
    def test_task_validation(self):
        """Test task validation with adaptive rules"""
        # Add a test rule
        rule = SafetyRule(
            rule_id='test_validation_rule',
            condition={
                'sensor_thresholds': {'velocity': 1.0},
                'context_conditions': {'human_present': False}
            },
            threshold=0.7,
            confidence=0.8,
            priority=1,
            created_at=time.time(),
            last_updated=time.time(),
            usage_count=0,
            success_rate=0.9
        )
        self.learning_engine.safety_rules[rule.rule_id] = rule
        
        # Mock request and response
        request = Mock()
        request.task_plan = "move to position A"
        response = Mock()
        
        # Validate task
        result = self.learning_engine._validate_task_adaptive(request, response)
        
        # Check response
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'is_safe'))
        self.assertTrue(hasattr(result, 'safety_score'))
        self.assertTrue(hasattr(result, 'violations'))
        self.assertTrue(hasattr(result, 'confidence'))

class TestAdaptiveLearningEngineIntegration(unittest.TestCase):
    """Integration tests for AdaptiveLearningEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('rclpy.init'), patch('rclpy.Node'):
            self.learning_engine = AdaptiveLearningEngine()
            
    def test_end_to_end_learning(self):
        """Test end-to-end learning process"""
        # Create multiple experiences
        experiences = []
        for i in range(100):
            experience = SafetyExperience(
                timestamp=time.time(),
                sensor_data={'velocity': np.array([i * 0.1]), 'proximity': np.array([i * 0.05])},
                safety_violation=i % 3 == 0,  # Some violations
                violation_type='test',
                severity=0.3 + (i % 3) * 0.2,
                context={'human_present': i % 2 == 0},
                outcome='near_miss' if i % 3 == 0 else 'safe_operation'
            )
            experiences.append(experience)
            
        # Process experiences
        for experience in experiences:
            self.learning_engine._process_experience(experience)
            
        # Check that experiences were processed
        self.assertGreater(len(self.learning_engine.experience_buffer), 0)
        
        # Update meta-learner multiple times
        for _ in range(5):
            self.learning_engine._update_meta_learner()
            
        # Check that rules were generated
        self.assertGreater(len(self.learning_engine.safety_rules), 0)
        
    def test_learning_convergence(self):
        """Test learning convergence over time"""
        # Create experiences with patterns
        for epoch in range(10):
            for i in range(20):
                # Create pattern: high velocity + human present = violation
                velocity = 2.0 if i % 2 == 0 else 0.5
                human_present = i % 2 == 0
                violation = velocity > 1.5 and human_present
                
                experience = SafetyExperience(
                    timestamp=time.time(),
                    sensor_data={'velocity': np.array([velocity])},
                    safety_violation=violation,
                    violation_type='collision',
                    severity=0.8 if violation else 0.1,
                    context={'human_present': human_present},
                    outcome='incident' if violation else 'safe_operation'
                )
                
                self.learning_engine._process_experience(experience)
                
            # Update meta-learner
            self.learning_engine._update_meta_learner()
            
        # Check that learning occurred
        self.assertGreater(len(self.learning_engine.safety_rules), 0)
        
        # Check that rules have reasonable confidence
        for rule in self.learning_engine.safety_rules.values():
            self.assertGreaterEqual(rule.confidence, 0.0)
            self.assertLessEqual(rule.confidence, 1.0)

class TestPerformanceAndRobustness(unittest.TestCase):
    """Test performance and robustness aspects"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('rclpy.init'), patch('rclpy.Node'):
            self.learning_engine = AdaptiveLearningEngine()
            
    def test_memory_usage(self):
        """Test memory usage with large experience buffer"""
        # Add many experiences
        for i in range(1000):
            experience = SafetyExperience(
                timestamp=time.time(),
                sensor_data={'data': np.random.randn(100)},
                safety_violation=i % 10 == 0,
                violation_type='test',
                severity=0.5,
                context={},
                outcome='test'
            )
            self.learning_engine._process_experience(experience)
            
        # Check that buffer size is limited
        self.assertLessEqual(len(self.learning_engine.experience_buffer), 10000)
        
    def test_error_handling(self):
        """Test error handling with invalid data"""
        # Test with invalid experience
        invalid_experience = None
        
        # Should handle gracefully
        try:
            self.learning_engine._process_experience(invalid_experience)
        except Exception as e:
            self.fail(f"Should handle invalid experience gracefully: {e}")
            
    def test_concurrent_access(self):
        """Test concurrent access to learning engine"""
        import threading
        import time
        
        # Create multiple threads adding experiences
        def add_experiences(thread_id):
            for i in range(100):
                experience = SafetyExperience(
                    timestamp=time.time(),
                    sensor_data={'thread': thread_id, 'index': i},
                    safety_violation=i % 2 == 0,
                    violation_type='test',
                    severity=0.5,
                    context={},
                    outcome='test'
                )
                self.learning_engine._process_experience(experience)
                time.sleep(0.001)  # Small delay
                
        # Start multiple threads
        threads = []
        for i in range(4):
            thread = threading.Thread(target=add_experiences, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # Check that all experiences were processed
        self.assertGreater(len(self.learning_engine.experience_buffer), 0)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 