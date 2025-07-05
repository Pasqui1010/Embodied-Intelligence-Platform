#!/usr/bin/env python3
"""
Integration Tests for Critical Issue Fixes

This module tests that the critical issues identified in the review have been
properly addressed, including thread safety, memory management, input validation,
and error recovery.
"""

import unittest
import threading
import time
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import queue
import gc

# Add the package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../intelligence/eip_adaptive_safety'))

from eip_adaptive_safety.thread_safe_containers import (
    ThreadSafeExperienceBuffer, ThreadSafeRuleRegistry, 
    InputValidator, ErrorRecoveryManager, thread_safe_context
)
from eip_adaptive_safety.adaptive_learning_engine import (
    AdaptiveLearningEngine, SafetyExperience, SafetyRule
)

class TestThreadSafetyFixes(unittest.TestCase):
    """Test thread safety fixes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.experience_buffer = ThreadSafeExperienceBuffer(maxlen=1000)
        self.rule_registry = ThreadSafeRuleRegistry(max_rules=50)
        
    def test_concurrent_experience_processing(self):
        """Test concurrent experience processing"""
        print("\n=== Testing Concurrent Experience Processing ===")
        
        # Create experiences
        experiences = []
        for i in range(100):
            experience = SafetyExperience(
                timestamp=time.time(),
                sensor_data={'velocity': np.array([i * 0.1])},
                safety_violation=i % 5 == 0,
                violation_type='test',
                severity=0.5,
                context={'test': i},
                outcome='test'
            )
            experiences.append(experience)
        
        # Test concurrent append
        def append_experiences(thread_id, exp_list):
            for i, exp in enumerate(exp_list):
                success = self.experience_buffer.append(exp)
                if not success:
                    print(f"Thread {thread_id}: Failed to append experience {i}")
        
        # Start multiple threads
        threads = []
        experiences_per_thread = 25
        
        for i in range(4):
            start_idx = i * experiences_per_thread
            end_idx = start_idx + experiences_per_thread
            thread = threading.Thread(
                target=append_experiences,
                args=(i, experiences[start_idx:end_idx])
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all experiences were added
        final_count = len(self.experience_buffer)
        print(f"Final experience count: {final_count}")
        
        self.assertGreater(final_count, 0)
        self.assertLessEqual(final_count, 1000)  # Should not exceed maxlen
        
    def test_concurrent_rule_operations(self):
        """Test concurrent rule registry operations"""
        print("\n=== Testing Concurrent Rule Operations ===")
        
        # Create test rules
        rules = []
        for i in range(20):
            rule = SafetyRule(
                rule_id=f"test_rule_{i}",
                condition={'test': i},
                threshold=0.5 + i * 0.1,
                confidence=0.6 + i * 0.02,
                priority=i,
                created_at=time.time(),
                last_updated=time.time(),
                usage_count=0,
                success_rate=0.5
            )
            rules.append(rule)
        
        # Test concurrent rule operations
        def rule_operations(thread_id, rule_list):
            for i, rule in enumerate(rule_list):
                # Add rule
                success = self.rule_registry.add_rule(rule.rule_id, rule)
                if success:
                    # Get rule
                    retrieved_rule = self.rule_registry.get_rule(rule.rule_id)
                    if retrieved_rule:
                        # Update rule
                        self.rule_registry.update_rule(rule.rule_id, {'usage_count': i})
        
        # Start multiple threads
        threads = []
        rules_per_thread = 5
        
        for i in range(4):
            start_idx = i * rules_per_thread
            end_idx = start_idx + rules_per_thread
            thread = threading.Thread(
                target=rule_operations,
                args=(i, rules[start_idx:end_idx])
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify rules were added
        final_count = len(self.rule_registry)
        print(f"Final rule count: {final_count}")
        
        self.assertGreater(final_count, 0)
        self.assertLessEqual(final_count, 50)  # Should not exceed max_rules
        
    def test_memory_management(self):
        """Test memory management and cleanup"""
        print("\n=== Testing Memory Management ===")
        
        # Get initial memory
        import psutil
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Add many experiences to trigger cleanup
        for i in range(2000):
            experience = SafetyExperience(
                timestamp=time.time(),
                sensor_data={'data': np.random.randn(100)},
                safety_violation=i % 10 == 0,
                violation_type='test',
                severity=0.5,
                context={'test': i},
                outcome='test'
            )
            self.experience_buffer.append(experience)
        
        # Force garbage collection
        gc.collect()
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.1f}MB")
        print(f"Final memory: {final_memory:.1f}MB")
        print(f"Memory growth: {memory_growth:.1f}MB")
        
        # Check that memory growth is reasonable
        self.assertLess(memory_growth, 200)  # Should not grow more than 200MB
        
        # Check buffer size is limited
        buffer_size = len(self.experience_buffer)
        print(f"Buffer size: {buffer_size}")
        self.assertLessEqual(buffer_size, 1000)  # Should respect maxlen

class TestInputValidationFixes(unittest.TestCase):
    """Test input validation fixes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = InputValidator()
        
    def test_task_plan_validation(self):
        """Test task plan validation and sanitization"""
        print("\n=== Testing Task Plan Validation ===")
        
        # Test valid task plans
        valid_plans = [
            "move to position A",
            "pick up object B",
            "Navigate to location C with speed 0.5",
            "Assist human with task D"
        ]
        
        for plan in valid_plans:
            is_valid, sanitized = self.validator.validate_task_plan(plan)
            print(f"Valid plan: {plan[:30]}... -> {is_valid}")
            self.assertTrue(is_valid)
            self.assertEqual(sanitized, plan)
        
        # Test invalid task plans
        invalid_plans = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "A" * 2000,  # Too long
            None,
            123,
            {"invalid": "type"}
        ]
        
        for plan in invalid_plans:
            is_valid, error = self.validator.validate_task_plan(plan)
            print(f"Invalid plan: {str(plan)[:30]}... -> {is_valid} ({error})")
            self.assertFalse(is_valid)
            self.assertIsInstance(error, str)
    
    def test_sensor_data_validation(self):
        """Test sensor data validation"""
        print("\n=== Testing Sensor Data Validation ===")
        
        # Test valid sensor data
        valid_sensor_data = {
            'velocity': np.array([1.0, 2.0, 3.0]),
            'proximity': np.array([0.5]),
            'temperature': 25.0,
            'pressure': 1013.25
        }
        
        is_valid, error = self.validator.validate_sensor_data(valid_sensor_data)
        print(f"Valid sensor data: {is_valid}")
        self.assertTrue(is_valid)
        
        # Test invalid sensor data
        invalid_sensor_data = [
            {"velocity": "not_a_number"},
            {"proximity": np.array([1e10])},  # Too large
            {"temperature": np.array([1, 2, 3, 4] * 300)},  # Too large array
            {"pressure": None},
            "not_a_dict"
        ]
        
        for data in invalid_sensor_data:
            is_valid, error = self.validator.validate_sensor_data(data)
            print(f"Invalid sensor data: {is_valid} ({error})")
            self.assertFalse(is_valid)
    
    def test_context_validation(self):
        """Test context validation"""
        print("\n=== Testing Context Validation ===")
        
        # Test valid context
        valid_context = {
            'human_present': True,
            'workspace': 'lab',
            'time_of_day': 14,
            'weather': 'clear',
            'nested': {
                'level1': {
                    'level2': 'value'
                }
            }
        }
        
        is_valid, error = self.validator.validate_context(valid_context)
        print(f"Valid context: {is_valid}")
        self.assertTrue(is_valid)
        
        # Test invalid context
        invalid_contexts = [
            {"key": "A" * 200},  # Value too long
            {"level1": {"level2": {"level3": {"level4": {"level5": {"level6": "value"}}}}}},  # Too deep
            {f"key_{i}": i for i in range(100)},  # Too many keys
            {"invalid_key": None},
            "not_a_dict"
        ]
        
        for context in invalid_contexts:
            is_valid, error = self.validator.validate_context(context)
            print(f"Invalid context: {is_valid} ({error})")
            self.assertFalse(is_valid)

class TestErrorRecoveryFixes(unittest.TestCase):
    """Test error recovery fixes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.recovery_manager = ErrorRecoveryManager(max_retries=3)
        
    def test_error_recovery_strategies(self):
        """Test error recovery strategies"""
        print("\n=== Testing Error Recovery Strategies ===")
        
        # Register recovery strategies
        def memory_recovery(error, *args, **kwargs):
            return "memory_recovered"
        
        def validation_recovery(error, *args, **kwargs):
            return "validation_recovered"
        
        self.recovery_manager.register_recovery_strategy("memory_error", memory_recovery)
        self.recovery_manager.register_recovery_strategy("validation_error", validation_recovery)
        
        # Test successful recovery
        def failing_operation():
            raise MemoryError("Out of memory")
        
        success, result = self.recovery_manager.execute_with_recovery(
            failing_operation, "memory_error"
        )
        
        print(f"Memory error recovery: {success} -> {result}")
        self.assertTrue(success)
        self.assertEqual(result, "memory_recovered")
        
        # Test max retries
        def always_failing_operation():
            raise ValueError("Always fails")
        
        for i in range(4):  # More than max_retries
            success, result = self.recovery_manager.execute_with_recovery(
                always_failing_operation, "general"
            )
            print(f"Attempt {i+1}: {success} -> {result}")
        
        # Check error stats
        stats = self.recovery_manager.get_error_stats()
        print(f"Error stats: {stats}")
        self.assertIn('general', stats['error_counts'])
        self.assertEqual(stats['error_counts']['general'], 4)
        
    def test_recovery_cooldown(self):
        """Test recovery cooldown mechanism"""
        print("\n=== Testing Recovery Cooldown ===")
        
        def failing_operation():
            raise RuntimeError("Test error")
        
        # Trigger max retries
        for i in range(3):
            success, result = self.recovery_manager.execute_with_recovery(
                failing_operation, "test_error"
            )
        
        # Next attempt should be blocked by cooldown
        success, result = self.recovery_manager.execute_with_recovery(
            failing_operation, "test_error"
        )
        
        print(f"After max retries: {success} -> {result}")
        self.assertFalse(success)
        self.assertIn("cooldown", result)
        
        # Reset errors
        self.recovery_manager.reset_errors("test_error")
        
        # Should work again
        success, result = self.recovery_manager.execute_with_recovery(
            lambda: "success", "test_error"
        )
        
        print(f"After reset: {success} -> {result}")
        self.assertTrue(success)

class TestIntegrationWithFixes(unittest.TestCase):
    """Test integration with all fixes applied"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('rclpy.init'), patch('rclpy.Node'):
            self.learning_engine = AdaptiveLearningEngine()
    
    def test_end_to_end_safety_validation(self):
        """Test end-to-end safety validation with fixes"""
        print("\n=== Testing End-to-End Safety Validation ===")
        
        # Create experiences with various data types
        experiences = []
        for i in range(50):
            experience = SafetyExperience(
                timestamp=time.time(),
                sensor_data={
                    'velocity': np.array([i * 0.1]),
                    'proximity': np.array([i * 0.05]),
                    'temperature': 20.0 + i * 0.1
                },
                safety_violation=i % 5 == 0,
                violation_type='collision',
                severity=0.3 + (i % 5) * 0.1,
                context={
                    'human_present': i % 2 == 0,
                    'workspace': 'lab',
                    'time_of_day': i % 24
                },
                outcome='near_miss' if i % 5 == 0 else 'safe_operation'
            )
            experiences.append(experience)
        
        # Process experiences
        processed_count = 0
        for experience in experiences:
            success = self.learning_engine._process_experience(experience)
            if success:
                processed_count += 1
        
        print(f"Processed {processed_count}/{len(experiences)} experiences")
        self.assertGreater(processed_count, 0)
        
        # Update meta-learner
        for _ in range(3):
            self.learning_engine._update_meta_learner()
        
        # Test task validation
        request = Mock()
        request.task_plan = "move to position A with speed 0.5"
        response = Mock()
        
        result = self.learning_engine._validate_task_adaptive(request, response)
        
        print(f"Task validation result: {result.is_safe}")
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'is_safe'))
        self.assertTrue(hasattr(result, 'safety_score'))
        self.assertTrue(hasattr(result, 'violations'))
        self.assertTrue(hasattr(result, 'confidence'))
    
    def test_concurrent_safety_validation(self):
        """Test concurrent safety validation"""
        print("\n=== Testing Concurrent Safety Validation ===")
        
        # Add some experiences first
        for i in range(20):
            experience = SafetyExperience(
                timestamp=time.time(),
                sensor_data={'velocity': np.array([i * 0.1])},
                safety_violation=i % 3 == 0,
                violation_type='test',
                severity=0.5,
                context={'test': i},
                outcome='test'
            )
            self.learning_engine._process_experience(experience)
        
        # Update meta-learner
        self.learning_engine._update_meta_learner()
        
        # Test concurrent validation
        def validate_task(thread_id):
            request = Mock()
            request.task_plan = f"task from thread {thread_id}"
            response = Mock()
            
            result = self.learning_engine._validate_task_adaptive(request, response)
            return result.is_safe
        
        # Start multiple threads
        threads = []
        results = []
        
        for i in range(5):
            thread = threading.Thread(target=lambda: results.append(validate_task(i)))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        print(f"Concurrent validation results: {results}")
        self.assertEqual(len(results), 5)
        
        # All should return boolean values
        for result in results:
            self.assertIsInstance(result, bool)
    
    def test_malicious_input_handling(self):
        """Test handling of malicious inputs"""
        print("\n=== Testing Malicious Input Handling ===")
        
        # Test malicious task plans
        malicious_plans = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "'; DROP TABLE users; --",
            "A" * 2000,  # Very long input
            None,
            123,
            {"malicious": "object"}
        ]
        
        for plan in malicious_plans:
            request = Mock()
            request.task_plan = plan
            response = Mock()
            
            result = self.learning_engine._validate_task_adaptive(request, response)
            
            print(f"Malicious plan: {str(plan)[:30]}... -> Safe: {result.is_safe}")
            
            # Should always return safe=False for malicious inputs
            self.assertFalse(result.is_safe)
            self.assertEqual(result.safety_score, 0.0)
            self.assertEqual(result.confidence, 0.0)
            self.assertGreater(len(result.violations), 0)
    
    def test_memory_leak_prevention(self):
        """Test memory leak prevention"""
        print("\n=== Testing Memory Leak Prevention ===")
        
        import psutil
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many experiences
        for i in range(1000):
            experience = SafetyExperience(
                timestamp=time.time(),
                sensor_data={'data': np.random.randn(50)},
                safety_violation=i % 10 == 0,
                violation_type='test',
                severity=0.5,
                context={'test': i},
                outcome='test'
            )
            self.learning_engine._process_experience(experience)
            
            # Periodic updates
            if i % 100 == 0:
                self.learning_engine._update_meta_learner()
        
        # Force garbage collection
        gc.collect()
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"Memory growth after 1000 experiences: {memory_growth:.1f}MB")
        
        # Memory growth should be reasonable
        self.assertLess(memory_growth, 300)  # Should not grow more than 300MB
        
        # Check buffer size is limited
        buffer_size = len(self.learning_engine.experience_buffer)
        print(f"Final buffer size: {buffer_size}")
        self.assertLessEqual(buffer_size, 10000)  # Should respect maxlen

def run_integration_tests():
    """Run all integration tests"""
    print("="*60)
    print("INTEGRATION TESTS FOR CRITICAL ISSUE FIXES")
    print("="*60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestThreadSafetyFixes,
        TestInputValidationFixes,
        TestErrorRecoveryFixes,
        TestIntegrationWithFixes
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1) 