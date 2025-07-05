#!/usr/bin/env python3
"""
Performance Benchmarks for Adaptive Safety Orchestration

This module benchmarks the performance of the ASO system including:
- Learning speed and convergence
- Safety validation latency
- Memory usage and efficiency
- Rule generation quality
"""

import unittest
import numpy as np
import time
import psutil
import os
import sys
from unittest.mock import Mock, patch
import json
import threading

# Add the package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../intelligence/eip_adaptive_safety'))

from eip_adaptive_safety.adaptive_learning_engine import (
    AdaptiveLearningEngine, SafetyExperience, SafetyRule, MetaLearner
)

class AdaptiveSafetyPerformanceBenchmark(unittest.TestCase):
    """Performance benchmarks for Adaptive Safety Orchestration"""
    
    def setUp(self):
        """Set up benchmark fixtures"""
        with patch('rclpy.init'), patch('rclpy.Node'):
            self.learning_engine = AdaptiveLearningEngine()
            
        # Performance tracking
        self.performance_metrics = {
            'learning_speed': [],
            'validation_latency': [],
            'memory_usage': [],
            'rule_quality': [],
            'convergence_time': []
        }
        
    def test_learning_speed_benchmark(self):
        """Benchmark learning speed"""
        print("\n=== Learning Speed Benchmark ===")
        
        # Generate test experiences
        num_experiences = 1000
        experiences = []
        
        start_time = time.time()
        for i in range(num_experiences):
            experience = SafetyExperience(
                timestamp=time.time(),
                sensor_data={'velocity': np.array([i * 0.01]), 'proximity': np.array([i * 0.005])},
                safety_violation=i % 5 == 0,
                violation_type='collision',
                severity=0.3 + (i % 5) * 0.1,
                context={'human_present': i % 3 == 0},
                outcome='near_miss' if i % 5 == 0 else 'safe_operation'
            )
            experiences.append(experience)
        generation_time = time.time() - start_time
        
        # Process experiences and measure learning speed
        start_time = time.time()
        for experience in experiences:
            self.learning_engine._process_experience(experience)
        processing_time = time.time() - start_time
        
        # Update meta-learner and measure training speed
        start_time = time.time()
        for _ in range(10):  # Multiple updates
            self.learning_engine._update_meta_learner()
        training_time = time.time() - start_time
        
        # Calculate metrics
        experiences_per_second = num_experiences / processing_time
        training_updates_per_second = 10 / training_time
        
        print(f"Experience Generation: {generation_time:.3f}s")
        print(f"Experience Processing: {processing_time:.3f}s ({experiences_per_second:.1f} exp/s)")
        print(f"Meta-Learning Training: {training_time:.3f}s ({training_updates_per_second:.1f} updates/s)")
        
        # Performance assertions
        self.assertGreater(experiences_per_second, 100)  # At least 100 exp/s
        self.assertGreater(training_updates_per_second, 1)  # At least 1 update/s
        
        # Store metrics
        self.performance_metrics['learning_speed'].append({
            'experiences_per_second': experiences_per_second,
            'training_updates_per_second': training_updates_per_second,
            'total_time': generation_time + processing_time + training_time
        })
        
    def test_validation_latency_benchmark(self):
        """Benchmark safety validation latency"""
        print("\n=== Validation Latency Benchmark ===")
        
        # Generate some rules first
        for i in range(50):
            experience = SafetyExperience(
                timestamp=time.time(),
                sensor_data={'velocity': np.array([i * 0.1])},
                safety_violation=i % 3 == 0,
                violation_type='test',
                severity=0.5,
                context={},
                outcome='test'
            )
            self.learning_engine._process_experience(experience)
            
        # Update meta-learner
        self.learning_engine._update_meta_learner()
        
        # Benchmark validation latency
        num_validations = 100
        latencies = []
        
        for i in range(num_validations):
            # Create test request
            request = Mock()
            request.task_plan = f"move to position {i}"
            response = Mock()
            
            # Measure validation time
            start_time = time.time()
            result = self.learning_engine._validate_task_adaptive(request, response)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
            
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        
        print(f"Average Latency: {avg_latency*1000:.2f}ms")
        print(f"95th Percentile: {p95_latency*1000:.2f}ms")
        print(f"99th Percentile: {p99_latency*1000:.2f}ms")
        print(f"Maximum Latency: {max_latency*1000:.2f}ms")
        
        # Performance assertions
        self.assertLess(avg_latency, 0.1)  # Less than 100ms average
        self.assertLess(p95_latency, 0.2)  # Less than 200ms for 95%
        self.assertLess(p99_latency, 0.5)  # Less than 500ms for 99%
        
        # Store metrics
        self.performance_metrics['validation_latency'].append({
            'average_ms': avg_latency * 1000,
            'p95_ms': p95_latency * 1000,
            'p99_ms': p99_latency * 1000,
            'max_ms': max_latency * 1000
        })
        
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage"""
        print("\n=== Memory Usage Benchmark ===")
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Add experiences and measure memory growth
        memory_samples = [initial_memory]
        
        for batch in range(10):
            # Add batch of experiences
            for i in range(100):
                experience = SafetyExperience(
                    timestamp=time.time(),
                    sensor_data={'data': np.random.randn(50)},
                    safety_violation=i % 5 == 0,
                    violation_type='test',
                    severity=0.5,
                    context={},
                    outcome='test'
                )
                self.learning_engine._process_experience(experience)
                
            # Update meta-learner
            self.learning_engine._update_meta_learner()
            
            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
        # Calculate memory metrics
        final_memory = memory_samples[-1]
        memory_growth = final_memory - initial_memory
        peak_memory = max(memory_samples)
        
        print(f"Initial Memory: {initial_memory:.1f}MB")
        print(f"Final Memory: {final_memory:.1f}MB")
        print(f"Memory Growth: {memory_growth:.1f}MB")
        print(f"Peak Memory: {peak_memory:.1f}MB")
        
        # Performance assertions
        self.assertLess(memory_growth, 500)  # Less than 500MB growth
        self.assertLess(peak_memory, 1000)   # Less than 1GB peak
        
        # Store metrics
        self.performance_metrics['memory_usage'].append({
            'initial_mb': initial_memory,
            'final_mb': final_memory,
            'growth_mb': memory_growth,
            'peak_mb': peak_memory
        })
        
    def test_rule_quality_benchmark(self):
        """Benchmark rule generation quality"""
        print("\n=== Rule Quality Benchmark ===")
        
        # Create experiences with clear patterns
        patterns = [
            {'velocity': 2.0, 'human_present': True, 'violation': True},
            {'velocity': 0.5, 'human_present': False, 'violation': False},
            {'velocity': 1.5, 'human_present': True, 'violation': True},
            {'velocity': 0.8, 'human_present': False, 'violation': False},
        ]
        
        # Generate experiences based on patterns
        for i in range(200):
            pattern = patterns[i % len(patterns)]
            experience = SafetyExperience(
                timestamp=time.time(),
                sensor_data={'velocity': np.array([pattern['velocity']])},
                safety_violation=pattern['violation'],
                violation_type='collision',
                severity=0.8 if pattern['violation'] else 0.1,
                context={'human_present': pattern['human_present']},
                outcome='incident' if pattern['violation'] else 'safe_operation'
            )
            self.learning_engine._process_experience(experience)
            
        # Update meta-learner multiple times
        for _ in range(5):
            self.learning_engine._update_meta_learner()
            
        # Analyze rule quality
        rules = list(self.learning_engine.safety_rules.values())
        
        if rules:
            confidences = [rule.confidence for rule in rules]
            success_rates = [rule.success_rate for rule in rules]
            usage_counts = [rule.usage_count for rule in rules]
            
            avg_confidence = np.mean(confidences)
            avg_success_rate = np.mean(success_rates)
            total_usage = sum(usage_counts)
            
            print(f"Generated Rules: {len(rules)}")
            print(f"Average Confidence: {avg_confidence:.3f}")
            print(f"Average Success Rate: {avg_success_rate:.3f}")
            print(f"Total Rule Usage: {total_usage}")
            
            # Quality assertions
            self.assertGreater(len(rules), 0)
            self.assertGreater(avg_confidence, 0.5)
            self.assertGreater(avg_success_rate, 0.5)
            
            # Store metrics
            self.performance_metrics['rule_quality'].append({
                'num_rules': len(rules),
                'avg_confidence': avg_confidence,
                'avg_success_rate': avg_success_rate,
                'total_usage': total_usage
            })
        else:
            print("No rules generated")
            
    def test_convergence_benchmark(self):
        """Benchmark learning convergence"""
        print("\n=== Convergence Benchmark ===")
        
        # Create a learning scenario with clear convergence target
        target_pattern = {'velocity': 1.5, 'human_present': True, 'violation': True}
        
        convergence_metrics = []
        start_time = time.time()
        
        for epoch in range(20):
            # Generate experiences for this epoch
            for i in range(50):
                # Mix target pattern with noise
                if i % 3 == 0:  # Target pattern
                    velocity = target_pattern['velocity']
                    human_present = target_pattern['human_present']
                    violation = target_pattern['violation']
                else:  # Noise
                    velocity = np.random.uniform(0.1, 3.0)
                    human_present = np.random.choice([True, False])
                    violation = np.random.choice([True, False])
                    
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
            
            # Measure convergence metric (rule confidence)
            rules = list(self.learning_engine.safety_rules.values())
            if rules:
                avg_confidence = np.mean([rule.confidence for rule in rules])
                convergence_metrics.append(avg_confidence)
            else:
                convergence_metrics.append(0.0)
                
        convergence_time = time.time() - start_time
        
        # Analyze convergence
        if len(convergence_metrics) > 1:
            initial_confidence = convergence_metrics[0]
            final_confidence = convergence_metrics[-1]
            confidence_improvement = final_confidence - initial_confidence
            
            print(f"Convergence Time: {convergence_time:.2f}s")
            print(f"Initial Confidence: {initial_confidence:.3f}")
            print(f"Final Confidence: {final_confidence:.3f}")
            print(f"Confidence Improvement: {confidence_improvement:.3f}")
            
            # Convergence assertions
            self.assertGreater(confidence_improvement, 0.0)
            self.assertLess(convergence_time, 60.0)  # Less than 60 seconds
            
            # Store metrics
            self.performance_metrics['convergence_time'].append({
                'convergence_time_s': convergence_time,
                'confidence_improvement': confidence_improvement,
                'final_confidence': final_confidence
            })
        else:
            print("Insufficient data for convergence analysis")
            
    def test_concurrent_performance_benchmark(self):
        """Benchmark concurrent access performance"""
        print("\n=== Concurrent Performance Benchmark ===")
        
        # Test concurrent experience processing
        num_threads = 4
        experiences_per_thread = 250
        results = []
        
        def process_experiences(thread_id):
            thread_results = []
            for i in range(experiences_per_thread):
                experience = SafetyExperience(
                    timestamp=time.time(),
                    sensor_data={'thread': thread_id, 'index': i},
                    safety_violation=i % 5 == 0,
                    violation_type='test',
                    severity=0.5,
                    context={},
                    outcome='test'
                )
                
                start_time = time.time()
                self.learning_engine._process_experience(experience)
                end_time = time.time()
                
                thread_results.append(end_time - start_time)
            return thread_results
            
        # Start concurrent processing
        threads = []
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=lambda: results.extend(process_experiences(i)))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        total_time = time.time() - start_time
        
        # Calculate concurrent performance metrics
        total_experiences = num_threads * experiences_per_thread
        avg_latency = np.mean(results)
        throughput = total_experiences / total_time
        
        print(f"Total Experiences: {total_experiences}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.1f} exp/s")
        print(f"Average Latency: {avg_latency*1000:.2f}ms")
        
        # Performance assertions
        self.assertGreater(throughput, 50)  # At least 50 exp/s
        self.assertLess(avg_latency, 0.1)   # Less than 100ms average
        
    def test_stress_test_benchmark(self):
        """Stress test benchmark"""
        print("\n=== Stress Test Benchmark ===")
        
        # Stress test with high load
        num_experiences = 5000
        start_time = time.time()
        
        for i in range(num_experiences):
            # Create complex experience
            experience = SafetyExperience(
                timestamp=time.time(),
                sensor_data={
                    'velocity': np.random.randn(10),
                    'proximity': np.random.randn(5),
                    'force': np.random.randn(8),
                    'temperature': np.random.randn(3)
                },
                safety_violation=i % 10 == 0,
                violation_type='collision',
                severity=np.random.uniform(0.1, 0.9),
                context={
                    'human_present': i % 3 == 0,
                    'workspace': 'lab',
                    'time_of_day': i % 24,
                    'weather': 'clear'
                },
                outcome='near_miss' if i % 10 == 0 else 'safe_operation'
            )
            
            self.learning_engine._process_experience(experience)
            
            # Periodic updates
            if i % 100 == 0:
                self.learning_engine._update_meta_learner()
                
        total_time = time.time() - start_time
        throughput = num_experiences / total_time
        
        print(f"Stress Test Experiences: {num_experiences}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.1f} exp/s")
        print(f"Generated Rules: {len(self.learning_engine.safety_rules)}")
        
        # Stress test assertions
        self.assertGreater(throughput, 100)  # At least 100 exp/s under stress
        self.assertLess(total_time, 60)      # Complete within 60 seconds
        
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "="*60)
        print("ADAPTIVE SAFETY ORCHESTRATION PERFORMANCE REPORT")
        print("="*60)
        
        # Learning Speed
        if self.performance_metrics['learning_speed']:
            speed = self.performance_metrics['learning_speed'][0]
            print(f"\nLearning Speed:")
            print(f"  - Experience Processing: {speed['experiences_per_second']:.1f} exp/s")
            print(f"  - Training Updates: {speed['training_updates_per_second']:.1f} updates/s")
            
        # Validation Latency
        if self.performance_metrics['validation_latency']:
            latency = self.performance_metrics['validation_latency'][0]
            print(f"\nValidation Latency:")
            print(f"  - Average: {latency['average_ms']:.2f}ms")
            print(f"  - 95th Percentile: {latency['p95_ms']:.2f}ms")
            print(f"  - 99th Percentile: {latency['p99_ms']:.2f}ms")
            
        # Memory Usage
        if self.performance_metrics['memory_usage']:
            memory = self.performance_metrics['memory_usage'][0]
            print(f"\nMemory Usage:")
            print(f"  - Initial: {memory['initial_mb']:.1f}MB")
            print(f"  - Final: {memory['final_mb']:.1f}MB")
            print(f"  - Growth: {memory['growth_mb']:.1f}MB")
            
        # Rule Quality
        if self.performance_metrics['rule_quality']:
            quality = self.performance_metrics['rule_quality'][0]
            print(f"\nRule Quality:")
            print(f"  - Generated Rules: {quality['num_rules']}")
            print(f"  - Average Confidence: {quality['avg_confidence']:.3f}")
            print(f"  - Average Success Rate: {quality['avg_success_rate']:.3f}")
            
        # Convergence
        if self.performance_metrics['convergence_time']:
            convergence = self.performance_metrics['convergence_time'][0]
            print(f"\nConvergence:")
            print(f"  - Convergence Time: {convergence['convergence_time_s']:.2f}s")
            print(f"  - Confidence Improvement: {convergence['confidence_improvement']:.3f}")
            
        print("\n" + "="*60)
        
    def tearDown(self):
        """Clean up after tests"""
        # Generate performance report
        self.generate_performance_report()

if __name__ == '__main__':
    # Run performance benchmarks
    unittest.main(verbosity=2) 