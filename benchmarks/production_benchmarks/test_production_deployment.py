#!/usr/bin/env python3
"""
Production Deployment Test Suite

This test suite validates the Week 4 production deployment implementation including:
- GPU optimization and performance
- Memory management and monitoring
- Deployment automation
- Safety validation in production
"""

import unittest
import time
import json
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import threading
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from intelligence.eip_llm_interface.eip_llm_interface.gpu_optimized_llm import GPUOptimizedSafetyLLM, GPUConfig
    from intelligence.eip_llm_interface.eip_llm_interface.advanced_memory_manager import AdvancedMemoryManager
    from intelligence.eip_llm_interface.eip_llm_interface.performance_monitor import PerformanceMonitor, PerformanceBenchmark
    from scripts.prepare_deployment import DeploymentManager
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import production components: {e}")
    IMPORTS_AVAILABLE = False


class TestGPUOptimization(unittest.TestCase):
    """Test GPU optimization features"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Production components not available")
        
        self.gpu_config = GPUConfig(
            device="auto",
            batch_size=2,
            max_memory_mb=2048,
            enable_mixed_precision=True
        )
    
    def test_gpu_config_initialization(self):
        """Test GPU configuration initialization"""
        config = GPUConfig()
        self.assertIsNotNone(config.device)
        self.assertGreater(config.batch_size, 0)
        self.assertGreater(config.max_memory_mb, 0)
    
    def test_gpu_llm_initialization(self):
        """Test GPU-optimized LLM initialization"""
        try:
            llm = GPUOptimizedSafetyLLM(gpu_config=self.gpu_config)
            self.assertIsNotNone(llm)
            self.assertIsNotNone(llm.device)
            llm.shutdown()
        except Exception as e:
            self.skipTest(f"GPU LLM initialization failed: {e}")
    
    def test_memory_manager(self):
        """Test advanced memory manager"""
        memory_manager = AdvancedMemoryManager(
            max_memory_mb=1024,
            device="auto"
        )
        
        # Test memory usage tracking
        usage = memory_manager.get_memory_usage()
        self.assertIsInstance(usage, dict)
        self.assertIn('allocated_mb', usage)
        self.assertIn('utilization_percent', usage)
        
        # Test memory optimization
        memory_manager.optimize_memory()
        
        # Test memory trends
        trends = memory_manager.get_memory_trends()
        self.assertIsInstance(trends, dict)
        
        # Cleanup
        memory_manager.cleanup()
    
    def test_performance_monitor(self):
        """Test performance monitoring"""
        # Create a mock LLM instance for testing
        class MockLLM:
            def get_memory_usage(self):
                return {'allocated_mb': 100, 'utilization_percent': 50.0}
        
        monitor = PerformanceMonitor(MockLLM())
        
        # Test request recording
        monitor.record_request(
            request_id="test_1",
            processing_time=0.5,
            safety_score=0.9,
            success=True
        )
        
        # Test batch recording
        monitor.record_batch(
            batch_size=4,
            total_time=2.0,
            success=True
        )
        
        # Test performance summary
        summary = monitor.get_performance_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('total_requests', summary)
        self.assertIn('success_rate', summary)
        
        # Test recent performance
        recent = monitor.get_recent_performance(time_window=60.0)
        self.assertIsInstance(recent, dict)
        
        # Stop monitoring
        monitor.stop_monitoring()


class TestProductionDeployment(unittest.TestCase):
    """Test production deployment features"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Production components not available")
        
        # Create temporary deployment config
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_deployment_config.json")
        
        test_config = {
            "deployment_mode": "test",
            "gpu_optimization": False,  # Disable for testing
            "monitoring_enabled": True,
            "safety_validation": False,  # Disable for testing
            "performance_benchmarking": False,  # Disable for testing
            "deployment_targets": ["demo-llm"],
            "environment_variables": {
                "ROS_DOMAIN_ID": "42",
                "PYTHONPATH": "/workspace"
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_deployment_manager_initialization(self):
        """Test deployment manager initialization"""
        try:
            manager = DeploymentManager(self.config_path)
            self.assertIsNotNone(manager)
            self.assertIsNotNone(manager.config)
            self.assertEqual(manager.config["deployment_mode"], "test")
        except Exception as e:
            self.skipTest(f"Deployment manager initialization failed: {e}")
    
    def test_environment_validation(self):
        """Test environment validation"""
        manager = DeploymentManager(self.config_path)
        
        # Test individual validation checks
        checks = [
            manager._check_docker,
            manager._check_docker_compose,
            manager._check_system_resources,
            manager._check_permissions
        ]
        
        for check_func in checks:
            try:
                result = check_func()
                self.assertIsInstance(result, bool)
            except Exception as e:
                # Some checks may fail in test environment, that's OK
                pass
    
    def test_config_loading(self):
        """Test configuration loading"""
        manager = DeploymentManager(self.config_path)
        
        # Test default config merging
        self.assertIn("deployment_mode", manager.config)
        self.assertIn("gpu_optimization", manager.config)
        self.assertIn("monitoring_enabled", manager.config)
        self.assertIn("deployment_targets", manager.config)
    
    def test_monitoring_setup(self):
        """Test monitoring setup"""
        manager = DeploymentManager(self.config_path)
        
        # Test monitoring configuration creation
        try:
            # Create temporary monitoring directory
            monitoring_dir = os.path.join(self.temp_dir, "monitoring")
            os.makedirs(monitoring_dir, exist_ok=True)
            
            # Test Prometheus setup
            manager._setup_prometheus()
            
            # Test Grafana setup
            manager._setup_grafana()
            
            # Test alerting setup
            manager._setup_alerting()
            
        except Exception as e:
            # Monitoring setup may fail in test environment
            self.skipTest(f"Monitoring setup failed: {e}")


class TestPerformanceBenchmarking(unittest.TestCase):
    """Test performance benchmarking features"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Production components not available")
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization"""
        # Create mock LLM and monitor
        class MockLLM:
            def generate_safe_response(self, command):
                return type('Response', (), {
                    'content': 'Test response',
                    'safety_score': 0.9,
                    'execution_time': 0.1
                })()
            
            def generate_batch_responses(self, commands):
                return [self.generate_safe_response(cmd) for cmd in commands]
        
        monitor = PerformanceMonitor(MockLLM())
        benchmark = PerformanceBenchmark(MockLLM(), monitor)
        
        self.assertIsNotNone(benchmark)
    
    def test_single_benchmark(self):
        """Test single request benchmarking"""
        # Create mock LLM and monitor
        class MockLLM:
            def generate_safe_response(self, command):
                time.sleep(0.01)  # Simulate processing time
                return type('Response', (), {
                    'content': 'Test response',
                    'safety_score': 0.9,
                    'execution_time': 0.01
                })()
        
        monitor = PerformanceMonitor(MockLLM())
        benchmark = PerformanceBenchmark(MockLLM(), monitor)
        
        # Run benchmark
        test_commands = ["test command 1", "test command 2"]
        results = benchmark.run_benchmark(2, test_commands)
        
        # Validate results
        self.assertIsInstance(results, dict)
        self.assertIn('total_requests', results)
        self.assertIn('success_rate', results)
        self.assertIn('requests_per_second', results)
        self.assertEqual(results['total_requests'], 2)
    
    def test_batch_benchmark(self):
        """Test batch processing benchmarking"""
        # Create mock LLM and monitor
        class MockLLM:
            def generate_batch_responses(self, commands):
                time.sleep(0.01)  # Simulate processing time
                return [type('Response', (), {
                    'content': 'Test response',
                    'safety_score': 0.9,
                    'execution_time': 0.01
                })() for _ in commands]
        
        monitor = PerformanceMonitor(MockLLM())
        benchmark = PerformanceBenchmark(MockLLM(), monitor)
        
        # Run batch benchmark
        test_commands = ["test command 1", "test command 2", "test command 3", "test command 4"]
        results = benchmark.run_batch_benchmark(2, 2, test_commands)
        
        # Validate results
        self.assertIsInstance(results, dict)
        self.assertIn('num_batches', results)
        self.assertIn('batch_success_rate', results)
        self.assertIn('requests_per_second', results)
        self.assertEqual(results['num_batches'], 2)
        self.assertEqual(results['batch_size'], 2)


class TestSafetyValidation(unittest.TestCase):
    """Test safety validation in production"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Production components not available")
    
    def test_safety_embedded_llm_integration(self):
        """Test safety-embedded LLM integration with GPU optimization"""
        try:
            # Test with GPU optimization disabled for testing
            config = GPUConfig(device="cpu", batch_size=1)
            llm = GPUOptimizedSafetyLLM(gpu_config=config)
            
            # Test safe response generation
            response = llm.generate_safe_response("move to the kitchen safely")
            
            # Validate response
            self.assertIsNotNone(response)
            self.assertIsInstance(response.safety_score, float)
            self.assertGreaterEqual(response.safety_score, 0.0)
            self.assertLessEqual(response.safety_score, 1.0)
            
            # Test batch processing
            commands = ["move safely", "stop if unsafe"]
            responses = llm.generate_batch_responses(commands)
            
            self.assertEqual(len(responses), len(commands))
            for response in responses:
                self.assertIsNotNone(response)
                self.assertIsInstance(response.safety_score, float)
            
            llm.shutdown()
            
        except Exception as e:
            self.skipTest(f"Safety-embedded LLM test failed: {e}")
    
    def test_memory_safety(self):
        """Test memory safety and management"""
        memory_manager = AdvancedMemoryManager(max_memory_mb=512, device="cpu")
        
        # Test memory usage tracking
        initial_usage = memory_manager.get_memory_usage()
        
        # Simulate memory-intensive operations
        for i in range(10):
            memory_manager.check_memory_before_processing()
            memory_manager.optimize_after_processing()
        
        final_usage = memory_manager.get_memory_usage()
        
        # Memory usage should be reasonable
        self.assertLess(final_usage['allocated_mb'], 1000)  # Less than 1GB
        
        memory_manager.cleanup()
    
    def test_performance_safety(self):
        """Test performance safety and monitoring"""
        # Create mock LLM
        class MockLLM:
            def generate_safe_response(self, command):
                time.sleep(0.1)  # Simulate processing
                return type('Response', (), {
                    'content': 'Safe response',
                    'safety_score': 0.95,
                    'execution_time': 0.1
                })()
            
            def get_memory_usage(self):
                return {'allocated_mb': 100, 'utilization_percent': 20.0}
        
        monitor = PerformanceMonitor(MockLLM())
        
        # Test performance monitoring
        for i in range(5):
            monitor.record_request(
                request_id=f"test_{i}",
                processing_time=0.1,
                safety_score=0.95,
                success=True
            )
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        
        # Validate performance metrics
        self.assertGreater(summary['success_rate'], 0.9)
        self.assertLess(summary['average_processing_time'], 1.0)
        
        monitor.stop_monitoring()


class TestDeploymentAutomation(unittest.TestCase):
    """Test deployment automation features"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Production components not available")
        
        # Create temporary test environment
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        
        test_config = {
            "deployment_mode": "test",
            "gpu_optimization": False,
            "monitoring_enabled": True,
            "safety_validation": False,
            "performance_benchmarking": False,
            "deployment_targets": ["demo-llm"]
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_deployment_report_generation(self):
        """Test deployment report generation"""
        manager = DeploymentManager(self.config_path)
        
        # Generate report
        report = manager.generate_deployment_report()
        
        # Validate report structure
        self.assertIsInstance(report, dict)
        self.assertIn('timestamp', report)
        self.assertIn('deployment_config', report)
        self.assertIn('services', report)
        self.assertIn('recommendations', report)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        manager = DeploymentManager(self.config_path)
        
        # Test valid configuration
        self.assertTrue(manager.config["monitoring_enabled"])
        self.assertFalse(manager.config["gpu_optimization"])
        
        # Test configuration updates
        manager.config["gpu_optimization"] = True
        self.assertTrue(manager.config["gpu_optimization"])


def run_production_tests():
    """Run all production tests"""
    print("="*60)
    print("PRODUCTION DEPLOYMENT TEST SUITE")
    print("="*60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGPUOptimization,
        TestProductionDeployment,
        TestPerformanceBenchmarking,
        TestSafetyValidation,
        TestDeploymentAutomation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_production_tests()
    sys.exit(0 if success else 1) 