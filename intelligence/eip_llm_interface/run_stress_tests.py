#!/usr/bin/env python3
"""
Comprehensive Stress Test Runner

This script runs all stress tests for the GPU-optimized LLM and generates
detailed performance and memory analysis reports.
"""

import sys
import os
import time
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.test_advanced_memory_manager import TestAdvancedMemoryManager
from tests.test_stress_memory_leak import MemoryLeakDetector
from eip_llm_interface.gpu_optimized_llm import GPUOptimizedSafetyLLM, GPUConfig


class StressTestRunner:
    """Comprehensive stress test runner with reporting"""
    
    def __init__(self, output_dir: str = "stress_test_results"):
        """
        Initialize stress test runner
        
        Args:
            output_dir: Directory for test results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Test results storage
        self.test_results = {}
        self.start_time = None
        
    def run_all_tests(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run all stress tests"""
        self.start_time = time.time()
        self.logger.info("Starting comprehensive stress test suite")
        
        # Test 1: Advanced Memory Manager Tests
        self.logger.info("=== Test 1: Advanced Memory Manager ===")
        memory_manager_results = self._run_memory_manager_tests()
        self.test_results['memory_manager'] = memory_manager_results
        
        # Test 2: Memory Leak Detection
        self.logger.info("=== Test 2: Memory Leak Detection ===")
        memory_leak_results = self._run_memory_leak_tests(test_config)
        self.test_results['memory_leak'] = memory_leak_results
        
        # Test 3: GPU Performance Tests
        self.logger.info("=== Test 3: GPU Performance Tests ===")
        gpu_performance_results = self._run_gpu_performance_tests(test_config)
        self.test_results['gpu_performance'] = gpu_performance_results
        
        # Test 4: Concurrent Access Tests
        self.logger.info("=== Test 4: Concurrent Access Tests ===")
        concurrent_results = self._run_concurrent_tests(test_config)
        self.test_results['concurrent_access'] = concurrent_results
        
        # Test 5: Long-running Stability Tests
        self.logger.info("=== Test 5: Long-running Stability Tests ===")
        stability_results = self._run_stability_tests(test_config)
        self.test_results['stability'] = stability_results
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report()
        
        # Save results
        self._save_results(report)
        
        return report
    
    def _run_memory_manager_tests(self) -> Dict[str, Any]:
        """Run advanced memory manager unit tests"""
        try:
            import unittest
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(TestAdvancedMemoryManager)
            
            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            return {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1),
                'failure_details': [str(f) for f in result.failures],
                'error_details': [str(e) for e in result.errors]
            }
            
        except Exception as e:
            self.logger.error(f"Memory manager tests failed: {e}")
            return {
                'tests_run': 0,
                'failures': 1,
                'errors': 0,
                'success_rate': 0.0,
                'failure_details': [str(e)],
                'error_details': []
            }
    
    def _run_memory_leak_tests(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run memory leak detection tests"""
        try:
            # Initialize LLM for testing
            gpu_config = GPUConfig(
                max_memory_mb=test_config.get('max_memory_mb', 2048),
                batch_size=test_config.get('batch_size', 2),
                enable_mixed_precision=True,
                enable_memory_pooling=True
            )
            
            llm = GPUOptimizedSafetyLLM(
                model_name="microsoft/DialoGPT-medium",
                gpu_config=gpu_config
            )
            
            # Run memory leak detection
            detector = MemoryLeakDetector(
                test_duration=test_config.get('memory_leak_duration', 120),
                max_workers=test_config.get('max_workers', 4)
            )
            
            results = detector.run_memory_leak_test(llm)
            
            # Cleanup
            llm.shutdown()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Memory leak tests failed: {e}")
            return {
                'error': str(e),
                'test_summary': {'total_requests': 0, 'total_errors': 1},
                'memory_analysis': {'memory_leak_detected': True, 'memory_leak_severity': 'unknown'},
                'recommendations': [f"Test failed: {e}"]
            }
    
    def _run_gpu_performance_tests(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run GPU performance benchmarking tests"""
        try:
            import torch
            
            results = {
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'performance_benchmarks': {},
                'memory_benchmarks': {}
            }
            
            if torch.cuda.is_available():
                # GPU memory bandwidth test
                results['memory_benchmarks']['bandwidth'] = self._test_gpu_bandwidth()
                
                # Tensor operation performance
                results['performance_benchmarks']['tensor_ops'] = self._test_tensor_operations()
                
                # Model inference performance
                results['performance_benchmarks']['model_inference'] = self._test_model_inference()
            
            return results
            
        except Exception as e:
            self.logger.error(f"GPU performance tests failed: {e}")
            return {
                'error': str(e),
                'gpu_available': False,
                'performance_benchmarks': {},
                'memory_benchmarks': {}
            }
    
    def _test_gpu_bandwidth(self) -> Dict[str, float]:
        """Test GPU memory bandwidth"""
        import torch
        
        # Test memory allocation/deallocation speed
        start_time = time.time()
        tensors = []
        
        for i in range(100):
            tensor = torch.randn(1000, 1000, device='cuda')
            tensors.append(tensor)
        
        allocation_time = time.time() - start_time
        
        # Test deallocation
        start_time = time.time()
        for tensor in tensors:
            del tensor
        torch.cuda.empty_cache()
        deallocation_time = time.time() - start_time
        
        return {
            'allocation_time_per_tensor_ms': allocation_time * 1000 / 100,
            'deallocation_time_per_tensor_ms': deallocation_time * 1000 / 100,
            'total_memory_allocated_mb': 100 * 1000 * 1000 * 4 / (1024 * 1024)  # 4 bytes per float32
        }
    
    def _test_tensor_operations(self) -> Dict[str, float]:
        """Test tensor operation performance"""
        import torch
        
        # Matrix multiplication test
        size = 1000
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        
        # Warmup
        for _ in range(5):
            _ = torch.matmul(a, b)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        matmul_time = time.time() - start_time
        
        return {
            'matmul_time_ms': matmul_time * 1000 / 10,
            'matmul_gflops': (2 * size**3) / (matmul_time / 10) / 1e9
        }
    
    def _test_model_inference(self) -> Dict[str, float]:
        """Test model inference performance"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load small model for testing
            model_name = "microsoft/DialoGPT-small"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
            
            # Prepare input
            text = "Hello, how are you?"
            inputs = tokenizer(text, return_tensors="pt").to('cuda')
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model.generate(**inputs, max_length=50, do_sample=False)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):
                with torch.no_grad():
                    _ = model.generate(**inputs, max_length=50, do_sample=False)
            torch.cuda.synchronize()
            inference_time = time.time() - start_time
            
            return {
                'inference_time_ms': inference_time * 1000 / 10,
                'tokens_per_second': 50 / (inference_time / 10)
            }
            
        except Exception as e:
            self.logger.warning(f"Model inference test failed: {e}")
            return {
                'inference_time_ms': 0.0,
                'tokens_per_second': 0.0,
                'error': str(e)
            }
    
    def _run_concurrent_tests(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run concurrent access tests"""
        try:
            import threading
            import queue
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            results = {
                'concurrent_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'average_response_time': 0.0,
                'thread_safety_issues': []
            }
            
            # Initialize LLM
            gpu_config = GPUConfig(
                max_memory_mb=test_config.get('max_memory_mb', 2048),
                batch_size=test_config.get('batch_size', 2)
            )
            
            llm = GPUOptimizedSafetyLLM(
                model_name="microsoft/DialoGPT-medium",
                gpu_config=gpu_config
            )
            
            # Test concurrent requests
            num_threads = test_config.get('concurrent_threads', 4)
            requests_per_thread = test_config.get('requests_per_thread', 10)
            
            def worker(thread_id: int):
                thread_results = []
                for i in range(requests_per_thread):
                    try:
                        start_time = time.time()
                        response = llm.generate_safe_response(
                            f"Test request {i} from thread {thread_id}",
                            f"Context for thread {thread_id}"
                        )
                        response_time = time.time() - start_time
                        
                        thread_results.append({
                            'success': True,
                            'response_time': response_time,
                            'thread_id': thread_id,
                            'request_id': i
                        })
                        
                    except Exception as e:
                        thread_results.append({
                            'success': False,
                            'error': str(e),
                            'thread_id': thread_id,
                            'request_id': i
                        })
                
                return thread_results
            
            # Run concurrent workers
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker, i) for i in range(num_threads)]
                
                all_results = []
                for future in as_completed(futures):
                    all_results.extend(future.result())
            
            # Analyze results
            successful_requests = [r for r in all_results if r['success']]
            failed_requests = [r for r in all_results if not r['success']]
            
            results.update({
                'concurrent_requests': len(all_results),
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate': len(successful_requests) / len(all_results) if all_results else 0.0,
                'average_response_time': sum([r['response_time'] for r in successful_requests]) / len(successful_requests) if successful_requests else 0.0,
                'thread_safety_issues': [r['error'] for r in failed_requests]
            })
            
            # Cleanup
            llm.shutdown()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Concurrent tests failed: {e}")
            return {
                'error': str(e),
                'concurrent_requests': 0,
                'successful_requests': 0,
                'failed_requests': 1
            }
    
    def _run_stability_tests(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run long-running stability tests"""
        try:
            import psutil
            
            results = {
                'test_duration': 0.0,
                'requests_processed': 0,
                'memory_growth_mb': 0.0,
                'cpu_usage_percent': 0.0,
                'gpu_memory_growth_mb': 0.0,
                'errors_encountered': 0,
                'stability_score': 0.0
            }
            
            # Initialize LLM
            gpu_config = GPUConfig(
                max_memory_mb=test_config.get('max_memory_mb', 2048),
                batch_size=test_config.get('batch_size', 2)
            )
            
            llm = GPUOptimizedSafetyLLM(
                model_name="microsoft/DialoGPT-medium",
                gpu_config=gpu_config
            )
            
            # Record initial state
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)
            initial_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            
            start_time = time.time()
            test_duration = test_config.get('stability_duration', 300)  # 5 minutes
            
            requests_processed = 0
            errors_encountered = 0
            
            while time.time() - start_time < test_duration:
                try:
                    # Generate request
                    response = llm.generate_safe_response(
                        f"Stability test request {requests_processed}",
                        f"Long-running stability test at {time.time()}"
                    )
                    requests_processed += 1
                    
                    # Small delay
                    time.sleep(0.1)
                    
                except Exception as e:
                    errors_encountered += 1
                    self.logger.warning(f"Stability test error: {e}")
            
            # Record final state
            final_memory = process.memory_info().rss / (1024 * 1024)
            final_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            
            # Calculate results
            actual_duration = time.time() - start_time
            memory_growth = final_memory - initial_memory
            gpu_memory_growth = final_gpu_memory - initial_gpu_memory
            
            # Calculate stability score (0-100)
            error_rate = errors_encountered / max(requests_processed + errors_encountered, 1)
            memory_efficiency = 1.0 - min(memory_growth / 1000, 1.0)  # Penalize memory growth
            gpu_efficiency = 1.0 - min(gpu_memory_growth / 500, 1.0)  # Penalize GPU memory growth
            
            stability_score = (1.0 - error_rate) * 0.4 + memory_efficiency * 0.3 + gpu_efficiency * 0.3
            stability_score *= 100
            
            results.update({
                'test_duration': actual_duration,
                'requests_processed': requests_processed,
                'memory_growth_mb': memory_growth,
                'gpu_memory_growth_mb': gpu_memory_growth,
                'errors_encountered': errors_encountered,
                'error_rate': error_rate,
                'stability_score': stability_score
            })
            
            # Cleanup
            llm.shutdown()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Stability tests failed: {e}")
            return {
                'error': str(e),
                'test_duration': 0.0,
                'requests_processed': 0,
                'stability_score': 0.0
            }
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_duration = time.time() - self.start_time
        
        # Calculate overall metrics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if 'error' not in result)
        
        # Memory leak analysis
        memory_leak_detected = False
        if 'memory_leak' in self.test_results:
            memory_analysis = self.test_results['memory_leak'].get('memory_analysis', {})
            memory_leak_detected = memory_analysis.get('memory_leak_detected', False)
        
        # Performance analysis
        performance_issues = []
        if 'gpu_performance' in self.test_results:
            gpu_results = self.test_results['gpu_performance']
            if not gpu_results.get('gpu_available', False):
                performance_issues.append("GPU not available")
        
        # Stability analysis
        stability_score = 0.0
        if 'stability' in self.test_results:
            stability_score = self.test_results['stability'].get('stability_score', 0.0)
        
        # Generate recommendations
        recommendations = []
        
        if memory_leak_detected:
            recommendations.append("CRITICAL: Address memory leaks before production deployment")
        
        if stability_score < 70:
            recommendations.append("HIGH: Improve system stability - investigate error patterns")
        
        if performance_issues:
            recommendations.append("MEDIUM: Optimize performance - consider GPU acceleration")
        
        if not recommendations:
            recommendations.append("GOOD: System ready for production deployment")
        
        return {
            'test_summary': {
                'total_duration_seconds': total_duration,
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0.0,
                'timestamp': datetime.datetime.now().isoformat()
            },
            'critical_issues': {
                'memory_leak_detected': memory_leak_detected,
                'stability_issues': stability_score < 70,
                'performance_issues': len(performance_issues) > 0
            },
            'performance_metrics': {
                'stability_score': stability_score,
                'overall_health': 'good' if stability_score > 80 else 'warning' if stability_score > 60 else 'critical'
            },
            'recommendations': recommendations,
            'detailed_results': self.test_results
        }
    
    def _save_results(self, report: Dict[str, Any]):
        """Save test results to files"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive report
        report_file = self.output_dir / f"stress_test_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save summary report
        summary_file = self.output_dir / f"stress_test_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("=== STRESS TEST SUMMARY ===\n")
            f.write(f"Timestamp: {report['test_summary']['timestamp']}\n")
            f.write(f"Duration: {report['test_summary']['total_duration_seconds']:.1f}s\n")
            f.write(f"Tests: {report['test_summary']['successful_tests']}/{report['test_summary']['total_tests']}\n")
            f.write(f"Success Rate: {report['test_summary']['success_rate']:.1%}\n")
            f.write(f"Stability Score: {report['performance_metrics']['stability_score']:.1f}/100\n")
            f.write(f"Overall Health: {report['performance_metrics']['overall_health']}\n\n")
            
            f.write("=== CRITICAL ISSUES ===\n")
            for issue, detected in report['critical_issues'].items():
                f.write(f"{issue}: {'YES' if detected else 'NO'}\n")
            
            f.write("\n=== RECOMMENDATIONS ===\n")
            for rec in report['recommendations']:
                f.write(f"- {rec}\n")
        
        self.logger.info(f"Results saved to {self.output_dir}")
        self.logger.info(f"Report: {report_file}")
        self.logger.info(f"Summary: {summary_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Comprehensive Stress Test Runner")
    parser.add_argument("--output-dir", type=str, default="stress_test_results", help="Output directory")
    parser.add_argument("--max-memory-mb", type=int, default=2048, help="Maximum GPU memory in MB")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for testing")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum worker threads")
    parser.add_argument("--test-duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test configuration
    test_config = {
        'max_memory_mb': args.max_memory_mb,
        'batch_size': args.batch_size,
        'max_workers': args.max_workers,
        'memory_leak_duration': min(args.test_duration // 2, 120),
        'concurrent_threads': args.max_workers,
        'requests_per_thread': 10,
        'stability_duration': min(args.test_duration // 2, 300)
    }
    
    # Run tests
    runner = StressTestRunner(args.output_dir)
    
    try:
        report = runner.run_all_tests(test_config)
        
        # Print summary
        print("\n" + "="*50)
        print("STRESS TEST COMPLETED")
        print("="*50)
        print(f"Duration: {report['test_summary']['total_duration_seconds']:.1f}s")
        print(f"Success Rate: {report['test_summary']['success_rate']:.1%}")
        print(f"Stability Score: {report['performance_metrics']['stability_score']:.1f}/100")
        print(f"Overall Health: {report['performance_metrics']['overall_health'].upper()}")
        
        if report['critical_issues']['memory_leak_detected']:
            print("⚠️  CRITICAL: Memory leak detected!")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
        
        print(f"\nDetailed results saved to: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Stress test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 