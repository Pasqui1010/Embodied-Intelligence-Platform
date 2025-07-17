#!/usr/bin/env python3
"""
Performance Benchmarking Script for Code Improvements

This script benchmarks the performance improvements made to the EIP system
and provides detailed metrics on the enhancements.
"""

import time
import statistics
import json
import sys
import os
from typing import Dict, List, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import threading

# Add the module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'eip_llm_interface'))

from eip_llm_interface.performance_optimizations import (
    GPUMemoryOptimizer, ResponseCache, MemoryMonitor, 
    PerformanceProfiler, optimize_torch_settings
)
from eip_llm_interface.error_handling import ErrorHandler
from eip_llm_interface.config_manager import ConfigManager
from eip_llm_interface.testing_framework import MockSafetyLLM


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.setup_logging()
        
        # Initialize components
        self.memory_monitor = MemoryMonitor()
        self.profiler = PerformanceProfiler()
        self.error_handler = ErrorHandler()
        self.mock_llm = MockSafetyLLM()
        
        # Apply optimizations
        optimize_torch_settings()
    
    def setup_logging(self):
        """Set up logging for benchmarks"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def benchmark_response_cache(self) -> Dict[str, Any]:
        """Benchmark response caching performance"""
        self.logger.info("Benchmarking response cache performance...")
        
        cache_sizes = [32, 64, 128, 256]
        results = {}
        
        for cache_size in cache_sizes:
            cache = ResponseCache(max_size=cache_size)
            
            # Warm up cache
            for i in range(cache_size):
                cache.put(f"prompt_{i}", {"response": f"response_{i}"})
            
            # Benchmark cache hits
            start_time = time.time()
            for i in range(1000):
                key = f"prompt_{i % cache_size}"
                result = cache.get(key)
            
            cache_hit_time = time.time() - start_time
            
            # Benchmark cache misses
            start_time = time.time()
            for i in range(1000):
                key = f"new_prompt_{i}"
                result = cache.get(key)
            
            cache_miss_time = time.time() - start_time
            
            stats = cache.get_stats()
            
            results[f"cache_size_{cache_size}"] = {
                "cache_hit_time_ms": cache_hit_time * 1000,
                "cache_miss_time_ms": cache_miss_time * 1000,
                "hit_rate": stats['hit_rate'],
                "avg_hit_time_us": (cache_hit_time / 1000) * 1000000,
                "avg_miss_time_us": (cache_miss_time / 1000) * 1000000
            }
        
        return results
    
    def benchmark_error_handling(self) -> Dict[str, Any]:
        """Benchmark error handling performance"""
        self.logger.info("Benchmarking error handling performance...")
        
        # Test error handling overhead
        def normal_operation():
            time.sleep(0.001)  # Simulate work
            return "success"
        
        def operation_with_error():
            time.sleep(0.001)  # Simulate work
            raise Exception("Test error")
        
        # Benchmark normal operations
        start_time = time.time()
        for i in range(100):
            normal_operation()
        normal_time = time.time() - start_time
        
        # Benchmark operations with error handling
        start_time = time.time()
        for i in range(100):
            try:
                operation_with_error()
            except Exception as e:
                self.error_handler.handle_error(e)
        
        error_handling_time = time.time() - start_time
        
        # Get error statistics
        error_stats = self.error_handler.get_error_statistics()
        
        return {
            "normal_operation_time_ms": normal_time * 1000,
            "error_handling_time_ms": error_handling_time * 1000,
            "overhead_percentage": ((error_handling_time - normal_time) / normal_time) * 100,
            "errors_processed": error_stats['total_errors'],
            "recovery_success_rate": error_stats.get('recovery_success_rate', 0.0)
        }
    
    def benchmark_memory_monitoring(self) -> Dict[str, Any]:
        """Benchmark memory monitoring performance"""
        self.logger.info("Benchmarking memory monitoring performance...")
        
        # Benchmark memory usage tracking
        start_time = time.time()
        for i in range(1000):
            usage = self.memory_monitor.get_memory_usage()
        monitoring_time = time.time() - start_time
        
        # Benchmark memory delta calculation
        start_time = time.time()
        for i in range(1000):
            delta = self.memory_monitor.get_memory_delta()
        delta_time = time.time() - start_time
        
        # Test memory pressure detection
        start_time = time.time()
        for i in range(1000):
            pressure = self.memory_monitor.check_memory_pressure()
        pressure_time = time.time() - start_time
        
        return {
            "memory_usage_tracking_ms": monitoring_time * 1000,
            "memory_delta_calculation_ms": delta_time * 1000,
            "memory_pressure_detection_ms": pressure_time * 1000,
            "avg_monitoring_time_us": (monitoring_time / 1000) * 1000000,
            "current_memory_usage_mb": self.memory_monitor.get_memory_usage()['rss_mb']
        }
    
    def benchmark_configuration_management(self) -> Dict[str, Any]:
        """Benchmark configuration management performance"""
        self.logger.info("Benchmarking configuration management performance...")
        
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Benchmark config loading
            start_time = time.time()
            for i in range(10):
                config_manager = ConfigManager(temp_dir)
                config = config_manager.get_config()
            config_loading_time = time.time() - start_time
            
            # Benchmark config updates
            config_manager = ConfigManager(temp_dir)
            start_time = time.time()
            for i in range(100):
                updates = {'model': {'temperature': 0.5 + (i * 0.001)}}
                config_manager.update_config(updates)
            config_update_time = time.time() - start_time
            
            # Benchmark config validation
            start_time = time.time()
            for i in range(100):
                issues = config_manager.validate_config()
            validation_time = time.time() - start_time
            
            return {
                "config_loading_time_ms": config_loading_time * 1000,
                "config_update_time_ms": config_update_time * 1000,
                "config_validation_time_ms": validation_time * 1000,
                "avg_loading_time_ms": (config_loading_time / 10) * 1000,
                "avg_update_time_us": (config_update_time / 100) * 1000000,
                "avg_validation_time_us": (validation_time / 100) * 1000000
            }
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def benchmark_concurrent_operations(self) -> Dict[str, Any]:
        """Benchmark concurrent operations performance"""
        self.logger.info("Benchmarking concurrent operations performance...")
        
        def worker_task(worker_id: int, iterations: int):
            """Worker task for concurrent testing"""
            results = []
            for i in range(iterations):
                start_time = time.time()
                
                # Simulate LLM operation
                response = self.mock_llm.generate_safe_response(
                    f"Worker {worker_id} command {i}",
                    "test context"
                )
                
                execution_time = time.time() - start_time
                results.append(execution_time)
            
            return results
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        results = {}
        
        for num_workers in concurrency_levels:
            iterations_per_worker = 20
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(worker_task, i, iterations_per_worker)
                    for i in range(num_workers)
                ]
                
                all_results = []
                for future in as_completed(futures):
                    worker_results = future.result()
                    all_results.extend(worker_results)
            
            total_time = time.time() - start_time
            
            results[f"workers_{num_workers}"] = {
                "total_time_ms": total_time * 1000,
                "total_operations": len(all_results),
                "avg_operation_time_ms": statistics.mean(all_results) * 1000,
                "operations_per_second": len(all_results) / total_time,
                "min_operation_time_ms": min(all_results) * 1000,
                "max_operation_time_ms": max(all_results) * 1000,
                "std_dev_ms": statistics.stdev(all_results) * 1000 if len(all_results) > 1 else 0
            }
        
        return results
    
    def benchmark_memory_efficiency(self) -> Dict[str, Any]:
        """Benchmark memory efficiency improvements"""
        self.logger.info("Benchmarking memory efficiency...")
        
        # Reset memory baseline
        self.memory_monitor.reset_baseline()
        
        # Simulate memory-intensive operations
        operations = []
        for i in range(100):
            # Create some data structures
            data = {
                'operation_id': i,
                'data': [j for j in range(100)],
                'response': self.mock_llm.generate_safe_response(f"Operation {i}")
            }
            operations.append(data)
            
            if i % 10 == 0:
                # Check memory usage periodically
                usage = self.memory_monitor.get_memory_usage()
                delta = self.memory_monitor.get_memory_delta()
        
        # Final memory check
        final_usage = self.memory_monitor.get_memory_usage()
        final_delta = self.memory_monitor.get_memory_delta()
        
        # Clean up
        del operations
        import gc
        gc.collect()
        
        # Check memory after cleanup
        cleanup_usage = self.memory_monitor.get_memory_usage()
        cleanup_delta = self.memory_monitor.get_memory_delta()
        
        return {
            "peak_memory_usage_mb": final_usage['rss_mb'],
            "memory_growth_mb": final_delta['rss_delta_mb'],
            "memory_after_cleanup_mb": cleanup_usage['rss_mb'],
            "memory_recovered_mb": final_usage['rss_mb'] - cleanup_usage['rss_mb'],
            "memory_efficiency_score": max(0, 100 - (final_delta['rss_delta_mb'] / 10))  # Arbitrary scoring
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        self.logger.info("Starting comprehensive performance benchmark...")
        
        benchmark_start = time.time()
        
        # Run all benchmarks
        benchmarks = {
            'response_cache': self.benchmark_response_cache,
            'error_handling': self.benchmark_error_handling,
            'memory_monitoring': self.benchmark_memory_monitoring,
            'configuration_management': self.benchmark_configuration_management,
            'concurrent_operations': self.benchmark_concurrent_operations,
            'memory_efficiency': self.benchmark_memory_efficiency
        }
        
        results = {}
        
        for benchmark_name, benchmark_func in benchmarks.items():
            try:
                self.logger.info(f"Running {benchmark_name} benchmark...")
                benchmark_result = benchmark_func()
                results[benchmark_name] = benchmark_result
                self.logger.info(f"Completed {benchmark_name} benchmark")
            except Exception as e:
                self.logger.error(f"Benchmark {benchmark_name} failed: {e}")
                results[benchmark_name] = {"error": str(e)}
        
        total_time = time.time() - benchmark_start
        
        # Add summary
        results['summary'] = {
            'total_benchmark_time_ms': total_time * 1000,
            'benchmarks_completed': len([r for r in results.values() if 'error' not in r]),
            'benchmarks_failed': len([r for r in results.values() if 'error' in r]),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version
            }
        }
        
        return results
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable performance report"""
        report = []
        report.append("=" * 80)
        report.append("EIP CODE IMPROVEMENTS PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        summary = results.get('summary', {})
        report.append("SUMMARY:")
        report.append(f"  Total benchmark time: {summary.get('total_benchmark_time_ms', 0):.1f}ms")
        report.append(f"  Benchmarks completed: {summary.get('benchmarks_completed', 0)}")
        report.append(f"  Benchmarks failed: {summary.get('benchmarks_failed', 0)}")
        report.append("")
        
        # System info
        system_info = summary.get('system_info', {})
        report.append("SYSTEM INFO:")
        report.append(f"  CPU cores: {system_info.get('cpu_count', 'unknown')}")
        report.append(f"  Total memory: {system_info.get('memory_total_gb', 0):.1f}GB")
        report.append("")
        
        # Individual benchmark results
        for benchmark_name, benchmark_results in results.items():
            if benchmark_name == 'summary':
                continue
            
            report.append(f"{benchmark_name.upper().replace('_', ' ')}:")
            
            if 'error' in benchmark_results:
                report.append(f"  ❌ FAILED: {benchmark_results['error']}")
            else:
                # Format results based on benchmark type
                if benchmark_name == 'response_cache':
                    for cache_size, metrics in benchmark_results.items():
                        report.append(f"  {cache_size}:")
                        report.append(f"    Cache hit time: {metrics['avg_hit_time_us']:.1f}μs")
                        report.append(f"    Cache miss time: {metrics['avg_miss_time_us']:.1f}μs")
                
                elif benchmark_name == 'error_handling':
                    report.append(f"  Normal operations: {benchmark_results['normal_operation_time_ms']:.1f}ms")
                    report.append(f"  With error handling: {benchmark_results['error_handling_time_ms']:.1f}ms")
                    report.append(f"  Overhead: {benchmark_results['overhead_percentage']:.1f}%")
                
                elif benchmark_name == 'concurrent_operations':
                    for workers, metrics in benchmark_results.items():
                        report.append(f"  {workers}:")
                        report.append(f"    Operations/sec: {metrics['operations_per_second']:.1f}")
                        report.append(f"    Avg time: {metrics['avg_operation_time_ms']:.1f}ms")
                
                elif benchmark_name == 'memory_efficiency':
                    report.append(f"  Peak memory: {benchmark_results['peak_memory_usage_mb']:.1f}MB")
                    report.append(f"  Memory growth: {benchmark_results['memory_growth_mb']:.1f}MB")
                    report.append(f"  Efficiency score: {benchmark_results['memory_efficiency_score']:.1f}/100")
                
                else:
                    # Generic formatting
                    for key, value in benchmark_results.items():
                        if isinstance(value, (int, float)):
                            report.append(f"  {key}: {value:.2f}")
                        else:
                            report.append(f"  {key}: {value}")
            
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main benchmark execution"""
    print("Starting EIP Code Improvements Performance Benchmark...")
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark()
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Generate report
    report = benchmark.generate_performance_report(results)
    
    # Print report
    print(report)
    
    # Save results to file
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open('benchmark_report.txt', 'w') as f:
        f.write(report)
    
    print("\nBenchmark completed!")
    print("Results saved to: benchmark_results.json")
    print("Report saved to: benchmark_report.txt")
    
    # Return success/failure based on results
    failed_benchmarks = len([r for r in results.values() if isinstance(r, dict) and 'error' in r])
    if failed_benchmarks == 0:
        print("✅ All benchmarks completed successfully!")
        return 0
    else:
        print(f"❌ {failed_benchmarks} benchmarks failed!")
        return 1


if __name__ == '__main__':
    exit(main())