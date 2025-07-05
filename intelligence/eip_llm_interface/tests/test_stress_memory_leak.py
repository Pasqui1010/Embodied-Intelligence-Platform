#!/usr/bin/env python3
"""
Comprehensive Stress Test for Memory Leak Detection

This script performs intensive stress testing to detect memory leaks,
pressure issues, and performance degradation in the GPU-optimized LLM.
"""

import sys
import os
import time
import threading
import gc
import psutil
import torch
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import json
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eip_llm_interface.gpu_optimized_llm import GPUOptimizedSafetyLLM, GPUConfig
from eip_llm_interface.advanced_memory_manager import AdvancedMemoryManager


class MemoryLeakDetector:
    """Advanced memory leak detection and stress testing"""
    
    def __init__(self, test_duration: int = 300, max_workers: int = 8):
        """
        Initialize memory leak detector
        
        Args:
            test_duration: Test duration in seconds
            max_workers: Maximum number of worker threads
        """
        self.test_duration = test_duration
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Test configuration
        self.test_commands = [
            "Move the robot forward by 2 meters",
            "Turn the robot 90 degrees to the left",
            "Stop the robot immediately",
            "Check the robot's battery level",
            "Navigate to the charging station",
            "Perform a safety scan of the environment",
            "Return to the home position",
            "Execute emergency stop procedure",
            "Calibrate the robot's sensors",
            "Report current status"
        ]
        
        # Memory tracking
        self.memory_history = []
        self.performance_history = []
        self.error_count = 0
        self.success_count = 0
        
        # Test state
        self.test_running = False
        self.start_time = None
        
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics"""
        stats = {}
        
        # GPU memory stats
        if torch.cuda.is_available():
            stats.update({
                'gpu_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                'gpu_cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
                'gpu_memory_fragmentation': self._calculate_gpu_fragmentation()
            })
        
        # System memory stats
        process = psutil.Process()
        memory_info = process.memory_info()
        stats.update({
            'system_rss_mb': memory_info.rss / (1024 * 1024),
            'system_vms_mb': memory_info.vms / (1024 * 1024),
            'system_percent': process.memory_percent()
        })
        
        return stats
    
    def _calculate_gpu_fragmentation(self) -> float:
        """Calculate GPU memory fragmentation ratio"""
        if not torch.cuda.is_available():
            return 0.0
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        if reserved == 0:
            return 0.0
        
        return 1.0 - (allocated / reserved)
    
    def _record_memory_snapshot(self, test_phase: str):
        """Record memory snapshot with timestamp"""
        snapshot = {
            'timestamp': time.time(),
            'phase': test_phase,
            'memory_stats': self._get_memory_stats(),
            'error_count': self.error_count,
            'success_count': self.success_count
        }
        self.memory_history.append(snapshot)
        
        # Log memory usage
        stats = snapshot['memory_stats']
        self.logger.info(f"{test_phase}: "
                        f"GPU: {stats.get('gpu_allocated_mb', 0):.1f}MB, "
                        f"System: {stats['system_rss_mb']:.1f}MB, "
                        f"Errors: {self.error_count}, "
                        f"Success: {self.success_count}")
    
    def _stress_worker(self, worker_id: int, llm: GPUOptimizedSafetyLLM) -> Dict[str, Any]:
        """Worker function for stress testing"""
        worker_stats = {
            'worker_id': worker_id,
            'requests_processed': 0,
            'total_time': 0.0,
            'errors': 0,
            'memory_usage': []
        }
        
        start_time = time.time()
        
        try:
            while self.test_running and (time.time() - start_time) < self.test_duration:
                # Select random command
                command = np.random.choice(self.test_commands)
                
                # Add some context variation
                context = f"Worker {worker_id} context at {time.time()}"
                
                try:
                    # Generate response
                    response_start = time.time()
                    response = llm.generate_safe_response(command, context)
                    response_time = time.time() - response_start
                    
                    # Record performance
                    worker_stats['requests_processed'] += 1
                    worker_stats['total_time'] += response_time
                    
                    # Record memory usage periodically
                    if worker_stats['requests_processed'] % 10 == 0:
                        memory_stats = self._get_memory_stats()
                        worker_stats['memory_usage'].append({
                            'timestamp': time.time(),
                            'gpu_mb': memory_stats.get('gpu_allocated_mb', 0),
                            'system_mb': memory_stats['system_rss_mb']
                        })
                    
                    # Small delay to prevent overwhelming
                    time.sleep(0.01)
                    
                except Exception as e:
                    worker_stats['errors'] += 1
                    self.error_count += 1
                    self.logger.warning(f"Worker {worker_id} error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Worker {worker_id} failed: {e}")
            worker_stats['errors'] += 1
        
        self.success_count += worker_stats['requests_processed']
        return worker_stats
    
    def run_memory_leak_test(self, llm: GPUOptimizedSafetyLLM) -> Dict[str, Any]:
        """Run comprehensive memory leak detection test"""
        self.logger.info("Starting memory leak detection test...")
        
        # Initialize test state
        self.test_running = True
        self.start_time = time.time()
        self.memory_history.clear()
        self.performance_history.clear()
        self.error_count = 0
        self.success_count = 0
        
        # Record initial memory state
        self._record_memory_snapshot("initial")
        
        # Phase 1: Baseline testing
        self.logger.info("Phase 1: Baseline testing")
        baseline_workers = ThreadPoolExecutor(max_workers=2)
        baseline_futures = [
            baseline_workers.submit(self._stress_worker, i, llm)
            for i in range(2)
        ]
        
        # Wait for baseline completion
        baseline_results = []
        for future in as_completed(baseline_futures):
            baseline_results.append(future.result())
        
        baseline_workers.shutdown()
        self._record_memory_snapshot("baseline_complete")
        
        # Phase 2: Memory pressure testing
        self.logger.info("Phase 2: Memory pressure testing")
        pressure_workers = ThreadPoolExecutor(max_workers=self.max_workers)
        pressure_futures = [
            pressure_workers.submit(self._stress_worker, i, llm)
            for i in range(self.max_workers)
        ]
        
        # Monitor memory during pressure test
        pressure_start = time.time()
        while time.time() - pressure_start < 60:  # 1 minute pressure test
            self._record_memory_snapshot("pressure_test")
            time.sleep(5)  # Record every 5 seconds
        
        # Wait for pressure test completion
        pressure_results = []
        for future in as_completed(pressure_futures):
            pressure_results.append(future.result())
        
        pressure_workers.shutdown()
        self._record_memory_snapshot("pressure_complete")
        
        # Phase 3: Recovery testing
        self.logger.info("Phase 3: Recovery testing")
        
        # Force memory cleanup
        llm.optimize_memory()
        torch.cuda.empty_cache()
        gc.collect()
        
        self._record_memory_snapshot("recovery_start")
        
        # Test recovery with reduced load
        recovery_workers = ThreadPoolExecutor(max_workers=2)
        recovery_futures = [
            recovery_workers.submit(self._stress_worker, i, llm)
            for i in range(2)
        ]
        
        # Wait for recovery completion
        recovery_results = []
        for future in as_completed(recovery_futures):
            recovery_results.append(future.result())
        
        recovery_workers.shutdown()
        self._record_memory_snapshot("recovery_complete")
        
        # Phase 4: Long-running stability test
        self.logger.info("Phase 4: Long-running stability test")
        stability_start = time.time()
        
        while time.time() - stability_start < 120:  # 2 minutes stability test
            try:
                command = np.random.choice(self.test_commands)
                response = llm.generate_safe_response(command, "stability test")
                self.success_count += 1
                
                # Record memory every 10 seconds
                if int(time.time() - stability_start) % 10 == 0:
                    self._record_memory_snapshot("stability_test")
                
                time.sleep(0.1)
                
            except Exception as e:
                self.error_count += 1
                self.logger.warning(f"Stability test error: {e}")
        
        self._record_memory_snapshot("stability_complete")
        
        # Stop test
        self.test_running = False
        
        # Final cleanup
        llm.optimize_memory()
        torch.cuda.empty_cache()
        gc.collect()
        
        self._record_memory_snapshot("final")
        
        # Analyze results
        return self._analyze_test_results(baseline_results, pressure_results, recovery_results)
    
    def _analyze_test_results(self, baseline_results: List[Dict], 
                             pressure_results: List[Dict], 
                             recovery_results: List[Dict]) -> Dict[str, Any]:
        """Analyze test results for memory leaks and performance issues"""
        
        # Calculate memory growth
        initial_memory = self.memory_history[0]['memory_stats']
        final_memory = self.memory_history[-1]['memory_stats']
        
        gpu_memory_growth = final_memory.get('gpu_allocated_mb', 0) - initial_memory.get('gpu_allocated_mb', 0)
        system_memory_growth = final_memory['system_rss_mb'] - initial_memory['system_rss_mb']
        
        # Calculate performance metrics
        total_requests = sum(r['requests_processed'] for r in baseline_results + pressure_results + recovery_results)
        total_errors = sum(r['errors'] for r in baseline_results + pressure_results + recovery_results)
        total_time = sum(r['total_time'] for r in baseline_results + pressure_results + recovery_results)
        
        avg_response_time = total_time / max(total_requests, 1)
        error_rate = total_errors / max(total_requests + total_errors, 1)
        
        # Detect memory leaks
        memory_leak_detected = False
        memory_leak_severity = "none"
        
        if gpu_memory_growth > 100:  # 100MB growth threshold
            memory_leak_detected = True
            memory_leak_severity = "severe" if gpu_memory_growth > 500 else "moderate"
        elif gpu_memory_growth > 50:
            memory_leak_severity = "minor"
        
        if system_memory_growth > 200:  # 200MB system memory growth
            memory_leak_detected = True
            memory_leak_severity = "severe" if system_memory_growth > 1000 else "moderate"
        
        # Analyze memory patterns
        memory_patterns = self._analyze_memory_patterns()
        
        return {
            'test_summary': {
                'duration_seconds': time.time() - self.start_time,
                'total_requests': total_requests,
                'total_errors': total_errors,
                'error_rate': error_rate,
                'avg_response_time': avg_response_time
            },
            'memory_analysis': {
                'gpu_memory_growth_mb': gpu_memory_growth,
                'system_memory_growth_mb': system_memory_growth,
                'memory_leak_detected': memory_leak_detected,
                'memory_leak_severity': memory_leak_severity,
                'final_gpu_memory_mb': final_memory.get('gpu_allocated_mb', 0),
                'final_system_memory_mb': final_memory['system_rss_mb'],
                'gpu_fragmentation': final_memory.get('gpu_memory_fragmentation', 0)
            },
            'performance_analysis': {
                'baseline_avg_time': np.mean([r['total_time'] / max(r['requests_processed'], 1) for r in baseline_results]),
                'pressure_avg_time': np.mean([r['total_time'] / max(r['requests_processed'], 1) for r in pressure_results]),
                'recovery_avg_time': np.mean([r['total_time'] / max(r['requests_processed'], 1) for r in recovery_results]),
                'performance_degradation': self._calculate_performance_degradation()
            },
            'memory_patterns': memory_patterns,
            'recommendations': self._generate_recommendations(memory_leak_detected, error_rate, avg_response_time)
        }
    
    def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns over time"""
        if len(self.memory_history) < 2:
            return {}
        
        gpu_memory = [snapshot['memory_stats'].get('gpu_allocated_mb', 0) for snapshot in self.memory_history]
        system_memory = [snapshot['memory_stats']['system_rss_mb'] for snapshot in self.memory_history]
        timestamps = [snapshot['timestamp'] - self.start_time for snapshot in self.memory_history]
        
        # Calculate trends
        gpu_trend = np.polyfit(timestamps, gpu_memory, 1)[0]  # Linear trend slope
        system_trend = np.polyfit(timestamps, system_memory, 1)[0]
        
        # Calculate volatility
        gpu_volatility = np.std(gpu_memory)
        system_volatility = np.std(system_memory)
        
        return {
            'gpu_memory_trend_mb_per_second': gpu_trend,
            'system_memory_trend_mb_per_second': system_trend,
            'gpu_memory_volatility': gpu_volatility,
            'system_memory_volatility': system_volatility,
            'peak_gpu_memory_mb': max(gpu_memory),
            'peak_system_memory_mb': max(system_memory)
        }
    
    def _calculate_performance_degradation(self) -> float:
        """Calculate performance degradation over time"""
        if len(self.memory_history) < 10:
            return 0.0
        
        # Use memory usage as proxy for performance degradation
        gpu_memory = [snapshot['memory_stats'].get('gpu_allocated_mb', 0) for snapshot in self.memory_history]
        
        # Calculate if memory usage is consistently increasing
        first_half = gpu_memory[:len(gpu_memory)//2]
        second_half = gpu_memory[len(gpu_memory)//2:]
        
        if not first_half or not second_half:
            return 0.0
        
        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)
        
        if avg_first == 0:
            return 0.0
        
        return (avg_second - avg_first) / avg_first * 100  # Percentage degradation
    
    def _generate_recommendations(self, memory_leak_detected: bool, 
                                 error_rate: float, avg_response_time: float) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if memory_leak_detected:
            recommendations.append("CRITICAL: Memory leak detected - review tensor allocation/deallocation")
            recommendations.append("Implement more aggressive memory cleanup in exception handlers")
            recommendations.append("Consider reducing batch size or model complexity")
        
        if error_rate > 0.1:  # 10% error rate
            recommendations.append("HIGH: Error rate too high - investigate exception handling")
            recommendations.append("Add more robust input validation and error recovery")
        
        if avg_response_time > 1.0:  # 1 second average
            recommendations.append("MEDIUM: Response time too slow - optimize model inference")
            recommendations.append("Consider model quantization or caching strategies")
        
        if not recommendations:
            recommendations.append("GOOD: No critical issues detected")
            recommendations.append("Consider running longer tests for production validation")
        
        return recommendations


def main():
    """Main function for running memory leak stress tests"""
    parser = argparse.ArgumentParser(description="Memory Leak Stress Test")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--output", type=str, default="stress_test_results.json", help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Memory Leak Stress Test")
    
    try:
        # Initialize GPU-optimized LLM
        gpu_config = GPUConfig(
            max_memory_mb=2048,  # Conservative memory limit for testing
            batch_size=2,
            enable_mixed_precision=True,
            enable_memory_pooling=True
        )
        
        llm = GPUOptimizedSafetyLLM(
            model_name="microsoft/DialoGPT-medium",
            gpu_config=gpu_config
        )
        
        # Initialize memory leak detector
        detector = MemoryLeakDetector(
            test_duration=args.duration,
            max_workers=args.workers
        )
        
        # Run comprehensive stress test
        results = detector.run_memory_leak_test(llm)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        logger.info("=== STRESS TEST RESULTS ===")
        logger.info(f"Duration: {results['test_summary']['duration_seconds']:.1f}s")
        logger.info(f"Total Requests: {results['test_summary']['total_requests']}")
        logger.info(f"Error Rate: {results['test_summary']['error_rate']:.2%}")
        logger.info(f"GPU Memory Growth: {results['memory_analysis']['gpu_memory_growth_mb']:.1f}MB")
        logger.info(f"Memory Leak: {results['memory_analysis']['memory_leak_detected']}")
        logger.info(f"Leak Severity: {results['memory_analysis']['memory_leak_severity']}")
        
        logger.info("=== RECOMMENDATIONS ===")
        for rec in results['recommendations']:
            logger.info(f"- {rec}")
        
        # Cleanup
        llm.shutdown()
        
        logger.info(f"Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 