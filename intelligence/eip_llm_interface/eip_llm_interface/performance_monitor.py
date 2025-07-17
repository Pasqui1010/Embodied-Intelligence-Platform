#!/usr/bin/env python3
"""
Performance Monitor for GPU-Optimized LLM

This module provides real-time performance monitoring and benchmarking
for the GPU-optimized Safety-Embedded LLM.
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque
import statistics


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single request"""
    request_id: str
    processing_time: float
    safety_score: float
    success: bool
    timestamp: float
    error: Optional[str] = None


@dataclass
class BatchMetrics:
    """Performance metrics for batch processing"""
    batch_size: int
    total_time: float
    success: bool
    timestamp: float
    dynamic_batch_size: int
    parallel_config: Dict[str, Any]
    error: Optional[str] = None


class PerformanceMonitor:
    """
    Real-time performance monitor for GPU-optimized LLM
    """
    
    def __init__(self, llm_instance, monitoring_interval: float = 1.0):
        """
        Initialize performance monitor
        
        Args:
            llm_instance: Reference to the LLM instance being monitored
            monitoring_interval: Monitoring interval in seconds
        """
        self.llm_instance = llm_instance
        self.monitoring_interval = monitoring_interval
        
        # Performance tracking
        self.request_metrics: deque = deque(maxlen=10000)
        self.batch_metrics: deque = deque(maxlen=1000)
        
        # Real-time metrics
        self.current_requests_per_second = 0.0
        self.current_average_processing_time = 0.0
        self.current_success_rate = 1.0
        
        # Alert thresholds
        self.thresholds = {
            'max_processing_time': 5.0,  # seconds
            'min_success_rate': 0.95,    # 95%
            'max_memory_usage': 0.9,     # 90%
            'min_requests_per_second': 1.0
        }
        
        # Monitoring state
        self.monitoring_active = True
        self.monitoring_thread = None
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start the monitoring thread"""
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Performance monitoring started")
    
    def _monitoring_worker(self):
        """Monitoring worker thread"""
        while self.monitoring_active:
            try:
                self._update_real_time_metrics()
                self._check_alerts()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Monitoring worker error: {e}")
    
    def _update_real_time_metrics(self):
        """Update real-time performance metrics"""
        with self.lock:
            # Calculate requests per second (last 10 seconds)
            current_time = time.time()
            recent_requests = [
                m for m in self.request_metrics
                if current_time - m.timestamp <= 10.0
            ]
            
            if recent_requests:
                self.current_requests_per_second = len(recent_requests) / 10.0
                self.current_average_processing_time = statistics.mean(
                    [m.processing_time for m in recent_requests]
                )
                self.current_success_rate = sum(1 for m in recent_requests if m.success) / len(recent_requests)
            else:
                self.current_requests_per_second = 0.0
                self.current_average_processing_time = 0.0
                self.current_success_rate = 1.0
    
    def _check_alerts(self):
        """Check for performance alerts"""
        # Check processing time
        if self.current_average_processing_time > self.thresholds['max_processing_time']:
            self.logger.warning(
                f"High processing time: {self.current_average_processing_time:.3f}s "
                f"(threshold: {self.thresholds['max_processing_time']}s)"
            )
        
        # Check success rate
        if self.current_success_rate < self.thresholds['min_success_rate']:
            self.logger.warning(
                f"Low success rate: {self.current_success_rate:.1%} "
                f"(threshold: {self.thresholds['min_success_rate']:.1%})"
            )
        
        # Check requests per second
        if self.current_requests_per_second < self.thresholds['min_requests_per_second']:
            self.logger.warning(
                f"Low throughput: {self.current_requests_per_second:.2f} RPS "
                f"(threshold: {self.thresholds['min_requests_per_second']} RPS)"
            )
        
        # Check memory usage
        try:
            memory_usage = self.llm_instance.get_memory_usage()
            if memory_usage['utilization_percent'] > self.thresholds['max_memory_usage'] * 100:
                self.logger.warning(
                    f"High memory usage: {memory_usage['utilization_percent']:.1f}% "
                    f"(threshold: {self.thresholds['max_memory_usage'] * 100:.1f}%)"
                )
        except Exception as e:
            self.logger.warning(f"Could not check memory usage: {e}")
    
    def record_request(self, request_id: str, processing_time: float, safety_score: float, 
                      success: bool, error: Optional[str] = None):
        """
        Record performance metrics for a single request
        
        Args:
            request_id: Unique request identifier
            processing_time: Processing time in seconds
            safety_score: Safety score (0.0 to 1.0)
            success: Whether the request was successful
            error: Error message if failed
        """
        with self.lock:
            metrics = PerformanceMetrics(
                request_id=request_id,
                processing_time=processing_time,
                safety_score=safety_score,
                success=success,
                timestamp=time.time(),
                error=error
            )
            
            self.request_metrics.append(metrics)
    
    def record_batch(self, batch_size: int, total_time: float, success: bool, 
                    dynamic_batch_size: int, parallel_config: Dict[str, Any],
                    error: Optional[str] = None):
        """
        Record performance metrics for batch processing with dynamic sizing and parallel config
        
        Args:
            batch_size: Total number of requests in batch
            total_time: Total processing time in seconds
            success: Whether the batch was successful
            dynamic_batch_size: Actual batch size used
            parallel_config: Configuration for model parallelism
            error: Error message if failed
        """
        with self.lock:
            metrics = BatchMetrics(
                batch_size=batch_size,
                total_time=total_time,
                success=success,
                timestamp=time.time(),
                dynamic_batch_size=dynamic_batch_size,
                parallel_config=parallel_config,
                error=error
            )
            
            self.batch_metrics.append(metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary
        
        Returns:
            Dictionary containing performance metrics
        """
        with self.lock:
            if not self.request_metrics:
                return {
                    'total_requests': 0,
                    'success_rate': 1.0,
                    'average_processing_time': 0.0,
                    'requests_per_second': 0.0,
                    'current_metrics': {
                        'requests_per_second': self.current_requests_per_second,
                        'average_processing_time': self.current_average_processing_time,
                        'success_rate': self.current_success_rate
                    }
                }
            
            # Calculate overall metrics
            total_requests = len(self.request_metrics)
            successful_requests = sum(1 for m in self.request_metrics if m.success)
            success_rate = successful_requests / total_requests
            
            processing_times = [m.processing_time for m in self.request_metrics]
            average_processing_time = statistics.mean(processing_times)
            
            # Calculate requests per second over all time
            if len(self.request_metrics) >= 2:
                time_span = self.request_metrics[-1].timestamp - self.request_metrics[0].timestamp
                requests_per_second = total_requests / max(time_span, 1.0)
            else:
                requests_per_second = 0.0
            
            # Calculate safety score statistics
            safety_scores = [m.safety_score for m in self.request_metrics if m.success]
            average_safety_score = statistics.mean(safety_scores) if safety_scores else 0.0
            
            # Batch processing statistics
            if self.batch_metrics:
                batch_sizes = [m.batch_size for m in self.batch_metrics]
                batch_times = [m.total_time for m in self.batch_metrics]
                average_batch_size = statistics.mean(batch_sizes)
                average_batch_time = statistics.mean(batch_times)
                batch_success_rate = sum(1 for m in self.batch_metrics if m.success) / len(self.batch_metrics)
                
                # Calculate parallelism statistics
                parallel_configs = [m.parallel_config for m in self.batch_metrics]
                avg_num_splits = statistics.mean(
                    [config['num_splits'] for config in parallel_configs if config['enabled']]
                )
                parallel_usage_rate = sum(
                    1 for config in parallel_configs if config['enabled']
                ) / len(parallel_configs)
                
                avg_dynamic_batch_size = statistics.mean(
                    [m.dynamic_batch_size for m in self.batch_metrics]
                )
            else:
                average_batch_size = 0
                average_batch_time = 0.0
                batch_success_rate = 1.0
                avg_num_splits = 0.0
                parallel_usage_rate = 0.0
                avg_dynamic_batch_size = 0.0
            
            return {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'success_rate': success_rate,
                'average_processing_time': average_processing_time,
                'requests_per_second': requests_per_second,
                'average_safety_score': average_safety_score,
                'batch_processing': {
                    'total_batches': len(self.batch_metrics),
                    'average_batch_size': average_batch_size,
                    'average_batch_time': average_batch_time,
                    'batch_success_rate': batch_success_rate,
                    'average_dynamic_batch_size': avg_dynamic_batch_size,
                    'parallel_usage_rate': parallel_usage_rate,
                    'average_num_splits': avg_num_splits
                },
                'current_metrics': {
                    'requests_per_second': self.current_requests_per_second,
                    'average_processing_time': self.current_average_processing_time,
                    'success_rate': self.current_success_rate
                },
                'percentiles': {
                    'p50_processing_time': statistics.quantiles(processing_times, n=2)[0] if len(processing_times) > 1 else 0.0,
                    'p95_processing_time': statistics.quantiles(processing_times, n=20)[18] if len(processing_times) > 1 else 0.0,
                    'p99_processing_time': statistics.quantiles(processing_times, n=100)[98] if len(processing_times) > 1 else 0.0
                },
                'alerts': self._check_alerts()
            }
    
    def get_recent_performance(self, time_window: float = 60.0) -> Dict[str, Any]:
        """
        Get performance metrics for recent requests
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Performance metrics for the specified time window
        """
        with self.lock:
            current_time = time.time()
            recent_metrics = [
                m for m in self.request_metrics
                if current_time - m.timestamp <= time_window
            ]
            
            if not recent_metrics:
                return {
                    'requests_in_window': 0,
                    'success_rate': 1.0,
                    'average_processing_time': 0.0,
                    'requests_per_second': 0.0
                }
            
            requests_in_window = len(recent_metrics)
            successful_requests = sum(1 for m in recent_metrics if m.success)
            success_rate = successful_requests / requests_in_window
            
            processing_times = [m.processing_time for m in recent_metrics]
            average_processing_time = statistics.mean(processing_times)
            requests_per_second = requests_in_window / time_window
            
            return {
                'requests_in_window': requests_in_window,
                'success_rate': success_rate,
                'average_processing_time': average_processing_time,
                'requests_per_second': requests_per_second,
                'time_window': time_window
            }
    
    def set_thresholds(self, thresholds: Dict[str, float]):
        """
        Update alert thresholds
        
        Args:
            thresholds: Dictionary of threshold values
        """
        with self.lock:
            self.thresholds.update(thresholds)
            self.logger.info(f"Updated performance thresholds: {thresholds}")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")


class PerformanceBenchmark:
    """
    Performance benchmarking utility for GPU-optimized LLM
    """
    
    def __init__(self, llm_instance, monitor: PerformanceMonitor):
        """
        Initialize performance benchmark
        
        Args:
            llm_instance: LLM instance to benchmark
            monitor: Performance monitor instance
        """
        self.llm_instance = llm_instance
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
    
    def run_benchmark(self, num_requests: int, test_commands: List[str]) -> Dict[str, Any]:
        """
        Run comprehensive performance benchmark
        
        Args:
            num_requests: Number of requests to process
            test_commands: List of test commands
            
        Returns:
            Benchmark results
        """
        self.logger.info(f"Starting benchmark with {num_requests} requests")
        
        start_time = time.time()
        successful_requests = 0
        total_processing_time = 0.0
        
        for i in range(num_requests):
            command = test_commands[i % len(test_commands)]
            
            try:
                request_start = time.time()
                response = self.llm_instance.generate_safe_response(command)
                request_time = time.time() - request_start
                
                total_processing_time += request_time
                successful_requests += 1
                
                # Record metrics
                self.monitor.record_request(
                    request_id=f"benchmark_{i}",
                    processing_time=request_time,
                    safety_score=response.safety_score,
                    success=True
                )
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{num_requests} requests")
                
            except Exception as e:
                self.logger.error(f"Benchmark request {i} failed: {e}")
                
                # Record failed request
                self.monitor.record_request(
                    request_id=f"benchmark_{i}",
                    processing_time=time.time() - request_start if 'request_start' in locals() else 0.0,
                    safety_score=0.0,
                    success=False,
                    error=str(e)
                )
        
        total_time = time.time() - start_time
        success_rate = successful_requests / num_requests
        average_processing_time = total_processing_time / successful_requests if successful_requests > 0 else 0.0
        requests_per_second = successful_requests / total_time
        
        results = {
            'total_requests': num_requests,
            'successful_requests': successful_requests,
            'success_rate': success_rate,
            'total_time': total_time,
            'average_processing_time': average_processing_time,
            'requests_per_second': requests_per_second,
            'total_processing_time': total_processing_time
        }
        
        self.logger.info(f"Benchmark completed: {results}")
        return results
    
    def run_batch_benchmark(self, num_batches: int, batch_size: int, 
                           test_commands: List[str]) -> Dict[str, Any]:
        """
        Run batch processing benchmark
        
        Args:
            num_batches: Number of batches to process
            batch_size: Number of requests per batch
            test_commands: List of test commands
            
        Returns:
            Batch benchmark results
        """
        self.logger.info(f"Starting batch benchmark: {num_batches} batches of {batch_size} requests")
        
        start_time = time.time()
        successful_batches = 0
        total_requests = 0
        
        for i in range(num_batches):
            batch_commands = [
                test_commands[j % len(test_commands)]
                for j in range(i * batch_size, (i + 1) * batch_size)
            ]
            
            try:
                batch_start = time.time()
                responses = self.llm_instance.generate_batch_responses(batch_commands)
                batch_time = time.time() - batch_start
                
                successful_batches += 1
                total_requests += len(responses)
                
                # Record batch metrics
                self.monitor.record_batch(
                    batch_size=len(batch_commands),
                    total_time=batch_time,
                    success=True
                )
                
                self.logger.info(f"Processed batch {i + 1}/{num_batches} in {batch_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"Batch {i} failed: {e}")
                
                # Record failed batch
                self.monitor.record_batch(
                    batch_size=batch_size,
                    total_time=time.time() - batch_start if 'batch_start' in locals() else 0.0,
                    success=False,
                    error=str(e)
                )
        
        total_time = time.time() - start_time
        batch_success_rate = successful_batches / num_batches
        average_batch_time = total_time / successful_batches if successful_batches > 0 else 0.0
        requests_per_second = total_requests / total_time
        
        results = {
            'num_batches': num_batches,
            'batch_size': batch_size,
            'successful_batches': successful_batches,
            'batch_success_rate': batch_success_rate,
            'total_requests': total_requests,
            'total_time': total_time,
            'average_batch_time': average_batch_time,
            'requests_per_second': requests_per_second
        }
        
        self.logger.info(f"Batch benchmark completed: {results}")
        return results 