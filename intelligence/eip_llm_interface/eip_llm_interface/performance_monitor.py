#!/usr/bin/env python3
"""
Performance Monitor for GPU-Optimized Safety-Embedded LLM

This module provides real-time performance monitoring, benchmarking,
and alerting for the GPU-optimized LLM system.
"""

import time
import threading
import logging
import json
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import psutil
import torch

from .gpu_optimized_llm import GPUOptimizedSafetyLLM


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: float
    request_id: str
    processing_time: float
    gpu_memory_used: float
    cpu_memory_used: float
    gpu_utilization: float
    cpu_utilization: float
    device: str
    batch_size: int
    safety_score: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    timestamp: float
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metrics: Dict[str, Any]
    threshold: float
    current_value: float


class PerformanceMonitor:
    """Real-time performance monitoring and alerting system"""
    
    def __init__(self, llm: GPUOptimizedSafetyLLM, alert_callbacks: Optional[List[Callable]] = None):
        """
        Initialize performance monitor
        
        Args:
            llm: GPU-optimized LLM instance to monitor
            alert_callbacks: List of callback functions for alerts
        """
        self.llm = llm
        self.alert_callbacks = alert_callbacks or []
        
        # Performance thresholds
        self.thresholds = {
            'max_processing_time': 1.0,  # seconds
            'max_gpu_memory': 0.9,  # 90% of available memory
            'max_cpu_memory': 0.8,  # 80% of available memory
            'min_safety_score': 0.7,  # minimum safety score
            'max_error_rate': 0.05,  # 5% error rate
            'max_latency_p95': 0.5,  # 95th percentile latency
        }
        
        # Metrics storage
        self.metrics_history: List[PerformanceMetrics] = []
        self.alerts_history: List[PerformanceAlert] = []
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check for performance issues
                self._check_performance_alerts()
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                # Sleep for monitoring interval
                time.sleep(1.0)  # 1 second monitoring interval
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # Get LLM performance metrics
            llm_metrics = self.llm.get_performance_metrics()
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Get GPU metrics if available
            gpu_utilization = 0.0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                gpu_utilization = gpu_memory if not torch.isnan(gpu_memory) else 0.0
            
            # Store current system state
            self.current_system_state = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'gpu_utilization': gpu_utilization,
                'llm_metrics': llm_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _check_performance_alerts(self):
        """Check for performance issues and generate alerts"""
        if not hasattr(self, 'current_system_state'):
            return
        
        state = self.current_system_state
        
        # Check processing time
        if self.llm.performance_metrics['average_gpu_time'] > self.thresholds['max_processing_time']:
            self._create_alert(
                'high_processing_time',
                'high',
                f"GPU processing time ({self.llm.performance_metrics['average_gpu_time']:.3f}s) exceeds threshold ({self.thresholds['max_processing_time']}s)",
                {'average_gpu_time': self.llm.performance_metrics['average_gpu_time']},
                self.thresholds['max_processing_time'],
                self.llm.performance_metrics['average_gpu_time']
            )
        
        # Check GPU memory usage
        if state['gpu_utilization'] > self.thresholds['max_gpu_memory']:
            self._create_alert(
                'high_gpu_memory',
                'critical',
                f"GPU memory usage ({state['gpu_utilization']:.1%}) exceeds threshold ({self.thresholds['max_gpu_memory']:.1%})",
                {'gpu_utilization': state['gpu_utilization']},
                self.thresholds['max_gpu_memory'],
                state['gpu_utilization']
            )
        
        # Check CPU memory usage
        if state['memory_percent'] > self.thresholds['max_cpu_memory'] * 100:
            self._create_alert(
                'high_cpu_memory',
                'medium',
                f"CPU memory usage ({state['memory_percent']:.1f}%) exceeds threshold ({self.thresholds['max_cpu_memory'] * 100:.1f}%)",
                {'memory_percent': state['memory_percent']},
                self.thresholds['max_cpu_memory'] * 100,
                state['memory_percent']
            )
        
        # Check error rate
        if self.request_count > 0:
            error_rate = self.error_count / self.request_count
            if error_rate > self.thresholds['max_error_rate']:
                self._create_alert(
                    'high_error_rate',
                    'high',
                    f"Error rate ({error_rate:.1%}) exceeds threshold ({self.thresholds['max_error_rate']:.1%})",
                    {'error_rate': error_rate, 'total_requests': self.request_count, 'errors': self.error_count},
                    self.thresholds['max_error_rate'],
                    error_rate
                )
    
    def _create_alert(self, alert_type: str, severity: str, message: str, metrics: Dict[str, Any], threshold: float, current_value: float):
        """Create and dispatch a performance alert"""
        alert = PerformanceAlert(
            timestamp=time.time(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            metrics=metrics,
            threshold=threshold,
            current_value=current_value
        )
        
        # Store alert
        self.alerts_history.append(alert)
        
        # Log alert
        self.logger.warning(f"Performance Alert [{severity.upper()}]: {message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory bloat"""
        cutoff_time = time.time() - 3600  # Keep last hour
        self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        self.alerts_history = [a for a in self.alerts_history if a.timestamp > cutoff_time]
    
    def record_request(self, request_id: str, processing_time: float, safety_score: float, success: bool, error_message: Optional[str] = None):
        """Record a request for performance tracking"""
        self.request_count += 1
        if not success:
            self.error_count += 1
        
        self.total_processing_time += processing_time
        
        # Get current system state
        state = getattr(self, 'current_system_state', {})
        
        # Create metrics
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            request_id=request_id,
            processing_time=processing_time,
            gpu_memory_used=state.get('llm_metrics', {}).get('current_memory_usage', {}).get('allocated_mb', 0),
            cpu_memory_used=psutil.virtual_memory().percent,
            gpu_utilization=state.get('gpu_utilization', 0.0),
            cpu_utilization=state.get('cpu_percent', 0.0),
            device=self.llm.device,
            batch_size=self.llm.gpu_config.batch_size,
            safety_score=safety_score,
            success=success,
            error_message=error_message
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.metrics_history:
            return {'message': 'No metrics available'}
        
        # Calculate statistics
        processing_times = [m.processing_time for m in self.metrics_history]
        safety_scores = [m.safety_score for m in self.metrics_history if m.success]
        
        summary = {
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'average_processing_time': statistics.mean(processing_times),
            'p95_processing_time': statistics.quantiles(processing_times, n=20)[18] if len(processing_times) > 1 else 0,
            'p99_processing_time': statistics.quantiles(processing_times, n=100)[98] if len(processing_times) > 1 else 0,
            'average_safety_score': statistics.mean(safety_scores) if safety_scores else 0,
            'device_usage': {
                'gpu_requests': self.llm.performance_metrics['gpu_requests'],
                'cpu_fallback_requests': self.llm.performance_metrics['cpu_fallback_requests'],
                'gpu_utilization': self.llm.performance_metrics.get('current_memory_usage', {}).get('allocated_mb', 0)
            },
            'recent_alerts': len([a for a in self.alerts_history if a.timestamp > time.time() - 300]),  # Last 5 minutes
            'monitoring_active': self.monitoring_active
        }
        
        return summary
    
    def export_metrics(self, filename: str):
        """Export metrics to JSON file"""
        try:
            data = {
                'export_timestamp': time.time(),
                'metrics': [asdict(m) for m in self.metrics_history],
                'alerts': [asdict(a) for a in self.alerts_history],
                'summary': self.get_performance_summary()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
    
    def set_threshold(self, threshold_name: str, value: float):
        """Set a performance threshold"""
        if threshold_name in self.thresholds:
            self.thresholds[threshold_name] = value
            self.logger.info(f"Updated threshold {threshold_name} to {value}")
        else:
            self.logger.warning(f"Unknown threshold: {threshold_name}")
    
    def add_alert_callback(self, callback: Callable):
        """Add an alert callback function"""
        self.alert_callbacks.append(callback)
        self.logger.info("Added alert callback")


class PerformanceBenchmark:
    """Performance benchmarking for GPU-optimized LLM"""
    
    def __init__(self, llm: GPUOptimizedSafetyLLM, monitor: PerformanceMonitor):
        """
        Initialize performance benchmark
        
        Args:
            llm: GPU-optimized LLM instance
            monitor: Performance monitor instance
        """
        self.llm = llm
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
    
    def run_benchmark(self, num_requests: int = 100, commands: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run performance benchmark
        
        Args:
            num_requests: Number of requests to test
            commands: List of test commands (if None, uses default commands)
        
        Returns:
            Benchmark results
        """
        if commands is None:
            commands = [
                "move to the kitchen",
                "avoid obstacles while navigating",
                "stop if you detect a human nearby",
                "maintain safe velocity limits",
                "check workspace boundaries before moving"
            ]
        
        self.logger.info(f"Starting benchmark with {num_requests} requests")
        
        # Clear previous metrics
        self.monitor.metrics_history.clear()
        self.monitor.alerts_history.clear()
        
        # Run benchmark
        start_time = time.time()
        results = []
        
        for i in range(num_requests):
            command = commands[i % len(commands)]
            request_id = f"benchmark_{i}"
            
            try:
                # Generate response
                response = self.llm.generate_safe_response(command)
                
                # Record metrics
                self.monitor.record_request(
                    request_id=request_id,
                    processing_time=response.execution_time,
                    safety_score=response.safety_score,
                    success=True
                )
                
                results.append({
                    'request_id': request_id,
                    'command': command,
                    'processing_time': response.execution_time,
                    'safety_score': response.safety_score,
                    'success': True
                })
                
            except Exception as e:
                self.monitor.record_request(
                    request_id=request_id,
                    processing_time=0.0,
                    safety_score=0.0,
                    success=False,
                    error_message=str(e)
                )
                
                results.append({
                    'request_id': request_id,
                    'command': command,
                    'processing_time': 0.0,
                    'safety_score': 0.0,
                    'success': False,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        
        # Get performance summary
        summary = self.monitor.get_performance_summary()
        
        # Calculate benchmark metrics
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        benchmark_results = {
            'total_requests': num_requests,
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / num_requests,
            'total_time': total_time,
            'requests_per_second': num_requests / total_time,
            'average_processing_time': summary['average_processing_time'],
            'p95_processing_time': summary['p95_processing_time'],
            'p99_processing_time': summary['p99_processing_time'],
            'average_safety_score': summary['average_safety_score'],
            'device_usage': summary['device_usage'],
            'alerts_generated': len(self.monitor.alerts_history),
            'detailed_results': results
        }
        
        self.logger.info(f"Benchmark completed: {benchmark_results['requests_per_second']:.2f} req/s")
        
        return benchmark_results
    
    def export_benchmark_results(self, results: Dict[str, Any], filename: str):
        """Export benchmark results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Benchmark results exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to export benchmark results: {e}")


def create_performance_monitor(llm: GPUOptimizedSafetyLLM) -> PerformanceMonitor:
    """Factory function to create a performance monitor with default alert callbacks"""
    
    def log_alert(alert: PerformanceAlert):
        """Default alert logging callback"""
        logging.warning(f"PERFORMANCE ALERT [{alert.severity.upper()}]: {alert.message}")
    
    def memory_optimization_alert(alert: PerformanceAlert):
        """Memory optimization alert callback"""
        if alert.alert_type in ['high_gpu_memory', 'high_cpu_memory']:
            llm.optimize_memory()
            logging.info("Memory optimization triggered by alert")
    
    monitor = PerformanceMonitor(llm, alert_callbacks=[log_alert, memory_optimization_alert])
    return monitor 