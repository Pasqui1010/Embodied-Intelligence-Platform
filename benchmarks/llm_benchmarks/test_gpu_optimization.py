#!/usr/bin/env python3
"""
GPU Optimization Tests for Safety-Embedded LLM

Tests for GPU acceleration, memory management, batch processing,
and performance optimization features.
"""

import pytest
import time
import threading
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import json

from eip_llm_interface.gpu_optimized_llm import (
    GPUOptimizedSafetyLLM, GPUConfig, MemoryManager, 
    BatchRequest, BatchResponse
)
from eip_llm_interface.performance_monitor import (
    PerformanceMonitor, PerformanceBenchmark, 
    PerformanceMetrics, PerformanceAlert
)


class TestGPUConfig:
    """Test GPU configuration"""
    
    def test_default_config(self):
        """Test default GPU configuration"""
        config = GPUConfig()
        assert config.device == "auto"
        assert config.memory_fraction == 0.8
        assert config.batch_size == 4
        assert config.max_memory_mb == 8192
        assert config.enable_mixed_precision is True
        assert config.enable_kernel_fusion is True
        assert config.enable_memory_pooling is True
    
    def test_custom_config(self):
        """Test custom GPU configuration"""
        config = GPUConfig(
            device="cuda",
            memory_fraction=0.5,
            batch_size=8,
            max_memory_mb=4096,
            enable_mixed_precision=False
        )
        assert config.device == "cuda"
        assert config.memory_fraction == 0.5
        assert config.batch_size == 8
        assert config.max_memory_mb == 4096
        assert config.enable_mixed_precision is False


class TestMemoryManager:
    """Test GPU memory management"""
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization"""
        manager = MemoryManager(max_memory_mb=4096)
        assert manager.max_memory_mb == 4096
        assert manager.allocated_memory == 0
        assert len(manager.memory_pool) == 0
    
    def test_tensor_allocation_context_manager(self):
        """Test tensor allocation with context manager"""
        manager = MemoryManager(max_memory_mb=1024)
        
        with manager.allocate_tensor((100, 100), torch.float16) as tensor:
            assert tensor.shape == (100, 100)
            assert tensor.dtype == torch.float16
            assert tensor.device.type == 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Check that memory is cleaned up
        assert manager.allocated_memory == 0
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking"""
        manager = MemoryManager(max_memory_mb=1024)
        
        # Simulate tensor allocation
        with manager.allocate_tensor((50, 50), torch.float32) as tensor:
            initial_memory = manager.allocated_memory
            assert initial_memory > 0
        
        # Check memory is freed
        assert manager.allocated_memory == 0
    
    def test_memory_cleanup(self):
        """Test memory cleanup"""
        manager = MemoryManager(max_memory_mb=1024)
        
        # Allocate some memory
        with manager.allocate_tensor((100, 100), torch.float16):
            pass
        
        # Force cleanup
        manager.cleanup()
        assert manager.allocated_memory == 0
        assert len(manager.memory_pool) == 0


class TestGPUOptimizedSafetyLLM:
    """Test GPU-optimized Safety-Embedded LLM"""
    
    @pytest.fixture
    def gpu_config(self):
        """Create GPU configuration for testing"""
        return GPUConfig(
            device="cpu",  # Use CPU for testing
            batch_size=2,
            max_memory_mb=1024,
            enable_mixed_precision=False
        )
    
    @pytest.fixture
    def gpu_llm(self, gpu_config):
        """Create GPU-optimized LLM instance for testing"""
        with patch('torch.cuda.is_available', return_value=False):
            llm = GPUOptimizedSafetyLLM(
                model_name="microsoft/DialoGPT-medium",
                gpu_config=gpu_config
            )
            yield llm
            llm.shutdown()
    
    def test_device_determination_cpu_fallback(self):
        """Test device determination with CPU fallback"""
        with patch('torch.cuda.is_available', return_value=False):
            llm = GPUOptimizedSafetyLLM()
            assert llm.device == "cpu"
            llm.shutdown()
    
    def test_device_determination_gpu_available(self):
        """Test device determination with GPU available"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            # Mock GPU with sufficient memory
            mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
            
            llm = GPUOptimizedSafetyLLM()
            assert llm.device == "cuda"
            llm.shutdown()
    
    def test_device_determination_insufficient_gpu_memory(self):
        """Test device determination with insufficient GPU memory"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            # Mock GPU with insufficient memory
            mock_props.return_value.total_memory = 2 * 1024 * 1024 * 1024  # 2GB
            
            llm = GPUOptimizedSafetyLLM()
            assert llm.device == "cpu"
            llm.shutdown()
    
    def test_batch_request_creation(self):
        """Test batch request creation"""
        request = BatchRequest(
            id="test_001",
            command="move to kitchen",
            context="robot is in living room",
            priority=1,
            timestamp=time.time()
        )
        
        assert request.id == "test_001"
        assert request.command == "move to kitchen"
        assert request.context == "robot is in living room"
        assert request.priority == 1
        assert request.timestamp > 0
    
    def test_performance_metrics_initialization(self, gpu_llm):
        """Test performance metrics initialization"""
        metrics = gpu_llm.performance_metrics
        
        assert metrics['total_requests'] == 0
        assert metrics['gpu_requests'] == 0
        assert metrics['cpu_fallback_requests'] == 0
        assert metrics['average_gpu_time'] == 0.0
        assert metrics['average_cpu_time'] == 0.0
        assert isinstance(metrics['memory_usage_history'], list)
    
    def test_safe_response_generation(self, gpu_llm):
        """Test safe response generation"""
        response = gpu_llm.generate_safe_response("move to kitchen")
        
        assert isinstance(response.content, str)
        assert 0.0 <= response.safety_score <= 1.0
        assert isinstance(response.safety_tokens_used, list)
        assert isinstance(response.violations_detected, list)
        assert response.confidence > 0
        assert response.execution_time > 0
    
    def test_performance_metrics_update(self, gpu_llm):
        """Test performance metrics update"""
        # Generate a response
        response = gpu_llm.generate_safe_response("test command")
        
        # Check metrics are updated
        metrics = gpu_llm.performance_metrics
        assert metrics['total_requests'] > 0
        
        # Check memory usage history
        assert len(metrics['memory_usage_history']) > 0
    
    def test_memory_optimization(self, gpu_llm):
        """Test memory optimization"""
        # Generate some responses to use memory
        for i in range(5):
            gpu_llm.generate_safe_response(f"test command {i}")
        
        # Optimize memory
        gpu_llm.optimize_memory()
        
        # Check that optimization completed without errors
        assert True  # If we get here, optimization succeeded
    
    def test_get_performance_metrics(self, gpu_llm):
        """Test getting performance metrics"""
        # Generate a response
        gpu_llm.generate_safe_response("test command")
        
        # Get metrics
        metrics = gpu_llm.get_performance_metrics()
        
        assert 'total_requests' in metrics
        assert 'current_memory_usage' in metrics
        assert 'device' in metrics
        assert 'gpu_config' in metrics


class TestPerformanceMonitor:
    """Test performance monitoring"""
    
    @pytest.fixture
    def gpu_llm(self):
        """Create GPU-optimized LLM for testing"""
        with patch('torch.cuda.is_available', return_value=False):
            llm = GPUOptimizedSafetyLLM(device="cpu")
            yield llm
            llm.shutdown()
    
    @pytest.fixture
    def monitor(self, gpu_llm):
        """Create performance monitor for testing"""
        monitor = PerformanceMonitor(gpu_llm)
        yield monitor
        monitor.stop_monitoring()
    
    def test_monitor_initialization(self, gpu_llm):
        """Test performance monitor initialization"""
        monitor = PerformanceMonitor(gpu_llm)
        
        assert monitor.llm == gpu_llm
        assert monitor.alert_callbacks == []
        assert monitor.request_count == 0
        assert monitor.error_count == 0
        assert monitor.monitoring_active is True
        
        monitor.stop_monitoring()
    
    def test_threshold_configuration(self, monitor):
        """Test threshold configuration"""
        # Set custom threshold
        monitor.set_threshold('max_processing_time', 2.0)
        assert monitor.thresholds['max_processing_time'] == 2.0
        
        # Test unknown threshold
        monitor.set_threshold('unknown_threshold', 1.0)
        # Should log warning but not crash
    
    def test_request_recording(self, monitor):
        """Test request recording"""
        # Record a successful request
        monitor.record_request(
            request_id="test_001",
            processing_time=0.5,
            safety_score=0.9,
            success=True
        )
        
        assert monitor.request_count == 1
        assert monitor.error_count == 0
        assert len(monitor.metrics_history) == 1
        
        # Record a failed request
        monitor.record_request(
            request_id="test_002",
            processing_time=0.1,
            safety_score=0.0,
            success=False,
            error_message="Test error"
        )
        
        assert monitor.request_count == 2
        assert monitor.error_count == 1
        assert len(monitor.metrics_history) == 2
    
    def test_performance_summary(self, monitor):
        """Test performance summary generation"""
        # Record some requests
        for i in range(5):
            monitor.record_request(
                request_id=f"test_{i}",
                processing_time=0.1 + i * 0.1,
                safety_score=0.8 + i * 0.02,
                success=True
            )
        
        summary = monitor.get_performance_summary()
        
        assert summary['total_requests'] == 5
        assert summary['error_count'] == 0
        assert summary['error_rate'] == 0.0
        assert summary['average_processing_time'] > 0
        assert summary['monitoring_active'] is True
    
    def test_alert_callback(self, gpu_llm):
        """Test alert callback functionality"""
        alert_received = []
        
        def test_callback(alert):
            alert_received.append(alert)
        
        monitor = PerformanceMonitor(gpu_llm, alert_callbacks=[test_callback])
        
        # Trigger an alert by setting a very low threshold
        monitor.set_threshold('max_processing_time', 0.001)
        
        # Record a request that exceeds the threshold
        monitor.record_request(
            request_id="test_alert",
            processing_time=1.0,  # Exceeds 0.001 threshold
            safety_score=0.9,
            success=True
        )
        
        # Wait for monitoring loop to process
        time.sleep(2.0)
        
        # Check if alert was received
        assert len(alert_received) > 0
        assert alert_received[0].alert_type == 'high_processing_time'
        
        monitor.stop_monitoring()
    
    def test_metrics_export(self, monitor):
        """Test metrics export functionality"""
        # Record some requests
        for i in range(3):
            monitor.record_request(
                request_id=f"export_test_{i}",
                processing_time=0.1,
                safety_score=0.8,
                success=True
            )
        
        # Export metrics
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_file = f.name
        
        try:
            monitor.export_metrics(export_file)
            
            # Check file was created and contains data
            with open(export_file, 'r') as f:
                data = json.load(f)
            
            assert 'export_timestamp' in data
            assert 'metrics' in data
            assert 'alerts' in data
            assert 'summary' in data
            assert len(data['metrics']) == 3
            
        finally:
            import os
            os.unlink(export_file)


class TestPerformanceBenchmark:
    """Test performance benchmarking"""
    
    @pytest.fixture
    def gpu_llm(self):
        """Create GPU-optimized LLM for testing"""
        with patch('torch.cuda.is_available', return_value=False):
            llm = GPUOptimizedSafetyLLM(device="cpu")
            yield llm
            llm.shutdown()
    
    @pytest.fixture
    def monitor(self, gpu_llm):
        """Create performance monitor for testing"""
        monitor = PerformanceMonitor(gpu_llm)
        yield monitor
        monitor.stop_monitoring()
    
    @pytest.fixture
    def benchmark(self, gpu_llm, monitor):
        """Create performance benchmark for testing"""
        return PerformanceBenchmark(gpu_llm, monitor)
    
    def test_benchmark_initialization(self, benchmark):
        """Test benchmark initialization"""
        assert benchmark.llm is not None
        assert benchmark.monitor is not None
    
    def test_small_benchmark(self, benchmark):
        """Test small benchmark run"""
        results = benchmark.run_benchmark(num_requests=5)
        
        assert results['total_requests'] == 5
        assert results['successful_requests'] >= 0
        assert results['failed_requests'] >= 0
        assert results['success_rate'] >= 0.0
        assert results['total_time'] > 0
        assert results['requests_per_second'] > 0
        assert 'detailed_results' in results
    
    def test_benchmark_with_custom_commands(self, benchmark):
        """Test benchmark with custom commands"""
        custom_commands = [
            "move forward",
            "turn left",
            "stop immediately"
        ]
        
        results = benchmark.run_benchmark(num_requests=3, commands=custom_commands)
        
        assert results['total_requests'] == 3
        assert len(results['detailed_results']) == 3
    
    def test_benchmark_results_export(self, benchmark):
        """Test benchmark results export"""
        results = benchmark.run_benchmark(num_requests=2)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_file = f.name
        
        try:
            benchmark.export_benchmark_results(results, export_file)
            
            # Check file was created and contains data
            with open(export_file, 'r') as f:
                data = json.load(f)
            
            assert data['total_requests'] == 2
            assert 'detailed_results' in data
            
        finally:
            import os
            os.unlink(export_file)


class TestIntegration:
    """Integration tests for GPU optimization"""
    
    def test_end_to_end_gpu_optimization(self):
        """Test end-to-end GPU optimization workflow"""
        with patch('torch.cuda.is_available', return_value=False):
            # Create GPU-optimized LLM
            gpu_config = GPUConfig(device="cpu", batch_size=2)
            llm = GPUOptimizedSafetyLLM(gpu_config=gpu_config)
            
            # Create performance monitor
            monitor = PerformanceMonitor(llm)
            
            # Create benchmark
            benchmark = PerformanceBenchmark(llm, monitor)
            
            # Run benchmark
            results = benchmark.run_benchmark(num_requests=3)
            
            # Verify results
            assert results['total_requests'] == 3
            assert results['success_rate'] >= 0.0
            assert results['requests_per_second'] > 0
            
            # Get performance summary
            summary = monitor.get_performance_summary()
            assert summary['total_requests'] == 3
            
            # Cleanup
            monitor.stop_monitoring()
            llm.shutdown()
    
    def test_memory_management_integration(self):
        """Test memory management integration"""
        with patch('torch.cuda.is_available', return_value=False):
            llm = GPUOptimizedSafetyLLM()
            
            # Generate multiple responses to test memory management
            for i in range(10):
                response = llm.generate_safe_response(f"test command {i}")
                assert response.content is not None
            
            # Optimize memory
            llm.optimize_memory()
            
            # Verify system still works after optimization
            response = llm.generate_safe_response("final test")
            assert response.content is not None
            
            llm.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 