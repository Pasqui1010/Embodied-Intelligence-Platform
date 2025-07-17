#!/usr/bin/env python3
"""
Advanced Memory Manager for GPU Operations

This module provides advanced memory management for GPU-optimized operations
including memory allocation, optimization, and monitoring.
"""

import torch
import psutil
import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import threading
from collections import deque
from .gpu_memory_pool import GPUMemoryPool


@dataclass
class MemoryMetrics:
    """Memory usage metrics"""
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_percent: float
    timestamp: float


class AdvancedMemoryManager:
    """
    Advanced memory manager for GPU operations with optimization and monitoring
    """
    
    def __init__(self, max_memory_mb: int = 8192, device: str = "auto"):
        """
        Initialize memory manager
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            device: Device to monitor ("auto", "cuda", "cpu")
        """
        self.max_memory_mb = max_memory_mb
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize GPU memory pool if available
        self.gpu_pool = None
        if self.device == "cuda":
            self.gpu_pool = GPUMemoryPool(device=self.device, max_memory_mb=max_memory_mb)
        
        # Memory tracking
        self.memory_history = deque(maxlen=1000)
        self.allocation_history = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Performance tracking
        self.optimization_count = 0
        self.last_optimization = 0.0
        self.optimization_interval = 60.0  # seconds
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize device-specific settings
        self._initialize_device()
    
    def _initialize_device(self):
        """Initialize device-specific memory settings"""
        if self.device == "cuda":
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Enable memory pool
            torch.cuda.empty_cache()
            
            self.logger.info(f"GPU memory manager initialized with {self.max_memory_mb}MB limit")
        else:
            self.logger.info("CPU memory manager initialized")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        with self.lock:
            if self.device == "cuda":
                return self._get_gpu_memory_usage()
            else:
                return self._get_cpu_memory_usage()
    
    def _get_gpu_memory_usage(self) -> Dict[str, Any]:
        """Get GPU memory usage"""
        try:
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)   # MB
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
            free = total - reserved
            
            metrics = MemoryMetrics(
                allocated_mb=allocated,
                reserved_mb=reserved,
                free_mb=free,
                total_mb=total,
                utilization_percent=(reserved / total) * 100,
                timestamp=time.time()
            )
            
            # Store in history
            self.memory_history.append(metrics)
            
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'free_mb': free,
                'total_mb': total,
                'utilization_percent': metrics.utilization_percent,
                'device': 'cuda'
            }
            
        except Exception as e:
            self.logger.warning(f"Could not get GPU memory usage: {e}")
            return {
                'allocated_mb': 0.0,
                'reserved_mb': 0.0,
                'free_mb': 0.0,
                'total_mb': 0.0,
                'utilization_percent': 0.0,
                'device': 'cuda',
                'error': str(e)
            }
    
    def _get_cpu_memory_usage(self) -> Dict[str, Any]:
        """Get CPU memory usage"""
        try:
            memory = psutil.virtual_memory()
            
            metrics = MemoryMetrics(
                allocated_mb=memory.used / (1024 * 1024),
                reserved_mb=memory.used / (1024 * 1024),
                free_mb=memory.available / (1024 * 1024),
                total_mb=memory.total / (1024 * 1024),
                utilization_percent=memory.percent,
                timestamp=time.time()
            )
            
            # Store in history
            self.memory_history.append(metrics)
            
            return {
                'allocated_mb': metrics.allocated_mb,
                'reserved_mb': metrics.reserved_mb,
                'free_mb': metrics.free_mb,
                'total_mb': metrics.total_mb,
                'utilization_percent': metrics.utilization_percent,
                'device': 'cpu'
            }
            
        except Exception as e:
            self.logger.warning(f"Could not get CPU memory usage: {e}")
            return {
                'allocated_mb': 0.0,
                'reserved_mb': 0.0,
                'free_mb': 0.0,
                'total_mb': 0.0,
                'utilization_percent': 0.0,
                'device': 'cpu',
                'error': str(e)
            }
    
    def check_memory_before_processing(self):
        """Check memory before processing and optimize if needed"""
        with self.lock:
            current_usage = self.get_memory_usage()
            
            # Check if memory usage is high
            if current_usage['utilization_percent'] > 80:
                self.logger.warning(f"High memory usage: {current_usage['utilization_percent']:.1f}%")
                self.optimize_memory()
            
            # Check if we need periodic optimization
            if time.time() - self.last_optimization > self.optimization_interval:
                self.optimize_memory()
    
    def optimize_after_processing(self):
        """Optimize memory after processing"""
        with self.lock:
            # Light optimization after each processing
            if self.device == "cuda":
                torch.cuda.empty_cache()
    
    def optimize_memory(self):
        """Perform comprehensive memory optimization"""
        with self.lock:
            self.logger.info("Starting memory optimization...")
            
            if self.device == "cuda":
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Synchronize GPU
                torch.cuda.synchronize()
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Update optimization tracking
            self.optimization_count += 1
            self.last_optimization = time.time()
            
            # Log optimization results
            after_usage = self.get_memory_usage()
            self.logger.info(f"Memory optimization completed. Usage: {after_usage['utilization_percent']:.1f}%")
    
    def get_memory_trends(self) -> Dict[str, Any]:
        """Get memory usage trends"""
        with self.lock:
            if len(self.memory_history) < 2:
                return {'trend': 'insufficient_data'}
            
            recent = list(self.memory_history)[-10:]  # Last 10 measurements
            
            # Calculate trend
            if len(recent) >= 2:
                first_usage = recent[0].utilization_percent
                last_usage = recent[-1].utilization_percent
                trend = "increasing" if last_usage > first_usage else "decreasing" if last_usage < first_usage else "stable"
            else:
                trend = "stable"
            
            return {
                'trend': trend,
                'recent_usage': [m.utilization_percent for m in recent],
                'optimization_count': self.optimization_count,
                'last_optimization': self.last_optimization
            }
    
    def cleanup(self):
        """Cleanup memory manager resources"""
        with self.lock:
            self.logger.info("Cleaning up memory manager...")
            
            if self.device == "cuda" and self.gpu_pool:
                self.gpu_pool.cleanup()
                torch.cuda.empty_cache()
            
            # Clear history
            self.memory_history.clear()
            self.allocation_history.clear()
            
            self.logger.info("Memory manager cleanup completed")
    
    def _record_allocation(self, tensor: torch.Tensor):
        """Record tensor allocation in history"""
        with self.lock:
            self.allocation_history.append({
                'tensor_id': id(tensor),
                'size_mb': tensor.element_size() * tensor.numel() / (1024 * 1024),
                'timestamp': time.time()
            })
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            return psutil.Process().memory_info().rss / (1024 * 1024)
    
    def allocate_tensor(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """
        Allocate tensor with memory optimization
        
        Args:
            shape: Tensor shape
            dtype: Data type
            
        Returns:
            torch.Tensor
        """
        with self.lock:
            if self.device == "cuda" and self.gpu_pool:
                tensor = self.gpu_pool.allocate(shape, dtype)
                self._record_allocation(tensor)
                return tensor
            else:
                tensor = torch.zeros(shape, dtype=dtype, device=self.device)
                self._record_allocation(tensor)
                return tensor
    
    def release_tensor(self, tensor: torch.Tensor):
        """
        Release tensor memory
        
        Args:
            tensor: Tensor to release
        """
        with self.lock:
            if self.device == "cuda" and self.gpu_pool:
                self.gpu_pool.release(tensor)
            else:
                tensor.zero_()
                del tensor
                torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics
        
        Returns:
            Dict with memory statistics
        """
        with self.lock:
            stats = {
                'current_mb': self._get_current_memory_usage(),
                'max_mb': self.max_memory_mb,
                'optimization_count': self.optimization_count,
                'last_optimization': self.last_optimization,
                'allocation_count': len(self.allocation_history),
                'peak_memory_mb': self._get_peak_memory_usage()
            }
            
            if self.device == "cuda" and self.gpu_pool:
                stats.update(self._get_gpu_memory_usage())
                stats.update(self.gpu_pool.get_memory_stats())
            
            return stats
    
    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage in MB"""
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            return psutil.Process().memory_info().peak_rss / (1024 * 1024)