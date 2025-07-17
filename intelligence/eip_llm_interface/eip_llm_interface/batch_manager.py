from typing import List, Dict, Any, Optional
import torch
from dataclasses import dataclass
import logging
from collections import deque
import time

class BatchSizePolicy(Enum):
    """Available batch sizing policies"""
    DYNAMIC = "dynamic"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"

class BatchStats:
    """Statistics for batch processing"""
    def __init__(self):
        self.successful_batches = 0
        self.failed_batches = 0
        self.total_processing_time = 0.0
        self.total_requests = 0
        self.max_batch_size = 0
        self.min_batch_size = float('inf')
        self.avg_batch_size = 0.0
        self.batch_times = []
        self.batch_sizes = []

class BatchManager:
    """
    Manages dynamic batch sizing for GPU-optimized model inference
    """
    
    def __init__(
        self,
        max_batch_size: int = 16,
        min_batch_size: int = 1,
        base_batch_size: int = 4,
        adjustment_interval: int = 10,
        policy: BatchSizePolicy = BatchSizePolicy.ADAPTIVE,
        memory_threshold: float = 0.8,
        performance_threshold: float = 0.9
    ):
        """
        Initialize the batch manager
        
        Args:
            max_batch_size: Maximum batch size
            min_batch_size: Minimum batch size
            base_batch_size: Starting batch size
            adjustment_interval: How often to adjust batch size
            policy: Batch sizing policy
            memory_threshold: Memory usage threshold for adjustments
            performance_threshold: Performance threshold for adjustments
        """
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.base_batch_size = base_batch_size
        self.current_batch_size = base_batch_size
        self.adjustment_interval = adjustment_interval
        self.policy = policy
        self.memory_threshold = memory_threshold
        self.performance_threshold = performance_threshold
        
        self.batch_queue = deque(maxlen=max_batch_size * 2)
        self.batch_lock = threading.Lock()
        self.stats = BatchStats()
        self.last_adjustment_time = time.time()
        
        self.logger = logging.getLogger(__name__)
        
    def add_request(self, request: Any) -> None:
        """
        Add a request to the batch queue
        
        Args:
            request: The processing request
        """
        with self.batch_lock:
            self.batch_queue.append(request)
            self.stats.total_requests += 1
            
            # Check if we need to process batch
            if len(self.batch_queue) >= self.current_batch_size:
                self._process_batch()
                
    def _process_batch(self) -> None:
        """
        Process the current batch of requests
        """
        with self.batch_lock:
            if not self.batch_queue:
                return
                
            batch = list(self.batch_queue)
            batch_size = len(batch)
            
            try:
                start_time = time.time()
                # Process the batch (this would be replaced with actual model inference)
                self._execute_batch(batch)
                processing_time = time.time() - start_time
                
                self.stats.successful_batches += 1
                self.stats.total_processing_time += processing_time
                self.stats.batch_times.append(processing_time)
                self.stats.batch_sizes.append(batch_size)
                
                # Update batch size statistics
                self.stats.max_batch_size = max(self.stats.max_batch_size, batch_size)
                self.stats.min_batch_size = min(self.stats.min_batch_size, batch_size)
                self.stats.avg_batch_size = sum(self.stats.batch_sizes) / len(self.stats.batch_sizes)
                
                # Clear the queue
                self.batch_queue.clear()
                
                # Check if we should adjust batch size
                self._check_adjust_batch_size()
                
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                self.stats.failed_batches += 1
                
    def _execute_batch(self, batch: List[Any]) -> None:
        """
        Execute the batch of requests (to be implemented by subclass)
        
        Args:
            batch: List of requests to process
        """
        raise NotImplementedError("_execute_batch must be implemented by subclass")
    
    def _check_adjust_batch_size(self) -> None:
        """
        Check if we should adjust the batch size based on performance and memory usage
        """
        current_time = time.time()
        if current_time - self.last_adjustment_time < self.adjustment_interval:
            return
            
        try:
            # Get current GPU memory usage
            memory_usage = self._get_gpu_memory_usage()
            
            # Check memory usage
            if memory_usage > self.memory_threshold:
                self._reduce_batch_size()
                return
                
            # Check performance
            avg_batch_time = sum(self.stats.batch_times[-10:]) / len(self.stats.batch_times[-10:])
            if avg_batch_time > self.performance_threshold:
                self._reduce_batch_size()
            else:
                self._increase_batch_size()
                
            self.last_adjustment_time = current_time
            
        except Exception as e:
            self.logger.error(f"Error checking batch size adjustment: {e}")
            
    def _get_gpu_memory_usage(self) -> float:
        """
        Get current GPU memory usage as a fraction of total memory
        
        Returns:
            Memory usage as a float between 0 and 1
        """
        if not torch.cuda.is_available():
            return 0.0
            
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        return allocated / total
    
    def _reduce_batch_size(self) -> None:
        """
        Reduce the batch size while respecting min_batch_size
        """
        if self.current_batch_size > self.min_batch_size:
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size - 1)
            self.logger.info(f"Reducing batch size to {self.current_batch_size}")
    
    def _increase_batch_size(self) -> None:
        """
        Increase the batch size while respecting max_batch_size
        """
        if self.current_batch_size < self.max_batch_size:
            self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
            self.logger.info(f"Increasing batch size to {self.current_batch_size}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get batch processing statistics
        
        Returns:
            Dictionary of batch statistics
        """
        return {
            'successful_batches': self.stats.successful_batches,
            'failed_batches': self.stats.failed_batches,
            'total_processing_time': self.stats.total_processing_time,
            'total_requests': self.stats.total_requests,
            'max_batch_size': self.stats.max_batch_size,
            'min_batch_size': self.stats.min_batch_size,
            'avg_batch_size': self.stats.avg_batch_size,
            'current_batch_size': self.current_batch_size,
            'batch_times': self.stats.batch_times,
            'batch_sizes': self.stats.batch_sizes
        }
