import torch
import torch.cuda
from typing import Dict, List, Optional, Any
import logging
from collections import defaultdict
import threading

class GPUMemoryPool:
    """GPU Memory Pool Manager for efficient memory allocation and reuse"""
    
    def __init__(self, device: str = "cuda", max_memory_mb: int = 8192):
        """
        Initialize GPU memory pool
        
        Args:
            device: CUDA device to manage
            max_memory_mb: Maximum memory pool size in MB
        """
        self.device = device
        self.max_memory_mb = max_memory_mb
        self.logger = logging.getLogger(__name__)
        
        # Initialize memory pools
        self.tensor_pool = defaultdict(list)  # shape -> list of tensors
        self.lock = threading.Lock()
        self.current_memory_mb = 0
        
        # Track memory usage stats
        self.stats = {
            'allocations': 0,
            'reuses': 0,
            'peak_memory_mb': 0,
            'fragmentation': 0.0
        }
        
        self.logger.info(f"GPU Memory Pool initialized on {device} with max size {max_memory_mb}MB")
    
    def allocate(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """
        Allocate a tensor from the pool or create a new one if none available
        
        Args:
            shape: Shape of the tensor
            dtype: Data type of the tensor
            
        Returns:
            torch.Tensor: Allocated tensor
        """
        with self.lock:
            key = (shape, dtype)
            if self.tensor_pool[key]:
                # Reuse existing tensor
                tensor = self.tensor_pool[key].pop()
                self.stats['reuses'] += 1
                return tensor
            
            # Create new tensor
            tensor = torch.zeros(shape, dtype=dtype, device=self.device)
            self.stats['allocations'] += 1
            
            # Update memory usage
            tensor_size_mb = tensor.element_size() * tensor.numel() / (1024 * 1024)
            self.current_memory_mb += tensor_size_mb
            self.stats['peak_memory_mb'] = max(self.stats['peak_memory_mb'], self.current_memory_mb)
            
            return tensor
    
    def release(self, tensor: torch.Tensor):
        """
        Release a tensor back to the pool
        
        Args:
            tensor: Tensor to release
        """
        with self.lock:
            key = (tensor.shape, tensor.dtype)
            self.tensor_pool[key].append(tensor)
            
            # Update fragmentation stats
            self._update_fragmentation()
    
    def _update_fragmentation(self):
        """Calculate current memory fragmentation"""
        total_memory = 0
        used_memory = 0
        
        for tensors in self.tensor_pool.values():
            for tensor in tensors:
                tensor_size = tensor.element_size() * tensor.numel()
                total_memory += tensor_size
                
        current_memory = torch.cuda.memory_allocated(self.device)
        used_memory = current_memory - total_memory
        
        if current_memory > 0:
            self.stats['fragmentation'] = (used_memory / current_memory) * 100
        else:
            self.stats['fragmentation'] = 0.0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory pool statistics"""
        return {
            'allocations': self.stats['allocations'],
            'reuses': self.stats['reuses'],
            'peak_memory_mb': self.stats['peak_memory_mb'],
            'current_memory_mb': self.current_memory_mb,
            'fragmentation': self.stats['fragmentation'],
            'pool_size': sum(len(tensors) for tensors in self.tensor_pool.values())
        }
    
    def cleanup(self):
        """Cleanup all allocated memory"""
        with self.lock:
            for tensors in self.tensor_pool.values():
                for tensor in tensors:
                    del tensor
            self.tensor_pool.clear()
            torch.cuda.empty_cache()
            self.logger.info("GPU Memory Pool cleanup complete")
