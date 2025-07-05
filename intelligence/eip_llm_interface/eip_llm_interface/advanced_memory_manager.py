#!/usr/bin/env python3
"""
Advanced Memory Manager for GPU Optimization

This module implements a sophisticated memory management system inspired by
TensorFlow's BFC (Best-Fit with Coalescing) algorithm for efficient GPU memory
allocation and deallocation with automatic cleanup and fragmentation reduction.
"""

import torch
import gc
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import weakref
import numpy as np


@dataclass
class MemoryChunk:
    """Represents a memory chunk in the pool"""
    size: int
    ptr: Optional[torch.Tensor] = None
    is_free: bool = True
    allocation_id: int = -1
    requested_size: int = 0
    prev_chunk: Optional['MemoryChunk'] = None
    next_chunk: Optional['MemoryChunk'] = None
    timestamp: float = 0.0


@dataclass
class MemoryBin:
    """Represents a bin of memory chunks of similar size"""
    chunk_size: int
    chunks: List[MemoryChunk]
    free_chunks: List[MemoryChunk]


class MemoryAllocationStrategy(Enum):
    """Memory allocation strategies"""
    BEST_FIT = "best_fit"
    FIRST_FIT = "first_fit"
    WORST_FIT = "worst_fit"


class AdvancedMemoryManager:
    """
    Advanced GPU memory manager with BFC-inspired algorithms
    
    Features:
    - Best-fit allocation with coalescing
    - Automatic memory cleanup and garbage collection
    - Fragmentation reduction
    - Thread-safe operations
    - Memory pressure monitoring
    """
    
    def __init__(self, 
                 max_memory_mb: int = 8192,
                 min_allocation_size: int = 256,
                 fragmentation_fraction: float = 0.1,
                 allow_growth: bool = True,
                 enable_coalescing: bool = True):
        """
        Initialize advanced memory manager
        
        Args:
            max_memory_mb: Maximum memory in MB
            min_allocation_size: Minimum allocation size in bytes
            fragmentation_fraction: Fraction for chunk splitting
            allow_growth: Allow memory pool to grow
            enable_coalescing: Enable chunk coalescing
        """
        self.max_memory_mb = max_memory_mb
        self.min_allocation_size = min_allocation_size
        self.fragmentation_fraction = fragmentation_fraction
        self.allow_growth = allow_growth
        self.enable_coalescing = enable_coalescing
        
        # Memory tracking
        self.total_allocated = 0
        self.total_used = 0
        self.allocation_id_counter = 0
        
        # Bins for different chunk sizes (power of 2)
        self.bins: Dict[int, MemoryBin] = {}
        self.chunks: List[MemoryChunk] = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Memory pressure monitoring
        self.pressure_threshold = 0.8  # 80% memory usage
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 60.0  # 60 seconds
        
        # Weak references for automatic cleanup
        self._tensor_refs: weakref.WeakSet = weakref.WeakSet()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize bins
        self._initialize_bins()
        
        self.logger.info(f"Advanced Memory Manager initialized with {max_memory_mb}MB limit")
    
    def _initialize_bins(self):
        """Initialize memory bins for different chunk sizes"""
        current_size = self.min_allocation_size
        while current_size <= self.max_memory_mb * 1024 * 1024:
            self.bins[current_size] = MemoryBin(
                chunk_size=current_size,
                chunks=[],
                free_chunks=[]
            )
            current_size *= 2
    
    def _get_bin_size(self, requested_size: int) -> int:
        """Get appropriate bin size for requested allocation"""
        # Round up to next power of 2
        size = self.min_allocation_size
        while size < requested_size:
            size *= 2
        return min(size, self.max_memory_mb * 1024 * 1024)
    
    def _find_best_fit_chunk(self, requested_size: int) -> Optional[MemoryChunk]:
        """Find best-fit chunk for allocation"""
        bin_size = self._get_bin_size(requested_size)
        
        # Look in current bin
        if bin_size in self.bins:
            bin_obj = self.bins[bin_size]
            if bin_obj.free_chunks:
                return bin_obj.free_chunks.pop()
        
        # Look in larger bins
        for size in sorted(self.bins.keys()):
            if size >= bin_size:
                bin_obj = self.bins[size]
                if bin_obj.free_chunks:
                    chunk = bin_obj.free_chunks.pop()
                    # Split chunk if necessary
                    if size > bin_size * 2:  # Only split if significant difference
                        self._split_chunk(chunk, requested_size)
                    return chunk
        
        return None
    
    def _split_chunk(self, chunk: MemoryChunk, requested_size: int):
        """Split a chunk into smaller chunks"""
        if chunk.size <= requested_size * 2:
            return  # Don't split if too small
        
        # Create new chunk for remaining space
        remaining_size = chunk.size - requested_size
        if remaining_size >= self.min_allocation_size:
            new_chunk = MemoryChunk(
                size=remaining_size,
                is_free=True,
                timestamp=time.time()
            )
            
            # Update chunk sizes
            chunk.size = requested_size
            chunk.requested_size = requested_size
            
            # Link chunks
            new_chunk.next_chunk = chunk.next_chunk
            new_chunk.prev_chunk = chunk
            chunk.next_chunk = new_chunk
            
            if new_chunk.next_chunk:
                new_chunk.next_chunk.prev_chunk = new_chunk
            
            # Add to appropriate bin
            self._add_chunk_to_bin(new_chunk)
    
    def _add_chunk_to_bin(self, chunk: MemoryChunk):
        """Add chunk to appropriate bin"""
        bin_size = self._get_bin_size(chunk.size)
        if bin_size in self.bins:
            bin_obj = self.bins[bin_size]
            bin_obj.chunks.append(chunk)
            if chunk.is_free:
                bin_obj.free_chunks.append(chunk)
    
    def _coalesce_chunks(self, chunk: MemoryChunk) -> MemoryChunk:
        """Coalesce adjacent free chunks"""
        if not self.enable_coalescing:
            return chunk
        
        # Coalesce with previous chunk
        if chunk.prev_chunk and chunk.prev_chunk.is_free:
            prev_chunk = chunk.prev_chunk
            prev_chunk.size += chunk.size
            prev_chunk.next_chunk = chunk.next_chunk
            
            if chunk.next_chunk:
                chunk.next_chunk.prev_chunk = prev_chunk
            
            # Remove chunk from bins
            self._remove_chunk_from_bins(chunk)
            
            chunk = prev_chunk
        
        # Coalesce with next chunk
        if chunk.next_chunk and chunk.next_chunk.is_free:
            next_chunk = chunk.next_chunk
            chunk.size += next_chunk.size
            chunk.next_chunk = next_chunk.next_chunk
            
            if next_chunk.next_chunk:
                next_chunk.next_chunk.prev_chunk = chunk
            
            # Remove next_chunk from bins
            self._remove_chunk_from_bins(next_chunk)
        
        return chunk
    
    def _remove_chunk_from_bins(self, chunk: MemoryChunk):
        """Remove chunk from all bins"""
        bin_size = self._get_bin_size(chunk.size)
        if bin_size in self.bins:
            bin_obj = self.bins[bin_size]
            if chunk in bin_obj.chunks:
                bin_obj.chunks.remove(chunk)
            if chunk in bin_obj.free_chunks:
                bin_obj.free_chunks.remove(chunk)
    
    @contextmanager
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float16):
        """
        Context manager for tensor allocation with automatic cleanup
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
        
        Yields:
            torch.Tensor: Allocated tensor
        """
        tensor = None
        chunk = None
        
        try:
            # Calculate required size
            element_size = torch.tensor([], dtype=dtype).element_size()
            required_size = element_size * np.prod(shape)
            
            with self.lock:
                # Check memory pressure
                self._check_memory_pressure()
                
                # Find best-fit chunk
                chunk = self._find_best_fit_chunk(required_size)
                
                if chunk is None:
                    # Try to grow memory pool
                    if self.allow_growth and self._can_grow(required_size):
                        chunk = self._grow_memory_pool(required_size)
                    else:
                        # Force cleanup and retry
                        self._force_cleanup()
                        chunk = self._find_best_fit_chunk(required_size)
                
                if chunk is None:
                    raise RuntimeError(f"Failed to allocate {required_size} bytes")
                
                # Allocate tensor
                if chunk.ptr is None:
                    chunk.ptr = torch.empty(shape, dtype=dtype, device='cuda')
                else:
                    # Reshape existing tensor if possible
                    if chunk.ptr.numel() >= np.prod(shape):
                        tensor = chunk.ptr[:np.prod(shape)].reshape(shape)
                    else:
                        chunk.ptr = torch.empty(shape, dtype=dtype, device='cuda')
                        tensor = chunk.ptr
                else:
                    tensor = chunk.ptr
                
                # Update chunk metadata
                chunk.is_free = False
                chunk.allocation_id = self.allocation_id_counter
                chunk.requested_size = required_size
                chunk.timestamp = time.time()
                self.allocation_id_counter += 1
                
                # Update memory tracking
                self.total_used += required_size
                
                # Add to weak references for cleanup
                self._tensor_refs.add(tensor)
                
                yield tensor
                
        except Exception as e:
            self.logger.error(f"Tensor allocation failed: {e}")
            raise
        finally:
            if tensor is not None and chunk is not None:
                self._deallocate_chunk(chunk)
    
    def _can_grow(self, required_size: int) -> bool:
        """Check if memory pool can grow"""
        return (self.total_allocated + required_size) <= (self.max_memory_mb * 1024 * 1024)
    
    def _grow_memory_pool(self, required_size: int) -> Optional[MemoryChunk]:
        """Grow memory pool by allocating new chunk"""
        try:
            # Allocate new chunk
            chunk_size = self._get_bin_size(required_size)
            new_chunk = MemoryChunk(
                size=chunk_size,
                is_free=True,
                timestamp=time.time()
            )
            
            # Add to chunks list
            self.chunks.append(new_chunk)
            
            # Link with existing chunks
            if self.chunks:
                new_chunk.prev_chunk = self.chunks[-2] if len(self.chunks) > 1 else None
                if new_chunk.prev_chunk:
                    new_chunk.prev_chunk.next_chunk = new_chunk
            
            # Add to bin
            self._add_chunk_to_bin(new_chunk)
            
            self.total_allocated += chunk_size
            self.logger.info(f"Grew memory pool by {chunk_size} bytes")
            
            return new_chunk
            
        except Exception as e:
            self.logger.error(f"Failed to grow memory pool: {e}")
            return None
    
    def _deallocate_chunk(self, chunk: MemoryChunk):
        """Deallocate a memory chunk"""
        with self.lock:
            if not chunk.is_free:
                # Mark as free
                chunk.is_free = True
                chunk.allocation_id = -1
                
                # Update memory tracking
                self.total_used -= chunk.requested_size
                
                # Coalesce with adjacent chunks
                if self.enable_coalescing:
                    chunk = self._coalesce_chunks(chunk)
                
                # Add back to free chunks
                self._add_chunk_to_bin(chunk)
                
                # Clear tensor reference
                chunk.ptr = None
    
    def _check_memory_pressure(self):
        """Check memory pressure and trigger cleanup if necessary"""
        current_time = time.time()
        memory_usage_ratio = self.total_used / max(self.total_allocated, 1)
        
        if (memory_usage_ratio > self.pressure_threshold and 
            current_time - self.last_cleanup_time > self.cleanup_interval):
            self._force_cleanup()
            self.last_cleanup_time = current_time
    
    def _force_cleanup(self):
        """Force memory cleanup"""
        self.logger.info("Forcing memory cleanup due to pressure")
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        
        # Run garbage collection
        gc.collect()
        
        # Remove stale chunks
        self._remove_stale_chunks()
    
    def _remove_stale_chunks(self):
        """Remove chunks that haven't been used recently"""
        current_time = time.time()
        stale_threshold = 300.0  # 5 minutes
        
        chunks_to_remove = []
        for chunk in self.chunks:
            if (chunk.is_free and 
                current_time - chunk.timestamp > stale_threshold and
                chunk.size > self.min_allocation_size * 4):  # Only remove large chunks
                chunks_to_remove.append(chunk)
        
        for chunk in chunks_to_remove:
            self._remove_chunk_from_bins(chunk)
            if chunk in self.chunks:
                self.chunks.remove(chunk)
            self.total_allocated -= chunk.size
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage statistics"""
        with self.lock:
            return {
                'total_allocated_mb': self.total_allocated / (1024 * 1024),
                'total_used_mb': self.total_used / (1024 * 1024),
                'fragmentation_ratio': 1.0 - (self.total_used / max(self.total_allocated, 1)),
                'free_chunks_count': sum(len(bin_obj.free_chunks) for bin_obj in self.bins.values()),
                'total_chunks_count': len(self.chunks),
                'pressure_level': self.total_used / max(self.total_allocated, 1)
            }
    
    def optimize_memory(self):
        """Optimize memory usage"""
        with self.lock:
            self.logger.info("Starting memory optimization")
            
            # Force cleanup
            self._force_cleanup()
            
            # Defragment memory
            self._defragment_memory()
            
            self.logger.info("Memory optimization completed")
    
    def _defragment_memory(self):
        """Defragment memory by merging small free chunks"""
        # Find small free chunks that can be merged
        small_chunks = []
        for bin_obj in self.bins.values():
            for chunk in bin_obj.free_chunks:
                if chunk.size < self.min_allocation_size * 2:
                    small_chunks.append(chunk)
        
        # Merge adjacent small chunks
        for chunk in small_chunks:
            if chunk.is_free:
                self._coalesce_chunks(chunk)
    
    def shutdown(self):
        """Shutdown memory manager and cleanup all resources"""
        with self.lock:
            self.logger.info("Shutting down memory manager")
            
            # Clear all chunks
            for chunk in self.chunks:
                if chunk.ptr is not None:
                    del chunk.ptr
                chunk.ptr = None
            
            self.chunks.clear()
            self.bins.clear()
            
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            
            # Run garbage collection
            gc.collect()
            
            self.logger.info("Memory manager shutdown completed") 