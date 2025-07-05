#!/usr/bin/env python3
"""
Unit tests for Advanced Memory Manager

Tests the GPU memory management functionality including allocation,
deallocation, and memory pressure handling.
"""

import pytest
import torch
import time
import threading
from unittest.mock import Mock, patch
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eip_llm_interface'))

from advanced_memory_manager import AdvancedMemoryManager, MemoryChunk, MemoryBin, MemoryAllocationStrategy


class TestAdvancedMemoryManager:
    """Test cases for Advanced Memory Manager"""
    
    @pytest.fixture
    def memory_manager(self):
        """Create a memory manager instance for testing"""
        return AdvancedMemoryManager(max_memory_mb=1024, min_allocation_size=256)
    
    def test_initialization(self, memory_manager):
        """Test memory manager initialization"""
        assert memory_manager.max_memory_mb == 1024
        assert memory_manager.min_allocation_size == 256
        assert memory_manager.total_allocated == 0
        assert memory_manager.total_used == 0
        assert len(memory_manager.bins) > 0
    
    def test_bin_initialization(self, memory_manager):
        """Test that memory bins are properly initialized"""
        # Check that bins are created for power-of-2 sizes
        expected_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
        
        for size in expected_sizes:
            if size <= memory_manager.max_memory_mb * 1024 * 1024:
                assert size in memory_manager.bins
                bin_obj = memory_manager.bins[size]
                assert isinstance(bin_obj, MemoryBin)
                assert bin_obj.chunk_size == size
                assert len(bin_obj.chunks) == 0
                assert len(bin_obj.free_chunks) == 0
    
    def test_get_bin_size(self, memory_manager):
        """Test bin size calculation"""
        # Test exact matches
        assert memory_manager._get_bin_size(256) == 256
        assert memory_manager._get_bin_size(512) == 512
        
        # Test rounding up
        assert memory_manager._get_bin_size(300) == 512
        assert memory_manager._get_bin_size(1000) == 1024
        
        # Test maximum limit
        max_size = memory_manager.max_memory_mb * 1024 * 1024
        assert memory_manager._get_bin_size(max_size + 1000) == max_size
    
    @patch('torch.cuda.is_available')
    def test_tensor_allocation_context_manager(self, mock_cuda_available, memory_manager):
        """Test tensor allocation using context manager"""
        mock_cuda_available.return_value = False  # Use CPU for testing
        
        shape = (100, 100)
        dtype = torch.float32
        
        with memory_manager.allocate_tensor(shape, dtype) as tensor:
            assert tensor is not None
            assert tensor.shape == shape
            assert tensor.dtype == dtype
            assert tensor.device.type == 'cpu'
        
        # Tensor should be deallocated after context exit
        assert memory_manager.total_used == 0
    
    def test_memory_pressure_detection(self, memory_manager):
        """Test memory pressure detection and cleanup"""
        # Simulate high memory usage
        memory_manager.total_used = int(memory_manager.max_memory_mb * 1024 * 1024 * 0.9)
        
        # Trigger pressure check
        memory_manager._check_memory_pressure()
        
        # Should trigger cleanup
        assert memory_manager.total_used < memory_manager.max_memory_mb * 1024 * 1024 * 0.8
    
    def test_chunk_coalescing(self, memory_manager):
        """Test chunk coalescing functionality"""
        # Create adjacent free chunks
        chunk1 = MemoryChunk(size=256, is_free=True)
        chunk2 = MemoryChunk(size=256, is_free=True)
        chunk3 = MemoryChunk(size=256, is_free=True)
        
        # Link chunks
        chunk1.next_chunk = chunk2
        chunk2.prev_chunk = chunk1
        chunk2.next_chunk = chunk3
        chunk3.prev_chunk = chunk2
        
        # Test coalescing
        coalesced = memory_manager._coalesce_chunks(chunk1)
        
        # Should result in one large chunk
        assert coalesced.size == 768
        assert coalesced.is_free
    
    def test_thread_safety(self, memory_manager):
        """Test thread safety of memory operations"""
        results = []
        errors = []
        
        def allocate_tensors():
            try:
                for i in range(10):
                    with memory_manager.allocate_tensor((100, 100), torch.float32) as tensor:
                        results.append(tensor.shape)
                        time.sleep(0.001)  # Simulate work
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=allocate_tensors)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check for errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 50  # 5 threads * 10 allocations each
    
    def test_memory_metrics(self, memory_manager):
        """Test memory usage metrics"""
        initial_usage = memory_manager.get_memory_usage()
        
        # Allocate some memory
        with memory_manager.allocate_tensor((500, 500), torch.float32) as tensor:
            usage_during = memory_manager.get_memory_usage()
            assert usage_during['used_mb'] > initial_usage['used_mb']
        
        # Check final usage
        final_usage = memory_manager.get_memory_usage()
        assert final_usage['used_mb'] == initial_usage['used_mb']
    
    def test_cleanup_on_shutdown(self, memory_manager):
        """Test cleanup when shutting down"""
        # Allocate some memory
        with memory_manager.allocate_tensor((100, 100), torch.float32) as tensor:
            assert memory_manager.total_used > 0
        
        # Shutdown
        memory_manager.shutdown()
        
        # Should be cleaned up
        assert memory_manager.total_used == 0
        assert memory_manager.total_allocated == 0
    
    def test_error_handling(self, memory_manager):
        """Test error handling in memory operations"""
        # Test with invalid shape
        with pytest.raises(ValueError):
            with memory_manager.allocate_tensor((-1, 100), torch.float32) as tensor:
                pass
        
        # Test with very large allocation
        with pytest.raises(ValueError):
            with memory_manager.allocate_tensor((10000, 10000), torch.float32) as tensor:
                pass


class TestMemoryChunk:
    """Test cases for MemoryChunk class"""
    
    def test_memory_chunk_creation(self):
        """Test MemoryChunk creation and properties"""
        chunk = MemoryChunk(size=1024, ptr=None, is_free=True)
        
        assert chunk.size == 1024
        assert chunk.is_free is True
        assert chunk.allocation_id == -1
        assert chunk.prev_chunk is None
        assert chunk.next_chunk is None
    
    def test_memory_chunk_linking(self):
        """Test linking memory chunks"""
        chunk1 = MemoryChunk(size=256, is_free=True)
        chunk2 = MemoryChunk(size=256, is_free=True)
        
        chunk1.next_chunk = chunk2
        chunk2.prev_chunk = chunk1
        
        assert chunk1.next_chunk == chunk2
        assert chunk2.prev_chunk == chunk1


class TestMemoryBin:
    """Test cases for MemoryBin class"""
    
    def test_memory_bin_creation(self):
        """Test MemoryBin creation"""
        bin_obj = MemoryBin(chunk_size=1024, chunks=[], free_chunks=[])
        
        assert bin_obj.chunk_size == 1024
        assert len(bin_obj.chunks) == 0
        assert len(bin_obj.free_chunks) == 0
    
    def test_memory_bin_operations(self):
        """Test MemoryBin operations"""
        bin_obj = MemoryBin(chunk_size=1024, chunks=[], free_chunks=[])
        
        chunk = MemoryChunk(size=1024, is_free=True)
        bin_obj.chunks.append(chunk)
        bin_obj.free_chunks.append(chunk)
        
        assert len(bin_obj.chunks) == 1
        assert len(bin_obj.free_chunks) == 1
        assert bin_obj.chunks[0] == chunk
        assert bin_obj.free_chunks[0] == chunk


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 