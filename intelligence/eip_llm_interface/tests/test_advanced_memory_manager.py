#!/usr/bin/env python3
"""
Comprehensive Stress Tests for Advanced Memory Manager

This module provides extensive stress testing for the advanced memory manager,
including memory leak detection, concurrent access testing, pressure testing,
and edge case validation.
"""

import unittest
import threading
import time
import gc
import psutil
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eip_llm_interface.advanced_memory_manager import (
    AdvancedMemoryManager, 
    MemoryChunk, 
    MemoryBin,
    MemoryAllocationStrategy
)


class TestAdvancedMemoryManager(unittest.TestCase):
    """Comprehensive stress tests for Advanced Memory Manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.memory_manager = AdvancedMemoryManager(
            max_memory_mb=512,  # Smaller limit for testing
            min_allocation_size=64,
            fragmentation_fraction=0.1,
            allow_growth=True,
            enable_coalescing=True
        )
        
        # Enable detailed logging for debugging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        # Track initial memory state
        self.initial_memory = self._get_gpu_memory_usage()
    
    def tearDown(self):
        """Clean up after tests"""
        self.memory_manager.shutdown()
        torch.cuda.empty_cache()
        gc.collect()
        
        # Verify no memory leaks
        final_memory = self._get_gpu_memory_usage()
        memory_diff = final_memory['allocated'] - self.initial_memory['allocated']
        
        # Allow small tolerance for CUDA overhead
        if memory_diff > 10 * 1024 * 1024:  # 10MB tolerance
            self.logger.warning(f"Potential memory leak detected: {memory_diff / 1024 / 1024:.2f}MB")
    
    def _get_gpu_memory_usage(self) -> Dict[str, int]:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
        return {'allocated': 0, 'cached': 0, 'max_allocated': 0}
    
    def _get_system_memory_usage(self) -> Dict[str, float]:
        """Get system memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in tensor allocation/deallocation"""
        self.logger.info("Testing memory leak detection...")
        
        initial_memory = self._get_gpu_memory_usage()
        initial_usage = self.memory_manager.get_memory_usage()
        
        # Perform multiple allocation cycles
        for cycle in range(10):
            tensors = []
            
            # Allocate multiple tensors
            for i in range(20):
                with self.memory_manager.allocate_tensor((100, 100), torch.float16) as tensor:
                    tensors.append(tensor.clone())  # Create copy to test memory pressure
            
            # Simulate processing
            for tensor in tensors:
                _ = torch.sum(tensor)
            
            # Tensors are automatically deallocated when context exits
            
            # Check memory usage after each cycle
            current_memory = self._get_gpu_memory_usage()
            current_usage = self.memory_manager.get_memory_usage()
            
            self.logger.info(f"Cycle {cycle + 1}: "
                           f"GPU allocated: {current_memory['allocated'] / 1024 / 1024:.2f}MB, "
                           f"Manager used: {current_usage['total_used_mb']:.2f}MB")
        
        # Final memory check
        final_memory = self._get_gpu_memory_usage()
        final_usage = self.memory_manager.get_memory_usage()
        
        # Verify memory is properly cleaned up
        memory_diff = final_memory['allocated'] - initial_memory['allocated']
        usage_diff = final_usage['total_used_mb'] - initial_usage['total_used_mb']
        
        self.assertLess(memory_diff, 50 * 1024 * 1024,  # 50MB tolerance
                       f"Memory leak detected: {memory_diff / 1024 / 1024:.2f}MB")
        self.assertLess(usage_diff, 10,  # 10MB tolerance
                       f"Manager memory leak: {usage_diff:.2f}MB")
    
    def test_concurrent_access(self):
        """Test concurrent tensor allocation from multiple threads"""
        self.logger.info("Testing concurrent access...")
        
        num_threads = 8
        tensors_per_thread = 10
        results = []
        
        def worker(thread_id: int) -> Dict[str, Any]:
            """Worker function for concurrent testing"""
            thread_tensors = []
            thread_memory_usage = []
            
            try:
                for i in range(tensors_per_thread):
                    with self.memory_manager.allocate_tensor((50, 50), torch.float16) as tensor:
                        # Simulate work
                        result = torch.sum(tensor * 2.0)
                        thread_tensors.append(result.item())
                        
                        # Track memory usage
                        usage = self.memory_manager.get_memory_usage()
                        thread_memory_usage.append(usage['total_used_mb'])
                
                return {
                    'thread_id': thread_id,
                    'success': True,
                    'tensor_sum': sum(thread_tensors),
                    'max_memory_usage': max(thread_memory_usage),
                    'avg_memory_usage': np.mean(thread_memory_usage)
                }
                
            except Exception as e:
                return {
                    'thread_id': thread_id,
                    'success': False,
                    'error': str(e)
                }
        
        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        # Verify all threads completed successfully
        successful_results = [r for r in results if r['success']]
        self.assertEqual(len(successful_results), num_threads,
                        f"Some threads failed: {[r for r in results if not r['success']]}")
        
        # Verify memory consistency
        final_usage = self.memory_manager.get_memory_usage()
        self.assertLess(final_usage['total_used_mb'], 100,  # Should be near zero after cleanup
                       "Memory not properly cleaned up after concurrent access")
        
        self.logger.info(f"Concurrent access test completed: {len(successful_results)} threads successful")
    
    def test_memory_pressure(self):
        """Test behavior under memory pressure"""
        self.logger.info("Testing memory pressure handling...")
        
        # Create memory pressure by allocating large tensors
        large_tensors = []
        pressure_cycles = 5
        
        for cycle in range(pressure_cycles):
            try:
                # Try to allocate increasingly large tensors
                tensor_size = (1000, 1000)  # 4MB per tensor
                
                with self.memory_manager.allocate_tensor(tensor_size, torch.float16) as tensor:
                    large_tensors.append(tensor.clone())
                    
                    # Check memory pressure
                    usage = self.memory_manager.get_memory_usage()
                    self.logger.info(f"Pressure cycle {cycle + 1}: "
                                   f"Used: {usage['total_used_mb']:.2f}MB, "
                                   f"Pressure: {usage['pressure_level']:.2f}")
                    
                    # Verify pressure monitoring works
                    if usage['pressure_level'] > 0.8:
                        self.logger.info("High memory pressure detected")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.info(f"Expected OOM at cycle {cycle + 1}")
                    break
                else:
                    raise
        
        # Verify cleanup under pressure
        large_tensors.clear()
        torch.cuda.empty_cache()
        gc.collect()
        
        final_usage = self.memory_manager.get_memory_usage()
        self.assertLess(final_usage['pressure_level'], 0.5,
                       "Memory pressure not relieved after cleanup")
    
    def test_fragmentation_handling(self):
        """Test memory fragmentation handling"""
        self.logger.info("Testing fragmentation handling...")
        
        # Create fragmentation by allocating and deallocating different sized tensors
        tensor_sizes = [(10, 10), (50, 50), (100, 100), (200, 200), (500, 500)]
        
        for iteration in range(5):
            tensors = []
            
            # Allocate tensors of different sizes
            for size in tensor_sizes:
                with self.memory_manager.allocate_tensor(size, torch.float16) as tensor:
                    tensors.append(tensor.clone())
            
            # Deallocate some tensors to create fragmentation
            for i in range(0, len(tensors), 2):
                del tensors[i]
            
            # Allocate new tensors to test fragmentation handling
            new_tensors = []
            for size in tensor_sizes:
                try:
                    with self.memory_manager.allocate_tensor(size, torch.float16) as tensor:
                        new_tensors.append(tensor.clone())
                except RuntimeError:
                    self.logger.info(f"Expected allocation failure due to fragmentation at iteration {iteration}")
                    break
            
            # Clean up
            tensors.clear()
            new_tensors.clear()
            
            # Check fragmentation ratio
            usage = self.memory_manager.get_memory_usage()
            self.logger.info(f"Iteration {iteration + 1}: "
                           f"Fragmentation: {usage['fragmentation_ratio']:.3f}")
        
        # Test defragmentation
        self.memory_manager.optimize_memory()
        final_usage = self.memory_manager.get_memory_usage()
        
        self.logger.info(f"Final fragmentation after optimization: {final_usage['fragmentation_ratio']:.3f}")
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        self.logger.info("Testing edge cases...")
        
        # Test zero-size allocation
        with self.assertRaises(ValueError):
            with self.memory_manager.allocate_tensor((0, 0), torch.float16) as tensor:
                pass
        
        # Test very large allocation
        try:
            with self.memory_manager.allocate_tensor((10000, 10000), torch.float16) as tensor:
                self.logger.info("Large allocation succeeded")
        except RuntimeError as e:
            self.logger.info(f"Expected failure for very large allocation: {e}")
        
        # Test rapid allocation/deallocation
        for i in range(100):
            with self.memory_manager.allocate_tensor((10, 10), torch.float16) as tensor:
                _ = torch.sum(tensor)
        
        # Test mixed data types
        dtypes = [torch.float16, torch.float32, torch.int32, torch.int64]
        for dtype in dtypes:
            try:
                with self.memory_manager.allocate_tensor((50, 50), dtype) as tensor:
                    _ = torch.sum(tensor)
            except Exception as e:
                self.logger.info(f"Expected failure for dtype {dtype}: {e}")
        
        # Test concurrent access to same manager instance
        def concurrent_worker():
            for i in range(10):
                with self.memory_manager.allocate_tensor((20, 20), torch.float16) as tensor:
                    time.sleep(0.001)  # Small delay
        
        threads = [threading.Thread(target=concurrent_worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    
    def test_performance_benchmark(self):
        """Benchmark memory manager performance"""
        self.logger.info("Running performance benchmark...")
        
        # Benchmark allocation speed
        start_time = time.time()
        allocation_count = 1000
        
        for i in range(allocation_count):
            with self.memory_manager.allocate_tensor((100, 100), torch.float16) as tensor:
                _ = torch.sum(tensor)
        
        allocation_time = time.time() - start_time
        allocations_per_second = allocation_count / allocation_time
        
        self.logger.info(f"Allocation performance: {allocations_per_second:.2f} allocations/second")
        
        # Benchmark memory usage efficiency
        usage = self.memory_manager.get_memory_usage()
        efficiency = usage['total_used_mb'] / max(usage['total_allocated_mb'], 1)
        
        self.logger.info(f"Memory efficiency: {efficiency:.3f}")
        
        # Performance assertions
        self.assertGreater(allocations_per_second, 100,  # At least 100 allocations/second
                          "Allocation performance too low")
        self.assertGreater(efficiency, 0.5,  # At least 50% efficiency
                          "Memory efficiency too low")
    
    def test_stress_test_long_running(self):
        """Long-running stress test"""
        self.logger.info("Running long-running stress test...")
        
        test_duration = 30  # 30 seconds
        start_time = time.time()
        allocation_count = 0
        
        try:
            while time.time() - start_time < test_duration:
                # Allocate and use tensors
                with self.memory_manager.allocate_tensor((200, 200), torch.float16) as tensor:
                    # Simulate real work
                    result = torch.matmul(tensor, tensor.T)
                    _ = torch.sum(result)
                    allocation_count += 1
                
                # Periodic memory check
                if allocation_count % 100 == 0:
                    usage = self.memory_manager.get_memory_usage()
                    self.logger.info(f"Stress test: {allocation_count} allocations, "
                                   f"Memory used: {usage['total_used_mb']:.2f}MB")
                
                # Small delay to prevent overwhelming
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            self.logger.info("Stress test interrupted")
        
        self.logger.info(f"Long-running stress test completed: {allocation_count} allocations")
        
        # Verify system stability
        final_usage = self.memory_manager.get_memory_usage()
        self.assertLess(final_usage['total_used_mb'], 50,  # Should be near zero
                       "Memory not properly cleaned up after stress test")


class TestMemoryManagerIntegration(unittest.TestCase):
    """Integration tests for memory manager with real workloads"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.memory_manager = AdvancedMemoryManager(
            max_memory_mb=1024,
            min_allocation_size=128,
            allow_growth=True,
            enable_coalescing=True
        )
    
    def tearDown(self):
        """Clean up integration tests"""
        self.memory_manager.shutdown()
    
    def test_batch_processing_simulation(self):
        """Simulate batch processing workload"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Testing batch processing simulation...")
        
        batch_sizes = [32, 64, 128, 256]
        sequence_lengths = [512, 1024, 2048]
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                try:
                    # Simulate transformer attention computation
                    with self.memory_manager.allocate_tensor((batch_size, seq_len, 768), torch.float16) as hidden_states:
                        with self.memory_manager.allocate_tensor((768, 768), torch.float16) as weight_matrix:
                            # Simulate attention computation
                            attention_output = torch.matmul(hidden_states, weight_matrix)
                            
                            # Simulate residual connection
                            output = hidden_states + attention_output
                            
                            # Verify computation
                            self.assertIsNotNone(output)
                            self.assertEqual(output.shape, hidden_states.shape)
                    
                    self.logger.info(f"Batch {batch_size}x{seq_len} processed successfully")
                    
                except RuntimeError as e:
                    self.logger.info(f"Expected failure for large batch {batch_size}x{seq_len}: {e}")
                    break
    
    def test_memory_pressure_recovery(self):
        """Test recovery from memory pressure"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Testing memory pressure recovery...")
        
        # Create memory pressure
        tensors = []
        try:
            while True:
                tensor = self.memory_manager.allocate_tensor((500, 500), torch.float16).__enter__()
                tensors.append(tensor)
        except RuntimeError:
            self.logger.info("Memory pressure created")
        
        # Verify pressure detection
        usage = self.memory_manager.get_memory_usage()
        self.assertGreater(usage['pressure_level'], 0.8)
        
        # Release pressure
        tensors.clear()
        torch.cuda.empty_cache()
        gc.collect()
        
        # Verify recovery
        usage = self.memory_manager.get_memory_usage()
        self.assertLess(usage['pressure_level'], 0.5)
        
        self.logger.info("Memory pressure recovery successful")


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2) 