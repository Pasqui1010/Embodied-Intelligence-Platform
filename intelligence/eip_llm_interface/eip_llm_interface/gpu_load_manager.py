"""
GPU Load Manager for dynamic batch sizing and model parallelism
"""

import torch
from typing import Dict, List, Optional, Any, TypeVar, Generic
import time
import logging
from datetime import datetime
from collections import defaultdict

T = TypeVar('T')

class GPULoadManager(Generic[T]):
    """
    Manages GPU load and dynamically adjusts batch size based on memory and performance metrics
    """
    
    def __init__(self, gpu_config: Dict[str, Any]):
        self.config = gpu_config
        self.current_batch_size = self.config['batch_size']
        self.load_history = []
        self.memory_history = []
        self.peak_memory_usage = 0.0
        self.last_adjustment_time = datetime.now()
        self.load_thresholds = {
            'high': 0.8,
            'low': 0.6
        }
        self.memory_stats = {
            'current': 0.0,
            'peak': 0.0,
            'avg': 0.0
        }
        self.logger = logging.getLogger(__name__)
        
        # Initialize GPU statistics
        if torch.cuda.is_available():
            try:
                self.total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # Convert to MB
                # Initialize per-GPU statistics if model parallelism is enabled
                if self.config['enable_model_parallel']:
                    num_gpus = min(torch.cuda.device_count(), self.config['num_gpu_splits'])
                    self.gpu_stats = defaultdict(lambda: {
                        'memory': 0.0,
                        'load': 0.0,
                        'batch_size': self.config['batch_size'] // num_gpus
                    })
                    for i in range(num_gpus):
                        self.gpu_stats[i] = {
                            'memory': 0.0,
                            'load': 0.0,
                            'batch_size': self.config['batch_size'] // num_gpus
                        }
            except torch.cuda.CudaError as e:
                self.logger.error(f"CUDA error initializing GPU stats: {e}")
                self.total_memory = 0.0
                self.gpu_stats = defaultdict(lambda: {
                    'memory': 0.0,
                    'load': 0.0,
                    'batch_size': self.config['batch_size']
                })
        else:
            self.total_memory = 0.0
            self.gpu_stats = defaultdict(lambda: {
                'memory': 0.0,
                'load': 0.0,
                'batch_size': self.config['batch_size']
            })

    def get_gpu_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics with error handling"""
        try:
            if not torch.cuda.is_available():
                return {
                    'allocated': 0.0,
                    'cached': 0.0,
                    'utilization': 0.0,
                    'total_memory': 0.0
                }
                
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
                memory_cached = torch.cuda.memory_reserved() / 1024 / 1024  # Convert to MB
                memory_utilization = memory_allocated / self.total_memory
                
                self.memory_stats['current'] = memory_utilization
                self.memory_stats['peak'] = max(self.memory_stats['peak'], memory_utilization)
                
                return {
                    'allocated': float(memory_allocated),
                    'cached': float(memory_cached),
                    'utilization': float(memory_utilization),
                    'total_memory': float(self.total_memory)
                }
            except torch.cuda.CudaError as e:
                self.logger.error(f"CUDA error getting GPU memory stats: {e}")
                return {
                    'allocated': 0.0,
                    'cached': 0.0,
                    'utilization': 0.0,
                    'total_memory': 0.0
                }
            except TypeError as e:
                self.logger.error(f"Type error in GPU memory stats: {e}")
                return {
                    'allocated': 0.0,
                    'cached': 0.0,
                    'utilization': 0.0,
                    'total_memory': 0.0
                }
            except Exception as e:
                self.logger.error(f"Unexpected error in GPU memory stats: {e}")
                return {
                    'allocated': 0.0,
                    'cached': 0.0,
                    'utilization': 0.0,
                    'total_memory': 0.0
                }
        except Exception as e:
            self.logger.error(f"Error in get_gpu_memory_stats: {e}")
            return {
                'allocated': 0.0,
                'cached': 0.0,
                'utilization': 0.0,
                'total_memory': 0.0
            }

    def get_gpu_specific_batch_size(self, gpu_idx: int) -> int:
        """Get batch size for a specific GPU with error handling"""
        try:
            if not torch.cuda.is_available() or not self.config['enable_model_parallel']:
                return int(self.current_batch_size)
                
            try:
                # Get current GPU stats
                if gpu_idx not in self.gpu_stats:
                    self.logger.warning(f"No stats found for GPU {gpu_idx}")
                    return int(self.current_batch_size)
                    
                gpu_stat = self.gpu_stats[gpu_idx]
                memory_utilization = self.get_gpu_memory_stats()['utilization']
                
                # Validate batch size constraints
                if not isinstance(gpu_stat['batch_size'], (int, float)):
                    self.logger.error(f"Invalid batch size type for GPU {gpu_idx}")
                    return int(self.current_batch_size)
                    
                current_batch_size = int(gpu_stat['batch_size'])
                
                # Adjust batch size based on GPU-specific load
                if memory_utilization > self.load_thresholds['high'] and current_batch_size > self.config['min_batch_size']:
                    new_batch_size = max(current_batch_size // 2, self.config['min_batch_size'])
                    self.logger.info(f"Reducing batch size on GPU {gpu_idx} from {current_batch_size} to {new_batch_size}")
                elif memory_utilization < self.load_thresholds['low'] and current_batch_size < self.config['max_batch_size']:
                    new_batch_size = min(current_batch_size * 2, self.config['max_batch_size'])
                    self.logger.info(f"Increasing batch size on GPU {gpu_idx} from {current_batch_size} to {new_batch_size}")
                else:
                    new_batch_size = current_batch_size
                    
                gpu_stat['batch_size'] = int(new_batch_size)
                gpu_stat['load'] = float(memory_utilization)
                
                return int(new_batch_size)
            except KeyError as e:
                self.logger.error(f"Key error in GPU stats for GPU {gpu_idx}: {e}")
                return int(self.current_batch_size)
            except ValueError as e:
                self.logger.error(f"Value error in batch size calculation for GPU {gpu_idx}: {e}")
                return int(self.current_batch_size)
            except TypeError as e:
                self.logger.error(f"Type error in batch size adjustment for GPU {gpu_idx}: {e}")
                return int(self.current_batch_size)
            except Exception as e:
                self.logger.error(f"Unexpected error in batch size adjustment for GPU {gpu_idx}: {e}")
                return int(self.current_batch_size)
        except Exception as e:
            self.logger.error(f"Error in get_gpu_specific_batch_size for GPU {gpu_idx}: {e}")
            return int(self.current_batch_size)

    def adjust_batch_size(self) -> int:
        """Adjust batch size based on GPU load and memory utilization with error handling"""
        try:
            if not torch.cuda.is_available():
                return self.current_batch_size
                
            # Get current memory stats
            memory_stats = self.get_gpu_memory_stats()
            memory_utilization = memory_stats['utilization']
            
            # Check if enough time has passed since last adjustment
            time_since_last_adjustment = (datetime.now() - self.last_adjustment_time).total_seconds()
            if time_since_last_adjustment < self.config['batch_size_adjustment_interval']:
                return self.current_batch_size
                
            # Adjust batch size based on memory utilization
            if memory_utilization > self.load_thresholds['high'] and self.current_batch_size > self.config['min_batch_size']:
                new_batch_size = max(self.current_batch_size // 2, self.config['min_batch_size'])
                self.logger.info(f"Reducing batch size from {self.current_batch_size} to {new_batch_size} due to high memory utilization")
            elif memory_utilization < self.load_thresholds['low'] and self.current_batch_size < self.config['max_batch_size']:
                new_batch_size = min(self.current_batch_size * 2, self.config['max_batch_size'])
                self.logger.info(f"Increasing batch size from {self.current_batch_size} to {new_batch_size} due to low memory utilization")
            else:
                new_batch_size = self.current_batch_size
                
            # Update batch size and record adjustment
            self.current_batch_size = new_batch_size
            self.last_adjustment_time = datetime.now()
            self.load_history.append(memory_utilization)
            self.memory_history.append(memory_stats['current'])
            
            # Update GPU-specific batch sizes if model parallelism is enabled
            if self.config['enable_model_parallel']:
                num_gpus = min(torch.cuda.device_count(), self.config['num_gpu_splits'])
                for gpu_idx in range(num_gpus):
                    if gpu_idx in self.gpu_stats:
                        self.gpu_stats[gpu_idx]['batch_size'] = new_batch_size // num_gpus
            
            return self.current_batch_size
        except Exception as e:
            self.logger.error(f"Error adjusting batch size: {e}")
            return self.current_batch_size

    def update_load_metrics(self, processing_time: float):
        """Update load metrics after batch processing with error handling"""
        try:
            if torch.cuda.is_available():
                memory_stats = self.get_gpu_memory_stats()
                self.memory_stats['avg'] = (memory_stats['current'] + self.memory_stats['avg']) / 2
                self.peak_memory_usage = max(self.peak_memory_usage, memory_stats['current'])
                
                # Update GPU-specific metrics if model parallelism is enabled
                if self.config['enable_model_parallel']:
                    num_gpus = min(torch.cuda.device_count(), self.config['num_gpu_splits'])
                    for gpu_idx in range(num_gpus):
                        if gpu_idx in self.gpu_stats:
                            self.gpu_stats[gpu_idx]['load'] = memory_stats['utilization']
        except Exception as e:
            self.logger.error(f"Error updating load metrics: {e}")

    def get_model_parallel_config(self) -> Dict[str, Any]:
        """Get current model parallelism configuration with error handling"""
        try:
            config = {
                'enabled': self.config['enable_model_parallel'],
                'num_splits': self.config['num_gpu_splits'],
                'gpu_stats': dict(self.gpu_stats) if self.config['enable_model_parallel'] else None
            }
            return config
        except Exception as e:
            self.logger.error(f"Error getting model parallel config: {e}")
            return {
                'enabled': False,
                'num_splits': 1,
                'gpu_stats': None
            }
