#!/usr/bin/env python3
"""
Performance Optimizations for Safety-Embedded LLM

This module provides performance optimization utilities including:
- GPU memory management
- Response caching
- Batch processing
- Memory monitoring
"""

import torch
import gc
import psutil
import time
import hashlib
from typing import Dict, List, Optional, Any
from functools import lru_cache
from collections import OrderedDict
import logging


class GPUMemoryOptimizer:
    """Optimizes GPU memory usage for LLM inference"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def optimize_memory(self, model):
        """Apply GPU memory optimizations"""
        if self.device != "cuda" or not torch.cuda.is_available():
            return
            
        try:
            # Enable memory efficient attention if available
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # Enable memory pool
            torch.cuda.empty_cache()
            
            # Use mixed precision if supported
            if hasattr(torch.cuda, 'amp'):
                self.logger.info("GPU memory optimizations applied")
                
        except Exception as e:
            self.logger.warning(f"GPU optimization failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics"""
        if self.device != "cuda" or not torch.cuda.is_available():
            return {}
            
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'utilization': allocated / torch.cuda.get_device_properties(0).total_memory * 1024**3
            }
        except Exception:
            return {}
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


class ResponseCache:
    """LRU cache for LLM responses with safety validation"""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.logger = logging.getLogger(__name__)
    
    def _hash_input(self, prompt: str, context: str = "") -> str:
        """Create hash key for caching"""
        combined = f"{prompt}|{context}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, prompt: str, context: str = "") -> Optional[Any]:
        """Get cached response if available"""
        key = self._hash_input(prompt, context)
        
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, prompt: str, response: Any, context: str = ""):
        """Cache response"""
        key = self._hash_input(prompt, context)
        
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class BatchProcessor:
    """Batch processing for multiple safety evaluations"""
    
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    def process_batch(self, prompts: List[str], model, tokenizer, device: str) -> List[str]:
        """Process multiple prompts in batch for efficiency"""
        if not prompts:
            return []
        
        try:
            # Tokenize all prompts
            inputs = tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate responses in batch
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode responses
            responses = []
            for i, output in enumerate(outputs):
                response = tokenizer.decode(output, skip_special_tokens=True)
                # Extract new content
                prompt_len = len(tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True))
                new_content = response[prompt_len:].strip()
                responses.append(new_content)
            
            return responses
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return ["Error: Batch processing failed"] * len(prompts)


class MemoryMonitor:
    """Monitor system memory usage"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.baseline_memory = self.get_memory_usage()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def get_memory_delta(self) -> Dict[str, float]:
        """Get memory usage change from baseline"""
        current = self.get_memory_usage()
        
        return {
            'rss_delta_mb': current['rss_mb'] - self.baseline_memory['rss_mb'],
            'vms_delta_mb': current['vms_mb'] - self.baseline_memory['vms_mb'],
            'percent_delta': current['percent'] - self.baseline_memory['percent']
        }
    
    def check_memory_pressure(self, threshold_mb: float = 1000) -> bool:
        """Check if memory usage is too high"""
        current = self.get_memory_usage()
        return current['rss_mb'] > threshold_mb or current['available_mb'] < 500
    
    def reset_baseline(self):
        """Reset memory baseline"""
        self.baseline_memory = self.get_memory_usage()


class PerformanceProfiler:
    """Profile performance of safety operations"""
    
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.metrics[operation] = {
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024
        }
    
    def end_timer(self, operation: str) -> Dict[str, float]:
        """End timing and return metrics"""
        if operation not in self.metrics:
            return {}
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        start_data = self.metrics[operation]
        
        metrics = {
            'execution_time': end_time - start_data['start_time'],
            'memory_delta': end_memory - start_data['start_memory'],
            'timestamp': end_time
        }
        
        # Store for analysis
        self.metrics[f"{operation}_completed"] = metrics
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        completed_ops = {k: v for k, v in self.metrics.items() if k.endswith('_completed')}
        
        if not completed_ops:
            return {}
        
        execution_times = [v['execution_time'] for v in completed_ops.values()]
        memory_deltas = [v['memory_delta'] for v in completed_ops.values()]
        
        return {
            'total_operations': len(completed_ops),
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'max_execution_time': max(execution_times),
            'min_execution_time': min(execution_times),
            'avg_memory_delta': sum(memory_deltas) / len(memory_deltas),
            'max_memory_delta': max(memory_deltas)
        }
    
    def clear_metrics(self):
        """Clear all metrics"""
        self.metrics.clear()


@lru_cache(maxsize=64)
def cached_safety_evaluation(prompt_hash: str, safety_constraints: str) -> Dict[str, Any]:
    """Cached safety evaluation for repeated prompts"""
    # This would contain the actual safety evaluation logic
    # For now, return a placeholder
    return {
        'safety_score': 0.8,
        'violations': [],
        'cached': True
    }


def optimize_torch_settings():
    """Apply global PyTorch optimizations"""
    try:
        # Enable optimized attention if available
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable memory efficient attention
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        # Set number of threads for CPU operations
        torch.set_num_threads(min(4, torch.get_num_threads()))
        
        logging.getLogger(__name__).info("PyTorch optimizations applied")
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"PyTorch optimization failed: {e}")