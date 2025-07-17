from typing import Dict, Any, Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from dataclasses import dataclass
import time
from collections import defaultdict

class ModelParallelConfig:
    """Configuration for model parallelism"""
    def __init__(
        self,
        num_gpu_splits: int = 2,
        enable_model_parallel: bool = True,
        split_strategy: str = "balanced",
        memory_threshold: float = 0.8,
        performance_target: float = 0.9,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.num_gpu_splits = num_gpu_splits
        self.enable_model_parallel = enable_model_parallel
        self.split_strategy = split_strategy
        self.memory_threshold = memory_threshold
        self.performance_target = performance_target
        self.max_retries = max_retries
        self.retry_delay = retry_delay

class ModelParallelManager:
    """
    Manages model parallelism across multiple GPUs
    """
    
    def __init__(self, config: ModelParallelConfig):
        """
        Initialize the model parallel manager
        
        Args:
            config: Model parallel configuration
        """
        self.config = config
        self.device_map = self._create_device_map()
        self.model_parts = {}
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.stats = {
            'total_processing_time': 0.0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retries': 0,
            'avg_processing_time': 0.0,
            'memory_usage': defaultdict(float)
        }
        
    def _create_device_map(self) -> Dict[str, List[str]]:
        """
        Create device map for model parallelism
        
        Returns:
            Dictionary mapping model layers to devices
        """
        device_map = {}
        available_gpus = list(range(torch.cuda.device_count()))
        
        if not available_gpus:
            return {'cpu': ['all']}
            
        if self.config.split_strategy == "balanced":
            # Split model evenly across available GPUs
            layers = ['embeddings', 'encoder', 'decoder', 'output']
            layer_count = len(layers)
            split_size = layer_count // len(available_gpus)
            
            for i, gpu in enumerate(available_gpus):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < len(available_gpus) - 1 else layer_count
                device_map[f'cuda:{gpu}'] = layers[start_idx:end_idx]
                
        elif self.config.split_strategy == "memory":
            # Split based on available memory
            memory_per_gpu = self._get_gpu_memory()
            total_memory = sum(memory_per_gpu.values())
            
            for gpu, memory in memory_per_gpu.items():
                device_map[gpu] = []
                layers = ['embeddings', 'encoder', 'decoder', 'output']
                while layers and memory_per_gpu[gpu] > total_memory * self.config.memory_threshold:
                    layer = layers.pop(0)
                    device_map[gpu].append(layer)
                    memory_per_gpu[gpu] -= total_memory * 0.25  # Assume each layer takes ~25% of memory
                    
        return device_map
    
    def _get_gpu_memory(self) -> Dict[str, float]:
        """
        Get available memory for each GPU
        
        Returns:
            Dictionary of GPU memory in GB
        """
        if not torch.cuda.is_available():
            return {'cpu': float('inf')}
            
        memory = {}
        for i in range(torch.cuda.device_count()):
            device = f'cuda:{i}'
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # Convert to GB
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
            memory[device] = total - allocated
            
        return memory
    
    def load_model(
        self,
        model_name: str,
        tokenizer: Optional[AutoTokenizer] = None
    ) -> AutoModelForCausalLM:
        """
        Load and parallelize the model
        
        Args:
            model_name: Name of the model to load
            tokenizer: Optional tokenizer
            
        Returns:
            Parallelized model instance
        """
        try:
            if not self.config.enable_model_parallel:
                return AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
                
            # Load tokenizer if not provided
            if tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                self.tokenizer = tokenizer
                
            # Load model with parallel configuration
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Validate parallelization
            self._validate_parallelization(model)
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading parallel model: {e}")
            raise
    
    def _validate_parallelization(self, model: AutoModelForCausalLM) -> None:
        """
        Validate that model is properly parallelized
        
        Args:
            model: Model instance
        """
        try:
            # Check if model is on multiple devices
            devices = set()
            for param in model.parameters():
                devices.add(param.device)
                
            if len(devices) < len(self.device_map):
                self.logger.warning("Model not fully parallelized across available devices")
                
            # Check memory usage
            memory_usage = self._get_gpu_memory()
            for device, usage in memory_usage.items():
                if usage > self.config.memory_threshold:
                    self.logger.warning(f"High memory usage on {device}: {usage:.2f}GB")
                    
            self.logger.info("Model parallelization validated successfully")
            
        except Exception as e:
            self.logger.error(f"Error validating parallelization: {e}")
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request using parallel model
        
        Args:
            request: Request data
            
        Returns:
            Processed response
        """
        try:
            start_time = time.time()
            
            # Process request (this would be replaced with actual model inference)
            response = self._execute_request(request)
            
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            self.stats['successful_requests'] += 1
            self.stats['avg_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['successful_requests']
            )
            
            return response
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            self.logger.error(f"Error processing request: {e}")
            raise
    
    def _execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the request using parallel model
        
        Args:
            request: Request data
            
        Returns:
            Processed response
        """
        raise NotImplementedError("_execute_request must be implemented by subclass")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get model parallelism statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            'total_processing_time': self.stats['total_processing_time'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'retries': self.stats['retries'],
            'avg_processing_time': self.stats['avg_processing_time'],
            'memory_usage': dict(self.stats['memory_usage']),
            'device_map': self.device_map
        }
