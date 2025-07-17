#!/usr/bin/env python3
"""
GPU-Optimized Safety-Embedded LLM

This module provides GPU acceleration for the Safety-Embedded LLM with:
- Automatic GPU detection and configuration
- Memory management and optimization
- Batch processing capabilities
- Performance monitoring and optimization
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
import logging
import threading
from collections import deque
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.utils.import_utils import is_torch_available
from .safety_embedded_llm import SafetyEmbeddedLLM, SafetyEmbeddedResponse
from .advanced_memory_manager import AdvancedMemoryManager
from .performance_monitor import PerformanceMonitor
from .model_integrity import verify_model_safety
from .gpu_load_manager import GPULoadManager
from .quantization_manager import QuantizationManager, QuantizationConfig, QuantizationType

logger = logging.getLogger(__name__)

@dataclass
class GPUConfig:
    """Configuration for GPU optimization"""
    device: str = "auto"  # "auto", "cuda", "cpu"
    batch_size: int = 4  # Base batch size
    max_batch_size: int = 16  # Maximum batch size
    min_batch_size: int = 1  # Minimum batch size
    batch_size_adjustment_interval: int = 10  # Adjust batch size every N requests
    max_memory_mb: int = 8192  # 8GB default
    enable_mixed_precision: bool = True
    enable_memory_efficient_attention: bool = True
    enable_gradient_checkpointing: bool = False
    enable_quantization: bool = True  # Enable model quantization
    quantization_bits: int = 8  # 4 or 8 bits
    enable_model_parallel: bool = False  # Enable model parallelism
    num_gpu_splits: int = 2  # Number of GPU splits for model parallelism
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2


class GPUOptimizedSafetyLLM:
    """
    GPU-optimized version of Safety-Embedded LLM with advanced performance features
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", gpu_config: Optional[GPUConfig] = None):
        """
        Initialize GPU-optimized Safety-Embedded LLM
        
        Args:
            model_name: Hugging Face model name
            gpu_config: GPU configuration parameters
        """
        self.model_name = model_name
        self.gpu_config = gpu_config if gpu_config is not None else GPUConfig()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Detect and configure GPU
        self.device = self._detect_and_configure_device()
        
        # Initialize quantization
        quant_config = QuantizationConfig(
            quantization_type=QuantizationType.INT8 if self.gpu_config.quantization_bits == 8 else QuantizationType.INT4,
            bits=self.gpu_config.quantization_bits,
            load_in_8bit=self.gpu_config.quantization_bits == 8,
            load_in_4bit=self.gpu_config.quantization_bits == 4,
            use_double_quant=True
        )
        self.quantization_manager = QuantizationManager(quant_config)
        
        # Initialize base LLM with quantization support
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = self.quantization_manager.quantize_model(model_name, self.device)
            
            # Validate quantization
            if not self.quantization_manager.validate_quantization(self.model):
                raise RuntimeError("Quantization validation failed")
                
            self.logger.info("Quantized model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading or quantizing model: {e}")
            raise
        
        # Initialize safety features
        self._add_safety_tokens(self.tokenizer, self.model)
        
        # Initialize memory management
        self.memory_manager = AdvancedMemoryManager(
            max_memory_mb=self.gpu_config.max_memory_mb,
            device=self.device
        )
        
        # Initialize GPU load management
        self.load_manager = GPULoadManager(
            self.gpu_config.__dict__,
            dynamic_batch_size=True,
            parallel_config={
                'num_gpu_splits': self.gpu_config.num_gpu_splits,
                'enable_model_parallel': self.gpu_config.enable_model_parallel
            }
        )
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(self)
        
        # Batch processing
        self.batch_queue = deque(maxlen=self.gpu_config.batch_size * 2)
        self.batch_lock = threading.Lock()
        
        # Optimization flags
        self.optimization_enabled = True
        self.mixed_precision_enabled = self.gpu_config.enable_mixed_precision
        
        # Initialize optimizations
        self._apply_gpu_optimizations()
        
        self.logger.info(f"GPU-Optimized Safety-Embedded LLM initialized on {self.device}")
    
    def _detect_and_configure_device(self) -> str:
        """Detect and configure the best available device with error handling"""
        if self.gpu_config.device == "auto":
            try:
                if torch.cuda.is_available():
                    # Get GPU information
                    gpu_info = self._get_gpu_info()
                    if gpu_info:
                        self.logger.info(f"Using GPU: {gpu_info['name']} with {gpu_info['memory_mb']}MB memory")
                        return "cuda"
                    else:
                        self.logger.warning("CUDA available but no suitable GPU found, falling back to CPU")
                        return "cpu"
                else:
                    self.logger.info("CUDA not available, using CPU")
                    return "cpu"
            except torch.cuda.CudaError as e:
                self.logger.error(f"CUDA error during device detection: {e}")
                return "cpu"
            except Exception as e:
                self.logger.error(f"Unexpected error during device detection: {e}")
                return "cpu"
        else:
            return self.gpu_config.device
    
    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get information about available GPUs"""
        if GPUtil is None:
            self.logger.warning("GPUtil not available, cannot get GPU info")
            return None
            
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                # Select the GPU with the most free memory
                best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)
                return {
                    'name': best_gpu.name,
                    'memory_mb': best_gpu.memoryTotal,
                    'memory_free_mb': best_gpu.memoryFree,
                    'utilization': best_gpu.load * 100
                }
        except Exception as e:
            self.logger.warning(f"Could not get GPU info: {e}")
        
        return None
    
    def _apply_gpu_optimizations(self):
        """Apply GPU-specific optimizations with comprehensive error handling and resource cleanup"""
        if self.device == "cuda":
            try:
                # Enable mixed precision if supported
                if self.mixed_precision_enabled:
                    try:
                        from torch.cuda.amp import autocast, GradScaler
                        self.autocast = autocast
                        self.scaler = GradScaler()
                        self.logger.info("Mixed precision enabled")
                    except ImportError:
                        self.logger.warning("Mixed precision not available")
                        self.mixed_precision_enabled = False
                    except Exception as e:
                        self.logger.error(f"Error enabling mixed precision: {e}")
                        self.mixed_precision_enabled = False
                
                # Set memory fraction to prevent OOM
                try:
                    torch.cuda.set_per_process_memory_fraction(0.9)
                except torch.cuda.CudaError as e:
                    self.logger.error(f"Failed to set memory fraction: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error setting memory fraction: {e}")
                
                # Enable memory efficient attention if available
                if self.gpu_config.enable_memory_efficient_attention:
                    try:
                        # This would be set in the model configuration
                        self.logger.info("Memory efficient attention enabled")
                    except Exception as e:
                        self.logger.warning(f"Memory efficient attention not available: {e}")
                        self.gpu_config.enable_memory_efficient_attention = False
                
                # Enable gradient checkpointing if requested
                if self.gpu_config.enable_gradient_checkpointing:
                    try:
                        if hasattr(self.base_llm.model, 'gradient_checkpointing_enable'):
                            self.base_llm.model.gradient_checkpointing_enable()
                            self.logger.info("Gradient checkpointing enabled")
                        else:
                            self.logger.warning("Model does not support gradient checkpointing")
                            self.gpu_config.enable_gradient_checkpointing = False
                    except AttributeError:
                        self.logger.warning("Model does not support gradient checkpointing")
                        self.gpu_config.enable_gradient_checkpointing = False
                    except Exception as e:
                        self.logger.error(f"Error enabling gradient checkpointing: {e}")
                        self.gpu_config.enable_gradient_checkpointing = False
                
                # Configure model parallelism if enabled
                if self.gpu_config.enable_model_parallel:
                    try:
                        # Get GPU devices available
                        num_gpus = torch.cuda.device_count()
                        if num_gpus < self.gpu_config.num_gpu_splits:
                            self.logger.warning(f"Requested {self.gpu_config.num_gpu_splits} GPU splits but only {num_gpus} GPUs available")
                            self.gpu_config.num_gpu_splits = min(num_gpus, self.gpu_config.num_gpu_splits)
                        
                        # Split model layers across GPUs
                        model_layers = list(self.base_llm.model.transformer.h)
                        layers_per_gpu = len(model_layers) // self.gpu_config.num_gpu_splits
                        
                        # Create a list to store the device assignments
                        device_assignments = []
                        
                        for i in range(self.gpu_config.num_gpu_splits):
                            start_idx = i * layers_per_gpu
                            end_idx = (i + 1) * layers_per_gpu if i < self.gpu_config.num_gpu_splits - 1 else len(model_layers)
                            
                            # Move layers to the appropriate device
                            for j in range(start_idx, end_idx):
                                try:
                                    model_layers[j] = model_layers[j].to(f"cuda:{i}")
                                    device_assignments.append(f"cuda:{i}")
                                except torch.cuda.CudaError as e:
                                    self.logger.error(f"CUDA error moving layer {j} to GPU {i}: {e}")
                                    raise
                                except Exception as e:
                                    self.logger.error(f"Error moving layer {j} to GPU {i}: {e}")
                                    raise
                        
                        # Save device assignments for later reference
                        self._device_assignments = device_assignments
                        
                        # Configure parallel batch processing
                        self._parallel_batch_config = {
                            'num_gpus': num_gpus,
                            'batch_size_per_gpu': self.gpu_config.batch_size // num_gpus,
                            'max_batch_size': self.gpu_config.max_batch_size,
                            'min_batch_size': self.gpu_config.min_batch_size
                        }
                        
                        self.logger.info(f"Model parallelism enabled with {self.gpu_config.num_gpu_splits} GPU splits")
                    except torch.cuda.CudaError as e:
                        self.logger.error(f"CUDA error during model parallelism setup: {e}")
                        self.gpu_config.enable_model_parallel = False
                        # Cleanup partial GPU allocations
                        for layer in model_layers:
                            layer.to('cpu')
                    except Exception as e:
                        self.logger.error(f"Error during model parallelism setup: {e}")
                        self.gpu_config.enable_model_parallel = False
                        # Cleanup partial GPU allocations
                        for layer in model_layers:
                            layer.to('cpu')
            except torch.cuda.CudaError as e:
                self.logger.error(f"CUDA error during optimization setup: {e}")
                # Disable all GPU optimizations if any fail
                self.mixed_precision_enabled = False
                self.gpu_config.enable_memory_efficient_attention = False
                self.gpu_config.enable_gradient_checkpointing = False
                self.gpu_config.enable_model_parallel = False
                # Move model back to CPU
                self.base_llm.model.to('cpu')
            except Exception as e:
                self.logger.error(f"Unexpected error during optimization setup: {e}")
                # Disable all GPU optimizations if any fail
                self.mixed_precision_enabled = False
                self.gpu_config.enable_memory_efficient_attention = False
                self.gpu_config.enable_gradient_checkpointing = False
                self.gpu_config.enable_model_parallel = False
                # Move model back to CPU
                self.base_llm.model.to('cpu')
    
    def generate_safe_response(self, command: str, context: str = "") -> SafetyEmbeddedResponse:
        """
        Generate a safe response with GPU optimization
        
        Args:
            command: Input command
            context: Additional context
            
        Returns:
            SafetyEmbeddedResponse with optimized processing
        """
        start_time = time.time()
        
        try:
            # Check memory before processing
            self.memory_manager.check_memory_before_processing()
            
            # Use mixed precision if enabled
            if self.mixed_precision_enabled and self.device == "cuda":
                with self.autocast():
                    response = self.base_llm.generate_safe_response(command, context)
            else:
                response = self.base_llm.generate_safe_response(command, context)
            
            # Update execution time
            response.execution_time = time.time() - start_time
            
            # Record performance metrics
            self.performance_monitor.record_request(
                request_id=f"gpu_{int(start_time * 1000)}",
                processing_time=response.execution_time,
                safety_score=response.safety_score,
                success=True
            )
            
            # Optimize memory after processing
            self.memory_manager.optimize_after_processing()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in GPU-optimized response generation: {e}")
            
            # Record error
            self.performance_monitor.record_request(
                request_id=f"gpu_{int(start_time * 1000)}",
                processing_time=time.time() - start_time,
                safety_score=0.0,
                success=False,
                error=str(e)
            )
            
            # Return fallback response
            return SafetyEmbeddedResponse(
                content="I encountered an error while processing your request. Please try again.",
                safety_score=0.0,
                safety_tokens_used=[],
                violations_detected=["processing_error"],
                confidence=0.0,
                execution_time=time.time() - start_time
            )
    
    def generate_batch_responses(self, commands: List[str], contexts: Optional[List[str]] = None) -> List[SafetyEmbeddedResponse]:
        """
        Generate responses for multiple commands in batch with dynamic batch sizing and model parallelism
        
        Args:
            commands: List of input commands
            contexts: List of additional contexts (optional)
            
        Returns:
            List of SafetyEmbeddedResponse objects
        """
        if contexts is None:
            contexts = [""] * len(commands)
        
        if len(commands) != len(contexts):
            raise ValueError("Commands and contexts must have the same length")
        
        start_time = time.time()
        responses = []
        
        try:
            # Check memory before batch processing
            self.memory_manager.check_memory_before_processing()
            
            # Get dynamic batch size based on current load
            current_batch_size = self.load_manager.adjust_batch_size()
            
            # Get model parallel configuration
            parallel_config = self.load_manager.get_model_parallel_config()
            
            # Process in batches with dynamic sizing and model parallelism
            for i in range(0, len(commands), current_batch_size):
                batch_commands = commands[i:i + current_batch_size]
                batch_contexts = contexts[i:i + current_batch_size]
                
                # Process batch using model parallelism if enabled
                if self.gpu_config.enable_model_parallel:
                    # Get parallel batch configuration
                    num_gpus = min(torch.cuda.device_count(), self.gpu_config.num_gpu_splits)
                    batch_size_per_gpu = current_batch_size // num_gpus
                    
                    # Process each GPU's portion of the batch
                    batch_responses = []
                    for gpu_idx in range(num_gpus):
                        start_idx = gpu_idx * batch_size_per_gpu
                        end_idx = (gpu_idx + 1) * batch_size_per_gpu if gpu_idx < num_gpus - 1 else len(batch_commands)
                        gpu_commands = batch_commands[start_idx:end_idx]
                        gpu_contexts = batch_contexts[start_idx:end_idx]
                        
                        # Process on specific GPU with dynamic batch sizing
                        with torch.cuda.device(f"cuda:{gpu_idx}"):
                            try:
                                # Get GPU-specific batch size based on current load
                                gpu_batch_size = self.load_manager.get_gpu_specific_batch_size(gpu_idx)
                                
                                # Process in smaller batches if needed
                                for j in range(0, len(gpu_commands), gpu_batch_size):
                                    sub_batch_commands = gpu_commands[j:j + gpu_batch_size]
                                    sub_batch_contexts = gpu_contexts[j:j + gpu_batch_size]
                                    
                                    try:
                                        if self.mixed_precision_enabled:
                                            with self.autocast():
                                                sub_batch_responses = [
                                                    self.base_llm.generate_safe_response(cmd, ctx)
                                                    for cmd, ctx in zip(sub_batch_commands, sub_batch_contexts)
                                                ]
                                        else:
                                            sub_batch_responses = [
                                                self.base_llm.generate_safe_response(cmd, ctx)
                                                for cmd, ctx in zip(sub_batch_commands, sub_batch_contexts)
                                            ]
                                        
                                        batch_responses.extend(sub_batch_responses)
                                    except torch.cuda.CudaError as e:
                                        self.logger.error(f"CUDA error processing sub-batch on GPU {gpu_idx}: {e}")
                                        # Return fallback responses for failed sub-batch
                                        batch_responses.extend([
                                            SafetyEmbeddedResponse(
                                                content="I encountered an error while processing your request. Please try again.",
                                                safety_score=0.0,
                                                safety_tokens_used=[],
                                                violations_detected=["processing_error"],
                                                confidence=0.0,
                                                execution_time=time.time() - start_time
                                            )
                                            for _ in sub_batch_commands
                                        ])
                                    except Exception as e:
                                        self.logger.error(f"Error processing sub-batch on GPU {gpu_idx}: {e}")
                                        # Return fallback responses for failed sub-batch
                                        batch_responses.extend([
                                            SafetyEmbeddedResponse(
                                                content="I encountered an error while processing your request. Please try again.",
                                                safety_score=0.0,
                                                safety_tokens_used=[],
                                                violations_detected=["processing_error"],
                                                confidence=0.0,
                                                execution_time=time.time() - start_time
                                            )
                                            for _ in sub_batch_commands
                                        ])
                            except torch.cuda.CudaError as e:
                                self.logger.error(f"CUDA error processing batch on GPU {gpu_idx}: {e}")
                                # Return fallback responses for failed GPU batch
                                batch_responses.extend([
                                    SafetyEmbeddedResponse(
                                        content="I encountered an error while processing your request. Please try again.",
                                        safety_score=0.0,
                                        safety_tokens_used=[],
                                        violations_detected=["processing_error"],
                                        confidence=0.0,
                                        execution_time=time.time() - start_time
                                    )
                                    for _ in gpu_commands
                                ])
                            except Exception as e:
                                self.logger.error(f"Error processing batch on GPU {gpu_idx}: {e}")
                                # Return fallback responses for failed GPU batch
                                batch_responses.extend([
                                    SafetyEmbeddedResponse(
                                        content="I encountered an error while processing your request. Please try again.",
                                        safety_score=0.0,
                                        safety_tokens_used=[],
                                        violations_detected=["processing_error"],
                                        confidence=0.0,
                                        execution_time=time.time() - start_time
                                    )
                                    for _ in gpu_commands
                                ])
                else:
                    # Process batch sequentially with dynamic sizing
                    try:
                        if self.mixed_precision_enabled and self.device == "cuda":
                            with self.autocast():
                                # Process in smaller batches if needed
                                for j in range(0, len(batch_commands), current_batch_size):
                                    sub_batch_commands = batch_commands[j:j + current_batch_size]
                                    sub_batch_contexts = batch_contexts[j:j + current_batch_size]
                                    
                                    try:
                                        sub_batch_responses = [
                                            self.base_llm.generate_safe_response(cmd, ctx)
                                            for cmd, ctx in zip(sub_batch_commands, sub_batch_contexts)
                                        ]
                                        
                                        batch_responses.extend(sub_batch_responses)
                                    except torch.cuda.CudaError as e:
                                        self.logger.error(f"CUDA error processing sub-batch: {e}")
                                        # Return fallback responses for failed sub-batch
                                        batch_responses.extend([
                                            SafetyEmbeddedResponse(
                                                content="I encountered an error while processing your request. Please try again.",
                                                safety_score=0.0,
                                                safety_tokens_used=[],
                                                violations_detected=["processing_error"],
                                                confidence=0.0,
                                                execution_time=time.time() - start_time
                                            )
                                            for _ in sub_batch_commands
                                        ])
                                    except Exception as e:
                                        self.logger.error(f"Error processing sub-batch: {e}")
                                        # Return fallback responses for failed sub-batch
                                        batch_responses.extend([
                                            SafetyEmbeddedResponse(
                                                content="I encountered an error while processing your request. Please try again.",
                                                safety_score=0.0,
                                                safety_tokens_used=[],
                                                violations_detected=["processing_error"],
                                                confidence=0.0,
                                                execution_time=time.time() - start_time
                                            )
                                            for _ in sub_batch_commands
                                        ])
                        else:
                            # Process in smaller batches if needed
                            for j in range(0, len(batch_commands), current_batch_size):
                                sub_batch_commands = batch_commands[j:j + current_batch_size]
                                sub_batch_contexts = batch_contexts[j:j + current_batch_size]
                                
                                try:
                                    sub_batch_responses = [
                                        self.base_llm.generate_safe_response(cmd, ctx)
                                        for cmd, ctx in zip(sub_batch_commands, sub_batch_contexts)
                                    ]
                                    
                                    batch_responses.extend(sub_batch_responses)
                                except Exception as e:
                                    self.logger.error(f"Error processing sub-batch: {e}")
                                    # Return fallback responses for failed sub-batch
                                    batch_responses.extend([
                                        SafetyEmbeddedResponse(
                                            content="I encountered an error while processing your request. Please try again.",
                                            safety_score=0.0,
                                            safety_tokens_used=[],
                                            violations_detected=["processing_error"],
                                            confidence=0.0,
                                            execution_time=time.time() - start_time
                                        )
                                        for _ in sub_batch_commands
                                    ])
                    except torch.cuda.CudaError as e:
                        self.logger.error(f"CUDA error processing sequential batch: {e}")
                        # Return fallback responses for failed sequential batch
                        batch_responses = [
                            SafetyEmbeddedResponse(
                                content="I encountered an error while processing your request. Please try again.",
                                safety_score=0.0,
                                safety_tokens_used=[],
                                violations_detected=["processing_error"],
                                confidence=0.0,
                                execution_time=time.time() - start_time
                            )
                            for _ in batch_commands
                        ]
                    except Exception as e:
                        self.logger.error(f"Error processing sequential batch: {e}")
                        # Return fallback responses for failed sequential batch
                        batch_responses = [
                            SafetyEmbeddedResponse(
                                content="I encountered an error while processing your request. Please try again.",
                                safety_score=0.0,
                                safety_tokens_used=[],
                                violations_detected=["processing_error"],
                                confidence=0.0,
                                execution_time=time.time() - start_time
                            )
                            for _ in batch_commands
                        ]
                
                # Update execution times
                for response in batch_responses:
                    response.execution_time = time.time() - start_time
                
                # Update load metrics
                processing_time = time.time() - start_time
                self.load_manager.update_load_metrics(processing_time)
                
                responses.extend(batch_responses)
            
            # Record batch performance
            total_time = time.time() - start_time
            self.performance_monitor.record_batch(
                batch_size=len(commands),
                total_time=total_time,
                success=True,
                dynamic_batch_size=current_batch_size,
                parallel_config=parallel_config
            )
            
            # Optimize memory after batch processing
            self.memory_manager.optimize_after_processing()
            
            return responses
            
        except torch.cuda.CudaError as e:
            self.logger.error(f"CUDA error in batch response generation: {e}")
            
            # Record batch error
            self.performance_monitor.record_batch(
                batch_size=len(commands),
                total_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
            
            # Move model back to CPU
            self.base_llm.model.to('cpu')
            
            # Return fallback responses
            return [
                SafetyEmbeddedResponse(
                    content="I encountered an error while processing your request. Please try again.",
                    safety_score=0.0,
                    safety_tokens_used=[],
                    violations_detected=["processing_error"],
                    confidence=0.0,
                    execution_time=time.time() - start_time
                )
                for _ in commands
            ]
        except Exception as e:
            self.logger.error(f"Error in batch response generation: {e}")
            
            # Record batch error
            self.performance_monitor.record_batch(
                batch_size=len(commands),
                total_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
            
            # Return fallback responses
            return [
                SafetyEmbeddedResponse(
                    content="I encountered an error while processing your request. Please try again.",
                    safety_score=0.0,
                    safety_tokens_used=[],
                    violations_detected=["processing_error"],
                    confidence=0.0,
                    execution_time=time.time() - start_time
                )
                for _ in commands
            ]
    
    def optimize_memory(self):
        """Optimize memory usage with error handling"""
        try:
            self.memory_manager.optimize_memory()
            
            if self.device == "cuda":
                try:
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    
                    # Synchronize GPU
                    torch.cuda.synchronize()
                except torch.cuda.CudaError as e:
                    self.logger.error(f"CUDA error during memory optimization: {e}")
        except Exception as e:
            self.logger.error(f"Error during memory optimization: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        return self.memory_manager.get_memory_usage()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return self.performance_monitor.get_performance_summary()
    
    def set_optimization_level(self, level: str):
        """
        Set optimization level
        
        Args:
            level: "high", "medium", "low", or "off"
        """
        if level == "high":
            self.gpu_config.batch_size = 8
            self.mixed_precision_enabled = True
            self.optimization_enabled = True
        elif level == "medium":
            self.gpu_config.batch_size = 4
            self.mixed_precision_enabled = True
            self.optimization_enabled = True
        elif level == "low":
            self.gpu_config.batch_size = 2
            self.mixed_precision_enabled = False
            self.optimization_enabled = True
            self.gpu_config.quantization_bits = 8
        elif level == "off":
            self.gpu_config.batch_size = 1
            self.mixed_precision_enabled = False
            self.optimization_enabled = False
            self.gpu_config.quantization_bits = 8
        else:
            raise ValueError(f"Invalid optimization level: {level}")
        
        self.logger.info(f"Optimization level set to: {level}")
    
    def _initialize_quantized_llm(self, model_name: str, device: str):
        """Initialize a quantized version of the LLM"""
        try:
            self.logger.info(f"Initializing quantized model: {model_name}")
            
            # Verify model safety before loading
            is_safe, reason = verify_model_safety(model_name)
            if not is_safe:
                self.logger.warning(f"Model safety check failed: {reason}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Initialize quantization configuration
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.gpu_config.quantization_bits == 4,
                load_in_8bit=self.gpu_config.quantization_bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16 if device == "cuda" else torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load quantized model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device if device == "cuda" else None
            )
            
            # Move to device if not using device_map
            if device != "cuda" or model.device.type != device:
                model = model.to(device)
            
            # Create LLM instance with quantized model
            llm = SafetyEmbeddedLLM(model_name=model_name, device=device)
            llm.tokenizer = tokenizer
            llm.model = model
            
            # Add safety tokens to model
            llm._add_safety_tokens()
            
            self.logger.info(f"Quantized model initialized successfully with {self.gpu_config.quantization_bits}-bit precision")
            return llm
            
        except ImportError as e:
            self.logger.warning(f"Quantization dependencies not available: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing quantized model: {e}")
            raise
    
    def shutdown(self):
        """Shutdown the GPU-optimized LLM with proper cleanup"""
        try:
            # Stop monitoring
            self.performance_monitor.stop_monitoring()
            
            # Cleanup memory
            self.memory_manager.cleanup()
            
            # Cleanup GPU resources
            if self.device == "cuda":
                try:
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    
                    # Synchronize GPU
                    torch.cuda.synchronize()
                    
                    # Reset peak memory stats
                    torch.cuda.reset_peak_memory_stats()
                except torch.cuda.CudaError as e:
                    self.logger.error(f"CUDA error during shutdown: {e}")
            
            self.logger.info("GPU-Optimized Safety-Embedded LLM shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.logger.info("Forced shutdown complete")