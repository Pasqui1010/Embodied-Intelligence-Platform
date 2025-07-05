#!/usr/bin/env python3
"""
GPU-Optimized Safety-Embedded LLM

This module implements GPU acceleration for the Safety-Embedded LLM with
advanced memory management, batch processing, and performance optimization.
"""

import torch
import torch.nn.functional as F
import gc
import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from contextlib import contextmanager

from transformers import AutoTokenizer, AutoModelForCausalLM
from .safety_embedded_llm import SafetyEmbeddedLLM, SafetyToken, SafetyConstraint, SafetyEmbeddedResponse


@dataclass
class GPUConfig:
    """GPU configuration for optimization"""
    device: str = "auto"
    memory_fraction: float = 0.8
    batch_size: int = 4
    max_memory_mb: int = 8192
    enable_mixed_precision: bool = True
    enable_kernel_fusion: bool = True
    enable_memory_pooling: bool = True


@dataclass
class BatchRequest:
    """Batch request for GPU processing"""
    id: str
    command: str
    context: str = ""
    priority: int = 1
    timestamp: float = 0.0


@dataclass
class BatchResponse:
    """Batch response from GPU processing"""
    id: str
    response: SafetyEmbeddedResponse
    processing_time: float
    gpu_memory_used: float


class MemoryManager:
    """Advanced GPU memory management with pooling and garbage collection"""
    
    def __init__(self, max_memory_mb: int = 8192):
        self.max_memory_mb = max_memory_mb
        self.memory_pool = {}
        self.allocated_memory = 0
        self.lock = threading.Lock()
        
    @contextmanager
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float16):
        """Context manager for tensor allocation with automatic cleanup"""
        tensor = None
        try:
            tensor = torch.empty(shape, dtype=dtype, device='cuda')
            with self.lock:
                self.allocated_memory += tensor.element_size() * tensor.numel() / (1024 * 1024)
            yield tensor
        finally:
            if tensor is not None:
                del tensor
                torch.cuda.empty_cache()
                with self.lock:
                    self.allocated_memory -= tensor.element_size() * tensor.numel() / (1024 * 1024)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            cached = torch.cuda.memory_reserved() / (1024 * 1024)
            return {
                'allocated_mb': allocated,
                'cached_mb': cached,
                'pool_allocated_mb': self.allocated_memory
            }
        return {'allocated_mb': 0, 'cached_mb': 0, 'pool_allocated_mb': 0}
    
    def cleanup(self):
        """Force memory cleanup"""
        with self.lock:
            self.memory_pool.clear()
        torch.cuda.empty_cache()
        gc.collect()


class GPUOptimizedSafetyLLM(SafetyEmbeddedLLM):
    """
    GPU-Optimized Safety-Embedded LLM with advanced performance features
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", gpu_config: Optional[GPUConfig] = None):
        """
        Initialize GPU-optimized Safety-Embedded LLM
        
        Args:
            model_name: Hugging Face model name
            gpu_config: GPU configuration for optimization
        """
        # Initialize GPU configuration
        self.gpu_config = gpu_config or GPUConfig()
        self.device = self._determine_device()
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(self.gpu_config.max_memory_mb)
        
        # Batch processing infrastructure
        self.batch_queue = queue.PriorityQueue()
        self.batch_processor = None
        self.batch_processing = False
        
        # Performance monitoring
        self.performance_metrics = {
            'total_requests': 0,
            'gpu_requests': 0,
            'cpu_fallback_requests': 0,
            'average_gpu_time': 0.0,
            'average_cpu_time': 0.0,
            'memory_usage_history': []
        }
        
        # Initialize parent class
        super().__init__(model_name, self.device)
        
        # Start batch processor
        self._start_batch_processor()
        
        self.logger.info(f"GPU-Optimized Safety-Embedded LLM initialized on {self.device}")
    
    def _determine_device(self) -> str:
        """Determine optimal device for GPU acceleration"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        
        # Check available GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
        if gpu_memory < 4:  # Less than 4GB
            self.logger.warning(f"GPU memory ({gpu_memory:.1f}GB) may be insufficient for large models")
            return "cpu"
        
        # Set CUDA memory fraction
        torch.cuda.set_per_process_memory_fraction(self.gpu_config.memory_fraction)
        
        return "cuda"
    
    def _load_model(self):
        """Load model with GPU optimization"""
        try:
            self.logger.info(f"Loading model: {self.model_name} on {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._add_safety_tokens()
            
            # Load model with GPU optimization
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.gpu_config.enable_mixed_precision else torch.float32,
                    device_map="auto" if self.gpu_config.enable_kernel_fusion else None
                )
                
                # Enable optimizations
                if self.gpu_config.enable_mixed_precision:
                    self.model = self.model.half()
                
                # Compile model for better performance (PyTorch 2.0+)
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    self.logger.info("Model compiled for GPU optimization")
                except Exception as e:
                    self.logger.warning(f"Model compilation failed: {e}")
            else:
                # CPU fallback
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.model = self.model.to(self.device)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            # Fallback to mock model
            self._initialize_mock_model()
    
    def _start_batch_processor(self):
        """Start batch processing thread"""
        if self.device == "cuda":
            self.batch_processing = True
            self.batch_processor = threading.Thread(target=self._batch_worker, daemon=True)
            self.batch_processor.start()
            self.logger.info("Batch processor started")
    
    def _batch_worker(self):
        """Batch processing worker thread"""
        while self.batch_processing:
            try:
                # Collect batch requests
                batch_requests = []
                batch_start_time = time.time()
                
                # Wait for first request
                try:
                    first_request = self.batch_queue.get(timeout=0.1)
                    batch_requests.append(first_request)
                except queue.Empty:
                    continue
                
                # Collect additional requests for batch processing
                while (len(batch_requests) < self.gpu_config.batch_size and 
                       time.time() - batch_start_time < 0.05):  # 50ms batching window
                    try:
                        request = self.batch_queue.get_nowait()
                        batch_requests.append(request)
                    except queue.Empty:
                        break
                
                # Process batch
                if batch_requests:
                    self._process_batch(batch_requests)
                    
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
    
    def _process_batch(self, batch_requests: List[BatchRequest]):
        """Process a batch of requests"""
        try:
            # Prepare batch inputs
            commands = [req.command for req in batch_requests]
            contexts = [req.context for req in batch_requests]
            
            # Create safety-aware prompts
            prompts = [self._create_safety_aware_prompt(cmd, ctx) for cmd, ctx in zip(commands, contexts)]
            
            # Tokenize batch
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate responses with GPU optimization
            with torch.no_grad():
                start_time = time.time()
                
                # Use memory manager for tensor allocation
                with self.memory_manager.allocate_tensor(inputs['input_ids'].shape, torch.long):
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 100,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                processing_time = time.time() - start_time
            
            # Process responses
            for i, request in enumerate(batch_requests):
                response_text = self.tokenizer.decode(outputs[i][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                # Create safety-embedded response
                safety_tokens = self._extract_safety_tokens(response_text)
                safety_score = self._calculate_safety_score(response_text, safety_tokens)
                violations = self._detect_safety_violations(response_text, safety_tokens)
                
                response = SafetyEmbeddedResponse(
                    content=response_text,
                    safety_score=safety_score,
                    safety_tokens_used=safety_tokens,
                    violations_detected=violations,
                    confidence=0.9,
                    execution_time=processing_time
                )
                
                # Update performance metrics
                self._update_performance_metrics(processing_time, True)
                
                # Store result (in a real implementation, this would be returned to the caller)
                self.logger.info(f"Batch processed request {request.id} in {processing_time:.3f}s")
                
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            # Fallback to individual processing
            for request in batch_requests:
                try:
                    self._process_single_request(request)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback processing failed for {request.id}: {fallback_error}")
    
    def _process_single_request(self, request: BatchRequest):
        """Process a single request (fallback method)"""
        try:
            # Sanitize input
            sanitized_command = self._sanitize_input(request.command)
            
            # Create safety-aware prompt
            prompt = self._create_safety_aware_prompt(sanitized_command, request.context)
            
            # Generate response
            start_time = time.time()
            response_text = self._generate_real_response(prompt)
            processing_time = time.time() - start_time
            
            # Create safety-embedded response
            safety_tokens = self._extract_safety_tokens(response_text)
            safety_score = self._calculate_safety_score(response_text, safety_tokens)
            violations = self._detect_safety_violations(response_text, safety_tokens)
            
            response = SafetyEmbeddedResponse(
                content=response_text,
                safety_score=safety_score,
                safety_tokens_used=safety_tokens,
                violations_detected=violations,
                confidence=0.9,
                execution_time=processing_time
            )
            
            # Update performance metrics
            self._update_performance_metrics(processing_time, False)
            
            self.logger.info(f"Single request {request.id} processed in {processing_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Single request processing failed for {request.id}: {e}")
    
    def generate_safe_response(self, command: str, context: str = "", priority: int = 1) -> SafetyEmbeddedResponse:
        """
        Generate safe response with GPU optimization
        
        Args:
            command: Natural language command
            context: Additional context
            priority: Request priority (1=high, 2=medium, 3=low)
        
        Returns:
            SafetyEmbeddedResponse with safety analysis
        """
        try:
            # Sanitize input
            command = self._sanitize_input(command)
            
            # Check if GPU batch processing is available
            if self.device == "cuda" and self.batch_processing:
                # Add to batch queue
                request = BatchRequest(
                    id=f"req_{int(time.time() * 1000)}",
                    command=command,
                    context=context,
                    priority=priority,
                    timestamp=time.time()
                )
                
                self.batch_queue.put((priority, request))
                self.performance_metrics['gpu_requests'] += 1
                
                # For now, fall back to immediate processing
                # In a full implementation, this would wait for batch completion
                return self._generate_immediate_response(command, context)
            else:
                # CPU fallback
                self.performance_metrics['cpu_fallback_requests'] += 1
                return self._generate_immediate_response(command, context)
                
        except Exception as e:
            self.logger.error(f"GPU-optimized response generation failed: {e}")
            # Fallback to parent implementation
            return super().generate_safe_response(command, context)
    
    def _generate_immediate_response(self, command: str, context: str = "") -> SafetyEmbeddedResponse:
        """Generate immediate response (fallback method)"""
        start_time = time.time()
        
        # Create safety-aware prompt
        prompt = self._create_safety_aware_prompt(command, context)
        
        # Generate response
        if self.device == "cuda":
            response_text = self._generate_gpu_response(prompt)
        else:
            response_text = self._generate_real_response(prompt)
        
        processing_time = time.time() - start_time
        
        # Extract safety information
        safety_tokens = self._extract_safety_tokens(response_text)
        safety_score = self._calculate_safety_score(response_text, safety_tokens)
        violations = self._detect_safety_violations(response_text, safety_tokens)
        
        # Update performance metrics
        self._update_performance_metrics(processing_time, self.device == "cuda")
        
        return SafetyEmbeddedResponse(
            content=response_text,
            safety_score=safety_score,
            safety_tokens_used=safety_tokens,
            violations_detected=violations,
            confidence=0.9,
            execution_time=processing_time
        )
    
    def _generate_gpu_response(self, prompt: str) -> str:
        """Generate response using GPU optimization"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate with GPU optimization
            with torch.no_grad():
                with self.memory_manager.allocate_tensor(inputs['input_ids'].shape, torch.long):
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 100,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response_text
            
        except Exception as e:
            self.logger.error(f"GPU response generation failed: {e}")
            # Fallback to CPU
            return self._generate_real_response(prompt)
    
    def _update_performance_metrics(self, processing_time: float, gpu_used: bool):
        """Update performance metrics"""
        self.performance_metrics['total_requests'] += 1
        
        if gpu_used:
            # Update GPU average time
            current_avg = self.performance_metrics['average_gpu_time']
            count = self.performance_metrics['gpu_requests']
            self.performance_metrics['average_gpu_time'] = (current_avg * (count - 1) + processing_time) / count
        else:
            # Update CPU average time
            current_avg = self.performance_metrics['average_cpu_time']
            count = self.performance_metrics['cpu_fallback_requests']
            self.performance_metrics['average_cpu_time'] = (current_avg * (count - 1) + processing_time) / count
        
        # Update memory usage history
        memory_usage = self.memory_manager.get_memory_usage()
        self.performance_metrics['memory_usage_history'].append({
            'timestamp': time.time(),
            'usage': memory_usage
        })
        
        # Keep only last 100 entries
        if len(self.performance_metrics['memory_usage_history']) > 100:
            self.performance_metrics['memory_usage_history'] = self.performance_metrics['memory_usage_history'][-100:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = self.performance_metrics.copy()
        metrics['current_memory_usage'] = self.memory_manager.get_memory_usage()
        metrics['device'] = self.device
        metrics['gpu_config'] = self.gpu_config.__dict__
        return metrics
    
    def optimize_memory(self):
        """Optimize memory usage"""
        self.memory_manager.cleanup()
        self.logger.info("Memory optimization completed")
    
    def shutdown(self):
        """Shutdown GPU-optimized LLM"""
        self.batch_processing = False
        if self.batch_processor:
            self.batch_processor.join(timeout=5.0)
        
        self.memory_manager.cleanup()
        self.logger.info("GPU-optimized LLM shutdown completed") 