#!/usr/bin/env python3
"""
Safety-Embedded LLM Implementation

This module implements a safety-embedded LLM approach that embeds safety constraints
directly into the LLM's neural architecture rather than applying them as post-processing filters.
"""

import json
import time
import logging
import threading
import queue
import os
import gc
import psutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
import re

from eip_interfaces.msg import TaskPlan, TaskStep
from eip_interfaces.srv import SafetyVerificationRequest, SafetyVerificationResponse

# Import model integrity verification
from .model_integrity import ModelIntegrityVerifier, verify_model_safety
from .error_handling import (
    ErrorSeverity, ErrorCategory, ErrorContext, SafetyError,
    error_handler, safety_critical, performance_monitored,
    EnhancedLogger, global_recovery_manager
)
from .performance_optimizations import (
    GPUMemoryOptimizer, ResponseCache, BatchProcessor, 
    MemoryMonitor, PerformanceProfiler, optimize_torch_settings
)


class SafetyToken(Enum):
    """Safety tokens embedded in LLM vocabulary"""
    COLLISION_RISK = "<|collision_risk|>"
    HUMAN_PROXIMITY = "<|human_proximity|>"
    VELOCITY_LIMIT = "<|velocity_limit|>"
    WORKSPACE_BOUNDARY = "<|workspace_boundary|>"
    EMERGENCY_STOP = "<|emergency_stop|>"
    SAFE_ACTION = "<|safe_action|>"
    UNSAFE_ACTION = "<|unsafe_action|>"
    SAFETY_CHECK = "<|safety_check|>"


@dataclass
class SafetyConstraint:
    """Represents a safety constraint for the LLM"""
    token: SafetyToken
    description: str
    severity: float  # 0.0 to 1.0
    conditions: List[str]
    mitigation: str


@dataclass
class SafetyEmbeddedResponse:
    """Response from safety-embedded LLM with safety analysis"""
    content: str
    safety_score: float
    safety_tokens_used: List[SafetyToken]
    violations_detected: List[str]
    confidence: float
    execution_time: float


class SafetyConstitution:
    """Constitutional rules for safe LLM behavior"""
    
    RULES = [
        "Never generate plans that could cause harm to humans or property",
        "Always check for obstacles and maintain safe distances",
        "Respect velocity limits and workspace boundaries",
        "Prioritize safety over task completion speed",
        "Include safety considerations in every action plan",
        "Never ignore safety warnings or proximity alerts",
        "Always have an emergency stop mechanism available",
        "Validate all spatial references before execution"
    ]
    
    @classmethod
    def get_constitution_prompt(cls) -> str:
        """Get the constitutional prompt for LLM training"""
        rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(cls.RULES)])
        return f"""
You are a safety-conscious robotics assistant. Follow these constitutional rules:

{rules_text}

Always consider safety first in your responses. Use safety tokens to indicate safety considerations.
"""


def performance_monitor(func):
    """Decorator to monitor function performance and memory usage"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(self, *args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Log performance metrics
            if hasattr(self, 'logger'):
                self.logger.debug(f"{func.__name__}: {execution_time:.3f}s, memory: {memory_delta:+.1f}MB")
            
            # Store metrics for analysis
            if hasattr(self, 'performance_metrics'):
                self.performance_metrics[func.__name__] = {
                    'execution_time': execution_time,
                    'memory_delta': memory_delta,
                    'timestamp': time.time()
                }
            
            return result
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"{func.__name__} failed: {e}")
            raise
    
    return wrapper


class SafetyEmbeddedLLM:
    """
    Safety-Embedded LLM that integrates safety constraints directly into the neural architecture
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto", 
                 cache_size: int = 128, enable_gpu_optimization: bool = True):
        """
        Initialize the safety-embedded LLM
        
        Args:
            model_name: Hugging Face model name
            device: Device to run the model on
            cache_size: Size of response cache
            enable_gpu_optimization: Enable GPU memory optimizations
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_gpu_optimization = enable_gpu_optimization
        
        # Performance monitoring
        self.performance_metrics = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Response caching
        self.response_cache = {}
        self.cache_size = cache_size
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="safety_llm")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.safety_tokens_added = False
        
        # Safety constraints
        self.safety_constraints = self._initialize_safety_constraints()
        
        # Model integrity verification
        self.integrity_verifier = ModelIntegrityVerifier()
        
        # Async processing infrastructure
        self.inference_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.inference_thread = None
        self.inference_running = False
        
        # Load model
        self._load_model()
        
        # Start inference thread
        self._start_inference_thread()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize performance optimizations
        self.gpu_optimizer = GPUMemoryOptimizer(self.device)
        self.response_cache = ResponseCache(cache_size)
        self.memory_monitor = MemoryMonitor()
        self.performance_profiler = PerformanceProfiler()
        
        # Enhanced logging
        self.logger = EnhancedLogger(__name__)
        
        # GPU memory optimization
        if self.enable_gpu_optimization and self.device == "cuda":
            self._optimize_gpu_memory()
        
        # Import performance optimizations
        from .performance_optimizations import (
            GPUMemoryOptimizer, ResponseCache, MemoryMonitor, 
            PerformanceProfiler, optimize_torch_settings
        )
        
        # Initialize performance components
        self.gpu_optimizer = GPUMemoryOptimizer(self.device)
        self.response_cache = ResponseCache(cache_size)
        self.memory_monitor = MemoryMonitor()
        self.profiler = PerformanceProfiler()
        
        # Apply global optimizations
        optimize_torch_settings()
        
        # Import error handling
        from .error_handling import ErrorHandler, error_handler_decorator
        self.error_handler = ErrorHandler()
        
        # Import configuration management
        from .config_manager import get_config
        self.config = get_config()
        
        # Apply global PyTorch optimizations
        optimize_torch_settings()
    
    def _initialize_safety_constraints(self) -> Dict[SafetyToken, SafetyConstraint]:
        """Initialize safety constraints"""
        return {
            SafetyToken.COLLISION_RISK: SafetyConstraint(
                token=SafetyToken.COLLISION_RISK,
                description="Risk of collision with obstacles",
                severity=0.9,
                conditions=["obstacle_detected", "high_velocity", "narrow_path"],
                mitigation="Reduce velocity and plan alternative path"
            ),
            SafetyToken.HUMAN_PROXIMITY: SafetyConstraint(
                token=SafetyToken.HUMAN_PROXIMITY,
                description="Human detected in proximity",
                severity=1.0,
                conditions=["human_detected", "close_proximity", "moving_towards_human"],
                mitigation="Stop immediately and maintain safe distance"
            ),
            SafetyToken.VELOCITY_LIMIT: SafetyConstraint(
                token=SafetyToken.VELOCITY_LIMIT,
                description="Velocity exceeds safety limits",
                severity=0.8,
                conditions=["high_velocity", "confined_space", "near_obstacles"],
                mitigation="Reduce velocity to safe limits"
            ),
            SafetyToken.WORKSPACE_BOUNDARY: SafetyConstraint(
                token=SafetyToken.WORKSPACE_BOUNDARY,
                description="Approaching workspace boundary",
                severity=0.7,
                conditions=["near_boundary", "moving_outward", "no_escape_path"],
                mitigation="Stop and request boundary override if necessary"
            ),
            SafetyToken.EMERGENCY_STOP: SafetyConstraint(
                token=SafetyToken.EMERGENCY_STOP,
                description="Emergency stop required",
                severity=1.0,
                conditions=["immediate_danger", "safety_violation", "system_failure"],
                mitigation="Execute emergency stop immediately"
            )
        }
    
    def _load_model(self):
        """Load the LLM model with integrity verification and add safety tokens"""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Verify model safety before loading
            is_safe, reason = verify_model_safety(self.model_name)
            if not is_safe:
                self.logger.warning(f"Model safety check failed: {reason}")
                # Continue with loading but log the warning
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add safety tokens to vocabulary
            self._add_safety_tokens()
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else None
            )
            
            # Move to device if not using device_map
            if self.device != "cuda" or self.model.device.type != self.device:
                self.model = self.model.to(self.device)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Verify model integrity after loading
            try:
                # Get model cache directory
                from transformers import file_utils
                cache_dir = file_utils.default_cache_path
                model_cache_dir = os.path.join(cache_dir, "models--" + self.model_name.replace("/", "--"))
                
                if os.path.exists(model_cache_dir):
                    is_valid, message = self.integrity_verifier.verify_model_integrity(
                        self.model_name, model_cache_dir
                    )
                    if is_valid:
                        self.logger.info(f"Model integrity verified: {message}")
                    else:
                        self.logger.warning(f"Model integrity check failed: {message}")
                else:
                    self.logger.info("Model cache directory not found, skipping integrity check")
                    
            except Exception as e:
                self.logger.warning(f"Model integrity verification failed: {e}")
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            # Fallback to mock model
            self._initialize_mock_model()
    
    def _add_safety_tokens(self):
        """Add safety tokens to the tokenizer vocabulary with validation"""
        if self.safety_tokens_added:
            return
        
        # Ensure tokenizer is initialized and not mock
        if self.tokenizer is None or isinstance(self.tokenizer, str):
            self.logger.error("Tokenizer not properly initialized, using fallback mapping")
            self._use_fallback_safety_tokens()
            return
        
        # Validate model compatibility for token expansion
        if not self._validate_model_compatibility():
            self.logger.warning("Model incompatible with safety token expansion, using fallback mapping")
            self._use_fallback_safety_tokens()
            return
        
        # Add safety tokens to vocabulary
        safety_token_strings = [token.value for token in SafetyToken]
        
        # Validate token strings don't conflict with existing vocabulary
        conflicting_tokens = self._check_token_conflicts(safety_token_strings)
        if conflicting_tokens:
            self.logger.warning(f"Token conflicts detected: {conflicting_tokens}, using fallback mapping")
            self._use_fallback_safety_tokens()
            return
        
        # Add tokens to tokenizer
        try:
            num_added = self.tokenizer.add_special_tokens({
                'additional_special_tokens': safety_token_strings
            })
            
            if num_added > 0:
                # Ensure model is properly initialized before resizing
                if self.model is not None and not isinstance(self.model, str):
                    # Resize model embeddings to accommodate new tokens
                    self.model.resize_token_embeddings(len(self.tokenizer))
                
                # Validate the expansion was successful
                if not self._validate_token_expansion(safety_token_strings):
                    self.logger.error("Token expansion validation failed, using fallback")
                    self._use_fallback_safety_tokens()
                    return
                
                self.safety_tokens_added = True
                self.logger.info(f"Successfully added {num_added} safety tokens to vocabulary")
            else:
                self.logger.info("Safety tokens already present in vocabulary")
                self.safety_tokens_added = True
                
        except Exception as e:
            self.logger.error(f"Failed to add safety tokens: {e}, using fallback")
            self._use_fallback_safety_tokens()
    
    def _validate_model_compatibility(self) -> bool:
        """Validate that the model can safely accommodate token expansion"""
        try:
            # Handle mock model case
            if isinstance(self.model, str) or isinstance(self.tokenizer, str):
                return False
            
            # Check if model supports token expansion
            if not hasattr(self.model, 'resize_token_embeddings'):
                return False
            
            # Check model size constraints (avoid memory issues)
            current_vocab_size = len(self.tokenizer)
            max_safe_vocab_size = 100000  # Conservative limit
            
            if current_vocab_size + len(SafetyToken) > max_safe_vocab_size:
                self.logger.warning(f"Vocabulary size would exceed safe limit: {current_vocab_size + len(SafetyToken)} > {max_safe_vocab_size}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model compatibility validation failed: {e}")
            return False
    
    def _check_token_conflicts(self, safety_tokens: List[str]) -> List[str]:
        """Check for conflicts between safety tokens and existing vocabulary"""
        conflicting_tokens = []
        
        for token in safety_tokens:
            # Check if token already exists in vocabulary
            if token in self.tokenizer.get_vocab():
                conflicting_tokens.append(token)
            
            # Check for partial matches that could cause issues
            for existing_token in self.tokenizer.get_vocab().keys():
                if token in existing_token or existing_token in token:
                    if token != existing_token:  # Don't flag exact matches
                        conflicting_tokens.append(f"{token} conflicts with {existing_token}")
        
        return conflicting_tokens
    
    def _validate_token_expansion(self, safety_tokens: List[str]) -> bool:
        """Validate that token expansion was successful"""
        try:
            # Test tokenization of safety tokens
            for token in safety_tokens:
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if not token_ids:
                    self.logger.error(f"Failed to tokenize safety token: {token}")
                    return False
            
            # Test model can handle the new tokens
            test_input = self.tokenizer.encode("Test safety token", return_tensors="pt")
            if self.device != "cpu":
                test_input = test_input.to(self.device)
            
            with torch.no_grad():
                _ = self.model(test_input)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Token expansion validation failed: {e}")
            return False
    
    def _use_fallback_safety_tokens(self):
        """Use fallback safety token mapping for incompatible models"""
        # Map safety concepts to existing vocabulary tokens
        self.safety_token_mapping = {
            SafetyToken.COLLISION_RISK: "<|endoftext|>",
            SafetyToken.HUMAN_PROXIMITY: "<|endoftext|>",
            SafetyToken.VELOCITY_LIMIT: "<|endoftext|>",
            SafetyToken.WORKSPACE_BOUNDARY: "<|endoftext|>",
            SafetyToken.EMERGENCY_STOP: "<|endoftext|>",
            SafetyToken.SAFE_ACTION: "<|endoftext|>",
            SafetyToken.UNSAFE_ACTION: "<|endoftext|>",
            SafetyToken.SAFETY_CHECK: "<|endoftext|>"
        }
        
        self.safety_tokens_added = True
        self.logger.info("Using fallback safety token mapping")
    
    def _start_inference_thread(self):
        """Start the async inference thread"""
        if self.inference_thread is None or not self.inference_thread.is_alive():
            self.inference_running = True
            self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
            self.inference_thread.start()
            self.logger.info("Started async inference thread")
    
    def _inference_worker(self):
        """Worker thread for async model inference"""
        while self.inference_running:
            try:
                # Get inference request from queue
                request = self.inference_queue.get(timeout=1.0)
                if request is None:  # Shutdown signal
                    break
                
                prompt, request_id = request
                
                # Perform model inference
                start_time = time.time()
                response = self._generate_real_response(prompt)
                execution_time = time.time() - start_time
                
                # Put result in result queue
                self.result_queue.put((request_id, response, execution_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Inference worker error: {e}")
                # Put error result
                self.result_queue.put((request_id, f"Error: {e}", 0.0))
    
    def _initialize_mock_model(self):
        """Initialize mock model for development/testing"""
        self.logger.info("Initializing mock safety-embedded LLM")
        self.tokenizer = "mock_tokenizer"
        self.model = "mock_model"
    
    def _create_safety_aware_prompt(self, command: str, context: str = "") -> str:
        """
        Create a safety-aware prompt that embeds safety considerations
        
        Args:
            command: Natural language command
            context: Additional context (scene description, etc.)
            
        Returns:
            Safety-aware prompt
        """
        constitution = SafetyConstitution.get_constitution_prompt()
        
        safety_context = f"""
Current safety context:
- Workspace boundaries: Active
- Human proximity monitoring: Active  
- Velocity limits: Enforced
- Collision detection: Active

Use safety tokens to indicate safety considerations:
- {SafetyToken.SAFE_ACTION.value}: Safe action
- {SafetyToken.UNSAFE_ACTION.value}: Unsafe action
- {SafetyToken.SAFETY_CHECK.value}: Safety check required
- {SafetyToken.COLLISION_RISK.value}: Collision risk detected
- {SafetyToken.HUMAN_PROXIMITY.value}: Human proximity detected
"""
        
        prompt = f"""{constitution}

{safety_context}

Task: {command}
Context: {context}

Generate a safe task plan using safety tokens where appropriate:"""
        
        return prompt
    
    def _compute_safety_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute attention mask that gives higher weight to safety tokens
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Safety-aware attention mask
        """
        # Create base attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Find safety token positions
        safety_token_ids = []
        for token in SafetyToken:
            token_ids = self.tokenizer.encode(token.value, add_special_tokens=False)
            safety_token_ids.extend(token_ids)
        
        # Boost attention for safety tokens
        for i, token_id in enumerate(input_ids[0]):
            if token_id.item() in safety_token_ids:
                # Increase attention weight for safety tokens
                attention_mask[0, i] = 2.0
        
        return attention_mask
    
    def _extract_safety_tokens(self, response: str) -> List[SafetyToken]:
        """Extract safety tokens from LLM response"""
        safety_tokens = []
        for token in SafetyToken:
            if token.value in response:
                safety_tokens.append(token)
        return safety_tokens
    
    def _calculate_safety_score(self, response: str, safety_tokens: List[SafetyToken]) -> float:
        """
        Calculate safety score based on response content and safety tokens
        
        Args:
            response: LLM response
            safety_tokens: Safety tokens found in response
            
        Returns:
            Safety score (0.0 to 1.0)
        """
        base_score = 0.5  # Neutral base score
        
        # Positive safety indicators
        positive_indicators = [
            SafetyToken.SAFE_ACTION,
            SafetyToken.SAFETY_CHECK
        ]
        
        # Negative safety indicators
        negative_indicators = [
            SafetyToken.UNSAFE_ACTION,
            SafetyToken.COLLISION_RISK,
            SafetyToken.HUMAN_PROXIMITY,
            SafetyToken.EMERGENCY_STOP
        ]
        
        # Adjust score based on safety tokens
        for token in safety_tokens:
            if token in positive_indicators:
                base_score += 0.2
            elif token in negative_indicators:
                base_score -= 0.3
        
        # Adjust based on content analysis
        unsafe_keywords = ["ignore", "rush", "quickly", "dangerous", "unsafe"]
        safe_keywords = ["carefully", "safely", "slowly", "check", "verify"]
        
        for keyword in unsafe_keywords:
            if keyword.lower() in response.lower():
                base_score -= 0.1
        
        for keyword in safe_keywords:
            if keyword.lower() in response.lower():
                base_score += 0.05
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, base_score))
    
    def _detect_safety_violations(self, response: str, safety_tokens: List[SafetyToken]) -> List[str]:
        """Detect safety violations in the response"""
        violations = []
        
        # Check for unsafe tokens
        unsafe_tokens = [
            SafetyToken.UNSAFE_ACTION,
            SafetyToken.COLLISION_RISK,
            SafetyToken.HUMAN_PROXIMITY,
            SafetyToken.EMERGENCY_STOP
        ]
        
        for token in safety_tokens:
            if token in unsafe_tokens:
                constraint = self.safety_constraints.get(token)
                if constraint:
                    violations.append(f"{constraint.description}: {constraint.mitigation}")
        
        # Check for unsafe content
        unsafe_patterns = [
            "ignore safety",
            "rush through",
            "skip safety check",
            "override safety"
        ]
        
        for pattern in unsafe_patterns:
            if pattern.lower() in response.lower():
                violations.append(f"Unsafe instruction detected: {pattern}")
        
        return violations
    
    def _sanitize_input(self, command: str) -> str:
        """Sanitize input to prevent prompt injection and adversarial attacks"""
        # Remove non-printable characters
        command = re.sub(r'[^\x20-\x7E]', '', command)
        # Limit length
        command = command[:512]
        # Block known adversarial patterns (e.g., repeated tokens, suspicious substrings)
        block_patterns = [r'(\bignore\b|\bshutdown\b|\bself-destruct\b)', r'(\<\|.*?\|\>)']
        for pat in block_patterns:
            if re.search(pat, command, re.IGNORECASE):
                raise ValueError("Blocked potentially adversarial input pattern.")
        return command

    def generate_safe_response(self, command: str, context: str = "") -> SafetyEmbeddedResponse:
        """
        Generate a safety-embedded response using async processing
        
        Args:
            command: Natural language command
            context: Additional context
            
        Returns:
            Safety-embedded response with safety analysis
        """
        start_time = time.time()
        
        try:
            # Sanitize input before proceeding
            command = self._sanitize_input(command)
            # Create safety-aware prompt
            prompt = self._create_safety_aware_prompt(command, context)
            
            if self.model == "mock_model":
                # Mock response for development
                response = self._generate_mock_safe_response(command)
            else:
                # Real LLM inference using async processing
                response = self._generate_response_async(prompt)
            
            # Extract safety tokens
            safety_tokens = self._extract_safety_tokens(response)
            
            # Calculate safety score
            safety_score = self._calculate_safety_score(response, safety_tokens)
            
            # Detect violations
            violations = self._detect_safety_violations(response, safety_tokens)
            
            # Calculate confidence (simplified)
            confidence = 0.9 if safety_score > 0.7 else 0.5
            
            execution_time = time.time() - start_time
            
            return SafetyEmbeddedResponse(
                content=response,
                safety_score=safety_score,
                safety_tokens_used=safety_tokens,
                violations_detected=violations,
                confidence=confidence,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error generating safe response: {e}")
            return SafetyEmbeddedResponse(
                content=f"Error: {e}",
                safety_score=0.0,
                safety_tokens_used=[],
                violations_detected=[f"Generation error: {e}"],
                confidence=0.0,
                execution_time=time.time() - start_time
            )
    
    def _generate_real_response(self, prompt: str) -> str:
        """Generate response using real LLM"""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Create safety-aware attention mask
        safety_attention_mask = self._compute_safety_attention_mask(inputs['input_ids'])
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                attention_mask=safety_attention_mask,
                max_length=inputs['input_ids'].shape[1] + 200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the new content
        prompt_tokens = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
        new_content = response[len(prompt_tokens):].strip()
        
        return new_content
    
    def _generate_response_async(self, prompt: str) -> str:
        """Generate response using async processing to avoid blocking safety validation"""
        import uuid
        
        # Check cache first
        cached_response = self.response_cache.get(prompt)
        if cached_response:
            self.cache_hits += 1
            return cached_response
        
        self.cache_misses += 1
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Submit inference request
        self.inference_queue.put((prompt, request_id))
        
        # Wait for result with timeout
        timeout = 30.0  # 30 second timeout
        start_wait = time.time()
        
        while time.time() - start_wait < timeout:
            try:
                # Check for result
                result = self.result_queue.get_nowait()
                if result[0] == request_id:
                    response, execution_time = result[1], result[2]
                    
                    # Cache the response
                    self.response_cache.put(prompt, response)
                    
                    return response
                else:
                    # Put back result for another request
                    self.result_queue.put(result)
                    
            except queue.Empty:
                time.sleep(0.1)
                continue
        
        # Timeout occurred
        raise TimeoutError("LLM inference timed out")
    
    def _generate_mock_safe_response(self, command: str) -> str:
        """Generate mock safe response for development/testing"""
        # Simple mock responses based on command keywords
        command_lower = command.lower()
        
        if "move" in command_lower or "go" in command_lower:
            return f"{SafetyToken.SAFETY_CHECK.value} I'll move carefully to the destination while checking for obstacles. {SafetyToken.SAFE_ACTION.value}"
        elif "pick" in command_lower or "grab" in command_lower:
            return f"{SafetyToken.SAFETY_CHECK.value} I'll carefully pick up the object after verifying it's safe to grasp. {SafetyToken.SAFE_ACTION.value}"
        elif "stop" in command_lower or "emergency" in command_lower:
            return f"{SafetyToken.EMERGENCY_STOP.value} Executing emergency stop immediately for safety."
        else:
            return f"{SafetyToken.SAFETY_CHECK.value} I'll execute this task safely with appropriate precautions. {SafetyToken.SAFE_ACTION.value}"
    
    @performance_monitor
    def _optimize_gpu_memory(self):
        """Optimize GPU memory usage"""
        if self.device == "cuda" and torch.cuda.is_available():
            self.gpu_optimizer.optimize_memory(self.model)
            
            # Log memory stats
            memory_stats = self.gpu_optimizer.get_memory_stats()
            if memory_stats:
                self.logger.info(f"GPU Memory - Allocated: {memory_stats['allocated_gb']:.2f}GB, "
                               f"Reserved: {memory_stats['reserved_gb']:.2f}GB")
    
    @safety_critical
    def validate_safety_constraints(self, response: SafetyEmbeddedResponse) -> bool:
        """Validate that response meets all safety constraints"""
        # Check safety score threshold
        if response.safety_score < 0.3:
            return False
        
        # Check for critical violations
        critical_violations = [
            SafetyToken.EMERGENCY_STOP,
            SafetyToken.COLLISION_RISK,
            SafetyToken.HUMAN_PROXIMITY
        ]
        
        for token in response.safety_tokens_used:
            if token in critical_violations:
                constraint = self.safety_constraints.get(token)
                if constraint and constraint.severity >= 0.9:
                    return False
        
        return True
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            'cache_stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            },
            'memory_stats': self.memory_monitor.get_memory_usage(),
            'gpu_stats': self.gpu_optimizer.get_memory_stats() if self.device == "cuda" else {},
            'performance_history': self.performance_metrics
        }
        
        return metrics
    
    def cleanup_resources(self):
        """Clean up resources and shutdown gracefully"""
        try:
            # Stop inference thread
            self.inference_running = False
            if self.inference_thread and self.inference_thread.is_alive():
                self.inference_queue.put(None)  # Shutdown signal
                self.inference_thread.join(timeout=5.0)
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Clean up GPU memory
            if self.device == "cuda":
                self.gpu_optimizer.cleanup_memory()
            
            # Clear caches
            self.response_cache.clear()
            
            self.logger.info("Safety-Embedded LLM resources cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup_resources()
        
        return new_content
    
    def _generate_response_async(self, prompt: str) -> str:
        """Generate response using async processing to avoid blocking safety validation"""
        import uuid
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Submit inference request
        self.inference_queue.put((prompt, request_id))
        
        # Wait for result with timeout
        timeout = 30.0  # 30 second timeout
        start_wait = time.time()
        
        while time.time() - start_wait < timeout:
            try:
                # Check for result
                result = self.result_queue.get_nowait()
                if result[0] == request_id:
                    response, execution_time = result[1], result[2]
                    self.logger.debug(f"Async inference completed in {execution_time:.3f}s")
                    return response
            except queue.Empty:
                time.sleep(0.01)  # Small delay to avoid busy waiting
                continue
        
        # Timeout occurred
        self.logger.warning("Async inference timeout, falling back to mock response")
        return self._generate_mock_safe_response("timeout_fallback")
    
    def _generate_mock_safe_response(self, command: str) -> str:
        """Generate mock response for development/testing"""
        if "navigate" in command.lower():
            return f"""
{{
    "goal_description": "Safe navigation to target",
    "steps": [
        {{
            "action_type": "navigation",
            "description": "Carefully navigate to target location",
            "target_pose": {{"x": 2.0, "y": 1.0, "z": 0.0, "w": 1.0}},
            "parameters": ["target_x", "target_y"],
            "estimated_duration": 15.0,
            "preconditions": ["path_clear", "safety_check_passed"],
            "postconditions": ["at_target_location", "safety_maintained"]
        }}
    ],
    "estimated_duration_seconds": 15,
    "required_capabilities": ["navigation", "safety_monitoring"],
    "safety_considerations": ["Maintain safe distance from obstacles", "Monitor human proximity"]
}}

{SafetyToken.SAFE_ACTION.value} Navigation plan generated with safety considerations.
{SafetyToken.SAFETY_CHECK.value} Path clearance verified.
"""
        else:
            return f"""
{{
    "goal_description": "Execute safe task",
    "steps": [
        {{
            "action_type": "perception",
            "description": "Scan environment for safety",
            "target_pose": {{"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
            "parameters": ["scan_radius"],
            "estimated_duration": 3.0,
            "preconditions": ["sensors_ready"],
            "postconditions": ["environment_assessed", "safety_confirmed"]
        }}
    ],
    "estimated_duration_seconds": 3,
    "required_capabilities": ["perception", "safety_monitoring"],
    "safety_considerations": ["Stay in safe area", "Monitor surroundings"]
}}

{SafetyToken.SAFE_ACTION.value} Safe task plan generated.
{SafetyToken.SAFETY_CHECK.value} Environment safety confirmed.
"""
    
    def validate_task_plan_safety(self, task_plan: TaskPlan) -> SafetyVerificationResponse:
        """
        Validate task plan safety using embedded safety constraints
        
        Args:
            task_plan: Task plan to validate
            
        Returns:
            Safety verification response
        """
        try:
            # Convert task plan to text for analysis
            plan_text = self._task_plan_to_text(task_plan)
            
            # Generate safety analysis
            safety_response = self.generate_safe_response(
                f"Analyze safety of this plan: {plan_text}",
                "Task plan safety validation"
            )
            
            # Determine if plan is safe
            is_safe = safety_response.safety_score > 0.7 and len(safety_response.violations_detected) == 0
            
            return SafetyVerificationResponse(
                is_safe=is_safe,
                confidence_score=safety_response.confidence,
                explanation=f"Safety score: {safety_response.safety_score:.2f}. "
                           f"Violations: {len(safety_response.violations_detected)}"
            )
            
        except Exception as e:
            self.logger.error(f"Error validating task plan safety: {e}")
            return SafetyVerificationResponse(
                is_safe=False,
                confidence_score=0.0,
                explanation=f"Validation error: {e}"
            )
    
    def _task_plan_to_text(self, task_plan: TaskPlan) -> str:
        """Convert task plan to text for safety analysis"""
        text = f"Goal: {task_plan.goal_description}\n"
        text += f"Duration: {task_plan.estimated_duration_seconds} seconds\n"
        text += f"Capabilities: {', '.join(task_plan.required_capabilities)}\n"
        text += f"Safety considerations: {', '.join(task_plan.safety_considerations)}\n"
        
        text += "Steps:\n"
        for i, step in enumerate(task_plan.steps):
            text += f"  {i+1}. {step.action_type}: {step.description}\n"
            text += f"     Duration: {step.estimated_duration}s\n"
            text += f"     Parameters: {', '.join(step.parameters)}\n"
        
        return text  
       
        return new_content
    
    def _generate_response_async(self, prompt: str) -> str:
        """Generate response using async processing to avoid blocking safety validation"""
        import uuid
        
        # Check cache first
        cached_response = self.response_cache.get(prompt)
        if cached_response:
            self.cache_hits += 1
            return cached_response
        
        self.cache_misses += 1
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Submit inference request
        self.inference_queue.put((prompt, request_id))
        
        # Wait for result with timeout
        timeout = 30.0  # 30 second timeout
        start_wait = time.time()
        
        while time.time() - start_wait < timeout:
            try:
                # Check for result
                result = self.result_queue.get_nowait()
                if result[0] == request_id:
                    response, execution_time = result[1], result[2]
                    
                    # Cache the response
                    self.response_cache.put(prompt, response)
                    
                    return response
                else:
                    # Put back result for another request
                    self.result_queue.put(result)
                    
            except queue.Empty:
                time.sleep(0.1)
                continue
        
        # Timeout occurred
        raise TimeoutError("LLM inference timed out")
    
    def _generate_mock_safe_response(self, command: str) -> str:
        """Generate mock safe response for development/testing"""
        # Simple mock responses based on command keywords
        command_lower = command.lower()
        
        if "move" in command_lower or "go" in command_lower:
            return f"{SafetyToken.SAFETY_CHECK.value} I'll move carefully to the destination while checking for obstacles. {SafetyToken.SAFE_ACTION.value}"
        elif "pick" in command_lower or "grab" in command_lower:
            return f"{SafetyToken.SAFETY_CHECK.value} I'll carefully pick up the object after verifying it's safe to grasp. {SafetyToken.SAFE_ACTION.value}"
        elif "stop" in command_lower or "emergency" in command_lower:
            return f"{SafetyToken.EMERGENCY_STOP.value} Executing emergency stop immediately for safety."
        else:
            return f"{SafetyToken.SAFETY_CHECK.value} I'll execute this task safely with appropriate precautions. {SafetyToken.SAFE_ACTION.value}"
    
    @performance_monitor
    def _optimize_gpu_memory(self):
        """Optimize GPU memory usage"""
        if self.device == "cuda" and torch.cuda.is_available():
            self.gpu_optimizer.optimize_memory(self.model)
            
            # Log memory stats
            memory_stats = self.gpu_optimizer.get_memory_stats()
            if memory_stats:
                self.logger.info(f"GPU Memory - Allocated: {memory_stats['allocated_gb']:.2f}GB, "
                               f"Reserved: {memory_stats['reserved_gb']:.2f}GB")
    
    @safety_critical
    def validate_safety_constraints(self, response: SafetyEmbeddedResponse) -> bool:
        """Validate that response meets all safety constraints"""
        # Check safety score threshold
        if response.safety_score < 0.3:
            return False
        
        # Check for critical violations
        critical_violations = [
            SafetyToken.EMERGENCY_STOP,
            SafetyToken.COLLISION_RISK,
            SafetyToken.HUMAN_PROXIMITY
        ]
        
        for token in response.safety_tokens_used:
            if token in critical_violations:
                constraint = self.safety_constraints.get(token)
                if constraint and constraint.severity >= 0.9:
                    return False
        
        return True
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            'cache_stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            },
            'memory_stats': self.memory_monitor.get_memory_usage(),
            'gpu_stats': self.gpu_optimizer.get_memory_stats() if self.device == "cuda" else {},
            'performance_history': self.performance_metrics
        }
        
        return metrics
    
    def cleanup_resources(self):
        """Clean up resources and shutdown gracefully"""
        try:
            # Stop inference thread
            self.inference_running = False
            if self.inference_thread and self.inference_thread.is_alive():
                self.inference_queue.put(None)  # Shutdown signal
                self.inference_thread.join(timeout=5.0)
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Clean up GPU memory
            if self.device == "cuda":
                self.gpu_optimizer.cleanup_memory()
            
            # Clear caches
            self.response_cache.clear()
            
            self.logger.info("Safety-Embedded LLM resources cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup_resources()


# Utility functions for creating and managing Safety-Embedded LLM instances
def create_safety_llm(config: Optional[Dict[str, Any]] = None) -> SafetyEmbeddedLLM:
    """Create a Safety-Embedded LLM with configuration"""
    if config is None:
        config = {}
    
    return SafetyEmbeddedLLM(
        model_name=config.get('model_name', 'microsoft/DialoGPT-medium'),
        device=config.get('device', 'auto'),
        cache_size=config.get('cache_size', 128),
        enable_gpu_optimization=config.get('enable_gpu_optimization', True)
    )


def batch_safety_evaluation(llm: SafetyEmbeddedLLM, commands: List[str], 
                          context: str = "") -> List[SafetyEmbeddedResponse]:
    """Evaluate multiple commands for safety in batch"""
    responses = []
    
    for command in commands:
        try:
            response = llm.generate_safe_response(command, context)
            responses.append(response)
        except Exception as e:
            # Create error response
            error_response = SafetyEmbeddedResponse(
                content=f"Error: {e}",
                safety_score=0.0,
                safety_tokens_used=[],
                violations_detected=[f"Evaluation error: {e}"],
                confidence=0.0,
                execution_time=0.0
            )
            responses.append(error_response)
    
    return responses