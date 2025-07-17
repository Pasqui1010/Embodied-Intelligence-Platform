#!/usr/bin/env python3
"""
Enhanced Error Handling for Safety-Embedded LLM

This module provides comprehensive error handling, logging, and recovery mechanisms
for the safety-embedded LLM system.
"""

import logging
import traceback
import time
import functools
from typing import Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import threading


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    SAFETY_VIOLATION = "safety_violation"
    MODEL_ERROR = "model_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class ErrorContext:
    """Context information for errors"""
    timestamp: float
    function_name: str
    error_type: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    traceback_info: str
    recovery_attempted: bool = False
    recovery_successful: bool = False


class SafetyError(Exception):
    """Base exception for safety-related errors"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH, 
                 category: ErrorCategory = ErrorCategory.SAFETY_VIOLATION):
        super().__init__(message)
        self.severity = severity
        self.category = category
        self.timestamp = time.time()


class ModelLoadError(SafetyError):
    """Exception for model loading failures"""
    def __init__(self, message: str, model_name: str):
        super().__init__(message, ErrorSeverity.CRITICAL, ErrorCategory.MODEL_ERROR)
        self.model_name = model_name


class SafetyViolationError(SafetyError):
    """Exception for safety violations"""
    def __init__(self, message: str, violation_type: str):
        super().__init__(message, ErrorSeverity.CRITICAL, ErrorCategory.SAFETY_VIOLATION)
        self.violation_type = violation_type


class MemoryPressureError(SafetyError):
    """Exception for memory pressure situations"""
    def __init__(self, message: str, memory_usage: float):
        super().__init__(message, ErrorSeverity.HIGH, ErrorCategory.MEMORY_ERROR)
        self.memory_usage = memory_usage


class ErrorHandler:
    """Centralized error handling and recovery system"""
    
    def __init__(self, max_error_history: int = 100):
        self.error_history = []
        self.max_error_history = max_error_history
        self.recovery_strategies = {}
        self.error_counts = {}
        self.lock = threading.Lock()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _setup_logging(self):
        """Set up enhanced logging configuration"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # File handler for errors
        try:
            file_handler = logging.FileHandler('safety_llm_errors.log')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.ERROR)
            
            self.logger.addHandler(file_handler)
        except Exception:
            pass  # Fallback to console only
        
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
    
    def _register_default_strategies(self):
        """Register default error recovery strategies"""
        self.recovery_strategies = {
            ErrorCategory.MODEL_ERROR: self._recover_model_error,
            ErrorCategory.MEMORY_ERROR: self._recover_memory_error,
            ErrorCategory.TIMEOUT_ERROR: self._recover_timeout_error,
            ErrorCategory.VALIDATION_ERROR: self._recover_validation_error,
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """Handle an error with appropriate logging and recovery"""
        with self.lock:
            # Create error context
            error_context = self._create_error_context(error, context)
            
            # Log the error
            self._log_error(error_context)
            
            # Add to history
            self._add_to_history(error_context)
            
            # Attempt recovery if strategy exists
            if error_context.category in self.recovery_strategies:
                try:
                    error_context.recovery_attempted = True
                    recovery_result = self.recovery_strategies[error_context.category](error, context)
                    error_context.recovery_successful = recovery_result
                    
                    if recovery_result:
                        self.logger.info(f"Successfully recovered from {error_context.category.value}")
                    else:
                        self.logger.warning(f"Recovery failed for {error_context.category.value}")
                        
                except Exception as recovery_error:
                    self.logger.error(f"Recovery strategy failed: {recovery_error}")
                    error_context.recovery_successful = False
            
            return error_context
    
    def _create_error_context(self, error: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """Create error context from exception"""
        # Determine error category and severity
        if isinstance(error, SafetyError):
            category = error.category
            severity = error.severity
        else:
            category = self._classify_error(error)
            severity = self._determine_severity(error)
        
        return ErrorContext(
            timestamp=time.time(),
            function_name=context.get('function_name', 'unknown') if context else 'unknown',
            error_type=type(error).__name__,
            severity=severity,
            category=category,
            message=str(error),
            traceback_info=traceback.format_exc()
        )
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error into appropriate category"""
        error_type = type(error).__name__.lower()
        
        if 'memory' in error_type or 'cuda' in str(error).lower():
            return ErrorCategory.MEMORY_ERROR
        elif 'timeout' in error_type or 'timeout' in str(error).lower():
            return ErrorCategory.TIMEOUT_ERROR
        elif 'validation' in error_type or 'value' in error_type:
            return ErrorCategory.VALIDATION_ERROR
        elif 'model' in error_type or 'torch' in error_type:
            return ErrorCategory.MODEL_ERROR
        else:
            return ErrorCategory.SYSTEM_ERROR
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity"""
        error_str = str(error).lower()
        
        if any(word in error_str for word in ['critical', 'fatal', 'safety']):
            return ErrorSeverity.CRITICAL
        elif any(word in error_str for word in ['memory', 'cuda', 'timeout']):
            return ErrorSeverity.HIGH
        elif any(word in error_str for word in ['warning', 'deprecated']):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level"""
        log_message = (
            f"[{error_context.category.value}] {error_context.error_type}: "
            f"{error_context.message}"
        )
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            self.logger.debug(f"Traceback: {error_context.traceback_info}")
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _add_to_history(self, error_context: ErrorContext):
        """Add error to history with size limit"""
        self.error_history.append(error_context)
        
        # Maintain size limit
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
        
        # Update error counts
        error_key = f"{error_context.category.value}:{error_context.error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def _recover_model_error(self, error: Exception, context: Dict[str, Any] = None) -> bool:
        """Recovery strategy for model errors"""
        try:
            # Clear GPU memory
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("Applied model error recovery: cleared GPU memory and ran GC")
            return True
            
        except Exception as e:
            self.logger.error(f"Model error recovery failed: {e}")
            return False
    
    def _recover_memory_error(self, error: Exception, context: Dict[str, Any] = None) -> bool:
        """Recovery strategy for memory errors"""
        try:
            import gc
            import torch
            
            # Aggressive memory cleanup
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.logger.info("Applied memory error recovery: aggressive cleanup")
            return True
            
        except Exception as e:
            self.logger.error(f"Memory error recovery failed: {e}")
            return False
    
    def _recover_timeout_error(self, error: Exception, context: Dict[str, Any] = None) -> bool:
        """Recovery strategy for timeout errors"""
        # For timeout errors, we typically just log and continue
        self.logger.info("Timeout error noted, continuing with fallback behavior")
        return True
    
    def _recover_validation_error(self, error: Exception, context: Dict[str, Any] = None) -> bool:
        """Recovery strategy for validation errors"""
        # For validation errors, we typically sanitize input and retry
        self.logger.info("Validation error recovery: input sanitization recommended")
        return True
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        with self.lock:
            total_errors = len(self.error_history)
            
            if total_errors == 0:
                return {'total_errors': 0}
            
            # Count by category
            category_counts = {}
            severity_counts = {}
            recent_errors = 0
            current_time = time.time()
            
            for error in self.error_history:
                # Category counts
                category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
                
                # Severity counts
                severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
                
                # Recent errors (last hour)
                if current_time - error.timestamp < 3600:
                    recent_errors += 1
            
            return {
                'total_errors': total_errors,
                'recent_errors_1h': recent_errors,
                'category_breakdown': category_counts,
                'severity_breakdown': severity_counts,
                'error_counts': dict(self.error_counts),
                'recovery_success_rate': self._calculate_recovery_rate()
            }
    
    def _calculate_recovery_rate(self) -> float:
        """Calculate recovery success rate"""
        recovery_attempts = sum(1 for error in self.error_history if error.recovery_attempted)
        if recovery_attempts == 0:
            return 0.0
        
        successful_recoveries = sum(1 for error in self.error_history 
                                  if error.recovery_attempted and error.recovery_successful)
        
        return successful_recoveries / recovery_attempts


def error_handler_decorator(error_handler: ErrorHandler, 
                          function_name: Optional[str] = None,
                          reraise: bool = True):
    """Decorator for automatic error handling"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function_name': function_name or func.__name__,
                    'args': str(args)[:100],  # Truncate for logging
                    'kwargs': str(kwargs)[:100]
                }
                
                error_context = error_handler.handle_error(e, context)
                
                if reraise:
                    raise
                else:
                    return None
        
        return wrapper
    return decorator


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_error(error: Exception, context: Dict[str, Any] = None) -> ErrorContext:
    """Convenience function for global error handling"""
    return global_error_handler.handle_error(error, context)


def get_error_stats() -> Dict[str, Any]:
    """Get global error statistics"""
    return global_error_handler.get_error_statistics()