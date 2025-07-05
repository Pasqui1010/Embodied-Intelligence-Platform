#!/usr/bin/env python3
"""
Thread-Safe Containers for Adaptive Safety Orchestration

This module provides thread-safe data structures to address critical
thread safety issues in the ASO system.
"""

import threading
import time
import weakref
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import logging
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class MemoryMetrics:
    """Memory usage metrics for monitoring"""
    current_size: int
    peak_size: int
    total_allocations: int
    total_deallocations: int
    last_cleanup: float

class ThreadSafeExperienceBuffer:
    """Thread-safe experience buffer with memory management"""
    
    def __init__(self, maxlen: int = 10000, cleanup_threshold: float = 0.8):
        self._buffer = deque(maxlen=maxlen)
        self._lock = threading.RLock()
        self._maxlen = maxlen
        self._cleanup_threshold = cleanup_threshold
        self._metrics = MemoryMetrics(
            current_size=0,
            peak_size=0,
            total_allocations=0,
            total_deallocations=0,
            last_cleanup=time.time()
        )
        self._logger = logging.getLogger(__name__)
        
    def append(self, item: Any) -> bool:
        """Thread-safe append with memory management"""
        with self._lock:
            try:
                # Check if buffer is near capacity
                if len(self._buffer) >= self._maxlen * self._cleanup_threshold:
                    self._cleanup_old_experiences()
                
                # Add item
                self._buffer.append(item)
                self._metrics.current_size = len(self._buffer)
                self._metrics.total_allocations += 1
                self._metrics.peak_size = max(self._metrics.peak_size, len(self._buffer))
                
                return True
                
            except Exception as e:
                self._logger.error(f"Error appending to experience buffer: {e}")
                return False
    
    def extend(self, items: List[Any]) -> int:
        """Thread-safe extend with memory management"""
        with self._lock:
            try:
                added_count = 0
                for item in items:
                    if self.append(item):
                        added_count += 1
                    else:
                        break
                return added_count
                
            except Exception as e:
                self._logger.error(f"Error extending experience buffer: {e}")
                return 0
    
    def get_batch(self, batch_size: int) -> List[Any]:
        """Thread-safe batch retrieval"""
        with self._lock:
            try:
                if len(self._buffer) == 0:
                    return []
                
                # Get batch
                actual_batch_size = min(batch_size, len(self._buffer))
                batch = []
                
                for _ in range(actual_batch_size):
                    if self._buffer:
                        batch.append(self._buffer.popleft())
                        self._metrics.current_size = len(self._buffer)
                        self._metrics.total_deallocations += 1
                
                return batch
                
            except Exception as e:
                self._logger.error(f"Error getting batch from experience buffer: {e}")
                return []
    
    def clear(self) -> int:
        """Thread-safe clear with metrics update"""
        with self._lock:
            try:
                size = len(self._buffer)
                self._buffer.clear()
                self._metrics.current_size = 0
                self._metrics.total_deallocations += size
                return size
                
            except Exception as e:
                self._logger.error(f"Error clearing experience buffer: {e}")
                return 0
    
    def _cleanup_old_experiences(self):
        """Clean up old experiences to manage memory"""
        try:
            current_time = time.time()
            if current_time - self._metrics.last_cleanup < 60:  # Minimum 1 minute between cleanups
                return
            
            # Remove oldest 20% of experiences
            remove_count = int(len(self._buffer) * 0.2)
            for _ in range(remove_count):
                if self._buffer:
                    self._buffer.popleft()
                    self._metrics.total_deallocations += 1
            
            self._metrics.current_size = len(self._buffer)
            self._metrics.last_cleanup = current_time
            
            self._logger.info(f"Cleaned up {remove_count} old experiences")
            
        except Exception as e:
            self._logger.error(f"Error during cleanup: {e}")
    
    def get_metrics(self) -> MemoryMetrics:
        """Get current memory metrics"""
        with self._lock:
            return MemoryMetrics(
                current_size=self._metrics.current_size,
                peak_size=self._metrics.peak_size,
                total_allocations=self._metrics.total_allocations,
                total_deallocations=self._metrics.total_deallocations,
                last_cleanup=self._metrics.last_cleanup
            )
    
    def __len__(self) -> int:
        """Thread-safe length"""
        with self._lock:
            return len(self._buffer)
    
    def __contains__(self, item: Any) -> bool:
        """Thread-safe contains check"""
        with self._lock:
            return item in self._buffer

class ThreadSafeRuleRegistry:
    """Thread-safe safety rule registry with automatic cleanup"""
    
    def __init__(self, max_rules: int = 100, cleanup_interval: float = 300.0):
        self._rules: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._max_rules = max_rules
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._access_count: Dict[str, int] = {}
        self._logger = logging.getLogger(__name__)
        
    def add_rule(self, rule_id: str, rule: Any) -> bool:
        """Thread-safe rule addition with automatic pruning"""
        with self._lock:
            try:
                # Check if cleanup is needed
                current_time = time.time()
                if current_time - self._last_cleanup > self._cleanup_interval:
                    self._cleanup_low_priority_rules()
                
                # Add rule
                self._rules[rule_id] = rule
                self._access_count[rule_id] = 0
                
                # Prune if over limit
                if len(self._rules) > self._max_rules:
                    self._prune_rules()
                
                return True
                
            except Exception as e:
                self._logger.error(f"Error adding rule {rule_id}: {e}")
                return False
    
    def get_rule(self, rule_id: str) -> Optional[Any]:
        """Thread-safe rule retrieval with access tracking"""
        with self._lock:
            try:
                if rule_id in self._rules:
                    self._access_count[rule_id] += 1
                    return self._rules[rule_id]
                return None
                
            except Exception as e:
                self._logger.error(f"Error getting rule {rule_id}: {e}")
                return None
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Thread-safe rule update"""
        with self._lock:
            try:
                if rule_id in self._rules:
                    rule = self._rules[rule_id]
                    for key, value in updates.items():
                        if hasattr(rule, key):
                            setattr(rule, key, value)
                    return True
                return False
                
            except Exception as e:
                self._logger.error(f"Error updating rule {rule_id}: {e}")
                return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Thread-safe rule removal"""
        with self._lock:
            try:
                if rule_id in self._rules:
                    del self._rules[rule_id]
                    if rule_id in self._access_count:
                        del self._access_count[rule_id]
                    return True
                return False
                
            except Exception as e:
                self._logger.error(f"Error removing rule {rule_id}: {e}")
                return False
    
    def get_all_rules(self) -> Dict[str, Any]:
        """Thread-safe get all rules"""
        with self._lock:
            return self._rules.copy()
    
    def _cleanup_low_priority_rules(self):
        """Clean up low-priority rules"""
        try:
            current_time = time.time()
            
            # Remove rules with low access count and old timestamps
            rules_to_remove = []
            for rule_id, rule in self._rules.items():
                access_count = self._access_count.get(rule_id, 0)
                if hasattr(rule, 'last_updated'):
                    time_since_update = current_time - rule.last_updated
                    if access_count < 5 and time_since_update > 3600:  # 1 hour
                        rules_to_remove.append(rule_id)
            
            # Remove low-priority rules
            for rule_id in rules_to_remove:
                self.remove_rule(rule_id)
            
            self._last_cleanup = current_time
            
            if rules_to_remove:
                self._logger.info(f"Cleaned up {len(rules_to_remove)} low-priority rules")
                
        except Exception as e:
            self._logger.error(f"Error during rule cleanup: {e}")
    
    def _prune_rules(self):
        """Prune rules based on priority and usage"""
        try:
            if len(self._rules) <= self._max_rules:
                return
            
            # Calculate rule scores based on access count and priority
            rule_scores = []
            for rule_id, rule in self._rules.items():
                access_count = self._access_count.get(rule_id, 0)
                priority = getattr(rule, 'priority', 0)
                confidence = getattr(rule, 'confidence', 0.0)
                
                # Score = access_count * 0.3 + priority * 0.3 + confidence * 0.4
                score = access_count * 0.3 + priority * 0.3 + confidence * 0.4
                rule_scores.append((rule_id, score))
            
            # Sort by score and remove lowest scoring rules
            rule_scores.sort(key=lambda x: x[1])
            remove_count = len(self._rules) - self._max_rules
            
            for rule_id, _ in rule_scores[:remove_count]:
                self.remove_rule(rule_id)
            
            self._logger.info(f"Pruned {remove_count} rules")
            
        except Exception as e:
            self._logger.error(f"Error during rule pruning: {e}")
    
    def __len__(self) -> int:
        """Thread-safe length"""
        with self._lock:
            return len(self._rules)
    
    def __contains__(self, rule_id: str) -> bool:
        """Thread-safe contains check"""
        with self._lock:
            return rule_id in self._rules

class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._validation_rules = {
            'task_plan': {
                'max_length': 1000,
                'allowed_chars': set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-()[]{}:;"\'\\n\\t'),
                'forbidden_patterns': [
                    r'<script.*?>.*?</script>',
                    r'javascript:',
                    r'data:text/html',
                    r'vbscript:',
                    r'on\w+\s*=',
                    r'<iframe.*?>',
                    r'<object.*?>',
                    r'<embed.*?>'
                ]
            },
            'sensor_data': {
                'max_array_size': 1000,
                'max_value': 1e6,
                'min_value': -1e6,
                'allowed_types': (int, float, np.number)
            },
            'context': {
                'max_depth': 5,
                'max_keys': 50,
                'max_value_length': 100
            }
        }
    
    def validate_task_plan(self, task_plan: str) -> tuple[bool, str]:
        """Validate and sanitize task plan"""
        try:
            if not isinstance(task_plan, str):
                return False, "Task plan must be a string"
            
            if len(task_plan) > self._validation_rules['task_plan']['max_length']:
                return False, f"Task plan too long (max {self._validation_rules['task_plan']['max_length']} chars)"
            
            # Check for forbidden patterns
            import re
            for pattern in self._validation_rules['task_plan']['forbidden_patterns']:
                if re.search(pattern, task_plan, re.IGNORECASE):
                    return False, f"Forbidden pattern detected: {pattern}"
            
            # Sanitize by removing potentially dangerous characters
            sanitized = ''.join(
                char for char in task_plan 
                if char in self._validation_rules['task_plan']['allowed_chars']
            )
            
            return True, sanitized
            
        except Exception as e:
            self._logger.error(f"Error validating task plan: {e}")
            return False, f"Validation error: {str(e)}"
    
    def validate_sensor_data(self, sensor_data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate sensor data"""
        try:
            if not isinstance(sensor_data, dict):
                return False, "Sensor data must be a dictionary"
            
            for key, value in sensor_data.items():
                if not isinstance(key, str):
                    return False, f"Invalid sensor key type: {type(key)}"
                
                if isinstance(value, (list, np.ndarray)):
                    if len(value) > self._validation_rules['sensor_data']['max_array_size']:
                        return False, f"Array too large: {len(value)}"
                    
                    for item in value:
                        if not isinstance(item, self._validation_rules['sensor_data']['allowed_types']):
                            return False, f"Invalid array element type: {type(item)}"
                        
                        if not (self._validation_rules['sensor_data']['min_value'] <= 
                               item <= self._validation_rules['sensor_data']['max_value']):
                            return False, f"Value out of range: {item}"
                
                elif isinstance(value, self._validation_rules['sensor_data']['allowed_types']):
                    if not (self._validation_rules['sensor_data']['min_value'] <= 
                           value <= self._validation_rules['sensor_data']['max_value']):
                        return False, f"Value out of range: {value}"
                else:
                    return False, f"Invalid sensor data type: {type(value)}"
            
            return True, "Valid"
            
        except Exception as e:
            self._logger.error(f"Error validating sensor data: {e}")
            return False, f"Validation error: {str(e)}"
    
    def validate_context(self, context: Dict[str, Any], depth: int = 0) -> tuple[bool, str]:
        """Validate context data recursively"""
        try:
            if depth > self._validation_rules['context']['max_depth']:
                return False, f"Context too deep: {depth}"
            
            if len(context) > self._validation_rules['context']['max_keys']:
                return False, f"Too many context keys: {len(context)}"
            
            for key, value in context.items():
                if not isinstance(key, str):
                    return False, f"Invalid context key type: {type(key)}"
                
                if isinstance(value, dict):
                    is_valid, error = self.validate_context(value, depth + 1)
                    if not is_valid:
                        return False, error
                elif isinstance(value, str):
                    if len(value) > self._validation_rules['context']['max_value_length']:
                        return False, f"Context value too long: {len(value)}"
                elif not isinstance(value, (int, float, bool)):
                    return False, f"Invalid context value type: {type(value)}"
            
            return True, "Valid"
            
        except Exception as e:
            self._logger.error(f"Error validating context: {e}")
            return False, f"Validation error: {str(e)}"

class ErrorRecoveryManager:
    """Error recovery and system resilience management"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._error_counts: Dict[str, int] = {}
        self._last_errors: Dict[str, float] = {}
        self._recovery_strategies: Dict[str, Callable] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)
        
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register a recovery strategy for an error type"""
        with self._lock:
            self._recovery_strategies[error_type] = strategy
    
    def execute_with_recovery(self, operation: Callable, error_type: str = "general", 
                            *args, **kwargs) -> tuple[bool, Any]:
        """Execute operation with automatic error recovery"""
        with self._lock:
            current_time = time.time()
            
            # Check if we should attempt recovery
            if error_type in self._error_counts:
                if self._error_counts[error_type] >= self._max_retries:
                    time_since_last = current_time - self._last_errors.get(error_type, 0)
                    if time_since_last < 300:  # 5 minute cooldown
                        return False, f"Too many errors for {error_type}, in cooldown"
                    else:
                        # Reset error count after cooldown
                        self._error_counts[error_type] = 0
            
            try:
                # Execute operation
                result = operation(*args, **kwargs)
                return True, result
                
            except Exception as e:
                # Handle error
                self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
                self._last_errors[error_type] = current_time
                
                self._logger.error(f"Error in {error_type}: {e}")
                
                # Try recovery strategy
                if error_type in self._recovery_strategies:
                    try:
                        recovery_result = self._recovery_strategies[error_type](e, *args, **kwargs)
                        self._logger.info(f"Recovery successful for {error_type}")
                        return True, recovery_result
                    except Exception as recovery_error:
                        self._logger.error(f"Recovery failed for {error_type}: {recovery_error}")
                
                return False, str(e)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        with self._lock:
            return {
                'error_counts': self._error_counts.copy(),
                'last_errors': self._last_errors.copy(),
                'recovery_strategies': list(self._recovery_strategies.keys())
            }
    
    def reset_errors(self, error_type: str = None):
        """Reset error counts"""
        with self._lock:
            if error_type:
                self._error_counts[error_type] = 0
                self._last_errors[error_type] = 0
            else:
                self._error_counts.clear()
                self._last_errors.clear()

@contextmanager
def thread_safe_context(lock: threading.RLock):
    """Context manager for thread-safe operations"""
    try:
        with lock:
            yield
    except Exception as e:
        logging.getLogger(__name__).error(f"Error in thread-safe context: {e}")
        raise 