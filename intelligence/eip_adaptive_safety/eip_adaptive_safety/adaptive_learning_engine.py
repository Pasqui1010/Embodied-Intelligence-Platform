#!/usr/bin/env python3
"""
Adaptive Safety Orchestration (ASO) - Core Learning Engine

This module implements meta-learning algorithms for dynamic safety rule evolution
based on real-world interactions and near-miss scenarios.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import json
import logging
from datetime import datetime
import threading
import queue
import time
import re

# Local imports
from .thread_safe_containers import (
    ThreadSafeExperienceBuffer, ThreadSafeRuleRegistry, 
    InputValidator, ErrorRecoveryManager, thread_safe_context
)

# ROS 2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Float32MultiArray
from eip_interfaces.msg import SafetyViolation, SafetyVerificationRequest, SafetyVerificationResponse
from eip_interfaces.srv import ValidateTaskPlan

@dataclass
class SafetyExperience:
    """Represents a safety-related experience for learning"""
    timestamp: float
    sensor_data: Dict[str, np.ndarray]
    safety_violation: bool
    violation_type: str
    severity: float
    context: Dict[str, Any]
    outcome: str  # 'near_miss', 'incident', 'safe_operation'
    recovery_action: Optional[str] = None

@dataclass
class SafetyRule:
    """Represents a learned safety rule"""
    rule_id: str
    condition: Dict[str, Any]
    threshold: float
    confidence: float
    priority: int
    created_at: float
    last_updated: float
    usage_count: int
    success_rate: float

class MetaLearner(nn.Module):
    """Meta-learning neural network for safety rule evolution"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Meta-learning architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.meta_learner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()
        )
        
        self.rule_generator = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 8)  # Rule parameters
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through meta-learner"""
        features = self.feature_extractor(x)
        meta_features = self.meta_learner(features)
        rule_params = self.rule_generator(meta_features)
        return meta_features, rule_params

class AdaptiveLearningEngine(Node):
    """Core adaptive learning engine for safety orchestration"""
    
    def __init__(self):
        super().__init__('adaptive_learning_engine')
        
        # Initialize thread-safe components
        self.experience_buffer = ThreadSafeExperienceBuffer(maxlen=10000)
        self.safety_rules = ThreadSafeRuleRegistry(max_rules=100)
        self.input_validator = InputValidator()
        self.error_recovery = ErrorRecoveryManager(max_retries=3)
        
        # Initialize learning components
        self.meta_learner = MetaLearner(input_dim=256)  # Adjust based on sensor data
        self.optimizer = optim.Adam(self.meta_learner.parameters(), lr=0.001)
        
        # Learning parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.update_frequency = 100  # Update every 100 experiences
        self.min_confidence_threshold = 0.7
        
        # Threading for async learning
        self.learning_thread = None
        self.experience_queue = queue.Queue()
        self.is_learning = False
        
        # Register recovery strategies
        self._register_recovery_strategies()
        
        # Setup ROS 2 communication
        self._setup_ros_communication()
        
        # Start learning thread
        self._start_learning_thread()
        
        self.get_logger().info("Adaptive Learning Engine initialized with thread-safe components")
    
    def _register_recovery_strategies(self):
        """Register error recovery strategies"""
        # Recovery for memory issues
        self.error_recovery.register_recovery_strategy("memory_error", self._recover_memory_error)
        
        # Recovery for validation errors
        self.error_recovery.register_recovery_strategy("validation_error", self._recover_validation_error)
        
        # Recovery for learning errors
        self.error_recovery.register_recovery_strategy("learning_error", self._recover_learning_error)
    
    def _recover_memory_error(self, error: Exception, *args, **kwargs):
        """Recover from memory errors"""
        self.get_logger().warn("Attempting memory error recovery")
        
        # Clear old experiences
        cleared_count = self.experience_buffer.clear()
        self.get_logger().info(f"Cleared {cleared_count} experiences during recovery")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return "memory_recovered"
    
    def _recover_validation_error(self, error: Exception, *args, **kwargs):
        """Recover from validation errors"""
        self.get_logger().warn("Attempting validation error recovery")
        
        # Reset validation state
        self.input_validator = InputValidator()
        
        return "validation_recovered"
    
    def _recover_learning_error(self, error: Exception, *args, **kwargs):
        """Recover from learning errors"""
        self.get_logger().warn("Attempting learning error recovery")
        
        # Reset optimizer state
        self.optimizer = optim.Adam(self.meta_learner.parameters(), lr=self.learning_rate)
        
        return "learning_recovered"
    
    def _setup_ros_communication(self):
        """Setup ROS 2 publishers and subscribers"""
        
        # QoS for real-time safety communication
        safety_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers
        self.safety_violation_sub = self.create_subscription(
            SafetyViolation,
            '/safety/violation',
            self._handle_safety_violation,
            safety_qos
        )
        
        self.sensor_data_sub = self.create_subscription(
            Float32MultiArray,
            '/sensors/fused_data',
            self._handle_sensor_data,
            safety_qos
        )
        
        # Publishers
        self.safety_rules_pub = self.create_publisher(
            String,
            '/safety/adaptive_rules',
            safety_qos
        )
        
        self.learning_status_pub = self.create_publisher(
            String,
            '/safety/learning_status',
            safety_qos
        )
        
        # Services
        self.validate_task_service = self.create_service(
            ValidateTaskPlan,
            '/safety/validate_task_adaptive',
            self._validate_task_adaptive
        )
    
    def _start_learning_thread(self):
        """Start background learning thread"""
        self.is_learning = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
    
    def _learning_loop(self):
        """Background learning loop"""
        while self.is_learning:
            try:
                # Process experiences from queue
                while not self.experience_queue.empty():
                    experience = self.experience_queue.get_nowait()
                    self._process_experience(experience)
                
                # Periodic learning update
                if len(self.experience_buffer) >= self.batch_size:
                    self._update_meta_learner()
                
                # Publish learning status
                self._publish_learning_status()
                
                # Sleep to prevent busy waiting
                import time
                time.sleep(0.1)
                
            except Exception as e:
                self.get_logger().error(f"Error in learning loop: {e}")
    
    def _handle_safety_violation(self, msg: SafetyViolation):
        """Handle safety violation messages"""
        try:
            # Create safety experience
            experience = SafetyExperience(
                timestamp=msg.timestamp,
                sensor_data={},  # Will be filled from sensor data
                safety_violation=True,
                violation_type=msg.violation_type,
                severity=msg.severity,
                context=json.loads(msg.context) if msg.context else {},
                outcome='incident' if msg.severity > 0.7 else 'near_miss',
                recovery_action=msg.recovery_action
            )
            
            # Add to learning queue
            self.experience_queue.put(experience)
            
        except Exception as e:
            self.get_logger().error(f"Error handling safety violation: {e}")
    
    def _handle_sensor_data(self, msg: Float32MultiArray):
        """Handle fused sensor data"""
        try:
            # Store latest sensor data for context
            self.latest_sensor_data = np.array(msg.data)
            
        except Exception as e:
            self.get_logger().error(f"Error handling sensor data: {e}")
    
    def _process_experience(self, experience: SafetyExperience):
        """Process a safety experience for learning with thread safety and validation"""
        try:
            # Validate experience data
            is_valid, error_msg = self._validate_experience(experience)
            if not is_valid:
                self.get_logger().warn(f"Invalid experience: {error_msg}")
                return False
            
            # Add to thread-safe experience buffer
            success = self.experience_buffer.append(experience)
            if not success:
                self.get_logger().error("Failed to add experience to buffer")
                return False
            
            # Extract features from experience
            features = self._extract_features(experience)
            
            # Update meta-learner if enough experiences
            if len(self.experience_buffer) % self.update_frequency == 0:
                self._generate_new_rules(features)
            
            return True
                
        except Exception as e:
            self.get_logger().error(f"Error processing experience: {e}")
            return False
    
    def _validate_experience(self, experience: SafetyExperience) -> tuple[bool, str]:
        """Validate safety experience data"""
        try:
            # Validate sensor data
            is_valid, error = self.input_validator.validate_sensor_data(experience.sensor_data)
            if not is_valid:
                return False, f"Invalid sensor data: {error}"
            
            # Validate context
            is_valid, error = self.input_validator.validate_context(experience.context)
            if not is_valid:
                return False, f"Invalid context: {error}"
            
            # Validate basic fields
            if not isinstance(experience.timestamp, (int, float)):
                return False, "Invalid timestamp"
            
            if not isinstance(experience.safety_violation, bool):
                return False, "Invalid safety_violation field"
            
            if not isinstance(experience.severity, (int, float)):
                return False, "Invalid severity"
            
            if not (0.0 <= experience.severity <= 1.0):
                return False, "Severity out of range [0.0, 1.0]"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _extract_features(self, experience: SafetyExperience) -> np.ndarray:
        """Extract features from safety experience"""
        try:
            # Combine sensor data, context, and outcome
            sensor_features = np.array(list(experience.sensor_data.values())).flatten()
            context_features = np.array(list(experience.context.values())).flatten()
            
            # Create feature vector
            features = np.concatenate([
                sensor_features,
                context_features,
                [experience.severity],
                [1.0 if experience.safety_violation else 0.0],
                [hash(experience.violation_type) % 1000]  # Encode violation type
            ])
            
            # Pad or truncate to fixed size
            target_size = 256
            if len(features) < target_size:
                features = np.pad(features, (0, target_size - len(features)))
            else:
                features = features[:target_size]
            
            return features
            
        except Exception as e:
            self.get_logger().error(f"Error extracting features: {e}")
            return np.zeros(256)
    
    def _generate_new_rules(self, features: np.ndarray):
        """Generate new safety rules using meta-learner with thread safety"""
        try:
            # Convert features to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Forward pass through meta-learner
            with torch.no_grad():
                meta_features, rule_params = self.meta_learner(features_tensor)
            
            # Convert rule parameters to safety rule
            rule_params = rule_params.squeeze().numpy()
            
            # Create new safety rule
            new_rule = SafetyRule(
                rule_id=f"adaptive_rule_{len(self.safety_rules)}",
                condition=self._params_to_condition(rule_params),
                threshold=float(rule_params[0]),
                confidence=float(rule_params[1]),
                priority=int(rule_params[2] * 10),
                created_at=datetime.now().timestamp(),
                last_updated=datetime.now().timestamp(),
                usage_count=0,
                success_rate=0.5  # Initial neutral rate
            )
            
            # Add rule if confidence is high enough using thread-safe registry
            if new_rule.confidence > self.min_confidence_threshold:
                success = self.safety_rules.add_rule(new_rule.rule_id, new_rule)
                if success:
                    self._publish_safety_rules()
                else:
                    self.get_logger().warn(f"Failed to add rule {new_rule.rule_id}")
                
        except Exception as e:
            self.get_logger().error(f"Error generating new rules: {e}")
    
    def _params_to_condition(self, params: np.ndarray) -> Dict[str, Any]:
        """Convert rule parameters to condition dictionary"""
        return {
            'sensor_thresholds': {
                'velocity': float(params[3]),
                'proximity': float(params[4]),
                'force': float(params[5])
            },
            'context_conditions': {
                'human_present': bool(params[6] > 0.5),
                'workspace_boundary': bool(params[7] > 0.5)
            }
        }
    
    def _update_meta_learner(self):
        """Update meta-learner using batch of experiences with thread safety and error recovery"""
        try:
            if len(self.experience_buffer) < self.batch_size:
                return
            
            # Get batch from thread-safe buffer
            batch_experiences = self.experience_buffer.get_batch(self.batch_size)
            
            if not batch_experiences:
                self.get_logger().warn("No experiences available for training")
                return
            
            # Prepare training data with error recovery
            features_list = []
            targets_list = []
            
            for exp in batch_experiences:
                try:
                    features = self._extract_features(exp)
                    target = 1.0 if exp.safety_violation else 0.0
                    
                    features_list.append(features)
                    targets_list.append(target)
                except Exception as e:
                    self.get_logger().warn(f"Error processing experience for training: {e}")
                    continue
            
            if len(features_list) < 2:  # Need at least 2 samples for training
                self.get_logger().warn("Insufficient valid experiences for training")
                return
            
            # Convert to tensors
            features_tensor = torch.FloatTensor(features_list)
            targets_tensor = torch.FloatTensor(targets_list)
            
            # Training step with error recovery
            success, result = self.error_recovery.execute_with_recovery(
                self._perform_training_step, "learning_error", 
                features_tensor, targets_tensor
            )
            
            if success:
                self.get_logger().debug(f"Meta-learner updated successfully")
            else:
                self.get_logger().error(f"Meta-learner update failed: {result}")
            
        except Exception as e:
            self.get_logger().error(f"Error updating meta-learner: {e}")
    
    def _perform_training_step(self, features_tensor: torch.Tensor, targets_tensor: torch.Tensor):
        """Perform a single training step"""
        self.optimizer.zero_grad()
        meta_features, rule_params = self.meta_learner(features_tensor)
        
        # Simple loss function (can be enhanced)
        loss = nn.MSELoss()(rule_params[:, 0], targets_tensor)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _prune_rules(self):
        """Remove low-performing rules"""
        try:
            # Sort rules by success rate and usage count
            rule_scores = []
            for rule_id, rule in self.safety_rules.items():
                score = rule.success_rate * np.log(rule.usage_count + 1)
                rule_scores.append((rule_id, score))
            
            # Remove bottom 20% of rules
            rule_scores.sort(key=lambda x: x[1])
            num_to_remove = len(rule_scores) // 5
            
            for rule_id, _ in rule_scores[:num_to_remove]:
                del self.safety_rules[rule_id]
                
        except Exception as e:
            self.get_logger().error(f"Error pruning rules: {e}")
    
    def _validate_task_adaptive(self, request: ValidateTaskPlan.Request, 
                               response: ValidateTaskPlan.Response) -> ValidateTaskPlan.Response:
        """Validate task plan using adaptive safety rules with input validation and error recovery"""
        try:
            # Validate task plan input
            is_valid, sanitized_task = self.input_validator.validate_task_plan(request.task_plan)
            if not is_valid:
                response.is_safe = False
                response.safety_score = 0.0
                response.violations = [f"Invalid task plan: {sanitized_task}"]
                response.confidence = 0.0
                return response
            
            # Extract task features with error recovery
            success, task_features = self.error_recovery.execute_with_recovery(
                self._extract_task_features, "validation_error", sanitized_task
            )
            
            if not success:
                response.is_safe = False
                response.safety_score = 0.0
                response.violations = [f"Feature extraction failed: {task_features}"]
                response.confidence = 0.0
                return response
            
            # Apply adaptive safety rules with thread safety
            safety_score = 1.0
            violations = []
            
            rules = self.safety_rules.get_all_rules()
            for rule_id, rule in rules.items():
                if self._check_rule_violation(rule, task_features):
                    safety_score *= (1.0 - rule.confidence)
                    violations.append(f"Rule {rule_id}: {rule.condition}")
                    
                    # Update rule usage with thread safety
                    self.safety_rules.update_rule(rule_id, {'usage_count': rule.usage_count + 1})
            
            # Determine if task is safe
            is_safe = safety_score > 0.5
            
            # Update response
            response.is_safe = is_safe
            response.safety_score = safety_score
            response.violations = violations
            response.confidence = min(safety_score, 0.95)  # Conservative confidence
            
            return response
            
        except Exception as e:
            self.get_logger().error(f"Error validating task: {e}")
            response.is_safe = False
            response.safety_score = 0.0
            response.violations = [f"Validation error: {e}"]
            response.confidence = 0.0
            return response
    
    def _extract_task_features(self, task_plan: str) -> Dict[str, Any]:
        """Extract features from task plan"""
        # Simple feature extraction (can be enhanced with NLP)
        features = {
            'task_complexity': len(task_plan.split()),
            'contains_movement': 'move' in task_plan.lower(),
            'contains_manipulation': any(word in task_plan.lower() for word in ['grab', 'pick', 'place']),
            'contains_human_interaction': any(word in task_plan.lower() for word in ['human', 'person', 'assist']),
            'estimated_duration': len(task_plan.split()) * 0.5  # Rough estimate
        }
        return features
    
    def _check_rule_violation(self, rule: SafetyRule, task_features: Dict[str, Any]) -> bool:
        """Check if task violates a safety rule"""
        try:
            # Check sensor thresholds
            if 'velocity' in rule.condition['sensor_thresholds']:
                if task_features.get('contains_movement', False):
                    # Assume high velocity for movement tasks
                    if 1.0 > rule.condition['sensor_thresholds']['velocity']:
                        return True
            
            # Check context conditions
            if rule.condition['context_conditions'].get('human_present', False):
                if task_features.get('contains_human_interaction', False):
                    return True
            
            return False
            
        except Exception as e:
            self.get_logger().error(f"Error checking rule violation: {e}")
            return False
    
    def _publish_safety_rules(self):
        """Publish current safety rules"""
        try:
            rules_data = {
                'timestamp': datetime.now().isoformat(),
                'rules': [
                    {
                        'id': rule.rule_id,
                        'condition': rule.condition,
                        'threshold': rule.threshold,
                        'confidence': rule.confidence,
                        'priority': rule.priority,
                        'usage_count': rule.usage_count,
                        'success_rate': rule.success_rate
                    }
                    for rule in self.safety_rules.values()
                ]
            }
            
            rules_msg = String()
            rules_msg.data = json.dumps(rules_data)
            self.safety_rules_pub.publish(rules_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing safety rules: {e}")
    
    def _publish_learning_status(self):
        """Publish learning status"""
        try:
            status_data = {
                'timestamp': datetime.now().isoformat(),
                'experience_count': len(self.experience_buffer),
                'rule_count': len(self.safety_rules),
                'learning_active': self.is_learning,
                'average_confidence': np.mean([r.confidence for r in self.safety_rules.values()]) if self.safety_rules else 0.0
            }
            
            status_msg = String()
            status_msg.data = json.dumps(status_data)
            self.learning_status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing learning status: {e}")
    
    def shutdown(self):
        """Clean shutdown of learning engine"""
        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5.0)
        self.get_logger().info("Adaptive Learning Engine shutdown complete")

def main(args=None):
    rclpy.init(args=args)
    
    learning_engine = AdaptiveLearningEngine()
    
    try:
        rclpy.spin(learning_engine)
    except KeyboardInterrupt:
        pass
    finally:
        learning_engine.shutdown()
        learning_engine.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 