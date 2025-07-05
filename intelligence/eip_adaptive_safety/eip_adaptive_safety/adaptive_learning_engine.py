#!/usr/bin/env python3
"""
Adaptive Learning Engine

Implements online safety learning with pattern recognition, dynamic threshold adjustment,
and experience-based safety rule evolution for the embodied intelligence platform.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import json
import logging
import threading
from collections import deque, defaultdict
import pickle
import os


@dataclass
class SafetyExperience:
    """Represents a safety experience for learning"""
    experience_id: str
    timestamp: float
    sensor_data: Dict[str, Any]
    safety_level: float
    outcome: str  # 'safe', 'unsafe', 'violation', 'near_miss'
    context: Dict[str, Any]
    learning_value: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyPattern:
    """Represents a learned safety pattern"""
    pattern_id: str
    features: np.ndarray
    safety_threshold: float
    confidence: float
    creation_time: float
    usage_count: int = 0
    success_rate: float = 1.0
    adaptation_count: int = 0
    evolution_stage: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveRule:
    """Represents an adaptive safety rule"""
    rule_id: str
    condition: str
    action: str
    threshold: float
    confidence: float
    creation_time: float
    usage_count: int = 0
    success_rate: float = 1.0
    adaptation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class OnlineLearningNetwork(nn.Module):
    """Neural network for online safety learning"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Pattern recognition layers
        self.pattern_recognizer = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim),
            nn.Tanh()
        )
        
        # Safety prediction layers
        self.safety_predictor = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        features = self.feature_extractor(x)
        patterns = self.pattern_recognizer(features)
        safety_prediction = self.safety_predictor(patterns)
        confidence = self.confidence_estimator(patterns)
        
        return patterns, safety_prediction, confidence


class AdaptiveLearningEngine:
    """
    Adaptive Learning Engine
    
    Implements online safety learning with:
    - Real-time pattern recognition
    - Dynamic threshold adjustment
    - Experience-based rule evolution
    - Federated learning support
    """
    
    def __init__(self, learning_rate: float = 0.001, memory_size: int = 1000):
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        
        # Learning components
        self.input_dim = 512  # Will be adjusted based on input
        self.online_network = OnlineLearningNetwork(self.input_dim)
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        # Experience management
        self.experiences: deque = deque(maxlen=memory_size)
        self.safety_patterns: Dict[str, SafetyPattern] = {}
        self.adaptive_rules: Dict[str, AdaptiveRule] = {}
        
        # Pattern recognition
        self.pattern_clusterer = DBSCAN(eps=0.1, min_samples=5)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.feature_scaler = StandardScaler()
        
        # Learning parameters
        self.adaptation_rate = 0.1
        self.evolution_threshold = 0.8
        self.pattern_confidence_threshold = 0.7
        self.rule_confidence_threshold = 0.8
        
        # Performance tracking
        self.learning_rounds = 0
        self.pattern_discoveries = 0
        self.rule_evolutions = 0
        self.adaptation_count = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize learning components"""
        
        # Create initial safety patterns
        self._create_initial_patterns()
        
        # Create initial adaptive rules
        self._create_initial_rules()
        
        # Initialize anomaly detector
        self._initialize_anomaly_detector()
    
    def _create_initial_patterns(self):
        """Create initial safety patterns"""
        
        # Basic safety patterns
        initial_patterns = [
            {
                'name': 'low_velocity_safe',
                'features': np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]),
                'threshold': 0.8,
                'description': 'Low velocity operations are generally safe'
            },
            {
                'name': 'human_proximity_unsafe',
                'features': np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9]),
                'threshold': 0.3,
                'description': 'Human proximity requires caution'
            },
            {
                'name': 'high_acceleration_unsafe',
                'features': np.array([0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9]),
                'threshold': 0.4,
                'description': 'High acceleration is unsafe'
            }
        ]
        
        for pattern_data in initial_patterns:
            pattern = SafetyPattern(
                pattern_id=f"initial_{pattern_data['name']}",
                features=pattern_data['features'],
                safety_threshold=pattern_data['threshold'],
                confidence=0.8,
                creation_time=time.time(),
                metadata={'description': pattern_data['description']}
            )
            self.safety_patterns[pattern.pattern_id] = pattern
    
    def _create_initial_rules(self):
        """Create initial adaptive safety rules"""
        
        initial_rules = [
            {
                'condition': 'velocity > 0.5',
                'action': 'reduce_velocity',
                'threshold': 0.5,
                'description': 'Reduce velocity when too high'
            },
            {
                'condition': 'human_proximity < 1.0',
                'action': 'stop_motion',
                'threshold': 1.0,
                'description': 'Stop when human is too close'
            },
            {
                'condition': 'acceleration > 0.7',
                'action': 'limit_acceleration',
                'threshold': 0.7,
                'description': 'Limit acceleration when too high'
            }
        ]
        
        for rule_data in initial_rules:
            rule = AdaptiveRule(
                rule_id=f"initial_{rule_data['action']}",
                condition=rule_data['condition'],
                action=rule_data['action'],
                threshold=rule_data['threshold'],
                confidence=0.8,
                creation_time=time.time(),
                metadata={'description': rule_data['description']}
            )
            self.adaptive_rules[rule.rule_id] = rule
    
    def _initialize_anomaly_detector(self):
        """Initialize anomaly detector with initial data"""
        
        # Create synthetic initial data for anomaly detection
        initial_data = np.random.random((100, self.input_dim))
        self.feature_scaler.fit(initial_data)
        self.anomaly_detector.fit(initial_data)
    
    def learn_from_experience(self, experience: SafetyExperience):
        """
        Learn from a safety experience
        
        Args:
            experience: Safety experience to learn from
        """
        
        with self.lock:
            # Add experience to memory
            self.experiences.append(experience)
            
            # Extract features
            features = self._extract_features(experience.sensor_data)
            
            # Update neural network
            self._update_neural_network(features, experience.safety_level, experience.outcome)
            
            # Update pattern recognition
            self._update_pattern_recognition(features, experience)
            
            # Update adaptive rules
            self._update_adaptive_rules(experience)
            
            # Check for evolution opportunities
            if self._should_evolve():
                self._trigger_evolution()
            
            # Update performance tracking
            self.learning_rounds += 1
    
    def _extract_features(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from sensor data"""
        
        features = []
        
        # Extract features from different modalities
        for modality in ['vision', 'audio', 'tactile', 'proprioceptive']:
            if modality in sensor_data:
                modality_data = sensor_data[modality]
                if 'features' in modality_data:
                    features.extend(modality_data['features'])
                else:
                    # Create default features
                    features.extend([0.0] * 128)
            else:
                # No data for this modality
                features.extend([0.0] * 128)
        
        # Pad or truncate to standard size
        if len(features) < self.input_dim:
            features.extend([0.0] * (self.input_dim - len(features)))
        else:
            features = features[:self.input_dim]
        
        return np.array(features, dtype=np.float32)
    
    def _update_neural_network(self, features: np.ndarray, safety_level: float, outcome: str):
        """Update neural network with new experience"""
        
        # Prepare training data
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
        
        # Create target based on outcome
        if outcome == 'safe':
            target_safety = 1.0
        elif outcome == 'unsafe':
            target_safety = 0.0
        elif outcome == 'violation':
            target_safety = 0.0
        elif outcome == 'near_miss':
            target_safety = 0.3
        else:
            target_safety = safety_level
        
        target_tensor = torch.tensor([target_safety], dtype=torch.float32)
        
        # Forward pass
        patterns, safety_pred, confidence = self.online_network(features_tensor)
        
        # Calculate loss
        safety_loss = self.criterion(safety_pred, target_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        safety_loss.backward()
        self.optimizer.step()
    
    def _update_pattern_recognition(self, features: np.ndarray, experience: SafetyExperience):
        """Update pattern recognition with new experience"""
        
        # Check for anomalies
        features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        
        # If anomaly detected, create new pattern
        if anomaly_score < -0.5:  # Anomaly threshold
            self._create_new_pattern(features, experience)
        
        # Update existing patterns
        self._update_existing_patterns(features, experience)
        
        # Cluster patterns if enough data
        if len(self.experiences) % 50 == 0:  # Every 50 experiences
            self._cluster_patterns()
    
    def _create_new_pattern(self, features: np.ndarray, experience: SafetyExperience):
        """Create a new safety pattern"""
        
        pattern_id = f"pattern_{int(time.time() * 1000)}"
        
        # Determine threshold based on experience
        if experience.outcome in ['safe', 'near_miss']:
            threshold = 0.7
        else:
            threshold = 0.3
        
        pattern = SafetyPattern(
            pattern_id=pattern_id,
            features=features,
            safety_threshold=threshold,
            confidence=0.6,  # Initial confidence
            creation_time=time.time(),
            metadata={
                'outcome': experience.outcome,
                'safety_level': experience.safety_level,
                'context': experience.context
            }
        )
        
        self.safety_patterns[pattern_id] = pattern
        self.pattern_discoveries += 1
    
    def _update_existing_patterns(self, features: np.ndarray, experience: SafetyExperience):
        """Update existing patterns based on new experience"""
        
        for pattern_id, pattern in self.safety_patterns.items():
            # Calculate similarity
            similarity = self._calculate_similarity(features, pattern.features)
            
            if similarity > 0.8:  # High similarity
                pattern.usage_count += 1
                
                # Update confidence based on outcome
                if experience.outcome == 'safe' and experience.safety_level > pattern.safety_threshold:
                    pattern.success_rate = min(1.0, pattern.success_rate + 0.01)
                elif experience.outcome in ['unsafe', 'violation'] and experience.safety_level < pattern.safety_threshold:
                    pattern.success_rate = min(1.0, pattern.success_rate + 0.01)
                else:
                    pattern.success_rate = max(0.0, pattern.success_rate - 0.02)
                
                # Update threshold if needed
                if pattern.usage_count % 10 == 0:  # Every 10 uses
                    self._adapt_pattern_threshold(pattern, experience)
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two feature vectors"""
        
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _adapt_pattern_threshold(self, pattern: SafetyPattern, experience: SafetyExperience):
        """Adapt pattern threshold based on experience"""
        
        pattern.adaptation_count += 1
        
        # Adjust threshold based on experience
        if experience.outcome == 'safe' and experience.safety_level > pattern.safety_threshold:
            # Increase threshold slightly
            pattern.safety_threshold = min(1.0, pattern.safety_threshold + 0.01)
        elif experience.outcome in ['unsafe', 'violation'] and experience.safety_level < pattern.safety_threshold:
            # Decrease threshold slightly
            pattern.safety_threshold = max(0.0, pattern.safety_threshold - 0.01)
        
        # Update confidence
        pattern.confidence = (pattern.confidence + pattern.success_rate) / 2
    
    def _cluster_patterns(self):
        """Cluster patterns to identify similar ones"""
        
        if len(self.safety_patterns) < 5:
            return
        
        # Extract pattern features
        pattern_features = np.array([pattern.features for pattern in self.safety_patterns.values()])
        
        # Apply clustering
        clusters = self.pattern_clusterer.fit_predict(pattern_features)
        
        # Merge similar patterns
        cluster_groups = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            if cluster_id >= 0:  # Not noise
                cluster_groups[cluster_id].append(list(self.safety_patterns.keys())[i])
        
        # Merge patterns in same cluster
        for cluster_id, pattern_ids in cluster_groups.items():
            if len(pattern_ids) > 1:
                self._merge_patterns(pattern_ids)
    
    def _merge_patterns(self, pattern_ids: List[str]):
        """Merge similar patterns"""
        
        patterns = [self.safety_patterns[pid] for pid in pattern_ids]
        
        # Calculate merged features (average)
        merged_features = np.mean([p.features for p in patterns], axis=0)
        
        # Calculate merged threshold (weighted average)
        total_weight = sum(p.confidence for p in patterns)
        merged_threshold = sum(p.safety_threshold * p.confidence for p in patterns) / total_weight
        
        # Calculate merged confidence
        merged_confidence = np.mean([p.confidence for p in patterns])
        
        # Create merged pattern
        merged_pattern = SafetyPattern(
            pattern_id=f"merged_{int(time.time() * 1000)}",
            features=merged_features,
            safety_threshold=merged_threshold,
            confidence=merged_confidence,
            creation_time=time.time(),
            metadata={
                'merged_from': pattern_ids,
                'merge_reason': 'clustering'
            }
        )
        
        # Remove old patterns and add merged one
        for pid in pattern_ids:
            del self.safety_patterns[pid]
        
        self.safety_patterns[merged_pattern.pattern_id] = merged_pattern
    
    def _update_adaptive_rules(self, experience: SafetyExperience):
        """Update adaptive rules based on experience"""
        
        for rule_id, rule in self.adaptive_rules.items():
            # Check if rule condition is met
            if self._evaluate_rule_condition(rule, experience.sensor_data):
                rule.usage_count += 1
                
                # Evaluate rule effectiveness
                effectiveness = self._evaluate_rule_effectiveness(rule, experience)
                
                if effectiveness > 0.8:
                    rule.success_rate = min(1.0, rule.success_rate + 0.01)
                else:
                    rule.success_rate = max(0.0, rule.success_rate - 0.02)
                
                # Adapt rule threshold if needed
                if rule.usage_count % 20 == 0:  # Every 20 uses
                    self._adapt_rule_threshold(rule, experience)
    
    def _evaluate_rule_condition(self, rule: AdaptiveRule, sensor_data: Dict[str, Any]) -> bool:
        """Evaluate if a rule condition is met"""
        
        # Simple condition evaluation (can be extended)
        condition = rule.condition
        
        if 'velocity' in condition:
            velocity = sensor_data.get('proprioceptive', {}).get('velocity', 0.0)
            if '>' in condition:
                threshold = float(condition.split('>')[1].strip())
                return velocity > threshold
            elif '<' in condition:
                threshold = float(condition.split('<')[1].strip())
                return velocity < threshold
        
        elif 'human_proximity' in condition:
            proximity = sensor_data.get('vision', {}).get('human_proximity', 1.0)
            if '>' in condition:
                threshold = float(condition.split('>')[1].strip())
                return proximity > threshold
            elif '<' in condition:
                threshold = float(condition.split('<')[1].strip())
                return proximity < threshold
        
        elif 'acceleration' in condition:
            acceleration = sensor_data.get('proprioceptive', {}).get('acceleration', 0.0)
            if '>' in condition:
                threshold = float(condition.split('>')[1].strip())
                return acceleration > threshold
            elif '<' in condition:
                threshold = float(condition.split('<')[1].strip())
                return acceleration < threshold
        
        return False
    
    def _evaluate_rule_effectiveness(self, rule: AdaptiveRule, experience: SafetyExperience) -> float:
        """Evaluate rule effectiveness based on experience"""
        
        if rule.action == 'reduce_velocity':
            # Check if velocity was reduced and outcome was safe
            if experience.outcome == 'safe' and experience.safety_level > 0.7:
                return 1.0
            else:
                return 0.5
        
        elif rule.action == 'stop_motion':
            # Check if motion was stopped and outcome was safe
            if experience.outcome == 'safe' and experience.safety_level > 0.8:
                return 1.0
            else:
                return 0.5
        
        elif rule.action == 'limit_acceleration':
            # Check if acceleration was limited and outcome was safe
            if experience.outcome == 'safe' and experience.safety_level > 0.7:
                return 1.0
            else:
                return 0.5
        
        return 0.5  # Default effectiveness
    
    def _adapt_rule_threshold(self, rule: AdaptiveRule, experience: SafetyExperience):
        """Adapt rule threshold based on experience"""
        
        rule.adaptation_count += 1
        
        # Adjust threshold based on effectiveness
        if rule.success_rate > 0.8:
            # Rule is working well, make it more sensitive
            if '>' in rule.condition:
                rule.threshold = max(0.0, rule.threshold - 0.05)
            elif '<' in rule.condition:
                rule.threshold = min(1.0, rule.threshold + 0.05)
        elif rule.success_rate < 0.5:
            # Rule is not working well, make it less sensitive
            if '>' in rule.condition:
                rule.threshold = min(1.0, rule.threshold + 0.05)
            elif '<' in rule.condition:
                rule.threshold = max(0.0, rule.threshold - 0.05)
        
        # Update confidence
        rule.confidence = (rule.confidence + rule.success_rate) / 2
    
    def _should_evolve(self) -> bool:
        """Check if evolution should be triggered"""
        
        # Check learning rounds
        if self.learning_rounds > 100:
            return True
        
        # Check pattern discoveries
        if self.pattern_discoveries > 10:
            return True
        
        # Check rule evolutions
        if self.rule_evolutions > 5:
            return True
        
        # Check adaptation count
        if self.adaptation_count > 50:
            return True
        
        return False
    
    def _trigger_evolution(self):
        """Trigger learning evolution"""
        
        # Remove low-performing patterns
        self._cleanup_low_performing_patterns()
        
        # Remove low-performing rules
        self._cleanup_low_performing_rules()
        
        # Create new patterns from recent experiences
        self._create_patterns_from_experiences()
        
        # Create new rules from patterns
        self._create_rules_from_patterns()
        
        # Update learning parameters
        self._update_learning_parameters()
        
        self.rule_evolutions += 1
    
    def _cleanup_low_performing_patterns(self):
        """Remove low-performing patterns"""
        
        patterns_to_remove = []
        
        for pattern_id, pattern in self.safety_patterns.items():
            if (pattern.usage_count > 10 and pattern.success_rate < 0.3) or \
               (pattern.confidence < 0.2):
                patterns_to_remove.append(pattern_id)
        
        for pattern_id in patterns_to_remove:
            del self.safety_patterns[pattern_id]
    
    def _cleanup_low_performing_rules(self):
        """Remove low-performing rules"""
        
        rules_to_remove = []
        
        for rule_id, rule in self.adaptive_rules.items():
            if (rule.usage_count > 20 and rule.success_rate < 0.4) or \
               (rule.confidence < 0.3):
                rules_to_remove.append(rule_id)
        
        for rule_id in rules_to_remove:
            del self.adaptive_rules[rule_id]
    
    def _create_patterns_from_experiences(self):
        """Create new patterns from recent experiences"""
        
        if len(self.experiences) < 10:
            return
        
        # Get recent experiences
        recent_experiences = list(self.experiences)[-20:]
        
        # Group experiences by outcome
        outcome_groups = defaultdict(list)
        for exp in recent_experiences:
            outcome_groups[exp.outcome].append(exp)
        
        # Create patterns for each outcome group
        for outcome, exps in outcome_groups.items():
            if len(exps) >= 3:  # Need at least 3 experiences
                features = np.mean([self._extract_features(exp.sensor_data) for exp in exps], axis=0)
                
                pattern = SafetyPattern(
                    pattern_id=f"evolved_{outcome}_{int(time.time() * 1000)}",
                    features=features,
                    safety_threshold=0.5 if outcome == 'safe' else 0.3,
                    confidence=0.6,
                    creation_time=time.time(),
                    metadata={'outcome': outcome, 'source': 'evolution'}
                )
                
                self.safety_patterns[pattern.pattern_id] = pattern
    
    def _create_rules_from_patterns(self):
        """Create new rules from patterns"""
        
        for pattern_id, pattern in self.safety_patterns.items():
            if pattern.confidence > 0.7 and pattern.usage_count > 5:
                # Create rule based on pattern
                rule = self._create_rule_from_pattern(pattern)
                if rule:
                    self.adaptive_rules[rule.rule_id] = rule
    
    def _create_rule_from_pattern(self, pattern: SafetyPattern) -> Optional[AdaptiveRule]:
        """Create a rule from a pattern"""
        
        # Analyze pattern features to create rule
        features = pattern.features
        
        # Simple rule creation based on feature analysis
        if np.max(features[:128]) > 0.5:  # Vision features
            rule = AdaptiveRule(
                rule_id=f"rule_vision_{int(time.time() * 1000)}",
                condition="human_proximity < 0.5",
                action="stop_motion",
                threshold=0.5,
                confidence=pattern.confidence,
                creation_time=time.time(),
                metadata={'source_pattern': pattern.pattern_id}
            )
            return rule
        
        elif np.max(features[256:384]) > 0.5:  # Tactile features
            rule = AdaptiveRule(
                rule_id=f"rule_tactile_{int(time.time() * 1000)}",
                condition="contact_pressure > 0.7",
                action="reduce_force",
                threshold=0.7,
                confidence=pattern.confidence,
                creation_time=time.time(),
                metadata={'source_pattern': pattern.pattern_id}
            )
            return rule
        
        elif np.max(features[384:512]) > 0.5:  # Proprioceptive features
            rule = AdaptiveRule(
                rule_id=f"rule_proprioceptive_{int(time.time() * 1000)}",
                condition="acceleration > 0.6",
                action="limit_acceleration",
                threshold=0.6,
                confidence=pattern.confidence,
                creation_time=time.time(),
                metadata={'source_pattern': pattern.pattern_id}
            )
            return rule
        
        return None
    
    def _update_learning_parameters(self):
        """Update learning parameters based on evolution"""
        
        # Adjust learning rate
        if self.learning_rounds > 200:
            self.learning_rate *= 0.95
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        
        # Adjust adaptation rate
        if self.adaptation_count > 100:
            self.adaptation_rate = min(0.2, self.adaptation_rate * 1.05)
    
    def assess_safety(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess safety using learned patterns and rules
        
        Args:
            sensor_data: Current sensor data
            
        Returns:
            Safety assessment with confidence and recommendations
        """
        
        with self.lock:
            # Extract features
            features = self._extract_features(sensor_data)
            
            # Neural network prediction
            features_tensor = torch.from_numpy(features).float().unsqueeze(0)
            
            with torch.no_grad():
                patterns, safety_pred, confidence = self.online_network(features_tensor)
            
            neural_safety = safety_pred.item()
            neural_confidence = confidence.item()
            
            # Pattern-based assessment
            pattern_safety, pattern_confidence = self._assess_with_patterns(features)
            
            # Rule-based assessment
            rule_recommendations = self._assess_with_rules(sensor_data)
            
            # Combine assessments
            final_safety = 0.6 * neural_safety + 0.4 * pattern_safety
            final_confidence = 0.6 * neural_confidence + 0.4 * pattern_confidence
            
            return {
                'safety_level': final_safety,
                'confidence': final_confidence,
                'is_safe': final_safety > 0.7,
                'neural_safety': neural_safety,
                'neural_confidence': neural_confidence,
                'pattern_safety': pattern_safety,
                'pattern_confidence': pattern_confidence,
                'rule_recommendations': rule_recommendations,
                'active_patterns': len([p for p in self.safety_patterns.values() if p.confidence > 0.5]),
                'active_rules': len([r for r in self.adaptive_rules.values() if r.confidence > 0.5])
            }
    
    def _assess_with_patterns(self, features: np.ndarray) -> Tuple[float, float]:
        """Assess safety using learned patterns"""
        
        if not self.safety_patterns:
            return 0.5, 0.5
        
        # Find matching patterns
        matching_patterns = []
        
        for pattern in self.safety_patterns.values():
            if pattern.confidence > self.pattern_confidence_threshold:
                similarity = self._calculate_similarity(features, pattern.features)
                if similarity > 0.7:
                    matching_patterns.append((pattern, similarity))
        
        if not matching_patterns:
            return 0.5, 0.5
        
        # Weighted average based on similarity and confidence
        total_weight = 0
        weighted_safety = 0
        weighted_confidence = 0
        
        for pattern, similarity in matching_patterns:
            weight = similarity * pattern.confidence
            weighted_safety += weight * (1.0 if pattern.safety_threshold > 0.5 else 0.0)
            weighted_confidence += weight * pattern.confidence
            total_weight += weight
        
        if total_weight > 0:
            return weighted_safety / total_weight, weighted_confidence / total_weight
        else:
            return 0.5, 0.5
    
    def _assess_with_rules(self, sensor_data: Dict[str, Any]) -> List[str]:
        """Assess safety using adaptive rules"""
        
        recommendations = []
        
        for rule in self.adaptive_rules.values():
            if rule.confidence > self.rule_confidence_threshold:
                if self._evaluate_rule_condition(rule, sensor_data):
                    recommendations.append(f"Apply rule: {rule.action} ({rule.condition})")
        
        return recommendations
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of the adaptive learning engine"""
        
        return {
            'learning_rounds': self.learning_rounds,
            'pattern_discoveries': self.pattern_discoveries,
            'rule_evolutions': self.rule_evolutions,
            'adaptation_count': self.adaptation_count,
            'pattern_count': len(self.safety_patterns),
            'rule_count': len(self.adaptive_rules),
            'experience_count': len(self.experiences),
            'learning_rate': self.learning_rate,
            'adaptation_rate': self.adaptation_rate,
            'pattern_confidence_threshold': self.pattern_confidence_threshold,
            'rule_confidence_threshold': self.rule_confidence_threshold
        }
    
    def save_model(self, filepath: str):
        """Save the learned model to file"""
        
        with self.lock:
            model_data = {
                'safety_patterns': self.safety_patterns,
                'adaptive_rules': self.adaptive_rules,
                'online_network_state': self.online_network.state_dict(),
                'feature_scaler': self.feature_scaler,
                'anomaly_detector': self.anomaly_detector,
                'learning_parameters': {
                    'learning_rate': self.learning_rate,
                    'adaptation_rate': self.adaptation_rate,
                    'pattern_confidence_threshold': self.pattern_confidence_threshold,
                    'rule_confidence_threshold': self.rule_confidence_threshold
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load the learned model from file"""
        
        with self.lock:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.safety_patterns = model_data['safety_patterns']
            self.adaptive_rules = model_data['adaptive_rules']
            self.online_network.load_state_dict(model_data['online_network_state'])
            self.feature_scaler = model_data['feature_scaler']
            self.anomaly_detector = model_data['anomaly_detector']
            
            # Load learning parameters
            params = model_data['learning_parameters']
            self.learning_rate = params['learning_rate']
            self.adaptation_rate = params['adaptation_rate']
            self.pattern_confidence_threshold = params['pattern_confidence_threshold']
            self.rule_confidence_threshold = params['rule_confidence_threshold'] 