#!/usr/bin/env python3
"""
Adaptive Safety Learning Engine

This module implements online safety learning, pattern recognition, and dynamic
threshold adjustment for continuous safety improvement in robotics systems.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
import queue
import json
import pickle
from datetime import datetime, timedelta

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd


class LearningMethod(Enum):
    """Adaptive learning methods"""
    ONLINE_LEARNING = "online_learning"
    PATTERN_RECOGNITION = "pattern_recognition"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    FEDERATED_LEARNING = "federated_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class SafetyPattern(Enum):
    """Types of safety patterns"""
    COLLISION_RISK = "collision_risk"
    HUMAN_PROXIMITY = "human_proximity"
    VELOCITY_VIOLATION = "velocity_violation"
    WORKSPACE_BOUNDARY = "workspace_boundary"
    EMERGENCY_STOP = "emergency_stop"
    SENSOR_FAILURE = "sensor_failure"
    ENVIRONMENTAL_CHANGE = "environmental_change"


@dataclass
class SafetyExperience:
    """Container for safety learning experiences"""
    timestamp: float
    sensor_data: Dict[str, Any]
    safety_score: float
    safety_events: List[str]
    action_taken: str
    outcome: str  # 'success', 'failure', 'near_miss'
    environment_context: Dict[str, Any]
    robot_state: Dict[str, Any]
    pattern_type: Optional[SafetyPattern] = None
    confidence: float = 0.0


@dataclass
class SafetyPattern:
    """Identified safety pattern"""
    pattern_id: str
    pattern_type: SafetyPattern
    features: List[float]
    frequency: int
    confidence: float
    first_seen: float
    last_seen: float
    severity_distribution: List[float]
    success_rate: float
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AdaptiveThreshold:
    """Adaptive safety threshold"""
    threshold_name: str
    current_value: float
    base_value: float
    min_value: float
    max_value: float
    adaptation_rate: float
    confidence: float
    last_updated: float
    update_history: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class LearningResult:
    """Result of adaptive learning analysis"""
    new_patterns: List[SafetyPattern]
    updated_thresholds: Dict[str, AdaptiveThreshold]
    learning_confidence: float
    recommendations: List[str]
    performance_metrics: Dict[str, float]


class AdaptiveSafetyLearningEngine:
    """
    Adaptive Safety Learning Engine for continuous safety improvement
    
    This engine implements online learning, pattern recognition, and dynamic
    threshold adjustment to continuously improve safety performance.
    """
    
    def __init__(self, learning_methods: List[LearningMethod] = None):
        """
        Initialize the adaptive learning engine
        
        Args:
            learning_methods: List of learning methods to enable
        """
        if learning_methods is None:
            learning_methods = [
                LearningMethod.ONLINE_LEARNING,
                LearningMethod.PATTERN_RECOGNITION,
                LearningMethod.THRESHOLD_ADJUSTMENT
            ]
        
        self.learning_methods = learning_methods
        self.logger = logging.getLogger(__name__)
        
        # Experience storage
        self.experience_buffer = deque(maxlen=10000)
        self.pattern_database = {}
        self.threshold_registry = {}
        
        # Learning components
        self.pattern_detector = None
        self.threshold_optimizer = None
        self.feature_extractor = None
        
        # Performance tracking
        self.learning_metrics = {
            'total_experiences': 0,
            'patterns_identified': 0,
            'threshold_updates': 0,
            'success_rate': 0.0,
            'learning_accuracy': 0.0
        }
        
        # Initialize learning components
        self._initialize_learning_components()
        
        # Initialize default thresholds
        self._initialize_default_thresholds()
        
        # Processing thread
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        
        self.logger.info(f"Initialized adaptive learning engine with methods: {[m.value for m in learning_methods]}")
    
    def _initialize_learning_components(self):
        """Initialize learning components based on enabled methods"""
        if LearningMethod.PATTERN_RECOGNITION in self.learning_methods:
            self.pattern_detector = SafetyPatternDetector()
        
        if LearningMethod.THRESHOLD_ADJUSTMENT in self.learning_methods:
            self.threshold_optimizer = ThresholdOptimizer()
        
        if LearningMethod.ONLINE_LEARNING in self.learning_methods:
            self.feature_extractor = SafetyFeatureExtractor()
    
    def _initialize_default_thresholds(self):
        """Initialize default safety thresholds"""
        default_thresholds = {
            'collision_risk': 0.7,
            'human_proximity': 0.8,
            'velocity_limit': 0.6,
            'workspace_boundary': 0.5,
            'emergency_stop': 0.9
        }
        
        for name, value in default_thresholds.items():
            self.threshold_registry[name] = AdaptiveThreshold(
                threshold_name=name,
                current_value=value,
                base_value=value,
                min_value=value * 0.5,
                max_value=value * 1.5,
                adaptation_rate=0.1,
                confidence=0.5,
                last_updated=time.time()
            )
    
    def add_experience(self, experience: SafetyExperience):
        """
        Add a safety experience for learning
        
        Args:
            experience: Safety experience to learn from
        """
        try:
            # Validate experience
            if not self._validate_experience(experience):
                self.logger.warning("Invalid experience data, skipping")
                return
            
            # Add to experience buffer
            self.experience_buffer.append(experience)
            self.learning_metrics['total_experiences'] += 1
            
            # Add to processing queue
            self.processing_queue.put(experience)
            
            self.logger.debug(f"Added experience: {experience.outcome} at {experience.timestamp}")
            
        except Exception as e:
            self.logger.error(f"Error adding experience: {e}")
    
    def _validate_experience(self, experience: SafetyExperience) -> bool:
        """Validate safety experience data"""
        if experience.timestamp <= 0:
            return False
        
        if experience.safety_score < 0.0 or experience.safety_score > 1.0:
            return False
        
        if experience.outcome not in ['success', 'failure', 'near_miss']:
            return False
        
        if not experience.sensor_data:
            return False
        
        return True
    
    def start_learning(self):
        """Start the adaptive learning processing thread"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.running = True
            self.processing_thread = threading.Thread(target=self._learning_worker, daemon=True)
            self.processing_thread.start()
            self.logger.info("Started adaptive learning processing thread")
    
    def stop_learning(self):
        """Stop the adaptive learning processing thread"""
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            self.logger.info("Stopped adaptive learning processing thread")
    
    def _learning_worker(self):
        """Worker thread for adaptive learning processing"""
        while self.running:
            try:
                # Get experience from queue
                experience = self.processing_queue.get(timeout=1.0)
                
                # Process the experience
                self._process_experience(experience)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in learning worker: {e}")
    
    def _process_experience(self, experience: SafetyExperience):
        """Process individual safety experience"""
        try:
            # Extract features
            features = self._extract_safety_features(experience)
            
            # Update pattern detection
            if LearningMethod.PATTERN_RECOGNITION in self.learning_methods:
                self._update_pattern_detection(experience, features)
            
            # Update threshold optimization
            if LearningMethod.THRESHOLD_ADJUSTMENT in self.learning_methods:
                self._update_threshold_optimization(experience, features)
            
            # Update online learning
            if LearningMethod.ONLINE_LEARNING in self.learning_methods:
                self._update_online_learning(experience, features)
            
        except Exception as e:
            self.logger.error(f"Error processing experience: {e}")
    
    def _extract_safety_features(self, experience: SafetyExperience) -> List[float]:
        """Extract safety features from experience"""
        features = []
        
        # Sensor data features
        for sensor_name, sensor_data in experience.sensor_data.items():
            if isinstance(sensor_data, (int, float)):
                features.append(float(sensor_data))
            elif isinstance(sensor_data, dict):
                # Extract numerical values from sensor data
                for key, value in sensor_data.items():
                    if isinstance(value, (int, float)):
                        features.append(float(value))
        
        # Safety score features
        features.append(experience.safety_score)
        features.append(len(experience.safety_events))
        
        # Environment context features
        for key, value in experience.environment_context.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
        
        # Robot state features
        for key, value in experience.robot_state.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
        
        return features
    
    def _update_pattern_detection(self, experience: SafetyExperience, features: List[float]):
        """Update pattern detection with new experience"""
        if self.pattern_detector is None:
            return
        
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(features, experience)
        
        # Update pattern database
        for pattern in patterns:
            pattern_id = pattern.pattern_id
            if pattern_id in self.pattern_database:
                # Update existing pattern
                existing_pattern = self.pattern_database[pattern_id]
                existing_pattern.frequency += 1
                existing_pattern.last_seen = experience.timestamp
                existing_pattern.confidence = (existing_pattern.confidence + pattern.confidence) / 2
            else:
                # Add new pattern
                self.pattern_database[pattern_id] = pattern
                self.learning_metrics['patterns_identified'] += 1
    
    def _update_threshold_optimization(self, experience: SafetyExperience, features: List[float]):
        """Update threshold optimization with new experience"""
        if self.threshold_optimizer is None:
            return
        
        # Get threshold updates
        threshold_updates = self.threshold_optimizer.optimize_thresholds(
            experience, features, self.threshold_registry
        )
        
        # Apply updates
        for threshold_name, new_value in threshold_updates.items():
            if threshold_name in self.threshold_registry:
                threshold = self.threshold_registry[threshold_name]
                old_value = threshold.current_value
                
                # Update threshold
                threshold.current_value = new_value
                threshold.last_updated = time.time()
                threshold.update_history.append((time.time(), new_value))
                
                # Update confidence based on experience outcome
                if experience.outcome == 'success':
                    threshold.confidence = min(threshold.confidence + 0.01, 1.0)
                elif experience.outcome == 'failure':
                    threshold.confidence = max(threshold.confidence - 0.02, 0.0)
                
                self.learning_metrics['threshold_updates'] += 1
                
                self.logger.info(f"Updated threshold {threshold_name}: {old_value:.3f} -> {new_value:.3f}")
    
    def _update_online_learning(self, experience: SafetyExperience, features: List[float]):
        """Update online learning with new experience"""
        # Update success rate
        total_experiences = self.learning_metrics['total_experiences']
        if total_experiences > 0:
            success_count = sum(1 for exp in self.experience_buffer if exp.outcome == 'success')
            self.learning_metrics['success_rate'] = success_count / total_experiences
        
        # Update learning accuracy
        if len(self.experience_buffer) > 10:
            recent_experiences = list(self.experience_buffer)[-10:]
            accuracy = self._calculate_learning_accuracy(recent_experiences)
            self.learning_metrics['learning_accuracy'] = accuracy
    
    def _calculate_learning_accuracy(self, experiences: List[SafetyExperience]) -> float:
        """Calculate learning accuracy from recent experiences"""
        if not experiences:
            return 0.0
        
        correct_predictions = 0
        total_predictions = 0
        
        for experience in experiences:
            # Simple accuracy calculation based on safety score vs outcome
            predicted_safe = experience.safety_score > 0.5
            actual_safe = experience.outcome == 'success'
            
            if predicted_safe == actual_safe:
                correct_predictions += 1
            total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def get_learning_result(self) -> LearningResult:
        """
        Get current learning results and recommendations
        
        Returns:
            LearningResult with current learning state
        """
        try:
            # Get new patterns
            new_patterns = self._get_recent_patterns()
            
            # Get updated thresholds
            updated_thresholds = self._get_updated_thresholds()
            
            # Calculate learning confidence
            learning_confidence = self._calculate_learning_confidence()
            
            # Generate recommendations
            recommendations = self._generate_recommendations()
            
            # Get performance metrics
            performance_metrics = self.learning_metrics.copy()
            
            return LearningResult(
                new_patterns=new_patterns,
                updated_thresholds=updated_thresholds,
                learning_confidence=learning_confidence,
                recommendations=recommendations,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error getting learning result: {e}")
            return LearningResult(
                new_patterns=[],
                updated_thresholds={},
                learning_confidence=0.0,
                recommendations=[],
                performance_metrics=self.learning_metrics
            )
    
    def _get_recent_patterns(self) -> List[SafetyPattern]:
        """Get recently identified patterns"""
        current_time = time.time()
        recent_patterns = []
        
        for pattern in self.pattern_database.values():
            # Patterns seen in the last hour
            if current_time - pattern.last_seen < 3600:
                recent_patterns.append(pattern)
        
        return recent_patterns
    
    def _get_updated_thresholds(self) -> Dict[str, AdaptiveThreshold]:
        """Get recently updated thresholds"""
        current_time = time.time()
        updated_thresholds = {}
        
        for name, threshold in self.threshold_registry.items():
            # Thresholds updated in the last hour
            if current_time - threshold.last_updated < 3600:
                updated_thresholds[name] = threshold
        
        return updated_thresholds
    
    def _calculate_learning_confidence(self) -> float:
        """Calculate overall learning confidence"""
        if self.learning_metrics['total_experiences'] < 10:
            return 0.0
        
        # Combine multiple factors
        success_rate_weight = 0.4
        accuracy_weight = 0.3
        pattern_confidence_weight = 0.2
        threshold_confidence_weight = 0.1
        
        # Calculate pattern confidence
        pattern_confidence = 0.0
        if self.pattern_database:
            pattern_confidences = [p.confidence for p in self.pattern_database.values()]
            pattern_confidence = np.mean(pattern_confidences)
        
        # Calculate threshold confidence
        threshold_confidence = 0.0
        if self.threshold_registry:
            threshold_confidences = [t.confidence for t in self.threshold_registry.values()]
            threshold_confidence = np.mean(threshold_confidences)
        
        # Combine all factors
        learning_confidence = (
            self.learning_metrics['success_rate'] * success_rate_weight +
            self.learning_metrics['learning_accuracy'] * accuracy_weight +
            pattern_confidence * pattern_confidence_weight +
            threshold_confidence * threshold_confidence_weight
        )
        
        return min(learning_confidence, 1.0)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate safety recommendations based on learning"""
        recommendations = []
        
        # Check success rate
        if self.learning_metrics['success_rate'] < 0.8:
            recommendations.append("Consider adjusting safety thresholds to improve success rate")
        
        # Check for frequent patterns
        for pattern in self.pattern_database.values():
            if pattern.frequency > 10 and pattern.confidence > 0.7:
                recommendations.append(f"High-frequency pattern detected: {pattern.pattern_type.value}")
        
        # Check threshold confidence
        low_confidence_thresholds = [
            name for name, threshold in self.threshold_registry.items()
            if threshold.confidence < 0.5
        ]
        if low_confidence_thresholds:
            recommendations.append(f"Low confidence thresholds: {', '.join(low_confidence_thresholds)}")
        
        # Check learning accuracy
        if self.learning_metrics['learning_accuracy'] < 0.7:
            recommendations.append("Learning accuracy below target, consider feature engineering")
        
        return recommendations
    
    def save_learning_state(self, filepath: str):
        """Save learning state to file"""
        try:
            state = {
                'experience_buffer': list(self.experience_buffer),
                'pattern_database': self.pattern_database,
                'threshold_registry': self.threshold_registry,
                'learning_metrics': self.learning_metrics,
                'timestamp': time.time()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            self.logger.info(f"Saved learning state to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving learning state: {e}")
    
    def load_learning_state(self, filepath: str):
        """Load learning state from file"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.experience_buffer = deque(state['experience_buffer'], maxlen=10000)
            self.pattern_database = state['pattern_database']
            self.threshold_registry = state['threshold_registry']
            self.learning_metrics = state['learning_metrics']
            
            self.logger.info(f"Loaded learning state from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading learning state: {e}")
    
    def get_threshold(self, threshold_name: str) -> Optional[float]:
        """Get current threshold value"""
        if threshold_name in self.threshold_registry:
            return self.threshold_registry[threshold_name].current_value
        return None
    
    def get_patterns(self, pattern_type: SafetyPattern = None) -> List[SafetyPattern]:
        """Get safety patterns, optionally filtered by type"""
        if pattern_type is None:
            return list(self.pattern_database.values())
        
        return [
            pattern for pattern in self.pattern_database.values()
            if pattern.pattern_type == pattern_type
        ]


class SafetyPatternDetector:
    """Detector for safety patterns in sensor data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.clustering_model = DBSCAN(eps=0.5, min_samples=3)
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.feature_history = []
    
    def detect_patterns(self, features: List[float], experience: SafetyExperience) -> List[SafetyPattern]:
        """Detect safety patterns in features"""
        patterns = []
        
        # Add features to history
        self.feature_history.append(features)
        
        if len(self.feature_history) < 5:
            return patterns
        
        # Convert to numpy array
        feature_array = np.array(self.feature_history)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_array)
        
        # Detect clusters
        clusters = self.clustering_model.fit_predict(scaled_features)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.fit_predict(scaled_features)
        
        # Create patterns based on clusters and anomalies
        for i, (cluster_id, anomaly) in enumerate(zip(clusters, anomalies)):
            if cluster_id != -1 or anomaly == -1:  # Cluster or anomaly detected
                pattern = SafetyPattern(
                    pattern_id=f"pattern_{time.time()}_{i}",
                    pattern_type=self._classify_pattern(features),
                    features=features,
                    frequency=1,
                    confidence=0.7 if cluster_id != -1 else 0.5,
                    first_seen=experience.timestamp,
                    last_seen=experience.timestamp,
                    severity_distribution=[experience.safety_score],
                    success_rate=1.0 if experience.outcome == 'success' else 0.0
                )
                patterns.append(pattern)
        
        return patterns
    
    def _classify_pattern(self, features: List[float]) -> SafetyPattern:
        """Classify pattern type based on features"""
        # Simple classification based on feature values
        if len(features) >= 3:
            if features[0] > 0.8:  # High sensor value
                return SafetyPattern.COLLISION_RISK
            elif features[1] > 0.7:  # Medium sensor value
                return SafetyPattern.HUMAN_PROXIMITY
            else:
                return SafetyPattern.ENVIRONMENTAL_CHANGE
        
        return SafetyPattern.ENVIRONMENTAL_CHANGE


class ThresholdOptimizer:
    """Optimizer for safety thresholds"""
    
    def __init__(self):
        self.learning_rate = 0.01
        self.momentum = 0.9
    
    def optimize_thresholds(self, experience: SafetyExperience, features: List[float], 
                          thresholds: Dict[str, AdaptiveThreshold]) -> Dict[str, float]:
        """Optimize thresholds based on experience"""
        updates = {}
        
        for threshold_name, threshold in thresholds.items():
            # Calculate gradient based on experience outcome
            gradient = self._calculate_gradient(experience, threshold)
            
            # Apply momentum and learning rate
            update = self.learning_rate * gradient
            
            # Calculate new threshold value
            new_value = threshold.current_value + update
            
            # Apply bounds
            new_value = max(threshold.min_value, min(threshold.max_value, new_value))
            
            updates[threshold_name] = new_value
        
        return updates
    
    def _calculate_gradient(self, experience: SafetyExperience, threshold: AdaptiveThreshold) -> float:
        """Calculate gradient for threshold optimization"""
        # Simple gradient calculation based on outcome
        if experience.outcome == 'success':
            # If successful, we can be slightly more permissive
            return -0.01
        elif experience.outcome == 'failure':
            # If failed, we should be more conservative
            return 0.02
        else:  # near_miss
            # If near miss, slight adjustment
            return 0.005


class SafetyFeatureExtractor:
    """Extractor for safety-relevant features"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, experience: SafetyExperience) -> List[float]:
        """Extract safety-relevant features from experience"""
        features = []
        
        # Extract numerical features from sensor data
        for sensor_name, sensor_data in experience.sensor_data.items():
            if isinstance(sensor_data, (int, float)):
                features.append(float(sensor_data))
        
        # Add safety score
        features.append(experience.safety_score)
        
        # Add event count
        features.append(len(experience.safety_events))
        
        return features 