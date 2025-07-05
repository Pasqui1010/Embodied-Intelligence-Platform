#!/usr/bin/env python3
"""
Bio-Mimetic Safety Learning

Implements immune system-inspired safety learning with pattern recognition,
adaptation, and evolution for the swarm safety system.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import json
import logging
from collections import deque
import random
import threading


@dataclass
class SafetyAntigen:
    """Represents a safety pattern that needs to be learned (like an antigen)"""
    antigen_id: str
    features: np.ndarray
    safety_level: float
    timestamp: float
    violation_count: int = 0
    adaptation_strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyAntibody:
    """Represents a learned safety response (like an antibody)"""
    antibody_id: str
    features: np.ndarray
    safety_response: float
    confidence: float
    creation_time: float
    usage_count: int = 0
    success_rate: float = 1.0
    evolution_stage: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImmuneNetwork(nn.Module):
    """Neural network implementing immune system-inspired learning"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Feature extraction layers (like immune cell receptors)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Pattern recognition layers (like antibody-antigen binding)
        self.pattern_recognizer = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim),
            nn.Tanh()
        )
        
        # Safety assessment layer (like immune response)
        self.safety_assessor = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation layer
        self.confidence_estimator = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the immune network"""
        features = self.feature_extractor(x)
        patterns = self.pattern_recognizer(features)
        safety_level = self.safety_assessor(patterns)
        confidence = self.confidence_estimator(patterns)
        
        return patterns, safety_level, confidence


class BioMimeticSafetyLearner:
    """
    Bio-mimetic safety learning system
    
    Implements immune system-inspired learning with:
    - Pattern recognition (antigen-antibody binding)
    - Adaptation (antibody production and refinement)
    - Evolution (mutation and selection)
    - Memory (long-term safety pattern storage)
    """
    
    def __init__(self, learning_rate: float = 0.001, evolution_threshold: float = 0.8):
        self.learning_rate = learning_rate
        self.evolution_threshold = evolution_threshold
        
        # Immune system components
        self.antigens: Dict[str, SafetyAntigen] = {}
        self.antibodies: Dict[str, SafetyAntibody] = {}
        self.memory_cells: Dict[str, SafetyAntibody] = {}
        
        # Neural network
        self.input_dim = 512  # Will be adjusted based on input
        self.immune_network = ImmuneNetwork(self.input_dim)
        self.optimizer = torch.optim.Adam(self.immune_network.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        # Learning parameters
        self.adaptation_rate = 0.1
        self.mutation_rate = 0.05
        self.selection_pressure = 0.8
        self.memory_decay = 0.99
        
        # Evolution tracking
        self.evolution_stage = 0
        self.generation_count = 0
        self.adaptation_count = 0
        
        # Performance tracking
        self.success_history = deque(maxlen=100)
        self.adaptation_history = deque(maxlen=50)
        
        # Thread safety
        self.lock = threading.RLock()
    
    def assess_safety(self, sensor_data: Dict[str, Any], existing_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess safety using bio-mimetic learning
        
        Args:
            sensor_data: Fused sensor data
            existing_patterns: Previously learned safety patterns
            
        Returns:
            Safety assessment with confidence and recommendations
        """
        
        with self.lock:
            # Extract features
            features = self._extract_features(sensor_data)
            
            # Create antigen from current situation
            antigen = self._create_antigen(features, sensor_data)
            
            # Find matching antibodies
            matching_antibodies = self._find_matching_antibodies(antigen)
            
            # Generate immune response
            immune_response = self._generate_immune_response(antigen, matching_antibodies)
            
            # Update learning
            self._update_learning(antigen, immune_response)
            
            # Check for evolution opportunity
            if self._should_evolve():
                self._trigger_evolution()
            
            return immune_response
    
    def _extract_features(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from sensor data"""
        
        # Combine features from all modalities
        features = []
        
        if 'fused_features' in sensor_data:
            features.extend(sensor_data['fused_features'])
        else:
            # Fallback to individual modality features
            for modality in ['vision', 'audio', 'tactile', 'proprioceptive']:
                if modality in sensor_data and 'features' in sensor_data[modality]:
                    features.extend(sensor_data[modality]['features'])
        
        # Pad or truncate to standard size
        if len(features) < self.input_dim:
            features.extend([0.0] * (self.input_dim - len(features)))
        else:
            features = features[:self.input_dim]
        
        return np.array(features, dtype=np.float32)
    
    def _create_antigen(self, features: np.ndarray, sensor_data: Dict[str, Any]) -> SafetyAntigen:
        """Create a safety antigen from current situation"""
        
        antigen_id = f"antigen_{int(time.time() * 1000)}"
        
        # Determine safety level from sensor data
        safety_level = sensor_data.get('safety_score', 0.5)
        
        # Check for violations
        violations = sensor_data.get('safety_violations', [])
        violation_count = len(violations)
        
        antigen = SafetyAntigen(
            antigen_id=antigen_id,
            features=features,
            safety_level=safety_level,
            timestamp=time.time(),
            violation_count=violation_count,
            metadata={
                'sensor_data': sensor_data,
                'violations': violations
            }
        )
        
        # Store antigen
        self.antigens[antigen_id] = antigen
        
        return antigen
    
    def _find_matching_antibodies(self, antigen: SafetyAntigen) -> List[SafetyAntibody]:
        """Find antibodies that match the antigen (pattern recognition)"""
        
        matching_antibodies = []
        
        # Convert antigen features to tensor
        antigen_tensor = torch.from_numpy(antigen.features).float().unsqueeze(0)
        
        for antibody_id, antibody in self.antibodies.items():
            # Calculate similarity (cosine similarity)
            antibody_tensor = torch.from_numpy(antibody.features).float().unsqueeze(0)
            
            similarity = F.cosine_similarity(antigen_tensor, antibody_tensor, dim=1).item()
            
            # Consider it a match if similarity is above threshold
            if similarity > 0.7:
                matching_antibodies.append(antibody)
        
        # Also check memory cells
        for antibody_id, antibody in self.memory_cells.items():
            antibody_tensor = torch.from_numpy(antibody.features).float().unsqueeze(0)
            similarity = F.cosine_similarity(antigen_tensor, antibody_tensor, dim=1).item()
            
            if similarity > 0.8:  # Higher threshold for memory cells
                matching_antibodies.append(antibody)
        
        return matching_antibodies
    
    def _generate_immune_response(self, antigen: SafetyAntigen, matching_antibodies: List[SafetyAntibody]) -> Dict[str, Any]:
        """Generate immune response based on matching antibodies"""
        
        # Use neural network for primary assessment
        antigen_tensor = torch.from_numpy(antigen.features).float().unsqueeze(0)
        
        with torch.no_grad():
            patterns, safety_level, confidence = self.immune_network(antigen_tensor)
        
        # Combine with antibody-based assessment
        if matching_antibodies:
            # Weighted average of matching antibodies
            total_weight = 0.0
            weighted_safety = 0.0
            weighted_confidence = 0.0
            
            for antibody in matching_antibodies:
                weight = antibody.confidence * antibody.success_rate
                weighted_safety += weight * antibody.safety_response
                weighted_confidence += weight * antibody.confidence
                total_weight += weight
            
            if total_weight > 0:
                antibody_safety = weighted_safety / total_weight
                antibody_confidence = weighted_confidence / total_weight
                
                # Combine neural network and antibody assessments
                final_safety = 0.7 * safety_level.item() + 0.3 * antibody_safety
                final_confidence = 0.7 * confidence.item() + 0.3 * antibody_confidence
            else:
                final_safety = safety_level.item()
                final_confidence = confidence.item()
        else:
            final_safety = safety_level.item()
            final_confidence = confidence.item()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(antigen, matching_antibodies, final_safety)
        
        return {
            'safety_level': final_safety,
            'confidence': final_confidence,
            'is_safe': final_safety > 0.7,
            'matching_antibodies': len(matching_antibodies),
            'recommendations': recommendations,
            'antigen_id': antigen.antigen_id,
            'evolution_stage': self.evolution_stage,
            'metadata': {
                'neural_safety': safety_level.item(),
                'neural_confidence': confidence.item(),
                'antibody_safety': weighted_safety / total_weight if matching_antibodies else None,
                'antibody_confidence': weighted_confidence / total_weight if matching_antibodies else None
            }
        }
    
    def _generate_recommendations(self, antigen: SafetyAntigen, matching_antibodies: List[SafetyAntibody], safety_level: float) -> List[str]:
        """Generate safety recommendations"""
        
        recommendations = []
        
        # Check for violations
        if antigen.violation_count > 0:
            recommendations.append(f"Detected {antigen.violation_count} safety violations")
        
        # Check for low confidence
        if safety_level < 0.5:
            recommendations.append("Low safety confidence - recommend human supervision")
        
        # Check for high-risk patterns
        if antigen.safety_level < 0.3:
            recommendations.append("High-risk situation detected - immediate intervention required")
        
        # Check for learning opportunities
        if len(matching_antibodies) == 0:
            recommendations.append("Novel safety pattern - learning opportunity")
        
        # Check for antibody evolution
        if len(matching_antibodies) > 5:
            recommendations.append("Multiple antibody matches - consider evolution")
        
        return recommendations
    
    def _update_learning(self, antigen: SafetyAntigen, immune_response: Dict[str, Any]):
        """Update learning based on immune response"""
        
        # Update antigen with response
        antigen.adaptation_strength = immune_response['confidence']
        
        # Create new antibody if needed
        if len(self._find_matching_antibodies(antigen)) == 0:
            self._create_antibody(antigen, immune_response)
        
        # Update existing antibodies
        matching_antibodies = self._find_matching_antibodies(antigen)
        for antibody in matching_antibodies:
            self._update_antibody(antibody, antigen, immune_response)
        
        # Update neural network
        self._update_neural_network(antigen, immune_response)
        
        # Track performance
        self.success_history.append(immune_response['is_safe'])
        self.adaptation_count += 1
    
    def _create_antibody(self, antigen: SafetyAntigen, immune_response: Dict[str, Any]):
        """Create a new antibody from antigen"""
        
        antibody_id = f"antibody_{int(time.time() * 1000)}"
        
        # Create antibody with slight mutation
        mutated_features = self._mutate_features(antigen.features)
        
        antibody = SafetyAntibody(
            antibody_id=antibody_id,
            features=mutated_features,
            safety_response=immune_response['safety_level'],
            confidence=immune_response['confidence'],
            creation_time=time.time(),
            evolution_stage=self.evolution_stage,
            metadata={
                'parent_antigen': antigen.antigen_id,
                'creation_reason': 'novel_pattern'
            }
        )
        
        self.antibodies[antibody_id] = antibody
    
    def _update_antibody(self, antibody: SafetyAntibody, antigen: SafetyAntigen, immune_response: Dict[str, Any]):
        """Update existing antibody based on new experience"""
        
        antibody.usage_count += 1
        
        # Update success rate
        if immune_response['is_safe']:
            antibody.success_rate = min(1.0, antibody.success_rate + 0.01)
        else:
            antibody.success_rate = max(0.0, antibody.success_rate - 0.02)
        
        # Adaptive feature update
        if antibody.usage_count % 10 == 0:  # Update every 10 uses
            # Blend features with antigen
            blend_factor = 0.1
            antibody.features = (1 - blend_factor) * antibody.features + blend_factor * antigen.features
            
            # Update confidence
            antibody.confidence = (antibody.confidence + immune_response['confidence']) / 2
    
    def _mutate_features(self, features: np.ndarray) -> np.ndarray:
        """Apply mutation to features (like genetic mutation)"""
        
        mutation_mask = np.random.random(features.shape) < self.mutation_rate
        mutations = np.random.normal(0, 0.01, features.shape)
        
        mutated_features = features.copy()
        mutated_features[mutation_mask] += mutations[mutation_mask]
        
        return mutated_features
    
    def _update_neural_network(self, antigen: SafetyAntigen, immune_response: Dict[str, Any]):
        """Update neural network with new experience"""
        
        # Prepare training data
        features_tensor = torch.from_numpy(antigen.features).float().unsqueeze(0)
        target_safety = torch.tensor([immune_response['safety_level']], dtype=torch.float32)
        target_confidence = torch.tensor([immune_response['confidence']], dtype=torch.float32)
        
        # Forward pass
        patterns, safety_pred, confidence_pred = self.immune_network(features_tensor)
        
        # Calculate losses
        safety_loss = self.criterion(safety_pred, target_safety)
        confidence_loss = self.criterion(confidence_pred, target_confidence)
        total_loss = safety_loss + confidence_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
    def _should_evolve(self) -> bool:
        """Check if evolution should be triggered"""
        
        # Check performance threshold
        if len(self.success_history) >= 50:
            recent_success_rate = sum(self.success_history[-50:]) / 50
            if recent_success_rate > self.evolution_threshold:
                return True
        
        # Check adaptation count
        if self.adaptation_count > 100:
            return True
        
        # Check antibody diversity
        if len(self.antibodies) > 50:
            return True
        
        return False
    
    def _trigger_evolution(self):
        """Trigger bio-mimetic evolution"""
        
        self.evolution_stage += 1
        self.generation_count += 1
        
        # Selection: Keep best antibodies
        self._selection_phase()
        
        # Crossover: Create new antibodies from existing ones
        self._crossover_phase()
        
        # Mutation: Apply mutations to some antibodies
        self._mutation_phase()
        
        # Memory consolidation
        self._consolidate_memory()
        
        # Update learning parameters
        self._update_learning_parameters()
        
        # Clear old antigens
        self._cleanup_old_antigens()
        
        self.get_logger().info(f"Evolution triggered - stage {self.evolution_stage}")
    
    def _selection_phase(self):
        """Selection phase: Keep best antibodies"""
        
        # Sort antibodies by success rate and usage count
        sorted_antibodies = sorted(
            self.antibodies.items(),
            key=lambda x: (x[1].success_rate, x[1].usage_count),
            reverse=True
        )
        
        # Keep top antibodies
        keep_count = max(10, len(sorted_antibodies) // 2)
        selected_antibodies = sorted_antibodies[:keep_count]
        
        # Move others to memory or discard
        for antibody_id, antibody in sorted_antibodies[keep_count:]:
            if antibody.success_rate > 0.8:
                self.memory_cells[antibody_id] = antibody
            del self.antibodies[antibody_id]
    
    def _crossover_phase(self):
        """Crossover phase: Create new antibodies from existing ones"""
        
        antibody_list = list(self.antibodies.values())
        if len(antibody_list) < 2:
            return
        
        # Create new antibodies through crossover
        for _ in range(len(antibody_list) // 2):
            parent1, parent2 = random.sample(antibody_list, 2)
            
            # Crossover features
            crossover_point = len(parent1.features) // 2
            child_features = np.concatenate([
                parent1.features[:crossover_point],
                parent2.features[crossover_point:]
            ])
            
            # Create child antibody
            child_id = f"antibody_evolved_{int(time.time() * 1000)}"
            child_antibody = SafetyAntibody(
                antibody_id=child_id,
                features=child_features,
                safety_response=(parent1.safety_response + parent2.safety_response) / 2,
                confidence=(parent1.confidence + parent2.confidence) / 2,
                creation_time=time.time(),
                evolution_stage=self.evolution_stage,
                metadata={
                    'parent1': parent1.antibody_id,
                    'parent2': parent2.antibody_id,
                    'creation_reason': 'crossover'
                }
            )
            
            self.antibodies[child_id] = child_antibody
    
    def _mutation_phase(self):
        """Mutation phase: Apply mutations to antibodies"""
        
        for antibody in self.antibodies.values():
            if random.random() < self.mutation_rate:
                antibody.features = self._mutate_features(antibody.features)
                antibody.evolution_stage = self.evolution_stage
    
    def _consolidate_memory(self):
        """Consolidate important patterns into long-term memory"""
        
        # Move high-performing antibodies to memory
        for antibody_id, antibody in list(self.antibodies.items()):
            if antibody.success_rate > 0.9 and antibody.usage_count > 20:
                self.memory_cells[antibody_id] = antibody
                del self.antibodies[antibody_id]
        
        # Apply memory decay
        current_time = time.time()
        for antibody_id, antibody in list(self.memory_cells.items()):
            age = current_time - antibody.creation_time
            if age > 3600:  # 1 hour
                antibody.confidence *= self.memory_decay
                if antibody.confidence < 0.1:
                    del self.memory_cells[antibody_id]
    
    def _update_learning_parameters(self):
        """Update learning parameters based on evolution"""
        
        # Adjust mutation rate
        if self.generation_count % 10 == 0:
            self.mutation_rate = max(0.01, self.mutation_rate * 0.95)
        
        # Adjust selection pressure
        if len(self.success_history) >= 20:
            recent_success = sum(self.success_history[-20:]) / 20
            if recent_success > 0.8:
                self.selection_pressure = min(0.95, self.selection_pressure * 1.05)
            else:
                self.selection_pressure = max(0.5, self.selection_pressure * 0.95)
    
    def _cleanup_old_antigens(self):
        """Clean up old antigens"""
        
        current_time = time.time()
        old_antigens = [
            antigen_id for antigen_id, antigen in self.antigens.items()
            if current_time - antigen.timestamp > 300  # 5 minutes
        ]
        
        for antigen_id in old_antigens:
            del self.antigens[antigen_id]
    
    def adapt(self, trigger_data: Any):
        """Trigger adaptation based on external trigger"""
        
        with self.lock:
            self.adaptation_count += 1
            
            # Increase learning rate temporarily
            original_lr = self.learning_rate
            self.learning_rate *= 1.5
            
            # Update optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            
            # Reset learning rate after adaptation
            def reset_lr():
                time.sleep(10)  # Wait 10 seconds
                self.learning_rate = original_lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
            
            threading.Thread(target=reset_lr, daemon=True).start()
    
    def evolve(self, consensus_data: List[Any]):
        """Trigger evolution based on consensus data"""
        
        with self.lock:
            # Trigger evolution
            self._trigger_evolution()
            
            # Update based on consensus
            if consensus_data:
                # Learn from consensus patterns
                for data in consensus_data:
                    if hasattr(data, 'safety_level'):
                        # Create antigen from consensus
                        features = np.random.random(self.input_dim)  # Placeholder
                        antigen = SafetyAntigen(
                            antigen_id=f"consensus_{int(time.time() * 1000)}",
                            features=features,
                            safety_level=data.safety_level,
                            timestamp=time.time()
                        )
                        
                        # Create antibody from consensus
                        self._create_antibody(antigen, {
                            'safety_level': data.safety_level,
                            'confidence': getattr(data, 'confidence', 0.8)
                        })
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of the bio-mimetic learner"""
        
        return {
            'evolution_stage': self.evolution_stage,
            'generation_count': self.generation_count,
            'adaptation_count': self.adaptation_count,
            'antibody_count': len(self.antibodies),
            'memory_cell_count': len(self.memory_cells),
            'antigen_count': len(self.antigens),
            'success_rate': sum(self.success_history) / len(self.success_history) if self.success_history else 0.0,
            'learning_rate': self.learning_rate,
            'mutation_rate': self.mutation_rate,
            'selection_pressure': self.selection_pressure,
            'recent_adaptations': len(self.adaptation_history)
        } 