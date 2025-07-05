#!/usr/bin/env python3
"""
Sensor Fusion Engine

Integrates multi-modal sensor data (vision, audio, tactile, proprioceptive)
for comprehensive safety assessment in the swarm safety system.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import json
import logging

# ROS 2 imports
from sensor_msgs.msg import Image, PointCloud2, Imu
from geometry_msgs.msg import Twist, PoseStamped


@dataclass
class SensorData:
    """Represents fused sensor data"""
    timestamp: float
    vision_features: Optional[np.ndarray] = None
    audio_features: Optional[np.ndarray] = None
    tactile_features: Optional[np.ndarray] = None
    proprioceptive_features: Optional[np.ndarray] = None
    fusion_confidence: float = 0.0
    metadata: Dict[str, Any] = None


class VisionProcessor:
    """Processes vision data for safety assessment"""
    
    def __init__(self):
        self.feature_dim = 512
        self.confidence_threshold = 0.7
        
        # Initialize vision models (simplified for demo)
        self.object_detector = self._create_object_detector()
        self.human_detector = self._create_human_detector()
        self.scene_analyzer = self._create_scene_analyzer()
    
    def _create_object_detector(self):
        """Create object detection model"""
        # Simplified object detector
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.feature_dim)
        )
    
    def _create_human_detector(self):
        """Create human detection model"""
        # Simplified human detector
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128)
        )
    
    def _create_scene_analyzer(self):
        """Create scene analysis model"""
        # Simplified scene analyzer
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256)
        )
    
    def process(self, image_data: np.ndarray) -> Dict[str, Any]:
        """Process vision data and extract safety-relevant features"""
        
        # Convert to tensor
        if len(image_data.shape) == 3:
            image_tensor = torch.from_numpy(image_data).float().unsqueeze(0)
        else:
            image_tensor = torch.from_numpy(image_data).float()
        
        # Normalize
        image_tensor = image_tensor / 255.0
        
        # Extract features
        object_features = self.object_detector(image_tensor)
        human_features = self.human_detector(image_tensor)
        scene_features = self.scene_analyzer(image_tensor)
        
        # Combine features
        combined_features = torch.cat([
            object_features.flatten(),
            human_features.flatten(),
            scene_features.flatten()
        ]).detach().numpy()
        
        # Calculate confidence based on feature variance
        confidence = min(1.0, np.std(combined_features) * 10)
        
        return {
            'features': combined_features,
            'confidence': confidence,
            'object_count': len(object_features),
            'human_detected': torch.max(human_features) > self.confidence_threshold,
            'scene_complexity': torch.std(scene_features).item()
        }


class AudioProcessor:
    """Processes audio data for safety assessment"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.feature_dim = 128
        self.fft_size = 1024
        
        # Initialize audio models
        self.spectral_analyzer = self._create_spectral_analyzer()
        self.human_voice_detector = self._create_voice_detector()
    
    def _create_spectral_analyzer(self):
        """Create spectral analysis model"""
        return nn.Sequential(
            nn.Linear(self.fft_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.feature_dim)
        )
    
    def _create_voice_detector(self):
        """Create human voice detection model"""
        return nn.Sequential(
            nn.Linear(self.fft_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def process(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Process audio data and extract safety-relevant features"""
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_data).float()
        
        # Apply FFT
        fft_data = torch.fft.fft(audio_tensor, n=self.fft_size)
        fft_magnitude = torch.abs(fft_data[:self.fft_size//2])
        
        # Extract features
        spectral_features = self.spectral_analyzer(fft_magnitude)
        voice_probability = self.human_voice_detector(fft_magnitude)
        
        # Calculate confidence
        confidence = voice_probability.item()
        
        return {
            'features': spectral_features.detach().numpy(),
            'confidence': confidence,
            'voice_detected': confidence > 0.5,
            'audio_level': torch.mean(torch.abs(audio_tensor)).item(),
            'spectral_centroid': torch.mean(fft_magnitude).item()
        }


class TactileProcessor:
    """Processes tactile sensor data for safety assessment"""
    
    def __init__(self):
        self.sensor_count = 16
        self.feature_dim = 64
        
        # Initialize tactile models
        self.contact_analyzer = self._create_contact_analyzer()
        self.pressure_analyzer = self._create_pressure_analyzer()
    
    def _create_contact_analyzer(self):
        """Create contact analysis model"""
        return nn.Sequential(
            nn.Linear(self.sensor_count, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.feature_dim)
        )
    
    def _create_pressure_analyzer(self):
        """Create pressure analysis model"""
        return nn.Sequential(
            nn.Linear(self.sensor_count, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def process(self, tactile_data: np.ndarray) -> Dict[str, Any]:
        """Process tactile data and extract safety-relevant features"""
        
        # Convert to tensor
        tactile_tensor = torch.from_numpy(tactile_data).float()
        
        # Extract features
        contact_features = self.contact_analyzer(tactile_tensor)
        pressure_level = self.pressure_analyzer(tactile_tensor)
        
        # Calculate confidence
        confidence = pressure_level.item()
        
        return {
            'features': contact_features.detach().numpy(),
            'confidence': confidence,
            'contact_detected': confidence > 0.3,
            'pressure_level': confidence,
            'sensor_activation': torch.sum(tactile_tensor > 0.1).item()
        }


class ProprioceptiveProcessor:
    """Processes proprioceptive data for safety assessment"""
    
    def __init__(self):
        self.feature_dim = 128
        
        # Initialize proprioceptive models
        self.motion_analyzer = self._create_motion_analyzer()
        self.stability_analyzer = self._create_stability_analyzer()
    
    def _create_motion_analyzer(self):
        """Create motion analysis model"""
        return nn.Sequential(
            nn.Linear(6, 32),  # 6 DOF motion
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.feature_dim)
        )
    
    def _create_stability_analyzer(self):
        """Create stability analysis model"""
        return nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def process(self, proprioceptive_data: np.ndarray) -> Dict[str, Any]:
        """Process proprioceptive data and extract safety-relevant features"""
        
        # Convert to tensor
        proprioceptive_tensor = torch.from_numpy(proprioceptive_data).float()
        
        # Extract features
        motion_features = self.motion_analyzer(proprioceptive_tensor)
        stability_level = self.stability_analyzer(proprioceptive_tensor)
        
        # Calculate confidence
        confidence = stability_level.item()
        
        return {
            'features': motion_features.detach().numpy(),
            'confidence': confidence,
            'stable_motion': confidence > 0.7,
            'motion_magnitude': torch.norm(proprioceptive_tensor).item(),
            'acceleration_level': torch.norm(proprioceptive_tensor[3:]).item()
        }


class SensorFusionEngine:
    """
    Multi-modal sensor fusion engine
    
    Integrates vision, audio, tactile, and proprioceptive data
    for comprehensive safety assessment.
    """
    
    def __init__(self):
        # Initialize processors
        self.vision_processor = VisionProcessor()
        self.audio_processor = AudioProcessor()
        self.tactile_processor = TactileProcessor()
        self.proprioceptive_processor = ProprioceptiveProcessor()
        
        # Fusion weights (can be learned)
        self.fusion_weights = {
            'vision': 0.4,
            'audio': 0.2,
            'tactile': 0.2,
            'proprioceptive': 0.2
        }
        
        # Safety thresholds
        self.safety_thresholds = {
            'vision': 0.6,
            'audio': 0.5,
            'tactile': 0.7,
            'proprioceptive': 0.8
        }
    
    def process(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multi-modal sensor data and return fused safety assessment
        
        Args:
            sensor_data: Dictionary containing sensor data from different modalities
            
        Returns:
            Dictionary with fused safety assessment
        """
        
        timestamp = time.time()
        fusion_results = {}
        
        # Process vision data
        if 'vision' in sensor_data and sensor_data['vision'] is not None:
            vision_result = self.vision_processor.process(sensor_data['vision'])
            fusion_results['vision'] = vision_result
        else:
            fusion_results['vision'] = {
                'features': np.zeros(self.vision_processor.feature_dim),
                'confidence': 0.0,
                'object_count': 0,
                'human_detected': False,
                'scene_complexity': 0.0
            }
        
        # Process audio data
        if 'audio' in sensor_data and sensor_data['audio'] is not None:
            audio_result = self.audio_processor.process(sensor_data['audio'])
            fusion_results['audio'] = audio_result
        else:
            fusion_results['audio'] = {
                'features': np.zeros(self.audio_processor.feature_dim),
                'confidence': 0.0,
                'voice_detected': False,
                'audio_level': 0.0,
                'spectral_centroid': 0.0
            }
        
        # Process tactile data
        if 'tactile' in sensor_data and sensor_data['tactile'] is not None:
            tactile_result = self.tactile_processor.process(sensor_data['tactile'])
            fusion_results['tactile'] = tactile_result
        else:
            fusion_results['tactile'] = {
                'features': np.zeros(self.tactile_processor.feature_dim),
                'confidence': 0.0,
                'contact_detected': False,
                'pressure_level': 0.0,
                'sensor_activation': 0
            }
        
        # Process proprioceptive data
        if 'proprioceptive' in sensor_data and sensor_data['proprioceptive'] is not None:
            proprioceptive_result = self.proprioceptive_processor.process(sensor_data['proprioceptive'])
            fusion_results['proprioceptive'] = proprioceptive_result
        else:
            fusion_results['proprioceptive'] = {
                'features': np.zeros(self.proprioceptive_processor.feature_dim),
                'confidence': 0.0,
                'stable_motion': True,
                'motion_magnitude': 0.0,
                'acceleration_level': 0.0
            }
        
        # Perform sensor fusion
        fused_result = self._fuse_sensor_data(fusion_results)
        
        # Add metadata
        fused_result['timestamp'] = timestamp
        fused_result['sensor_availability'] = {
            'vision': 'vision' in sensor_data and sensor_data['vision'] is not None,
            'audio': 'audio' in sensor_data and sensor_data['audio'] is not None,
            'tactile': 'tactile' in sensor_data and sensor_data['tactile'] is not None,
            'proprioceptive': 'proprioceptive' in sensor_data and sensor_data['proprioceptive'] is not None
        }
        
        return fused_result
    
    def _fuse_sensor_data(self, fusion_results: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse sensor data from different modalities"""
        
        # Extract features and confidences
        features = {}
        confidences = {}
        safety_flags = {}
        
        for modality, result in fusion_results.items():
            features[modality] = result['features']
            confidences[modality] = result['confidence']
            
            # Determine safety flags based on modality-specific logic
            if modality == 'vision':
                safety_flags[modality] = not result['human_detected'] and result['confidence'] > self.safety_thresholds['vision']
            elif modality == 'audio':
                safety_flags[modality] = not result['voice_detected'] and result['confidence'] > self.safety_thresholds['audio']
            elif modality == 'tactile':
                safety_flags[modality] = not result['contact_detected'] and result['confidence'] > self.safety_thresholds['tactile']
            elif modality == 'proprioceptive':
                safety_flags[modality] = result['stable_motion'] and result['confidence'] > self.safety_thresholds['proprioceptive']
        
        # Weighted feature fusion
        fused_features = np.zeros_like(features['vision'])
        total_weight = 0.0
        
        for modality, weight in self.fusion_weights.items():
            if modality in features:
                fused_features += weight * features[modality]
                total_weight += weight
        
        if total_weight > 0:
            fused_features /= total_weight
        
        # Weighted confidence fusion
        fused_confidence = 0.0
        total_weight = 0.0
        
        for modality, weight in self.fusion_weights.items():
            if modality in confidences:
                fused_confidence += weight * confidences[modality]
                total_weight += weight
        
        if total_weight > 0:
            fused_confidence /= total_weight
        
        # Overall safety assessment
        safety_score = 1.0
        safety_violations = []
        
        for modality, is_safe in safety_flags.items():
            if not is_safe:
                safety_score *= 0.5
                safety_violations.append(f"{modality}_violation")
        
        # Additional safety checks
        if fusion_results['vision']['human_detected']:
            safety_score *= 0.3
            safety_violations.append("human_proximity")
        
        if fusion_results['tactile']['contact_detected']:
            safety_score *= 0.2
            safety_violations.append("physical_contact")
        
        if fusion_results['proprioceptive']['acceleration_level'] > 2.0:
            safety_score *= 0.7
            safety_violations.append("high_acceleration")
        
        return {
            'fused_features': fused_features,
            'fused_confidence': fused_confidence,
            'safety_score': safety_score,
            'safety_violations': safety_violations,
            'modality_scores': {
                modality: {
                    'confidence': confidences.get(modality, 0.0),
                    'is_safe': safety_flags.get(modality, True)
                }
                for modality in self.fusion_weights.keys()
            },
            'fusion_weights': self.fusion_weights.copy(),
            'raw_results': fusion_results
        }
    
    def update_fusion_weights(self, new_weights: Dict[str, float]):
        """Update fusion weights based on learning"""
        for modality, weight in new_weights.items():
            if modality in self.fusion_weights:
                self.fusion_weights[modality] = max(0.0, min(1.0, weight))
        
        # Normalize weights
        total_weight = sum(self.fusion_weights.values())
        if total_weight > 0:
            for modality in self.fusion_weights:
                self.fusion_weights[modality] /= total_weight
    
    def update_safety_thresholds(self, new_thresholds: Dict[str, float]):
        """Update safety thresholds based on learning"""
        for modality, threshold in new_thresholds.items():
            if modality in self.safety_thresholds:
                self.safety_thresholds[modality] = max(0.0, min(1.0, threshold))
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """Get status of all sensor processors"""
        return {
            'vision': {
                'feature_dim': self.vision_processor.feature_dim,
                'confidence_threshold': self.vision_processor.confidence_threshold
            },
            'audio': {
                'feature_dim': self.audio_processor.feature_dim,
                'sample_rate': self.audio_processor.sample_rate
            },
            'tactile': {
                'feature_dim': self.tactile_processor.feature_dim,
                'sensor_count': self.tactile_processor.sensor_count
            },
            'proprioceptive': {
                'feature_dim': self.proprioceptive_processor.feature_dim
            },
            'fusion_weights': self.fusion_weights.copy(),
            'safety_thresholds': self.safety_thresholds.copy()
        } 