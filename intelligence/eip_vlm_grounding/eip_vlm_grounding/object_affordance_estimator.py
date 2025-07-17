#!/usr/bin/env python3
"""
Object Affordance Estimator

This module estimates manipulation affordances for detected objects,
including grasp points, stability assessment, and safety-aware filtering.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import logging
from enum import Enum

# ROS 2 imports
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import String, Float32, Bool

# Import from spatial reference resolver
from .spatial_reference_resolver import ObjectDetection


class AffordanceType(Enum):
    """Types of object affordances"""
    GRASP = "grasp"
    PUSH = "push"
    PULL = "pull"
    LIFT = "lift"
    PLACE = "place"
    ROTATE = "rotate"
    SLIDE = "slide"


class GraspType(Enum):
    """Types of grasp strategies"""
    PINCH = "pinch"
    PALMAR = "palmar"
    POWER = "power"
    PRECISION = "precision"
    LATERAL = "lateral"


@dataclass
class GraspPoint:
    """Represents a grasp point on an object"""
    position: Tuple[float, float, float]  # x, y, z coordinates
    orientation: Tuple[float, float, float, float]  # quaternion
    grasp_type: GraspType
    confidence: float
    approach_direction: Tuple[float, float, float]
    gripper_width: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StabilityAssessment:
    """Assessment of object stability"""
    stability_score: float  # 0.0 to 1.0
    tipping_risk: float  # 0.0 to 1.0
    sliding_risk: float  # 0.0 to 1.0
    center_of_mass: Tuple[float, float, float]
    support_polygon: List[Tuple[float, float]]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ManipulationDifficulty:
    """Assessment of manipulation difficulty"""
    difficulty_score: float  # 0.0 to 1.0
    precision_required: float  # 0.0 to 1.0
    force_required: float  # 0.0 to 1.0
    dexterity_required: float  # 0.0 to 1.0
    time_estimate: float  # seconds
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AffordanceSet:
    """Complete set of affordances for an object"""
    object_detection: ObjectDetection
    grasp_points: List[GraspPoint]
    stability: StabilityAssessment
    difficulty: ManipulationDifficulty
    safety_score: float  # 0.0 to 1.0
    available_affordances: List[AffordanceType]
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


class GraspNet(nn.Module):
    """Neural network for grasp point detection"""
    
    def __init__(self, input_channels: int = 3, num_grasp_types: int = 5):
        super(GraspNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Decoder for grasp points
        self.grasp_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_grasp_types, 1),
            nn.Sigmoid()
        )
        
        # Orientation decoder
        self.orientation_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 1),  # sin and cos of angle
            nn.Tanh()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        grasp_probs = self.grasp_decoder(features)
        orientation = self.orientation_decoder(features)
        return grasp_probs, orientation


class ObjectAffordanceEstimator:
    """
    Estimates manipulation affordances for detected objects
    
    Provides grasp point detection, stability assessment, and
    manipulation difficulty estimation with safety-aware filtering.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize grasp detection model
        self.grasp_net = self._load_grasp_model(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.grasp_net:
            self.grasp_net.to(self.device)
        
        # Object affordance database (simplified)
        self.affordance_database = {
            "cup": [AffordanceType.GRASP, AffordanceType.LIFT, AffordanceType.PLACE],
            "bottle": [AffordanceType.GRASP, AffordanceType.LIFT, AffordanceType.ROTATE],
            "box": [AffordanceType.GRASP, AffordanceType.PUSH, AffordanceType.PULL, AffordanceType.LIFT],
            "book": [AffordanceType.GRASP, AffordanceType.LIFT, AffordanceType.PLACE],
            "pen": [AffordanceType.GRASP, AffordanceType.PRECISION],
            "plate": [AffordanceType.GRASP, AffordanceType.LIFT, AffordanceType.PLACE],
            "fork": [AffordanceType.GRASP, AffordanceType.PRECISION],
            "knife": [AffordanceType.GRASP, AffordanceType.PRECISION],
            "spoon": [AffordanceType.GRASP, AffordanceType.PRECISION],
            "chair": [AffordanceType.PUSH, AffordanceType.PULL],
            "table": [AffordanceType.PUSH, AffordanceType.PULL],
            "door": [AffordanceType.PUSH, AffordanceType.PULL, AffordanceType.ROTATE]
        }
        
        # Safety thresholds
        self.min_grasp_confidence = 0.7
        self.min_stability_score = 0.6
        self.max_difficulty_score = 0.8
        self.min_safety_score = 0.7
        
    def _load_grasp_model(self, model_path: Optional[str]) -> Optional[GraspNet]:
        """Load pre-trained grasp detection model"""
        try:
            if model_path and torch.cuda.is_available():
                model = GraspNet()
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                self.logger.info(f"Loaded grasp model from {model_path}")
                return model
            else:
                self.logger.info("Using fallback grasp detection (no model loaded)")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load grasp model: {e}")
            return None
    
    def estimate_affordances(self, object_detection: ObjectDetection) -> AffordanceSet:
        """
        Estimate manipulation affordances for detected objects
        
        Args:
            object_detection: Detected object with bounding box and class
            
        Returns:
            AffordanceSet with grasp points, stability, and safety scores
        """
        self.logger.info(f"Estimating affordances for {object_detection.class_name}")
        
        # Detect grasp points
        grasp_points = self._detect_grasp_points(object_detection)
        
        # Assess stability
        stability = self._assess_stability(object_detection)
        
        # Estimate manipulation difficulty
        difficulty = self._estimate_difficulty(object_detection)
        
        # Get available affordances
        available_affordances = self._get_available_affordances(object_detection.class_name)
        
        # Calculate safety score
        safety_score = self._calculate_safety_score(grasp_points, stability, difficulty)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(grasp_points, stability, difficulty)
        
        return AffordanceSet(
            object_detection=object_detection,
            grasp_points=grasp_points,
            stability=stability,
            difficulty=difficulty,
            safety_score=safety_score,
            available_affordances=available_affordances,
            confidence=confidence
        )
    
    def _detect_grasp_points(self, object_detection: ObjectDetection) -> List[GraspPoint]:
        """Detect grasp points on the object"""
        grasp_points = []
        
        if self.grasp_net is not None and object_detection.features is not None:
            # Use neural network for grasp detection
            grasp_points = self._detect_grasp_with_network(object_detection)
        else:
            # Use geometric heuristics for grasp detection
            grasp_points = self._detect_grasp_geometric(object_detection)
        
        # Filter by confidence
        grasp_points = [gp for gp in grasp_points if gp.confidence >= self.min_grasp_confidence]
        
        return grasp_points
    
    def _detect_grasp_with_network(self, object_detection: ObjectDetection) -> List[GraspPoint]:
        """Detect grasp points using neural network"""
        try:
            # Prepare input tensor
            if object_detection.features is None:
                return []
            
            # Convert features to tensor
            input_tensor = torch.from_numpy(object_detection.features).unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                grasp_probs, orientation = self.grasp_net(input_tensor)
            
            # Process outputs
            grasp_probs = grasp_probs.squeeze().cpu().numpy()
            orientation = orientation.squeeze().cpu().numpy()
            
            # Extract grasp points
            grasp_points = []
            for grasp_type_idx in range(grasp_probs.shape[0]):
                grasp_type = GraspType(list(GraspType)[grasp_type_idx])
                
                # Find local maxima in grasp probability
                prob_map = grasp_probs[grasp_type_idx]
                peaks = self._find_peaks(prob_map, threshold=0.5)
                
                for peak in peaks:
                    y, x = peak
                    confidence = prob_map[y, x]
                    
                    # Get orientation
                    angle_sin = orientation[0, y, x]
                    angle_cos = orientation[1, y, x]
                    angle = np.arctan2(angle_sin, angle_cos)
                    
                    # Convert to 3D position
                    position = self._pixel_to_3d(x, y, object_detection)
                    
                    # Create grasp point
                    grasp_point = GraspPoint(
                        position=position,
                        orientation=(0, 0, np.sin(angle/2), np.cos(angle/2)),  # quaternion
                        grasp_type=grasp_type,
                        confidence=confidence,
                        approach_direction=(0, 0, -1),  # approach from above
                        gripper_width=0.08  # default gripper width
                    )
                    grasp_points.append(grasp_point)
            
            return grasp_points
            
        except Exception as e:
            self.logger.error(f"Error in neural grasp detection: {e}")
            return []
    
    def _detect_grasp_geometric(self, object_detection: ObjectDetection) -> List[GraspPoint]:
        """Detect grasp points using geometric heuristics"""
        grasp_points = []
        
        # Get object center and dimensions
        center = object_detection.center
        bbox = object_detection.bbox
        width, height = bbox[2], bbox[3]
        
        # Generate grasp points based on object type and size
        if "cup" in object_detection.class_name.lower() or "bottle" in object_detection.class_name.lower():
            # Cylindrical objects - grasp from sides
            grasp_points.extend([
                GraspPoint(
                    position=(center[0] - width/4, center[1], 0),
                    orientation=(0, 0, 0, 1),
                    grasp_type=GraspType.PALMAR,
                    confidence=0.8,
                    approach_direction=(1, 0, 0),
                    gripper_width=min(width/2, 0.1)
                ),
                GraspPoint(
                    position=(center[0] + width/4, center[1], 0),
                    orientation=(0, 0, 0, 1),
                    grasp_type=GraspType.PALMAR,
                    confidence=0.8,
                    approach_direction=(-1, 0, 0),
                    gripper_width=min(width/2, 0.1)
                )
            ])
        
        elif "box" in object_detection.class_name.lower() or "book" in object_detection.class_name.lower():
            # Rectangular objects - grasp from top
            grasp_points.append(
                GraspPoint(
                    position=(center[0], center[1], 0),
                    orientation=(0, 0, 0, 1),
                    grasp_type=GraspType.POWER,
                    confidence=0.9,
                    approach_direction=(0, 0, -1),
                    gripper_width=min(width, 0.15)
                )
            )
        
        elif "pen" in object_detection.class_name.lower() or "fork" in object_detection.class_name.lower():
            # Small objects - precision grasp
            grasp_points.append(
                GraspPoint(
                    position=(center[0], center[1], 0),
                    orientation=(0, 0, 0, 1),
                    grasp_type=GraspType.PRECISION,
                    confidence=0.7,
                    approach_direction=(0, 0, -1),
                    gripper_width=0.02
                )
            )
        
        else:
            # Default grasp point
            grasp_points.append(
                GraspPoint(
                    position=(center[0], center[1], 0),
                    orientation=(0, 0, 0, 1),
                    grasp_type=GraspType.POWER,
                    confidence=0.6,
                    approach_direction=(0, 0, -1),
                    gripper_width=min(width, 0.1)
                )
            )
        
        return grasp_points
    
    def _assess_stability(self, object_detection: ObjectDetection) -> StabilityAssessment:
        """Assess object stability for manipulation"""
        # Get object properties
        bbox = object_detection.bbox
        width, height = bbox[2], bbox[3]
        area = object_detection.area
        
        # Calculate stability based on object properties
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Base stability score
        stability_score = min(1.0, area / 10000.0)  # Larger objects are more stable
        
        # Adjust for aspect ratio
        if aspect_ratio > 2.0:  # Very wide objects
            stability_score *= 0.8
        elif aspect_ratio < 0.5:  # Very tall objects
            stability_score *= 0.6
        
        # Calculate tipping risk
        tipping_risk = 1.0 - stability_score
        
        # Calculate sliding risk (simplified)
        sliding_risk = 0.2 if "smooth" in object_detection.class_name.lower() else 0.1
        
        # Estimate center of mass
        center_of_mass = (object_detection.center[0], object_detection.center[1], 0)
        
        # Estimate support polygon (simplified as bounding box)
        support_polygon = [
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            (bbox[0], bbox[1] + bbox[3])
        ]
        
        return StabilityAssessment(
            stability_score=stability_score,
            tipping_risk=tipping_risk,
            sliding_risk=sliding_risk,
            center_of_mass=center_of_mass,
            support_polygon=support_polygon
        )
    
    def _estimate_difficulty(self, object_detection: ObjectDetection) -> ManipulationDifficulty:
        """Estimate manipulation difficulty"""
        # Get object properties
        bbox = object_detection.bbox
        width, height = bbox[2], bbox[3]
        area = object_detection.area
        
        # Base difficulty score
        difficulty_score = 0.5
        
        # Adjust for object size
        if area < 1000:  # Very small objects
            difficulty_score += 0.3
            precision_required = 0.9
        elif area > 10000:  # Very large objects
            difficulty_score += 0.2
            precision_required = 0.3
        else:
            precision_required = 0.6
        
        # Adjust for object type
        if "fragile" in object_detection.class_name.lower():
            difficulty_score += 0.2
            precision_required += 0.2
        
        if "sharp" in object_detection.class_name.lower():
            difficulty_score += 0.3
            precision_required += 0.3
        
        # Estimate force requirements
        if area > 5000:
            force_required = 0.8
        elif area < 500:
            force_required = 0.2
        else:
            force_required = 0.5
        
        # Estimate dexterity requirements
        aspect_ratio = width / height if height > 0 else 1.0
        if aspect_ratio > 3.0 or aspect_ratio < 0.3:
            dexterity_required = 0.8
        else:
            dexterity_required = 0.4
        
        # Estimate time requirements
        time_estimate = 2.0 + difficulty_score * 3.0  # 2-5 seconds
        
        return ManipulationDifficulty(
            difficulty_score=min(difficulty_score, 1.0),
            precision_required=min(precision_required, 1.0),
            force_required=force_required,
            dexterity_required=dexterity_required,
            time_estimate=time_estimate
        )
    
    def _get_available_affordances(self, class_name: str) -> List[AffordanceType]:
        """Get available affordances for object class"""
        class_lower = class_name.lower()
        
        # Check database
        for key, affordances in self.affordance_database.items():
            if key in class_lower:
                return affordances
        
        # Default affordances
        return [AffordanceType.GRASP, AffordanceType.LIFT]
    
    def _calculate_safety_score(self, 
                               grasp_points: List[GraspPoint],
                               stability: StabilityAssessment,
                               difficulty: ManipulationDifficulty) -> float:
        """Calculate safety score for manipulation"""
        if not grasp_points:
            return 0.0
        
        # Base safety score
        safety_score = 1.0
        
        # Reduce for low stability
        safety_score *= stability.stability_score
        
        # Reduce for high difficulty
        safety_score *= (1.0 - difficulty.difficulty_score * 0.5)
        
        # Reduce for high tipping risk
        safety_score *= (1.0 - stability.tipping_risk * 0.3)
        
        # Reduce for high sliding risk
        safety_score *= (1.0 - stability.sliding_risk * 0.2)
        
        # Boost for high confidence grasp points
        max_grasp_confidence = max(gp.confidence for gp in grasp_points)
        safety_score *= (0.5 + max_grasp_confidence * 0.5)
        
        return max(0.0, min(1.0, safety_score))
    
    def _calculate_confidence(self,
                            grasp_points: List[GraspPoint],
                            stability: StabilityAssessment,
                            difficulty: ManipulationDifficulty) -> float:
        """Calculate overall confidence in affordance estimation"""
        if not grasp_points:
            return 0.0
        
        # Average grasp confidence
        grasp_confidence = np.mean([gp.confidence for gp in grasp_points])
        
        # Stability confidence
        stability_confidence = stability.stability_score
        
        # Difficulty confidence (inverse relationship)
        difficulty_confidence = 1.0 - difficulty.difficulty_score
        
        # Overall confidence
        confidence = (grasp_confidence + stability_confidence + difficulty_confidence) / 3.0
        
        return confidence
    
    def _find_peaks(self, prob_map: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int]]:
        """Find peaks in probability map"""
        peaks = []
        h, w = prob_map.shape
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                if prob_map[y, x] > threshold:
                    # Check if it's a local maximum
                    is_peak = True
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            if prob_map[y+dy, x+dx] >= prob_map[y, x]:
                                is_peak = False
                                break
                        if not is_peak:
                            break
                    
                    if is_peak:
                        peaks.append((y, x))
        
        return peaks
    
    def _pixel_to_3d(self, x: int, y: int, object_detection: ObjectDetection) -> Tuple[float, float, float]:
        """Convert pixel coordinates to 3D world coordinates"""
        # Simplified conversion - in practice, this would use camera intrinsics
        # and depth information
        bbox = object_detection.bbox
        
        # Normalize to object coordinates
        rel_x = (x - bbox[0]) / bbox[2]
        rel_y = (y - bbox[1]) / bbox[3]
        
        # Convert to world coordinates
        world_x = object_detection.center[0] + (rel_x - 0.5) * bbox[2]
        world_y = object_detection.center[1] + (rel_y - 0.5) * bbox[3]
        world_z = 0.0  # Default z-coordinate
        
        return (world_x, world_y, world_z)
    
    def filter_safe_affordances(self, affordance_set: AffordanceSet) -> AffordanceSet:
        """Filter affordances based on safety criteria"""
        # Filter grasp points by safety
        safe_grasp_points = [
            gp for gp in affordance_set.grasp_points
            if gp.confidence >= self.min_grasp_confidence
        ]
        
        # Check stability requirements
        if affordance_set.stability.stability_score < self.min_stability_score:
            safe_grasp_points = []
        
        # Check difficulty requirements
        if affordance_set.difficulty.difficulty_score > self.max_difficulty_score:
            safe_grasp_points = []
        
        # Recalculate safety score
        new_safety_score = self._calculate_safety_score(
            safe_grasp_points,
            affordance_set.stability,
            affordance_set.difficulty
        )
        
        # Create filtered affordance set
        return AffordanceSet(
            object_detection=affordance_set.object_detection,
            grasp_points=safe_grasp_points,
            stability=affordance_set.stability,
            difficulty=affordance_set.difficulty,
            safety_score=new_safety_score,
            available_affordances=affordance_set.available_affordances,
            confidence=affordance_set.confidence
        ) 