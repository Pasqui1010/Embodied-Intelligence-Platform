#!/usr/bin/env python3
"""
Spatial Reference Resolver

This module handles spatial reference resolution in natural language commands
using vision-language models for grounding spatial relationships.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import logging
import re
from enum import Enum

# ROS 2 imports
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import String, Float32

# Vision-Language Model imports
try:
    import clip
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available, using fallback spatial reference resolution")


class SpatialRelationType(Enum):
    """Types of spatial relationships"""
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    BEHIND = "behind"
    IN_FRONT_OF = "in_front_of"
    NEXT_TO = "next_to"
    NEAR = "near"
    FAR_FROM = "far_from"
    ABOVE = "above"
    BELOW = "below"
    INSIDE = "inside"
    OUTSIDE = "outside"


@dataclass
class ObjectDetection:
    """Represents a detected object in the scene"""
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    center: Tuple[float, float]
    area: float
    features: Optional[np.ndarray] = None


@dataclass
class SceneData:
    """Represents scene understanding data"""
    timestamp: float
    objects: List[ObjectDetection]
    image: Optional[np.ndarray] = None
    point_cloud: Optional[np.ndarray] = None
    camera_pose: Optional[Pose] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SpatialReference:
    """Represents a resolved spatial reference"""
    position: Tuple[float, float, float]  # x, y, z coordinates
    confidence: float
    reference_type: SpatialRelationType
    reference_object: Optional[str] = None
    description: str = ""
    metadata: Optional[Dict[str, Any]] = None


class SpatialReferenceResolver:
    """
    Resolves spatial references in natural language commands
    
    Uses vision-language models to ground spatial relationships
    between objects in the scene.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.logger = logging.getLogger(__name__)
        
        # Initialize CLIP model for vision-language grounding
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_processor = self._load_clip_model(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model.to(self.device)
        else:
            self.clip_model = None
            self.clip_processor = None
            self.device = "cpu"
        
        # Spatial relationship patterns
        self.spatial_patterns = {
            SpatialRelationType.LEFT_OF: [
                r"left\s+of", r"to\s+the\s+left", r"on\s+the\s+left\s+side"
            ],
            SpatialRelationType.RIGHT_OF: [
                r"right\s+of", r"to\s+the\s+right", r"on\s+the\s+right\s+side"
            ],
            SpatialRelationType.BEHIND: [
                r"behind", r"in\s+back\s+of", r"at\s+the\s+back\s+of"
            ],
            SpatialRelationType.IN_FRONT_OF: [
                r"in\s+front\s+of", r"ahead\s+of", r"before"
            ],
            SpatialRelationType.NEXT_TO: [
                r"next\s+to", r"beside", r"adjacent\s+to"
            ],
            SpatialRelationType.NEAR: [
                r"near", r"close\s+to", r"by", r"around"
            ],
            SpatialRelationType.FAR_FROM: [
                r"far\s+from", r"away\s+from", r"distant\s+from"
            ],
            SpatialRelationType.ABOVE: [
                r"above", r"over", r"on\s+top\s+of"
            ],
            SpatialRelationType.BELOW: [
                r"below", r"under", r"beneath", r"underneath"
            ]
        }
        
        # Confidence thresholds
        self.confidence_threshold = 0.7
        self.spatial_confidence_threshold = 0.6
        
    def _load_clip_model(self, model_name: str):
        """Load CLIP model for vision-language grounding"""
        try:
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            self.logger.info(f"Loaded CLIP model: {model_name}")
            return model, processor
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
            return None, None
    
    def resolve_reference(self, command: str, scene_data: SceneData) -> SpatialReference:
        """
        Resolve spatial references in natural language commands
        
        Args:
            command: Natural language command (e.g., "move to the left of the red cup")
            scene_data: Current scene understanding with object detections
            
        Returns:
            SpatialReference with resolved position and confidence
        """
        self.logger.info(f"Resolving spatial reference: {command}")
        
        # Extract spatial relationship and target object
        spatial_relation, target_object = self._extract_spatial_relation(command)
        
        if spatial_relation is None:
            # No spatial relation found, return default position
            default_pos = self._get_default_position(scene_data)
            return SpatialReference(
                position=default_pos,
                confidence=0.0,
                reference_type=SpatialRelationType.NEAR,
                description="default position"
            )
        
        # Find target object in scene
        if target_object is None:
            self.logger.warning("No target object specified")
            default_pos = self._get_default_position(scene_data)
            return SpatialReference(
                position=default_pos,
                confidence=0.0,
                reference_type=spatial_relation,
                description="no target object"
            )
            
        target_detection = self._find_target_object(target_object, scene_data)
        
        if target_detection is None:
            self.logger.warning(f"Target object '{target_object}' not found in scene")
            default_pos = self._get_default_position(scene_data)
            return SpatialReference(
                position=default_pos,
                confidence=0.0,
                reference_type=spatial_relation,
                reference_object=target_object,
                description=f"object not found: {target_object}"
            )
        
        # Calculate spatial position based on relationship
        position = self._calculate_spatial_position(
            spatial_relation, target_detection, scene_data
        )
        
        # Calculate confidence based on object detection and spatial reasoning
        confidence = self._calculate_confidence(target_detection, spatial_relation)
        
        return SpatialReference(
            position=position,
            confidence=confidence,
            reference_type=spatial_relation,
            reference_object=target_object,
            description=f"{spatial_relation.value} {target_object}",
            metadata={
                "target_confidence": target_detection.confidence,
                "spatial_relation": spatial_relation.value
            }
        )
    
    def _extract_spatial_relation(self, command: str) -> Tuple[Optional[SpatialRelationType], Optional[str]]:
        """Extract spatial relationship and target object from command"""
        command_lower = command.lower()
        
        # Find matching spatial pattern
        for relation_type, patterns in self.spatial_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command_lower)
                if match:
                    # Extract target object (simplified extraction)
                    # In practice, this would use more sophisticated NLP
                    words = command_lower.split()
                    target_words = []
                    found_relation = False
                    
                    for word in words:
                        if pattern.replace(r"\s+", " ") in word or any(p.replace(r"\s+", " ") in word for p in patterns):
                            found_relation = True
                            continue
                        if found_relation and word not in ["the", "a", "an", "of", "to", "in", "on", "at"]:
                            target_words.append(word)
                    
                    target_object = " ".join(target_words) if target_words else None
                    return relation_type, target_object
        
        return None, None
    
    def _find_target_object(self, target_object: str, scene_data: SceneData) -> Optional[ObjectDetection]:
        """Find target object in scene using CLIP or fallback methods"""
        if not target_object or not scene_data.objects:
            return None
        
        if self.clip_model is not None and scene_data.image is not None:
            return self._find_object_with_clip(target_object, scene_data)
        else:
            return self._find_object_fallback(target_object, scene_data)
    
    def _find_object_with_clip(self, target_object: str, scene_data: SceneData) -> Optional[ObjectDetection]:
        """Find object using CLIP vision-language grounding"""
        try:
            # Prepare text prompt
            text_prompts = [
                f"a {target_object}",
                f"the {target_object}",
                target_object,
                f"photo of a {target_object}",
                f"image of {target_object}"
            ]
            
            # Process image and text
            inputs = self.clip_processor(
                text=text_prompts,
                images=scene_data.image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get CLIP embeddings
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)
            
            # Find best matching object
            best_match_idx = torch.argmax(probs[0]).item()
            best_confidence = probs[0][best_match_idx].item()
            
            if best_confidence > self.confidence_threshold:
                # Return the object with highest confidence
                return max(scene_data.objects, key=lambda obj: obj.confidence)
            
        except Exception as e:
            self.logger.error(f"Error in CLIP object detection: {e}")
        
        return None
    
    def _find_object_fallback(self, target_object: str, scene_data: SceneData) -> Optional[ObjectDetection]:
        """Fallback object detection using simple text matching"""
        target_lower = target_object.lower()
        
        # Simple text matching
        for obj in scene_data.objects:
            if target_lower in obj.class_name.lower():
                return obj
        
        # Partial matching
        for obj in scene_data.objects:
            if any(word in obj.class_name.lower() for word in target_lower.split()):
                return obj
        
        return None
    
    def _calculate_spatial_position(self, 
                                  relation: SpatialRelationType, 
                                  target: ObjectDetection,
                                  scene_data: SceneData) -> Tuple[float, float, float]:
        """Calculate spatial position based on relationship and target object"""
        target_center = target.center
        
        # Get scene dimensions for relative positioning
        if scene_data.image is not None:
            height, width = scene_data.image.shape[:2]
        else:
            width, height = 640, 480  # Default dimensions
        
        # Calculate offset based on relationship type
        offset_x, offset_y = 0.0, 0.0
        
        if relation == SpatialRelationType.LEFT_OF:
            offset_x = -50.0  # 50 pixels to the left
        elif relation == SpatialRelationType.RIGHT_OF:
            offset_x = 50.0   # 50 pixels to the right
        elif relation == SpatialRelationType.BEHIND:
            offset_y = -50.0  # 50 pixels behind
        elif relation == SpatialRelationType.IN_FRONT_OF:
            offset_y = 50.0   # 50 pixels in front
        elif relation == SpatialRelationType.NEXT_TO:
            offset_x = 30.0   # 30 pixels to the side
        elif relation == SpatialRelationType.NEAR:
            offset_x = 20.0   # 20 pixels nearby
            offset_y = 20.0
        elif relation == SpatialRelationType.ABOVE:
            offset_y = -30.0  # 30 pixels above
        elif relation == SpatialRelationType.BELOW:
            offset_y = 30.0   # 30 pixels below
        
        # Calculate final position
        x = target_center[0] + offset_x
        y = target_center[1] + offset_y
        z = 0.0  # Default z-coordinate
        
        # Ensure position is within scene bounds
        x = max(0, min(x, width))
        y = max(0, min(y, height))
        
        return (x, y, z)
    
    def _calculate_confidence(self, target: ObjectDetection, relation: SpatialRelationType) -> float:
        """Calculate confidence in spatial reference resolution"""
        # Base confidence on object detection confidence
        base_confidence = target.confidence
        
        # Adjust based on spatial relationship complexity
        relation_confidence = {
            SpatialRelationType.NEAR: 0.9,
            SpatialRelationType.NEXT_TO: 0.8,
            SpatialRelationType.LEFT_OF: 0.7,
            SpatialRelationType.RIGHT_OF: 0.7,
            SpatialRelationType.IN_FRONT_OF: 0.6,
            SpatialRelationType.BEHIND: 0.6,
            SpatialRelationType.ABOVE: 0.5,
            SpatialRelationType.BELOW: 0.5,
            SpatialRelationType.FAR_FROM: 0.4
        }
        
        spatial_confidence = relation_confidence.get(relation, 0.5)
        
        # Combine confidences
        final_confidence = (base_confidence + spatial_confidence) / 2.0
        
        return min(final_confidence, 1.0)
    
    def _get_default_position(self, scene_data: SceneData) -> Tuple[float, float, float]:
        """Get default position when spatial reference cannot be resolved"""
        if scene_data.image is not None:
            height, width = scene_data.image.shape[:2]
            return (width / 2, height / 2, 0.0)
        else:
            return (320.0, 240.0, 0.0)  # Default center position
    
    def resolve_multiple_references(self, command: str, scene_data: SceneData) -> List[SpatialReference]:
        """Resolve multiple spatial references in a single command"""
        # This is a simplified implementation
        # In practice, this would use more sophisticated NLP parsing
        references = []
        
        # Try to resolve the main reference
        main_reference = self.resolve_reference(command, scene_data)
        if main_reference.confidence > self.spatial_confidence_threshold:
            references.append(main_reference)
        
        return references
    
    def validate_spatial_reference(self, reference: SpatialReference, scene_data: SceneData) -> bool:
        """Validate if a spatial reference is still valid given current scene"""
        if reference.reference_object is None:
            return False
        
        # Check if reference object still exists
        target = self._find_target_object(reference.reference_object, scene_data)
        if target is None:
            return False
        
        # Check if position is still reasonable
        current_position = self._calculate_spatial_position(
            reference.reference_type, target, scene_data
        )
        
        # Calculate distance between original and current position
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(reference.position, current_position)))
        
        # Position is valid if within reasonable distance
        return distance < 100.0  # 100 pixel threshold 