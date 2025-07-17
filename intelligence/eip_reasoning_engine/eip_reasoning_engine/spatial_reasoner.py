#!/usr/bin/env python3
"""
Spatial Reasoner

This module implements spatial reasoning capabilities for understanding object relationships,
navigation paths, and spatial constraints in robotic environments.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from enum import Enum

from .multi_modal_reasoner import VisualContext, SpatialContext


class SpatialRelation(Enum):
    """Types of spatial relationships"""
    NEAR = "near"
    FAR = "far"
    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BEHIND = "behind"
    INSIDE = "inside"
    OUTSIDE = "outside"
    ON = "on"
    UNDER = "under"


@dataclass
class SpatialUnderstanding:
    """Result of spatial reasoning"""
    object_relationships: Dict[str, List[Tuple[str, SpatialRelation, float]]]
    navigation_paths: Dict[str, List[Tuple[float, float, float]]]
    spatial_constraints: List[str]
    affordance_map: Dict[str, List[str]]
    summary: str
    confidence: float


@dataclass
class ObjectSpatialInfo:
    """Spatial information about an object"""
    position: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]
    orientation: float
    affordances: List[str]
    spatial_relations: Dict[str, List[str]]


class SpatialReasoner:
    """
    Spatial reasoning engine for understanding object relationships and navigation
    """
    
    def __init__(self):
        """Initialize the spatial reasoner"""
        self.logger = logging.getLogger(__name__)
        
        # Spatial reasoning parameters
        self.near_threshold = 0.5  # meters
        self.far_threshold = 2.0   # meters
        self.height_threshold = 0.3  # meters
        
        # Navigation parameters
        self.path_resolution = 0.1  # meters
        self.max_path_length = 10.0  # meters
        
        # Performance tracking
        self.reasoning_times = []
        
        self.logger.info("Spatial Reasoner initialized successfully")
    
    def analyze_scene(self, 
                     visual_context: VisualContext,
                     spatial_context: SpatialContext) -> SpatialUnderstanding:
        """
        Analyze spatial relationships in the scene
        
        Args:
            visual_context: Visual understanding of the scene
            spatial_context: Spatial context information
            
        Returns:
            SpatialUnderstanding with relationships and constraints
        """
        start_time = time.time()
        
        try:
            # 1. Extract object spatial information
            object_spatial_info = self._extract_object_spatial_info(
                visual_context, spatial_context
            )
            
            # 2. Analyze spatial relationships
            relationships = self._analyze_spatial_relationships(object_spatial_info)
            
            # 3. Generate navigation paths
            navigation_paths = self._generate_navigation_paths(
                spatial_context, object_spatial_info
            )
            
            # 4. Identify spatial constraints
            spatial_constraints = self._identify_spatial_constraints(
                object_spatial_info, spatial_context
            )
            
            # 5. Create affordance map
            affordance_map = self._create_affordance_map(object_spatial_info)
            
            # 6. Generate summary
            summary = self._generate_spatial_summary(
                relationships, spatial_constraints, affordance_map
            )
            
            # 7. Calculate confidence
            confidence = self._calculate_spatial_confidence(
                visual_context, object_spatial_info
            )
            
            execution_time = time.time() - start_time
            self.reasoning_times.append(execution_time)
            
            result = SpatialUnderstanding(
                object_relationships=relationships,
                navigation_paths=navigation_paths,
                spatial_constraints=spatial_constraints,
                affordance_map=affordance_map,
                summary=summary,
                confidence=confidence
            )
            
            self.logger.info(f"Spatial analysis completed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in spatial reasoning: {e}")
            return self._generate_fallback_spatial_understanding()
    
    def _extract_object_spatial_info(self, 
                                   visual_context: VisualContext,
                                   spatial_context: SpatialContext) -> Dict[str, ObjectSpatialInfo]:
        """Extract spatial information for objects"""
        object_info = {}
        
        # Process objects from visual context
        for obj in visual_context.objects:
            obj_name = obj.get('name', 'unknown_object')
            
            # Get position from spatial context
            position = spatial_context.object_positions.get(
                obj_name, 
                (0.0, 0.0, 0.0)
            )
            
            # Extract dimensions (default if not available)
            dimensions = obj.get('dimensions', (0.1, 0.1, 0.1))
            
            # Extract orientation
            orientation = obj.get('orientation', 0.0)
            
            # Get affordances from visual context
            affordances = visual_context.affordances.get(obj_name, [])
            
            # Get spatial relations
            spatial_relations = visual_context.spatial_relationships.get(obj_name, {})
            
            object_info[obj_name] = ObjectSpatialInfo(
                position=position,
                dimensions=dimensions,
                orientation=orientation,
                affordances=affordances,
                spatial_relations=spatial_relations
            )
        
        return object_info
    
    def _analyze_spatial_relationships(self, 
                                     object_info: Dict[str, ObjectSpatialInfo]) -> Dict[str, List[Tuple[str, SpatialRelation, float]]]:
        """Analyze spatial relationships between objects"""
        relationships = {}
        
        object_names = list(object_info.keys())
        
        for i, obj1_name in enumerate(object_names):
            obj1 = object_info[obj1_name]
            relationships[obj1_name] = []
            
            for j, obj2_name in enumerate(object_names):
                if i == j:
                    continue
                
                obj2 = object_info[obj2_name]
                
                # Calculate spatial relationship
                relation, confidence = self._calculate_spatial_relation(obj1, obj2)
                
                if relation and confidence > 0.3:  # Only include confident relationships
                    relationships[obj1_name].append((obj2_name, relation, confidence))
        
        return relationships
    
    def _calculate_spatial_relation(self, 
                                  obj1: ObjectSpatialInfo,
                                  obj2: ObjectSpatialInfo) -> Tuple[Optional[SpatialRelation], float]:
        """Calculate spatial relationship between two objects"""
        pos1 = np.array(obj1.position)
        pos2 = np.array(obj2.position)
        
        # Calculate distance
        distance = np.linalg.norm(pos1[:2] - pos2[:2])  # 2D distance
        
        # Calculate height difference
        height_diff = pos1[2] - pos2[2]
        
        # Determine relationship based on distance and height
        if distance < self.near_threshold:
            if abs(height_diff) < self.height_threshold:
                return SpatialRelation.NEAR, 0.9
            elif height_diff > self.height_threshold:
                return SpatialRelation.ABOVE, 0.8
            else:
                return SpatialRelation.BELOW, 0.8
        elif distance > self.far_threshold:
            return SpatialRelation.FAR, 0.7
        else:
            # Medium distance - determine left/right/front/behind
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            
            if abs(dx) > abs(dy):
                if dx > 0:
                    return SpatialRelation.RIGHT, 0.6
                else:
                    return SpatialRelation.LEFT, 0.6
            else:
                if dy > 0:
                    return SpatialRelation.FRONT, 0.6
                else:
                    return SpatialRelation.BEHIND, 0.6
        
        return None, 0.0
    
    def _generate_navigation_paths(self, 
                                 spatial_context: SpatialContext,
                                 object_info: Dict[str, ObjectSpatialInfo]) -> Dict[str, List[Tuple[float, float, float]]]:
        """Generate navigation paths between objects"""
        navigation_paths = {}
        
        # Get robot position
        robot_pos = spatial_context.robot_pose
        robot_position = (robot_pos.get('x', 0.0), robot_pos.get('y', 0.0), robot_pos.get('z', 0.0))
        
        # Generate paths to each object
        for obj_name, obj in object_info.items():
            path = self._calculate_path_to_object(robot_position, obj.position, object_info)
            if path:
                navigation_paths[obj_name] = path
        
        return navigation_paths
    
    def _calculate_path_to_object(self, 
                                start_pos: Tuple[float, float, float],
                                target_pos: Tuple[float, float, float],
                                object_info: Dict[str, ObjectSpatialInfo]) -> List[Tuple[float, float, float]]:
        """Calculate path to target object avoiding obstacles"""
        # Simple straight-line path with obstacle avoidance
        path = []
        
        # Check if direct path is possible
        if self._is_path_clear(start_pos, target_pos, object_info):
            # Direct path
            num_steps = int(np.linalg.norm(np.array(target_pos) - np.array(start_pos)) / self.path_resolution)
            for i in range(num_steps + 1):
                t = i / num_steps
                pos = tuple(np.array(start_pos) * (1 - t) + np.array(target_pos) * t)
                path.append(pos)
        else:
            # Find alternative path
            path = self._find_alternative_path(start_pos, target_pos, object_info)
        
        return path
    
    def _is_path_clear(self, 
                      start_pos: Tuple[float, float, float],
                      target_pos: Tuple[float, float, float],
                      object_info: Dict[str, ObjectSpatialInfo]) -> bool:
        """Check if direct path between two points is clear of obstacles"""
        # Simple collision check
        path_vector = np.array(target_pos) - np.array(start_pos)
        path_length = np.linalg.norm(path_vector)
        
        if path_length == 0:
            return True
        
        # Check multiple points along the path
        num_checks = int(path_length / self.path_resolution)
        for i in range(num_checks + 1):
            t = i / num_checks
            check_pos = tuple(np.array(start_pos) + path_vector * t)
            
            # Check distance to all objects
            for obj in object_info.values():
                obj_distance = np.linalg.norm(np.array(check_pos) - np.array(obj.position))
                if obj_distance < 0.2:  # Minimum clearance
                    return False
        
        return True
    
    def _find_alternative_path(self, 
                             start_pos: Tuple[float, float, float],
                             target_pos: Tuple[float, float, float],
                             object_info: Dict[str, ObjectSpatialInfo]) -> List[Tuple[float, float, float]]:
        """Find alternative path avoiding obstacles"""
        # Simple waypoint-based path finding
        path = [start_pos]
        
        # Find intermediate waypoints
        waypoints = self._find_waypoints(start_pos, target_pos, object_info)
        
        for waypoint in waypoints:
            path.append(waypoint)
        
        path.append(target_pos)
        return path
    
    def _find_waypoints(self, 
                       start_pos: Tuple[float, float, float],
                       target_pos: Tuple[float, float, float],
                       object_info: Dict[str, ObjectSpatialInfo]) -> List[Tuple[float, float, float]]:
        """Find waypoints for navigation"""
        waypoints = []
        
        # Simple strategy: go around obstacles
        mid_point = tuple((np.array(start_pos) + np.array(target_pos)) / 2)
        
        # Check if midpoint is clear
        if self._is_path_clear(start_pos, mid_point, object_info) and \
           self._is_path_clear(mid_point, target_pos, object_info):
            waypoints.append(mid_point)
        else:
            # Find offset waypoint
            offset = np.array([0.5, 0.5, 0.0])  # Simple offset
            offset_waypoint = tuple(np.array(mid_point) + offset)
            waypoints.append(offset_waypoint)
        
        return waypoints
    
    def _identify_spatial_constraints(self, 
                                    object_info: Dict[str, ObjectSpatialInfo],
                                    spatial_context: SpatialContext) -> List[str]:
        """Identify spatial constraints in the environment"""
        constraints = []
        
        # Check workspace boundaries
        robot_pos = spatial_context.robot_pose
        robot_x, robot_y = robot_pos.get('x', 0.0), robot_pos.get('y', 0.0)
        
        boundaries = spatial_context.workspace_boundaries
        if robot_x < boundaries.get('min_x', -5.0) + 0.5:
            constraints.append("Robot near left workspace boundary")
        if robot_x > boundaries.get('max_x', 5.0) - 0.5:
            constraints.append("Robot near right workspace boundary")
        if robot_y < boundaries.get('min_y', -5.0) + 0.5:
            constraints.append("Robot near bottom workspace boundary")
        if robot_y > boundaries.get('max_y', 5.0) - 0.5:
            constraints.append("Robot near top workspace boundary")
        
        # Check object proximity constraints
        for obj_name, obj in object_info.items():
            robot_pos_array = np.array([robot_x, robot_y, robot_pos.get('z', 0.0)])
            obj_distance = np.linalg.norm(robot_pos_array - np.array(obj.position))
            
            if obj_distance < 0.3:
                constraints.append(f"Robot too close to {obj_name}")
            elif obj_distance < 0.5:
                constraints.append(f"Robot near {obj_name} - limited maneuverability")
        
        return constraints
    
    def _create_affordance_map(self, 
                             object_info: Dict[str, ObjectSpatialInfo]) -> Dict[str, List[str]]:
        """Create map of object affordances"""
        affordance_map = {}
        
        for obj_name, obj in object_info.items():
            affordance_map[obj_name] = obj.affordances.copy()
        
        return affordance_map
    
    def _generate_spatial_summary(self, 
                                relationships: Dict[str, List[Tuple[str, SpatialRelation, float]]],
                                constraints: List[str],
                                affordance_map: Dict[str, List[str]]) -> str:
        """Generate summary of spatial understanding"""
        summary_parts = []
        
        # Count objects and relationships
        num_objects = len(relationships)
        total_relationships = sum(len(rels) for rels in relationships.values())
        
        summary_parts.append(f"Scene contains {num_objects} objects with {total_relationships} spatial relationships")
        
        # Key relationships
        key_relationships = []
        for obj_name, rels in relationships.items():
            if rels:
                # Get most confident relationship
                best_rel = max(rels, key=lambda x: x[2])
                key_relationships.append(f"{obj_name} is {best_rel[1].value} {best_rel[0]}")
        
        if key_relationships:
            summary_parts.append(f"Key relationships: {', '.join(key_relationships[:3])}")
        
        # Constraints
        if constraints:
            summary_parts.append(f"Spatial constraints: {len(constraints)} active")
        
        # Affordances
        total_affordances = sum(len(affs) for affs in affordance_map.values())
        summary_parts.append(f"Total affordances: {total_affordances}")
        
        return ". ".join(summary_parts)
    
    def _calculate_spatial_confidence(self, 
                                    visual_context: VisualContext,
                                    object_info: Dict[str, ObjectSpatialInfo]) -> float:
        """Calculate confidence in spatial understanding"""
        confidence_factors = []
        
        # Visual confidence
        confidence_factors.append(visual_context.confidence)
        
        # Object detection confidence
        if object_info:
            avg_object_confidence = np.mean([
                obj.get('confidence', 0.5) for obj in visual_context.objects
            ])
            confidence_factors.append(avg_object_confidence)
        
        # Spatial relationship confidence
        if object_info:
            relationship_confidences = []
            for obj_name, obj in object_info.items():
                for rel_name, rels in obj.spatial_relations.items():
                    if rels:
                        relationship_confidences.append(0.7)  # Assume good relationship detection
            
            if relationship_confidences:
                confidence_factors.append(np.mean(relationship_confidences))
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _generate_fallback_spatial_understanding(self) -> SpatialUnderstanding:
        """Generate fallback spatial understanding when reasoning fails"""
        return SpatialUnderstanding(
            object_relationships={},
            navigation_paths={},
            spatial_constraints=['Spatial reasoning failed - proceed with caution'],
            affordance_map={},
            summary="Fallback spatial understanding - limited spatial awareness",
            confidence=0.1
        )
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.reasoning_times:
            return {}
        
        times = np.array(self.reasoning_times)
        return {
            'avg_reasoning_time': np.mean(times),
            'max_reasoning_time': np.max(times),
            'min_reasoning_time': np.min(times),
            'std_reasoning_time': np.std(times)
        } 