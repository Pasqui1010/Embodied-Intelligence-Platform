#!/usr/bin/env python3
"""
Test Spatial Reasoning

This module tests the spatial reasoning capabilities of the
Advanced Multi-Modal Reasoning Engine.
"""

import unittest
import time
import numpy as np
from unittest.mock import Mock, patch

from eip_reasoning_engine.spatial_reasoner import (
    SpatialReasoner, SpatialUnderstanding, ObjectSpatialInfo, SpatialRelation
)
from eip_reasoning_engine.multi_modal_reasoner import VisualContext, SpatialContext


class TestSpatialReasoner(unittest.TestCase):
    """Test cases for SpatialReasoner"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.reasoner = SpatialReasoner()
        
        # Create test data
        self.test_visual_context = VisualContext(
            objects=[
                {'name': 'red_cube', 'position': [1.0, 0.0, 0.5], 'dimensions': [0.1, 0.1, 0.1]},
                {'name': 'blue_sphere', 'position': [0.0, 1.0, 0.3], 'dimensions': [0.05, 0.05, 0.05]},
                {'name': 'green_cylinder', 'position': [-1.0, 0.5, 0.4], 'dimensions': [0.08, 0.08, 0.15]}
            ],
            scene_description="A table with multiple objects",
            spatial_relationships={
                'red_cube': ['near_table', 'above_surface'],
                'blue_sphere': ['near_table', 'above_surface'],
                'green_cylinder': ['near_table', 'above_surface']
            },
            affordances={
                'red_cube': ['graspable', 'movable'],
                'blue_sphere': ['graspable', 'movable'],
                'green_cylinder': ['graspable', 'movable']
            },
            confidence=0.8
        )
        
        self.test_spatial_context = SpatialContext(
            robot_pose={'x': 0.0, 'y': 0.0, 'z': 0.0},
            object_positions={
                'red_cube': (1.0, 0.0, 0.5),
                'blue_sphere': (0.0, 1.0, 0.3),
                'green_cylinder': (-1.0, 0.5, 0.4)
            },
            workspace_boundaries={
                'min_x': -5.0, 'max_x': 5.0,
                'min_y': -5.0, 'max_y': 5.0,
                'min_z': 0.0, 'max_z': 2.0
            },
            navigation_graph={},
            occupancy_grid=None
        )
    
    def test_initialization(self):
        """Test reasoner initialization"""
        self.assertIsNotNone(self.reasoner)
        self.assertEqual(self.reasoner.near_threshold, 0.5)
        self.assertEqual(self.reasoner.far_threshold, 2.0)
        self.assertEqual(self.reasoner.height_threshold, 0.3)
        self.assertEqual(self.reasoner.path_resolution, 0.1)
        self.assertEqual(self.reasoner.max_path_length, 10.0)
    
    def test_analyze_scene_basic(self):
        """Test basic scene analysis"""
        result = self.reasoner.analyze_scene(
            self.test_visual_context,
            self.test_spatial_context
        )
        
        self.assertIsInstance(result, SpatialUnderstanding)
        self.assertIsInstance(result.object_relationships, dict)
        self.assertIsInstance(result.navigation_paths, dict)
        self.assertIsInstance(result.spatial_constraints, list)
        self.assertIsInstance(result.affordance_map, dict)
        self.assertIsInstance(result.summary, str)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_extract_object_spatial_info(self):
        """Test object spatial information extraction"""
        object_info = self.reasoner._extract_object_spatial_info(
            self.test_visual_context,
            self.test_spatial_context
        )
        
        self.assertIsInstance(object_info, dict)
        self.assertIn('red_cube', object_info)
        self.assertIn('blue_sphere', object_info)
        self.assertIn('green_cylinder', object_info)
        
        # Check object info structure
        for obj_name, obj_info in object_info.items():
            self.assertIsInstance(obj_info, ObjectSpatialInfo)
            self.assertIsInstance(obj_info.position, tuple)
            self.assertEqual(len(obj_info.position), 3)
            self.assertIsInstance(obj_info.dimensions, tuple)
            self.assertEqual(len(obj_info.dimensions), 3)
            self.assertIsInstance(obj_info.orientation, float)
            self.assertIsInstance(obj_info.affordances, list)
            self.assertIsInstance(obj_info.spatial_relations, dict)
    
    def test_analyze_spatial_relationships(self):
        """Test spatial relationship analysis"""
        object_info = self.reasoner._extract_object_spatial_info(
            self.test_visual_context,
            self.test_spatial_context
        )
        
        relationships = self.reasoner._analyze_spatial_relationships(object_info)
        
        self.assertIsInstance(relationships, dict)
        self.assertIn('red_cube', relationships)
        self.assertIn('blue_sphere', relationships)
        self.assertIn('green_cylinder', relationships)
        
        # Check relationship structure
        for obj_name, rels in relationships.items():
            self.assertIsInstance(rels, list)
            for rel in rels:
                self.assertIsInstance(rel, tuple)
                self.assertEqual(len(rel), 3)
                self.assertIsInstance(rel[0], str)  # related object
                self.assertIsInstance(rel[1], SpatialRelation)  # relation type
                self.assertIsInstance(rel[2], float)  # confidence
                self.assertGreaterEqual(rel[2], 0.0)
                self.assertLessEqual(rel[2], 1.0)
    
    def test_calculate_spatial_relation(self):
        """Test spatial relation calculation"""
        obj1 = ObjectSpatialInfo(
            position=(0.0, 0.0, 0.0),
            dimensions=(0.1, 0.1, 0.1),
            orientation=0.0,
            affordances=['graspable'],
            spatial_relations={}
        )
        
        obj2 = ObjectSpatialInfo(
            position=(0.3, 0.0, 0.0),  # Near obj1
            dimensions=(0.1, 0.1, 0.1),
            orientation=0.0,
            affordances=['graspable'],
            spatial_relations={}
        )
        
        relation, confidence = self.reasoner._calculate_spatial_relation(obj1, obj2)
        
        self.assertIsInstance(relation, SpatialRelation)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Test different spatial relationships
        obj3 = ObjectSpatialInfo(
            position=(0.0, 0.0, 0.5),  # Above obj1
            dimensions=(0.1, 0.1, 0.1),
            orientation=0.0,
            affordances=['graspable'],
            spatial_relations={}
        )
        
        relation3, confidence3 = self.reasoner._calculate_spatial_relation(obj1, obj3)
        self.assertIsInstance(relation3, SpatialRelation)
    
    def test_generate_navigation_paths(self):
        """Test navigation path generation"""
        object_info = self.reasoner._extract_object_spatial_info(
            self.test_visual_context,
            self.test_spatial_context
        )
        
        paths = self.reasoner._generate_navigation_paths(
            self.test_spatial_context,
            object_info
        )
        
        self.assertIsInstance(paths, dict)
        self.assertIn('red_cube', paths)
        self.assertIn('blue_sphere', paths)
        self.assertIn('green_cylinder', paths)
        
        # Check path structure
        for obj_name, path in paths.items():
            self.assertIsInstance(path, list)
            for point in path:
                self.assertIsInstance(point, tuple)
                self.assertEqual(len(point), 3)
                for coord in point:
                    self.assertIsInstance(coord, float)
    
    def test_calculate_path_to_object(self):
        """Test path calculation to specific object"""
        start_pos = (0.0, 0.0, 0.0)
        target_pos = (1.0, 0.0, 0.5)
        
        object_info = self.reasoner._extract_object_spatial_info(
            self.test_visual_context,
            self.test_spatial_context
        )
        
        path = self.reasoner._calculate_path_to_object(
            start_pos,
            target_pos,
            object_info
        )
        
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)
        
        # Check path starts at start position
        self.assertEqual(path[0], start_pos)
        
        # Check path ends at target position
        self.assertEqual(path[-1], target_pos)
        
        # Check all points are tuples of 3 coordinates
        for point in path:
            self.assertIsInstance(point, tuple)
            self.assertEqual(len(point), 3)
    
    def test_is_path_clear(self):
        """Test path clearance checking"""
        start_pos = (0.0, 0.0, 0.0)
        target_pos = (1.0, 0.0, 0.0)
        
        object_info = self.reasoner._extract_object_spatial_info(
            self.test_visual_context,
            self.test_spatial_context
        )
        
        is_clear = self.reasoner._is_path_clear(
            start_pos,
            target_pos,
            object_info
        )
        
        self.assertIsInstance(is_clear, bool)
    
    def test_identify_spatial_constraints(self):
        """Test spatial constraint identification"""
        object_info = self.reasoner._extract_object_spatial_info(
            self.test_visual_context,
            self.test_spatial_context
        )
        
        constraints = self.reasoner._identify_spatial_constraints(
            object_info,
            self.test_spatial_context
        )
        
        self.assertIsInstance(constraints, list)
        
        # Check constraint format
        for constraint in constraints:
            self.assertIsInstance(constraint, str)
            self.assertGreater(len(constraint), 0)
    
    def test_create_affordance_map(self):
        """Test affordance map creation"""
        object_info = self.reasoner._extract_object_spatial_info(
            self.test_visual_context,
            self.test_spatial_context
        )
        
        affordance_map = self.reasoner._create_affordance_map(object_info)
        
        self.assertIsInstance(affordance_map, dict)
        self.assertIn('red_cube', affordance_map)
        self.assertIn('blue_sphere', affordance_map)
        self.assertIn('green_cylinder', affordance_map)
        
        # Check affordance lists
        for obj_name, affordances in affordance_map.items():
            self.assertIsInstance(affordances, list)
            for affordance in affordances:
                self.assertIsInstance(affordance, str)
    
    def test_generate_spatial_summary(self):
        """Test spatial summary generation"""
        object_info = self.reasoner._extract_object_spatial_info(
            self.test_visual_context,
            self.test_spatial_context
        )
        
        relationships = self.reasoner._analyze_spatial_relationships(object_info)
        constraints = self.reasoner._identify_spatial_constraints(
            object_info,
            self.test_spatial_context
        )
        affordance_map = self.reasoner._create_affordance_map(object_info)
        
        summary = self.reasoner._generate_spatial_summary(
            relationships,
            constraints,
            affordance_map
        )
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertIn('objects', summary.lower())
    
    def test_calculate_spatial_confidence(self):
        """Test spatial confidence calculation"""
        object_info = self.reasoner._extract_object_spatial_info(
            self.test_visual_context,
            self.test_spatial_context
        )
        
        confidence = self.reasoner._calculate_spatial_confidence(
            self.test_visual_context,
            object_info
        )
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_fallback_spatial_understanding(self):
        """Test fallback spatial understanding generation"""
        result = self.reasoner._generate_fallback_spatial_understanding()
        
        self.assertIsInstance(result, SpatialUnderstanding)
        self.assertEqual(result.confidence, 0.1)
        self.assertIn('Fallback', result.summary)
        self.assertIn('limited spatial awareness', result.summary)
    
    def test_performance_stats(self):
        """Test performance statistics"""
        # Simulate some reasoning operations
        for _ in range(5):
            self.reasoner.analyze_scene(
                self.test_visual_context,
                self.test_spatial_context
            )
        
        stats = self.reasoner.get_performance_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('avg_reasoning_time', stats)
        self.assertIn('max_reasoning_time', stats)
        self.assertIn('min_reasoning_time', stats)
        self.assertIn('std_reasoning_time', stats)
        
        # Check that stats are reasonable
        self.assertGreaterEqual(stats['avg_reasoning_time'], 0.0)
        self.assertGreaterEqual(stats['max_reasoning_time'], stats['min_reasoning_time'])
    
    def test_error_handling(self):
        """Test error handling in spatial reasoning"""
        # Test with invalid inputs
        with patch.object(self.reasoner, '_extract_object_spatial_info', side_effect=Exception("Test error")):
            result = self.reasoner.analyze_scene(
                self.test_visual_context,
                self.test_spatial_context
            )
            
            # Should return fallback understanding
            self.assertIsInstance(result, SpatialUnderstanding)
            self.assertEqual(result.confidence, 0.1)
    
    def test_complex_spatial_scene(self):
        """Test reasoning with complex spatial scene"""
        # Create more complex visual context
        complex_visual_context = VisualContext(
            objects=[
                {'name': 'red_cube', 'position': [1.0, 0.0, 0.5], 'dimensions': [0.1, 0.1, 0.1]},
                {'name': 'blue_sphere', 'position': [0.0, 1.0, 0.3], 'dimensions': [0.05, 0.05, 0.05]},
                {'name': 'green_cylinder', 'position': [-1.0, 0.5, 0.4], 'dimensions': [0.08, 0.08, 0.15]},
                {'name': 'yellow_pyramid', 'position': [0.5, -0.5, 0.2], 'dimensions': [0.12, 0.12, 0.1]},
                {'name': 'purple_cube', 'position': [2.0, 1.0, 0.6], 'dimensions': [0.15, 0.15, 0.15]}
            ],
            scene_description="Complex spatial scene with multiple objects",
            spatial_relationships={
                'red_cube': ['near_table', 'above_surface', 'left_of_blue_sphere'],
                'blue_sphere': ['near_table', 'above_surface', 'right_of_red_cube'],
                'green_cylinder': ['near_table', 'above_surface', 'behind_red_cube'],
                'yellow_pyramid': ['near_table', 'above_surface', 'front_of_red_cube'],
                'purple_cube': ['near_table', 'above_surface', 'far_from_red_cube']
            },
            affordances={
                'red_cube': ['graspable', 'movable', 'stackable'],
                'blue_sphere': ['graspable', 'movable', 'rollable'],
                'green_cylinder': ['graspable', 'movable', 'stackable'],
                'yellow_pyramid': ['graspable', 'movable', 'pointed'],
                'purple_cube': ['graspable', 'movable', 'stackable']
            },
            confidence=0.9
        )
        
        complex_spatial_context = SpatialContext(
            robot_pose={'x': 0.0, 'y': 0.0, 'z': 0.0},
            object_positions={
                'red_cube': (1.0, 0.0, 0.5),
                'blue_sphere': (0.0, 1.0, 0.3),
                'green_cylinder': (-1.0, 0.5, 0.4),
                'yellow_pyramid': (0.5, -0.5, 0.2),
                'purple_cube': (2.0, 1.0, 0.6)
            },
            workspace_boundaries={
                'min_x': -5.0, 'max_x': 5.0,
                'min_y': -5.0, 'max_y': 5.0,
                'min_z': 0.0, 'max_z': 2.0
            },
            navigation_graph={},
            occupancy_grid=None
        )
        
        result = self.reasoner.analyze_scene(
            complex_visual_context,
            complex_spatial_context
        )
        
        self.assertIsInstance(result, SpatialUnderstanding)
        self.assertGreater(len(result.object_relationships), 0)
        self.assertGreater(len(result.navigation_paths), 0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


if __name__ == '__main__':
    unittest.main() 