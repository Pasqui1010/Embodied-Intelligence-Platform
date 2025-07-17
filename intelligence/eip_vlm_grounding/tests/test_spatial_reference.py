#!/usr/bin/env python3
"""
Test Spatial Reference Resolver

Tests for spatial reference resolution functionality.
"""

import unittest
import numpy as np
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eip_vlm_grounding'))

from spatial_reference_resolver import (
    SpatialReferenceResolver, SceneData, ObjectDetection, 
    SpatialReference, SpatialRelationType
)


class TestSpatialReferenceResolver(unittest.TestCase):
    """Test cases for SpatialReferenceResolver"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.resolver = SpatialReferenceResolver()
        
        # Create test scene data
        self.test_objects = [
            ObjectDetection(
                class_name="cup",
                confidence=0.9,
                bbox=(100, 100, 50, 50),
                center=(125, 125),
                area=2500
            ),
            ObjectDetection(
                class_name="bottle",
                confidence=0.8,
                bbox=(200, 150, 30, 80),
                center=(215, 190),
                area=2400
            ),
            ObjectDetection(
                class_name="book",
                confidence=0.7,
                bbox=(300, 200, 60, 40),
                center=(330, 220),
                area=2400
            )
        ]
        
        self.test_scene_data = SceneData(
            timestamp=time.time(),
            objects=self.test_objects,
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            point_cloud=None,
            camera_pose=None
        )
    
    def test_resolve_left_of_reference(self):
        """Test resolving 'left of' spatial reference"""
        command = "move to the left of the cup"
        reference = self.resolver.resolve_reference(command, self.test_scene_data)
        
        self.assertIsInstance(reference, SpatialReference)
        self.assertEqual(reference.reference_type, SpatialRelationType.LEFT_OF)
        self.assertEqual(reference.reference_object, "cup")
        self.assertGreater(reference.confidence, 0.0)
        self.assertLess(reference.confidence, 1.0)
    
    def test_resolve_right_of_reference(self):
        """Test resolving 'right of' spatial reference"""
        command = "move to the right of the bottle"
        reference = self.resolver.resolve_reference(command, self.test_scene_data)
        
        self.assertIsInstance(reference, SpatialReference)
        self.assertEqual(reference.reference_type, SpatialRelationType.RIGHT_OF)
        self.assertEqual(reference.reference_object, "bottle")
        self.assertGreater(reference.confidence, 0.0)
    
    def test_resolve_near_reference(self):
        """Test resolving 'near' spatial reference"""
        command = "move near the book"
        reference = self.resolver.resolve_reference(command, self.test_scene_data)
        
        self.assertIsInstance(reference, SpatialReference)
        self.assertEqual(reference.reference_type, SpatialRelationType.NEAR)
        self.assertEqual(reference.reference_object, "book")
        self.assertGreater(reference.confidence, 0.0)
    
    def test_resolve_next_to_reference(self):
        """Test resolving 'next to' spatial reference"""
        command = "move next to the cup"
        reference = self.resolver.resolve_reference(command, self.test_scene_data)
        
        self.assertIsInstance(reference, SpatialReference)
        self.assertEqual(reference.reference_type, SpatialRelationType.NEXT_TO)
        self.assertEqual(reference.reference_object, "cup")
        self.assertGreater(reference.confidence, 0.0)
    
    def test_resolve_behind_reference(self):
        """Test resolving 'behind' spatial reference"""
        command = "move behind the bottle"
        reference = self.resolver.resolve_reference(command, self.test_scene_data)
        
        self.assertIsInstance(reference, SpatialReference)
        self.assertEqual(reference.reference_type, SpatialRelationType.BEHIND)
        self.assertEqual(reference.reference_object, "bottle")
        self.assertGreater(reference.confidence, 0.0)
    
    def test_resolve_in_front_of_reference(self):
        """Test resolving 'in front of' spatial reference"""
        command = "move in front of the book"
        reference = self.resolver.resolve_reference(command, self.test_scene_data)
        
        self.assertIsInstance(reference, SpatialReference)
        self.assertEqual(reference.reference_type, SpatialRelationType.IN_FRONT_OF)
        self.assertEqual(reference.reference_object, "book")
        self.assertGreater(reference.confidence, 0.0)
    
    def test_resolve_above_reference(self):
        """Test resolving 'above' spatial reference"""
        command = "move above the cup"
        reference = self.resolver.resolve_reference(command, self.test_scene_data)
        
        self.assertIsInstance(reference, SpatialReference)
        self.assertEqual(reference.reference_type, SpatialRelationType.ABOVE)
        self.assertEqual(reference.reference_object, "cup")
        self.assertGreater(reference.confidence, 0.0)
    
    def test_resolve_below_reference(self):
        """Test resolving 'below' spatial reference"""
        command = "move below the bottle"
        reference = self.resolver.resolve_reference(command, self.test_scene_data)
        
        self.assertIsInstance(reference, SpatialReference)
        self.assertEqual(reference.reference_type, SpatialRelationType.BELOW)
        self.assertEqual(reference.reference_object, "bottle")
        self.assertGreater(reference.confidence, 0.0)
    
    def test_resolve_far_from_reference(self):
        """Test resolving 'far from' spatial reference"""
        command = "move far from the book"
        reference = self.resolver.resolve_reference(command, self.test_scene_data)
        
        self.assertIsInstance(reference, SpatialReference)
        self.assertEqual(reference.reference_type, SpatialRelationType.FAR_FROM)
        self.assertEqual(reference.reference_object, "book")
        self.assertGreater(reference.confidence, 0.0)
    
    def test_resolve_multiple_references(self):
        """Test resolving multiple spatial references"""
        command = "move to the left of the cup and near the bottle"
        references = self.resolver.resolve_multiple_references(command, self.test_scene_data)
        
        self.assertIsInstance(references, list)
        self.assertGreater(len(references), 0)
        
        for reference in references:
            self.assertIsInstance(reference, SpatialReference)
            self.assertGreater(reference.confidence, 0.0)
    
    def test_resolve_unknown_object(self):
        """Test resolving reference to unknown object"""
        command = "move to the left of the unknown_object"
        reference = self.resolver.resolve_reference(command, self.test_scene_data)
        
        self.assertIsInstance(reference, SpatialReference)
        self.assertEqual(reference.reference_object, "unknown_object")
        self.assertEqual(reference.confidence, 0.0)  # Should have zero confidence
    
    def test_resolve_no_spatial_relation(self):
        """Test resolving command without spatial relation"""
        command = "move forward"
        reference = self.resolver.resolve_reference(command, self.test_scene_data)
        
        self.assertIsInstance(reference, SpatialReference)
        self.assertEqual(reference.reference_type, SpatialRelationType.NEAR)
        self.assertEqual(reference.confidence, 0.0)
    
    def test_resolve_empty_scene(self):
        """Test resolving reference in empty scene"""
        empty_scene = SceneData(
            timestamp=time.time(),
            objects=[],
            image=np.zeros((480, 640, 3), dtype=np.uint8)
        )
        
        command = "move to the left of the cup"
        reference = self.resolver.resolve_reference(command, empty_scene)
        
        self.assertIsInstance(reference, SpatialReference)
        self.assertEqual(reference.confidence, 0.0)
    
    def test_validate_spatial_reference(self):
        """Test spatial reference validation"""
        command = "move to the left of the cup"
        reference = self.resolver.resolve_reference(command, self.test_scene_data)
        
        # Should be valid initially
        is_valid = self.resolver.validate_spatial_reference(reference, self.test_scene_data)
        self.assertTrue(is_valid)
        
        # Test with modified scene (object removed)
        modified_scene = SceneData(
            timestamp=time.time(),
            objects=[],  # No objects
            image=np.zeros((480, 640, 3), dtype=np.uint8)
        )
        
        is_valid = self.resolver.validate_spatial_reference(reference, modified_scene)
        self.assertFalse(is_valid)
    
    def test_calculate_spatial_position(self):
        """Test spatial position calculation"""
        # Test left of position
        cup_obj = self.test_objects[0]  # cup
        position = self.resolver._calculate_spatial_position(
            SpatialRelationType.LEFT_OF, cup_obj, self.test_scene_data
        )
        
        self.assertIsInstance(position, tuple)
        self.assertEqual(len(position), 3)
        self.assertLess(position[0], cup_obj.center[0])  # Should be to the left
        
        # Test right of position
        position = self.resolver._calculate_spatial_position(
            SpatialRelationType.RIGHT_OF, cup_obj, self.test_scene_data
        )
        
        self.assertGreater(position[0], cup_obj.center[0])  # Should be to the right
        
        # Test near position
        position = self.resolver._calculate_spatial_position(
            SpatialRelationType.NEAR, cup_obj, self.test_scene_data
        )
        
        # Should be close to the object
        distance = np.sqrt((position[0] - cup_obj.center[0])**2 + 
                          (position[1] - cup_obj.center[1])**2)
        self.assertLess(distance, 50.0)  # Within 50 pixels
    
    def test_extract_spatial_relation(self):
        """Test spatial relation extraction"""
        # Test left of
        relation, obj = self.resolver._extract_spatial_relation("move to the left of the red cup")
        self.assertEqual(relation, SpatialRelationType.LEFT_OF)
        self.assertEqual(obj, "red cup")
        
        # Test right of
        relation, obj = self.resolver._extract_spatial_relation("go to the right of the bottle")
        self.assertEqual(relation, SpatialRelationType.RIGHT_OF)
        self.assertEqual(obj, "bottle")
        
        # Test near
        relation, obj = self.resolver._extract_spatial_relation("move near the book")
        self.assertEqual(relation, SpatialRelationType.NEAR)
        self.assertEqual(obj, "book")
        
        # Test no relation
        relation, obj = self.resolver._extract_spatial_relation("move forward")
        self.assertIsNone(relation)
        self.assertIsNone(obj)
    
    def test_find_target_object(self):
        """Test target object finding"""
        # Test exact match
        obj = self.resolver._find_target_object("cup", self.test_scene_data)
        self.assertIsNotNone(obj)
        self.assertEqual(obj.class_name, "cup")
        
        # Test partial match
        obj = self.resolver._find_target_object("red cup", self.test_scene_data)
        self.assertIsNotNone(obj)
        self.assertEqual(obj.class_name, "cup")
        
        # Test no match
        obj = self.resolver._find_target_object("unknown", self.test_scene_data)
        self.assertIsNone(obj)
    
    def test_calculate_confidence(self):
        """Test confidence calculation"""
        cup_obj = self.test_objects[0]
        
        # Test high confidence object with simple relation
        confidence = self.resolver._calculate_confidence(cup_obj, SpatialRelationType.NEAR)
        self.assertGreater(confidence, 0.8)
        
        # Test high confidence object with complex relation
        confidence = self.resolver._calculate_confidence(cup_obj, SpatialRelationType.BEHIND)
        self.assertLess(confidence, 0.8)
        
        # Test low confidence object
        low_conf_obj = ObjectDetection(
            class_name="unknown",
            confidence=0.3,
            bbox=(100, 100, 50, 50),
            center=(125, 125),
            area=2500
        )
        
        confidence = self.resolver._calculate_confidence(low_conf_obj, SpatialRelationType.NEAR)
        self.assertLess(confidence, 0.7)


if __name__ == '__main__':
    import time
    unittest.main() 