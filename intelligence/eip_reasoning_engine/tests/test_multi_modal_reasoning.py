#!/usr/bin/env python3
"""
Test Multi-Modal Reasoning

This module tests the multi-modal reasoning capabilities of the
Advanced Multi-Modal Reasoning Engine.
"""

import unittest
import time
import numpy as np
from unittest.mock import Mock, patch

from eip_reasoning_engine.multi_modal_reasoner import (
    MultiModalReasoner, VisualContext, SpatialContext, 
    SafetyConstraints, ReasoningResult
)


class TestMultiModalReasoner(unittest.TestCase):
    """Test cases for MultiModalReasoner"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.reasoner = MultiModalReasoner()
        
        # Create test data
        self.test_visual_context = VisualContext(
            objects=[
                {'name': 'red_cube', 'position': [1.0, 0.0, 0.5], 'dimensions': [0.1, 0.1, 0.1]},
                {'name': 'blue_sphere', 'position': [0.0, 1.0, 0.3], 'dimensions': [0.05, 0.05, 0.05]}
            ],
            scene_description="A table with a red cube and blue sphere",
            spatial_relationships={
                'red_cube': ['near_table', 'above_surface'],
                'blue_sphere': ['near_table', 'above_surface']
            },
            affordances={
                'red_cube': ['graspable', 'movable'],
                'blue_sphere': ['graspable', 'movable']
            },
            confidence=0.8
        )
        
        self.test_spatial_context = SpatialContext(
            robot_pose={'x': 0.0, 'y': 0.0, 'z': 0.0},
            object_positions={
                'red_cube': (1.0, 0.0, 0.5),
                'blue_sphere': (0.0, 1.0, 0.3)
            },
            workspace_boundaries={
                'min_x': -5.0, 'max_x': 5.0,
                'min_y': -5.0, 'max_y': 5.0,
                'min_z': 0.0, 'max_z': 2.0
            },
            navigation_graph={},
            occupancy_grid=None
        )
        
        self.test_safety_constraints = SafetyConstraints(
            collision_threshold=0.7,
            human_proximity_threshold=0.8,
            velocity_limits={'linear': 1.0, 'angular': 1.0},
            workspace_boundaries={
                'min_x': -5.0, 'max_x': 5.0,
                'min_y': -5.0, 'max_y': 5.0
            },
            emergency_stop_conditions=['collision_detected', 'human_proximity']
        )
    
    def test_initialization(self):
        """Test reasoner initialization"""
        self.assertIsNotNone(self.reasoner)
        self.assertIsNotNone(self.reasoner.spatial_reasoner)
        self.assertIsNotNone(self.reasoner.temporal_reasoner)
        self.assertIsNotNone(self.reasoner.causal_reasoner)
        self.assertTrue(len(self.reasoner.reasoning_capabilities) > 0)
    
    def test_reason_about_scene_basic(self):
        """Test basic reasoning about a scene"""
        result = self.reasoner.reason_about_scene(
            self.test_visual_context,
            "Move to the red cube",
            self.test_spatial_context,
            self.test_safety_constraints
        )
        
        self.assertIsInstance(result, ReasoningResult)
        self.assertIsNotNone(result.plan)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertGreaterEqual(result.safety_score, 0.0)
        self.assertLessEqual(result.safety_score, 1.0)
        self.assertIsInstance(result.reasoning_steps, list)
        self.assertIsInstance(result.alternative_plans, list)
        self.assertGreaterEqual(result.execution_time, 0.0)
    
    def test_language_grounding(self):
        """Test language command grounding"""
        grounded = self.reasoner._ground_language_command(
            "Pick up the blue sphere",
            self.test_visual_context,
            self.test_spatial_context
        )
        
        self.assertIsInstance(grounded, dict)
        self.assertIn('action', grounded)
        self.assertIn('objects', grounded)
        self.assertIn('spatial_references', grounded)
        self.assertIn('confidence', grounded)
        self.assertGreaterEqual(grounded['confidence'], 0.0)
        self.assertLessEqual(grounded['confidence'], 1.0)
    
    def test_safe_plan_generation(self):
        """Test safe plan generation"""
        # Create a mock temporal plan
        from eip_interfaces.msg import TaskPlan, TaskStep
        
        temporal_plan = TaskPlan()
        temporal_plan.plan_id = "test_plan"
        temporal_plan.goal_description = "Test goal"
        temporal_plan.steps = []
        temporal_plan.estimated_duration_seconds = 5.0
        temporal_plan.required_capabilities = ['navigation']
        temporal_plan.safety_considerations = []
        
        # Add a test step
        step = TaskStep()
        step.action_type = 'move'
        step.description = 'Move to target'
        step.estimated_duration = 2.0
        step.preconditions = []
        step.postconditions = ['at_target']
        temporal_plan.steps.append(step)
        
        # Create mock causal analysis
        mock_causal_analysis = Mock()
        mock_causal_analysis.risk_level = 'medium'
        mock_causal_analysis.risk_score = 0.4
        
        safe_plan = self.reasoner._generate_safe_plan(
            temporal_plan,
            mock_causal_analysis,
            self.test_safety_constraints
        )
        
        self.assertIsInstance(safe_plan, TaskPlan)
        self.assertGreater(len(safe_plan.steps), 0)
        self.assertGreaterEqual(safe_plan.safety_score, 0.0)
        self.assertLessEqual(safe_plan.safety_score, 1.0)
    
    def test_confidence_calculation(self):
        """Test confidence calculation"""
        # Create mock inputs
        mock_spatial_understanding = Mock()
        mock_spatial_understanding.confidence = 0.8
        
        language_understanding = {
            'action': 'move',
            'confidence': 0.7
        }
        
        mock_causal_analysis = Mock()
        mock_causal_analysis.confidence = 0.6
        
        confidence = self.reasoner._calculate_confidence(
            mock_spatial_understanding,
            language_understanding,
            mock_causal_analysis
        )
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        # Should be approximately the average of the confidences
        expected_confidence = (0.8 + 0.7 + 0.6) / 3
        self.assertAlmostEqual(confidence, expected_confidence, places=1)
    
    def test_alternative_plan_generation(self):
        """Test alternative plan generation"""
        # Create a mock primary plan
        from eip_interfaces.msg import TaskPlan
        
        primary_plan = TaskPlan()
        primary_plan.plan_id = "primary_plan"
        primary_plan.goal_description = "Test goal"
        primary_plan.steps = []
        primary_plan.estimated_duration_seconds = 5.0
        primary_plan.required_capabilities = ['navigation']
        primary_plan.safety_considerations = []
        
        # Create mock spatial understanding
        mock_spatial_understanding = Mock()
        
        alternatives = self.reasoner._generate_alternatives(
            primary_plan,
            mock_spatial_understanding,
            self.test_safety_constraints
        )
        
        self.assertIsInstance(alternatives, list)
        self.assertGreater(len(alternatives), 0)
        
        for alt_plan in alternatives:
            self.assertIsInstance(alt_plan, TaskPlan)
            self.assertNotEqual(alt_plan.plan_id, primary_plan.plan_id)
    
    def test_fallback_plan_generation(self):
        """Test fallback plan generation"""
        result = self.reasoner._generate_fallback_plan(
            "Test command",
            self.test_safety_constraints
        )
        
        self.assertIsInstance(result, ReasoningResult)
        self.assertIsNotNone(result.plan)
        self.assertEqual(result.confidence, 0.1)
        self.assertEqual(result.safety_score, 0.9)
        self.assertIn('Fallback', result.reasoning_steps[0])
    
    def test_performance_tracking(self):
        """Test performance tracking"""
        # Track some performance metrics
        self.reasoner._track_performance(0.5, 0.8)
        self.reasoner._track_performance(0.3, 0.9)
        self.reasoner._track_performance(0.7, 0.6)
        
        stats = self.reasoner.get_performance_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('avg_reasoning_time', stats)
        self.assertIn('max_reasoning_time', stats)
        self.assertIn('min_reasoning_time', stats)
        self.assertIn('avg_confidence', stats)
        self.assertIn('min_confidence', stats)
        self.assertIn('max_confidence', stats)
        
        # Check that stats are reasonable
        self.assertGreaterEqual(stats['avg_reasoning_time'], 0.0)
        self.assertGreaterEqual(stats['max_reasoning_time'], stats['min_reasoning_time'])
        self.assertGreaterEqual(stats['avg_confidence'], 0.0)
        self.assertLessEqual(stats['avg_confidence'], 1.0)
    
    def test_error_handling(self):
        """Test error handling in reasoning"""
        # Test with invalid inputs
        with patch.object(self.reasoner.spatial_reasoner, 'analyze_scene', side_effect=Exception("Test error")):
            result = self.reasoner.reason_about_scene(
                self.test_visual_context,
                "Test command",
                self.test_spatial_context,
                self.test_safety_constraints
            )
            
            # Should return fallback plan
            self.assertIsInstance(result, ReasoningResult)
            self.assertEqual(result.confidence, 0.1)
    
    def test_reasoning_capabilities(self):
        """Test reasoning capabilities configuration"""
        capabilities = self.reasoner.reasoning_capabilities
        
        self.assertIsInstance(capabilities, dict)
        self.assertIn('spatial', capabilities)
        self.assertIn('temporal', capabilities)
        self.assertIn('causal', capabilities)
        self.assertIn('social', capabilities)
        self.assertIn('safety', capabilities)
        
        # All capabilities should be enabled by default
        for capability in capabilities.values():
            self.assertTrue(capability)
    
    def test_complex_scene_reasoning(self):
        """Test reasoning with complex scene"""
        # Create more complex visual context
        complex_visual_context = VisualContext(
            objects=[
                {'name': 'red_cube', 'position': [1.0, 0.0, 0.5], 'dimensions': [0.1, 0.1, 0.1]},
                {'name': 'blue_sphere', 'position': [0.0, 1.0, 0.3], 'dimensions': [0.05, 0.05, 0.05]},
                {'name': 'green_cylinder', 'position': [-1.0, 0.5, 0.4], 'dimensions': [0.08, 0.08, 0.15]},
                {'name': 'yellow_pyramid', 'position': [0.5, -0.5, 0.2], 'dimensions': [0.12, 0.12, 0.1]}
            ],
            scene_description="Complex scene with multiple objects",
            spatial_relationships={
                'red_cube': ['near_table', 'above_surface', 'left_of_blue_sphere'],
                'blue_sphere': ['near_table', 'above_surface', 'right_of_red_cube'],
                'green_cylinder': ['near_table', 'above_surface', 'behind_red_cube'],
                'yellow_pyramid': ['near_table', 'above_surface', 'front_of_red_cube']
            },
            affordances={
                'red_cube': ['graspable', 'movable', 'stackable'],
                'blue_sphere': ['graspable', 'movable', 'rollable'],
                'green_cylinder': ['graspable', 'movable', 'stackable'],
                'yellow_pyramid': ['graspable', 'movable', 'pointed']
            },
            confidence=0.9
        )
        
        result = self.reasoner.reason_about_scene(
            complex_visual_context,
            "Arrange the objects in a line from left to right",
            self.test_spatial_context,
            self.test_safety_constraints
        )
        
        self.assertIsInstance(result, ReasoningResult)
        self.assertGreater(len(result.reasoning_steps), 0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


if __name__ == '__main__':
    unittest.main() 