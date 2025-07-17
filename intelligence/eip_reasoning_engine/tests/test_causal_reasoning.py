#!/usr/bin/env python3
"""
Test Causal Reasoning

This module tests the causal reasoning capabilities of the
Advanced Multi-Modal Reasoning Engine.
"""

import unittest
import time
import numpy as np
from unittest.mock import Mock, patch

from eip_reasoning_engine.causal_reasoner import (
    CausalReasoner, CausalAnalysis, CausalEffect, RiskAssessment,
    RiskLevel, EffectType
)
from eip_reasoning_engine.spatial_reasoner import SpatialUnderstanding
from eip_reasoning_engine.multi_modal_reasoner import SafetyConstraints
from eip_interfaces.msg import TaskPlan, TaskStep


class TestCausalReasoner(unittest.TestCase):
    """Test cases for CausalReasoner"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.reasoner = CausalReasoner()
        
        # Create test data
        self.test_task_plan = TaskPlan()
        self.test_task_plan.plan_id = "test_plan"
        self.test_task_plan.goal_description = "Move to target"
        self.test_task_plan.steps = []
        self.test_task_plan.estimated_duration_seconds = 5.0
        self.test_task_plan.required_capabilities = ['navigation']
        self.test_task_plan.safety_considerations = []
        
        # Add test steps
        step1 = TaskStep()
        step1.action_type = 'move'
        step1.description = 'Move to target position'
        step1.estimated_duration = 3.0
        step1.preconditions = ['path_clear']
        step1.postconditions = ['at_target']
        self.test_task_plan.steps.append(step1)
        
        step2 = TaskStep()
        step2.action_type = 'verify'
        step2.description = 'Verify arrival'
        step2.estimated_duration = 1.0
        step2.preconditions = ['at_target']
        step2.postconditions = ['target_reached']
        self.test_task_plan.steps.append(step2)
        
        self.test_spatial_understanding = SpatialUnderstanding(
            object_relationships={
                'robot': [('obstacle', 'near', 0.8)],
                'obstacle': [('robot', 'near', 0.8)]
            },
            navigation_paths={
                'target': [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
            },
            spatial_constraints=['Robot near obstacle', 'Narrow path'],
            affordance_map={
                'robot': ['movable'],
                'obstacle': ['avoidable']
            },
            summary="Robot near obstacle with narrow path to target",
            confidence=0.7
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
        self.assertEqual(self.reasoner.max_chain_length, 5)
        self.assertEqual(self.reasoner.min_effect_probability, 0.1)
        self.assertEqual(self.reasoner.max_analysis_time, 3.0)
        
        # Check risk thresholds
        self.assertIn(RiskLevel.LOW, self.reasoner.risk_thresholds)
        self.assertIn(RiskLevel.MEDIUM, self.reasoner.risk_thresholds)
        self.assertIn(RiskLevel.HIGH, self.reasoner.risk_thresholds)
        self.assertIn(RiskLevel.CRITICAL, self.reasoner.risk_thresholds)
        
        # Check causal knowledge
        self.assertIsInstance(self.reasoner.causal_knowledge, dict)
        self.assertIn('move', self.reasoner.causal_knowledge)
        self.assertIn('pick', self.reasoner.causal_knowledge)
        self.assertIn('place', self.reasoner.causal_knowledge)
    
    def test_analyze_effects_basic(self):
        """Test basic effect analysis"""
        result = self.reasoner.analyze_effects(
            self.test_task_plan,
            self.test_spatial_understanding,
            self.test_safety_constraints
        )
        
        self.assertIsInstance(result, CausalAnalysis)
        self.assertIsInstance(result.effects, list)
        self.assertIsInstance(result.risk_assessment, RiskAssessment)
        self.assertIsInstance(result.causal_chains, list)
        self.assertIsInstance(result.safety_implications, list)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_extract_causal_factors(self):
        """Test causal factor extraction"""
        factors = self.reasoner._extract_causal_factors(
            self.test_task_plan,
            self.test_spatial_understanding
        )
        
        self.assertIsInstance(factors, dict)
        self.assertIn('actions', factors)
        self.assertIn('objects', factors)
        self.assertIn('spatial_constraints', factors)
        self.assertIn('environmental_factors', factors)
        
        # Check actions
        self.assertIsInstance(factors['actions'], list)
        self.assertGreater(len(factors['actions']), 0)
        
        # Check objects
        self.assertIsInstance(factors['objects'], list)
        self.assertIn('robot', factors['objects'])
        self.assertIn('obstacle', factors['objects'])
        
        # Check spatial constraints
        self.assertIsInstance(factors['spatial_constraints'], list)
        self.assertGreater(len(factors['spatial_constraints']), 0)
    
    def test_predict_effects(self):
        """Test effect prediction"""
        factors = self.reasoner._extract_causal_factors(
            self.test_task_plan,
            self.test_spatial_understanding
        )
        
        effects = self.reasoner._predict_effects(
            factors,
            self.test_spatial_understanding
        )
        
        self.assertIsInstance(effects, list)
        
        for effect in effects:
            self.assertIsInstance(effect, CausalEffect)
            self.assertIsInstance(effect.action, str)
            self.assertIsInstance(effect.effect_type, EffectType)
            self.assertIsInstance(effect.description, str)
            self.assertGreaterEqual(effect.probability, 0.0)
            self.assertLessEqual(effect.probability, 1.0)
            self.assertGreaterEqual(effect.severity, 0.0)
            self.assertLessEqual(effect.severity, 1.0)
            self.assertIsInstance(effect.affected_objects, list)
            self.assertGreaterEqual(effect.time_delay, 0.0)
    
    def test_adjust_effect_probability(self):
        """Test effect probability adjustment"""
        base_probability = 0.5
        
        adjusted_prob = self.reasoner._adjust_effect_probability(
            base_probability,
            'move',
            self.test_spatial_understanding
        )
        
        self.assertIsInstance(adjusted_prob, float)
        self.assertGreaterEqual(adjusted_prob, 0.0)
        self.assertLessEqual(adjusted_prob, 1.0)
        
        # Test with different action types
        for action_type in ['move', 'pick', 'place', 'observe']:
            adjusted = self.reasoner._adjust_effect_probability(
                base_probability,
                action_type,
                self.test_spatial_understanding
            )
            self.assertGreaterEqual(adjusted, 0.0)
            self.assertLessEqual(adjusted, 1.0)
    
    def test_adjust_effect_severity(self):
        """Test effect severity adjustment"""
        base_severity = 0.5
        
        adjusted_severity = self.reasoner._adjust_effect_severity(
            base_severity,
            'move',
            self.test_spatial_understanding
        )
        
        self.assertIsInstance(adjusted_severity, float)
        self.assertGreaterEqual(adjusted_severity, 0.0)
        self.assertLessEqual(adjusted_severity, 1.0)
        
        # Test with different action types
        for action_type in ['move', 'pick', 'place', 'observe']:
            adjusted = self.reasoner._adjust_effect_severity(
                base_severity,
                action_type,
                self.test_spatial_understanding
            )
            self.assertGreaterEqual(adjusted, 0.0)
            self.assertLessEqual(adjusted, 1.0)
    
    def test_identify_affected_objects(self):
        """Test affected object identification"""
        objects = ['robot', 'obstacle', 'target']
        
        # Test different action types
        for action_type in ['move', 'pick', 'place', 'observe']:
            affected = self.reasoner._identify_affected_objects(
                action_type,
                objects
            )
            
            self.assertIsInstance(affected, list)
            
            if action_type in ['pick', 'place']:
                # These actions should affect objects
                self.assertGreater(len(affected), 0)
    
    def test_estimate_effect_delay(self):
        """Test effect delay estimation"""
        # Test different action types and descriptions
        test_cases = [
            ('move', 'Robot position changes'),
            ('pick', 'Object grasped'),
            ('place', 'Object placed'),
            ('observe', 'Scene understanding improved')
        ]
        
        for action_type, description in test_cases:
            delay = self.reasoner._estimate_effect_delay(action_type, description)
            
            self.assertIsInstance(delay, float)
            self.assertGreaterEqual(delay, 0.0)
    
    def test_assess_risks(self):
        """Test risk assessment"""
        # Create test effects
        effects = [
            CausalEffect(
                action='move',
                effect_type=EffectType.NEGATIVE,
                description='Risk of collision',
                probability=0.3,
                severity=0.8,
                affected_objects=['robot'],
                time_delay=0.1
            ),
            CausalEffect(
                action='move',
                effect_type=EffectType.NEUTRAL,
                description='Energy consumption',
                probability=1.0,
                severity=0.2,
                affected_objects=['robot'],
                time_delay=0.0
            )
        ]
        
        risk_assessment = self.reasoner._assess_risks(
            effects,
            self.test_safety_constraints,
            self.test_spatial_understanding
        )
        
        self.assertIsInstance(risk_assessment, RiskAssessment)
        self.assertIsInstance(risk_assessment.risk_level, RiskLevel)
        self.assertGreaterEqual(risk_assessment.risk_score, 0.0)
        self.assertLessEqual(risk_assessment.risk_score, 1.0)
        self.assertIsInstance(risk_assessment.risk_factors, list)
        self.assertIsInstance(risk_assessment.mitigation_strategies, list)
        self.assertGreaterEqual(risk_assessment.confidence, 0.0)
        self.assertLessEqual(risk_assessment.confidence, 1.0)
    
    def test_determine_risk_level(self):
        """Test risk level determination"""
        # Test different risk scores
        test_cases = [
            (0.1, RiskLevel.LOW),
            (0.5, RiskLevel.MEDIUM),
            (0.8, RiskLevel.HIGH),
            (0.95, RiskLevel.CRITICAL)
        ]
        
        for risk_score, expected_level in test_cases:
            level = self.reasoner._determine_risk_level(risk_score)
            self.assertEqual(level, expected_level)
    
    def test_generate_mitigation_strategies(self):
        """Test mitigation strategy generation"""
        effects = [
            CausalEffect(
                action='move',
                effect_type=EffectType.NEGATIVE,
                description='Risk of collision',
                probability=0.3,
                severity=0.8,
                affected_objects=['robot'],
                time_delay=0.1
            )
        ]
        
        risk_factors = ['obstacles_present', 'narrow_path']
        
        strategies = self.reasoner._generate_mitigation_strategies(
            effects,
            risk_factors,
            self.test_safety_constraints
        )
        
        self.assertIsInstance(strategies, list)
        self.assertGreater(len(strategies), 0)
        
        for strategy in strategies:
            self.assertIsInstance(strategy, str)
            self.assertGreater(len(strategy), 0)
    
    def test_calculate_risk_confidence(self):
        """Test risk confidence calculation"""
        effects = [
            CausalEffect(
                action='move',
                effect_type=EffectType.NEGATIVE,
                description='Risk of collision',
                probability=0.8,
                severity=0.7,
                affected_objects=['robot'],
                time_delay=0.1
            )
        ]
        
        confidence = self.reasoner._calculate_risk_confidence(
            effects,
            self.test_spatial_understanding
        )
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_identify_causal_chains(self):
        """Test causal chain identification"""
        effects = [
            CausalEffect(
                action='move',
                effect_type=EffectType.NEGATIVE,
                description='Risk of collision',
                probability=0.3,
                severity=0.8,
                affected_objects=['robot'],
                time_delay=0.1
            )
        ]
        
        chains = self.reasoner._identify_causal_chains(effects)
        
        self.assertIsInstance(chains, list)
        
        for chain in chains:
            self.assertIsInstance(chain, list)
            self.assertGreater(len(chain), 0)
            for element in chain:
                self.assertIsInstance(element, str)
    
    def test_analyze_safety_implications(self):
        """Test safety implications analysis"""
        effects = [
            CausalEffect(
                action='move',
                effect_type=EffectType.NEGATIVE,
                description='Risk of collision',
                probability=0.3,
                severity=0.8,
                affected_objects=['robot'],
                time_delay=0.1
            )
        ]
        
        risk_assessment = RiskAssessment(
            risk_level=RiskLevel.MEDIUM,
            risk_score=0.6,
            risk_factors=['obstacles_present'],
            mitigation_strategies=['path_planning'],
            confidence=0.7
        )
        
        implications = self.reasoner._analyze_safety_implications(
            effects,
            risk_assessment,
            self.test_safety_constraints
        )
        
        self.assertIsInstance(implications, list)
        self.assertGreater(len(implications), 0)
        
        for implication in implications:
            self.assertIsInstance(implication, str)
            self.assertGreater(len(implication), 0)
    
    def test_calculate_causal_confidence(self):
        """Test causal confidence calculation"""
        effects = [
            CausalEffect(
                action='move',
                effect_type=EffectType.NEGATIVE,
                description='Risk of collision',
                probability=0.8,
                severity=0.7,
                affected_objects=['robot'],
                time_delay=0.1
            )
        ]
        
        confidence = self.reasoner._calculate_causal_confidence(
            effects,
            self.test_spatial_understanding
        )
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_fallback_analysis(self):
        """Test fallback analysis generation"""
        result = self.reasoner._generate_fallback_analysis()
        
        self.assertIsInstance(result, CausalAnalysis)
        self.assertEqual(len(result.effects), 1)
        self.assertEqual(result.effects[0].effect_type, EffectType.UNKNOWN)
        self.assertEqual(result.risk_assessment.risk_level, RiskLevel.MEDIUM)
        self.assertEqual(result.risk_assessment.confidence, 0.1)
        self.assertEqual(result.confidence, 0.1)
    
    def test_performance_stats(self):
        """Test performance statistics"""
        # Simulate some analysis operations
        for _ in range(5):
            self.reasoner.analyze_effects(
                self.test_task_plan,
                self.test_spatial_understanding,
                self.test_safety_constraints
            )
        
        stats = self.reasoner.get_performance_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('avg_analysis_time', stats)
        self.assertIn('max_analysis_time', stats)
        self.assertIn('min_analysis_time', stats)
        self.assertIn('std_analysis_time', stats)
        
        # Check that stats are reasonable
        self.assertGreaterEqual(stats['avg_analysis_time'], 0.0)
        self.assertGreaterEqual(stats['max_analysis_time'], stats['min_analysis_time'])
    
    def test_error_handling(self):
        """Test error handling in causal reasoning"""
        # Test with invalid inputs
        with patch.object(self.reasoner, '_extract_causal_factors', side_effect=Exception("Test error")):
            result = self.reasoner.analyze_effects(
                self.test_task_plan,
                self.test_spatial_understanding,
                self.test_safety_constraints
            )
            
            # Should return fallback analysis
            self.assertIsInstance(result, CausalAnalysis)
            self.assertEqual(result.confidence, 0.1)
    
    def test_complex_causal_analysis(self):
        """Test complex causal analysis"""
        # Create more complex task plan
        complex_plan = TaskPlan()
        complex_plan.plan_id = "complex_plan"
        complex_plan.goal_description = "Pick and place object"
        complex_plan.steps = []
        complex_plan.estimated_duration_seconds = 10.0
        complex_plan.required_capabilities = ['navigation', 'manipulation']
        complex_plan.safety_considerations = []
        
        # Add complex steps
        step1 = TaskStep()
        step1.action_type = 'move'
        step1.description = 'Move to object'
        step1.estimated_duration = 2.0
        step1.preconditions = ['path_clear']
        step1.postconditions = ['near_object']
        complex_plan.steps.append(step1)
        
        step2 = TaskStep()
        step2.action_type = 'pick'
        step2.description = 'Pick up object'
        step2.estimated_duration = 3.0
        step2.preconditions = ['near_object', 'object_graspable']
        step2.postconditions = ['object_grasped']
        complex_plan.steps.append(step2)
        
        step3 = TaskStep()
        step3.action_type = 'move'
        step3.description = 'Move to target'
        step3.estimated_duration = 3.0
        step3.preconditions = ['object_grasped']
        step3.postconditions = ['near_target']
        complex_plan.steps.append(step3)
        
        step4 = TaskStep()
        step4.action_type = 'place'
        step4.description = 'Place object'
        step4.estimated_duration = 2.0
        step4.preconditions = ['near_target', 'object_grasped']
        step4.postconditions = ['object_placed']
        complex_plan.steps.append(step4)
        
        result = self.reasoner.analyze_effects(
            complex_plan,
            self.test_spatial_understanding,
            self.test_safety_constraints
        )
        
        self.assertIsInstance(result, CausalAnalysis)
        self.assertGreater(len(result.effects), 0)
        self.assertIsInstance(result.risk_assessment, RiskAssessment)
        self.assertGreater(len(result.causal_chains), 0)
        self.assertGreater(len(result.safety_implications), 0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


if __name__ == '__main__':
    unittest.main() 