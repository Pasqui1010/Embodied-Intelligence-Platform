#!/usr/bin/env python3
"""
Safety Simulator Tests

Comprehensive test suite for the Digital Twin Safety Ecosystem,
validating safety simulation, scenario generation, and validation capabilities.
"""

import unittest
import json
import time
import yaml
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# ROS 2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose, Twist
from eip_interfaces.msg import TaskPlan, SafetyViolation

# Import the safety simulator components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../intelligence/eip_safety_simulator/eip_safety_simulator'))

from safety_simulator_node import (
    SafetySimulatorNode, 
    SimulationState, 
    SafetyScenario, 
    SafetyMetrics, 
    SimulationConfig
)


class TestSafetySimulator(unittest.TestCase):
    """Test suite for Safety Simulator Node"""

    def setUp(self):
        """Set up test environment"""
        rclpy.init()
        self.node = SafetySimulatorNode()
        
        # Test data
        self.test_scenario = {
            'name': 'test_collision_avoidance',
            'type': SafetyScenario.COLLISION_AVOIDANCE.value,
            'description': 'Test collision avoidance scenario',
            'obstacles': [
                {'position': [2.0, 1.0], 'size': 0.5},
                {'position': [3.0, 2.0], 'size': 0.3}
            ],
            'target': [4.0, 4.0],
            'duration': 30.0
        }
        
        self.test_task_plan = TaskPlan()
        self.test_task_plan.goal_description = "Navigate to target while avoiding obstacles"
        self.test_task_plan.safety_considerations = "Maintain safe distance from obstacles"
        self.test_task_plan.steps = [
            Mock(action_type="move_forward", estimated_duration=10.0),
            Mock(action_type="turn_left", estimated_duration=5.0),
            Mock(action_type="move_forward", estimated_duration=15.0)
        ]

    def tearDown(self):
        """Clean up test environment"""
        self.node.destroy_node()
        rclpy.shutdown()

    def test_node_initialization(self):
        """Test safety simulator node initialization"""
        self.assertIsNotNone(self.node)
        self.assertEqual(self.node.simulation_state, SimulationState.IDLE)
        self.assertIsNone(self.node.current_scenario)
        self.assertIsNone(self.node.simulation_start_time)
        self.assertEqual(len(self.node.metrics_collector), 0)

    def test_config_loading(self):
        """Test simulation configuration loading"""
        config = self.node._load_simulation_config()
        self.assertIsInstance(config, SimulationConfig)
        self.assertIsInstance(config.scenario_type, SafetyScenario)
        self.assertGreater(config.safety_threshold, 0.0)
        self.assertLessEqual(config.safety_threshold, 1.0)

    def test_scenario_generation(self):
        """Test scenario generation for different types"""
        scenarios = [
            SafetyScenario.COLLISION_AVOIDANCE,
            SafetyScenario.HUMAN_PROXIMITY,
            SafetyScenario.VELOCITY_LIMITS,
            SafetyScenario.WORKSPACE_BOUNDARY,
            SafetyScenario.EMERGENCY_STOP
        ]
        
        for scenario_type in scenarios:
            scenario = self.node._generate_scenario(scenario_type.value)
            self.assertIsNotNone(scenario)
            self.assertEqual(scenario['type'], scenario_type.value)
            self.assertIn('name', scenario)
            self.assertIn('description', scenario)
            self.assertIn('duration', scenario)

    def test_simulation_start_stop(self):
        """Test simulation start and stop functionality"""
        # Test start simulation
        success = self.node._start_simulation(self.test_scenario)
        self.assertTrue(success)
        self.assertEqual(self.node.simulation_state, SimulationState.RUNNING)
        self.assertIsNotNone(self.node.simulation_start_time)
        self.assertEqual(self.node.current_scenario, self.test_scenario)
        
        # Test stop simulation
        success = self.node._stop_simulation()
        self.assertTrue(success)
        self.assertEqual(self.node.simulation_state, SimulationState.COMPLETED)

    def test_task_plan_safety_validation(self):
        """Test task plan safety validation"""
        safety_score = self.node._validate_task_plan_safety(self.test_task_plan)
        self.assertIsInstance(safety_score, float)
        self.assertGreaterEqual(safety_score, 0.0)
        self.assertLessEqual(safety_score, 1.0)

    def test_metrics_recording(self):
        """Test metrics recording functionality"""
        initial_count = len(self.node.metrics_collector)
        
        # Record task plan metrics
        self.node._record_task_plan_metrics(self.test_task_plan, 0.85)
        
        self.assertEqual(len(self.node.metrics_collector), initial_count + 1)
        
        # Check metrics structure
        metrics = self.node.metrics_collector[-1]
        self.assertIsInstance(metrics, SafetyMetrics)
        self.assertEqual(metrics.safety_score, 0.85)
        self.assertEqual(metrics.total_actions, 3)

    def test_safety_violation_handling(self):
        """Test safety violation handling"""
        # Create test violation
        violation = SafetyViolation()
        violation.explanation = "Collision risk detected"
        violation.severity = 0.8
        
        # Record violation
        self.node._record_task_plan_metrics(self.test_task_plan, 0.9)
        self.node._record_safety_violation(violation)
        
        # Check that violation was recorded
        metrics = self.node.metrics_collector[-1]
        self.assertEqual(metrics.violations_detected, 1)
        self.assertLess(metrics.safety_score, 0.9)  # Should be reduced due to violation

    def test_collision_scenario_generation(self):
        """Test collision avoidance scenario generation"""
        scenario = self.node._generate_collision_scenario()
        
        self.assertEqual(scenario['name'], 'collision_avoidance')
        self.assertEqual(scenario['type'], SafetyScenario.COLLISION_AVOIDANCE.value)
        self.assertIn('obstacles', scenario)
        self.assertIn('target', scenario)
        self.assertIn('duration', scenario)
        
        # Check obstacles structure
        obstacles = scenario['obstacles']
        self.assertIsInstance(obstacles, list)
        for obstacle in obstacles:
            self.assertIn('position', obstacle)
            self.assertIn('size', obstacle)

    def test_human_proximity_scenario_generation(self):
        """Test human proximity scenario generation"""
        scenario = self.node._generate_human_proximity_scenario()
        
        self.assertEqual(scenario['name'], 'human_proximity')
        self.assertEqual(scenario['type'], SafetyScenario.HUMAN_PROXIMITY.value)
        self.assertIn('humans', scenario)
        self.assertIn('target', scenario)
        self.assertIn('duration', scenario)
        
        # Check humans structure
        humans = scenario['humans']
        self.assertIsInstance(humans, list)
        for human in humans:
            self.assertIn('position', human)
            self.assertIn('movement_pattern', human)

    def test_velocity_scenario_generation(self):
        """Test velocity limits scenario generation"""
        scenario = self.node._generate_velocity_scenario()
        
        self.assertEqual(scenario['name'], 'velocity_limits')
        self.assertEqual(scenario['type'], SafetyScenario.VELOCITY_LIMITS.value)
        self.assertIn('max_velocity', scenario)
        self.assertIn('confined_area', scenario)
        self.assertIn('target', scenario)
        self.assertIn('duration', scenario)

    def test_workspace_boundary_scenario_generation(self):
        """Test workspace boundary scenario generation"""
        scenario = self.node._generate_boundary_scenario()
        
        self.assertEqual(scenario['name'], 'workspace_boundary')
        self.assertEqual(scenario['type'], SafetyScenario.WORKSPACE_BOUNDARY.value)
        self.assertIn('boundaries', scenario)
        self.assertIn('target', scenario)
        self.assertIn('duration', scenario)
        
        # Check boundaries structure
        boundaries = scenario['boundaries']
        self.assertIn('x_min', boundaries)
        self.assertIn('x_max', boundaries)
        self.assertIn('y_min', boundaries)
        self.assertIn('y_max', boundaries)

    def test_emergency_scenario_generation(self):
        """Test emergency stop scenario generation"""
        scenario = self.node._generate_emergency_scenario()
        
        self.assertEqual(scenario['name'], 'emergency_stop')
        self.assertEqual(scenario['type'], SafetyScenario.EMERGENCY_STOP.value)
        self.assertIn('emergency_triggers', scenario)
        self.assertIn('target', scenario)
        self.assertIn('duration', scenario)
        
        # Check emergency triggers structure
        triggers = scenario['emergency_triggers']
        self.assertIsInstance(triggers, list)
        for trigger in triggers:
            self.assertIn('type', trigger)
            self.assertIn('position', trigger)

    def test_multi_agent_scenario_generation(self):
        """Test multi-agent scenario generation"""
        scenario = self.node._generate_multi_agent_scenario()
        
        self.assertEqual(scenario['name'], 'multi_agent')
        self.assertEqual(scenario['type'], SafetyScenario.MULTI_AGENT.value)
        self.assertIn('other_robots', scenario)
        self.assertIn('target', scenario)
        self.assertIn('duration', scenario)
        
        # Check other robots structure
        robots = scenario['other_robots']
        self.assertIsInstance(robots, list)
        for robot in robots:
            self.assertIn('id', robot)
            self.assertIn('position', robot)
            self.assertIn('behavior', robot)

    def test_dynamic_obstacles_scenario_generation(self):
        """Test dynamic obstacles scenario generation"""
        scenario = self.node._generate_dynamic_obstacles_scenario()
        
        self.assertEqual(scenario['name'], 'dynamic_obstacles')
        self.assertEqual(scenario['type'], SafetyScenario.DYNAMIC_OBSTACLES.value)
        self.assertIn('dynamic_obstacles', scenario)
        self.assertIn('target', scenario)
        self.assertIn('duration', scenario)
        
        # Check dynamic obstacles structure
        obstacles = scenario['dynamic_obstacles']
        self.assertIsInstance(obstacles, list)
        for obstacle in obstacles:
            self.assertIn('position', obstacle)
            self.assertIn('velocity', obstacle)
            self.assertIn('trajectory', obstacle)

    def test_complex_environment_scenario_generation(self):
        """Test complex environment scenario generation"""
        scenario = self.node._generate_complex_environment_scenario()
        
        self.assertEqual(scenario['name'], 'complex_environment')
        self.assertEqual(scenario['type'], SafetyScenario.COMPLEX_ENVIRONMENT.value)
        self.assertIn('environment_features', scenario)
        self.assertIn('target', scenario)
        self.assertIn('duration', scenario)
        
        # Check environment features
        features = scenario['environment_features']
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_safety_validation_methods(self):
        """Test safety validation methods"""
        # Test collision risk validation
        robot_pose = Pose()
        obstacles = [{'position': [2.0, 1.0], 'size': 0.5}]
        collision_score = self.node._validate_collision_risk(robot_pose, obstacles)
        self.assertIsInstance(collision_score, float)
        self.assertGreaterEqual(collision_score, 0.0)
        self.assertLessEqual(collision_score, 1.0)
        
        # Test human proximity validation
        humans = [{'position': [2.0, 1.0], 'movement_pattern': 'stationary'}]
        proximity_score = self.node._validate_human_proximity(robot_pose, humans)
        self.assertIsInstance(proximity_score, float)
        self.assertGreaterEqual(proximity_score, 0.0)
        self.assertLessEqual(proximity_score, 1.0)
        
        # Test velocity limits validation
        robot_velocity = Twist()
        limits = {'max_velocity': 1.0}
        velocity_score = self.node._validate_velocity_limits(robot_velocity, limits)
        self.assertIsInstance(velocity_score, float)
        self.assertGreaterEqual(velocity_score, 0.0)
        self.assertLessEqual(velocity_score, 1.0)
        
        # Test workspace boundary validation
        boundaries = {'x_min': 0.0, 'x_max': 5.0, 'y_min': 0.0, 'y_max': 5.0}
        boundary_score = self.node._validate_workspace_boundary(robot_pose, boundaries)
        self.assertIsInstance(boundary_score, float)
        self.assertGreaterEqual(boundary_score, 0.0)
        self.assertLessEqual(boundary_score, 1.0)
        
        # Test emergency stop validation
        emergency_conditions = [{'type': 'sudden_obstacle', 'position': [2.0, 2.0]}]
        emergency_score = self.node._validate_emergency_stop(emergency_conditions)
        self.assertIsInstance(emergency_score, float)
        self.assertGreaterEqual(emergency_score, 0.0)
        self.assertLessEqual(emergency_score, 1.0)

    def test_metrics_calculation(self):
        """Test metrics calculation functionality"""
        # Add some test metrics
        for i in range(5):
            metrics = SafetyMetrics(
                scenario_name=f'test_scenario_{i}',
                safety_score=0.8 + (i * 0.02),
                violations_detected=i,
                response_time_ms=100.0 + (i * 10.0),
                success_rate=0.9,
                false_positives=0,
                false_negatives=0,
                total_actions=5,
                safe_actions=4,
                unsafe_actions=1,
                timestamp=time.time()
            )
            self.node.metrics_collector.append(metrics)
        
        # Calculate final metrics
        self.node._calculate_final_metrics()
        
        # Check that metrics were calculated (this would log the results)
        self.assertEqual(len(self.node.metrics_collector), 5)

    def test_simulation_violation_handling(self):
        """Test simulation violation handling"""
        # Start simulation
        self.node._start_simulation(self.test_scenario)
        
        # Create critical violation
        violation = SafetyViolation()
        violation.explanation = "Emergency stop triggered - collision imminent"
        violation.severity = 0.9
        
        # Handle violation
        self.node._handle_simulation_violation(violation)
        
        # Check that simulation was stopped due to critical violation
        self.assertEqual(self.node.simulation_state, SimulationState.COMPLETED)

    def test_status_publishing(self):
        """Test status publishing functionality"""
        # Start simulation
        self.node._start_simulation(self.test_scenario)
        
        # Publish status
        self.node._publish_status()
        
        # Check that status was published (this would be verified in integration tests)
        self.assertEqual(self.node.simulation_state, SimulationState.RUNNING)

    def test_metrics_publishing(self):
        """Test metrics publishing functionality"""
        # Add test metrics
        metrics = SafetyMetrics(
            scenario_name='test_scenario',
            safety_score=0.85,
            violations_detected=1,
            response_time_ms=150.0,
            success_rate=0.9,
            false_positives=0,
            false_negatives=0,
            total_actions=3,
            safe_actions=2,
            unsafe_actions=1,
            timestamp=time.time()
        )
        self.node.metrics_collector.append(metrics)
        
        # Publish metrics
        self.node._publish_metrics()
        
        # Check that metrics were published (this would be verified in integration tests)
        self.assertEqual(len(self.node.metrics_collector), 1)

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test invalid scenario generation
        scenario = self.node._generate_scenario("invalid_scenario")
        self.assertIsNone(scenario)
        
        # Test invalid config loading
        with patch.object(self.node, 'config_file', 'nonexistent_file.yaml'):
            config = self.node._load_simulation_config()
            self.assertIsInstance(config, SimulationConfig)  # Should return default config
        
        # Test metrics recording with invalid data
        self.node._record_task_plan_metrics(None, 0.5)
        # Should not crash and should handle gracefully

    def test_performance_characteristics(self):
        """Test performance characteristics of the simulator"""
        # Test scenario generation performance
        start_time = time.time()
        for _ in range(100):
            self.node._generate_collision_scenario()
        generation_time = time.time() - start_time
        
        # Should complete within reasonable time (less than 1 second for 100 scenarios)
        self.assertLess(generation_time, 1.0)
        
        # Test safety validation performance
        start_time = time.time()
        for _ in range(1000):
            self.node._validate_task_plan_safety(self.test_task_plan)
        validation_time = time.time() - start_time
        
        # Should complete within reasonable time (less than 1 second for 1000 validations)
        self.assertLess(validation_time, 1.0)

    def test_integration_with_safety_embedded_llm(self):
        """Test integration with Safety-Embedded LLM"""
        # Test that task plans are properly validated
        task_plan = TaskPlan()
        task_plan.goal_description = "Navigate safely to target"
        task_plan.safety_considerations = "Maintain safe distance from obstacles and humans"
        task_plan.steps = [
            Mock(action_type="safe_move_forward", estimated_duration=10.0),
            Mock(action_type="safe_turn", estimated_duration=5.0)
        ]
        
        safety_score = self.node._validate_task_plan_safety(task_plan)
        self.assertIsInstance(safety_score, float)
        self.assertGreaterEqual(safety_score, 0.0)
        self.assertLessEqual(safety_score, 1.0)
        
        # Test that safety violations are properly handled
        violation = SafetyViolation()
        violation.explanation = "Safety-Embedded LLM detected unsafe action"
        violation.severity = 0.7
        
        self.node._record_task_plan_metrics(task_plan, 0.8)
        self.node._record_safety_violation(violation)
        
        metrics = self.node.metrics_collector[-1]
        self.assertEqual(metrics.violations_detected, 1)
        self.assertLess(metrics.safety_score, 0.8)  # Should be reduced due to violation


class TestSafetySimulatorIntegration(unittest.TestCase):
    """Integration tests for Safety Simulator with other components"""

    def setUp(self):
        """Set up integration test environment"""
        rclpy.init()
        self.node = SafetySimulatorNode()

    def tearDown(self):
        """Clean up integration test environment"""
        self.node.destroy_node()
        rclpy.shutdown()

    def test_full_simulation_workflow(self):
        """Test complete simulation workflow"""
        # 1. Generate scenario
        scenario = self.node._generate_collision_scenario()
        self.assertIsNotNone(scenario)
        
        # 2. Start simulation
        success = self.node._start_simulation(scenario)
        self.assertTrue(success)
        
        # 3. Process task plan
        task_plan = TaskPlan()
        task_plan.goal_description = "Navigate to target"
        task_plan.safety_considerations = "Avoid obstacles"
        task_plan.steps = [Mock(action_type="move", estimated_duration=10.0)]
        
        safety_score = self.node._validate_task_plan_safety(task_plan)
        self.assertIsInstance(safety_score, float)
        
        # 4. Record metrics
        self.node._record_task_plan_metrics(task_plan, safety_score)
        self.assertEqual(len(self.node.metrics_collector), 1)
        
        # 5. Handle safety violation
        violation = SafetyViolation()
        violation.explanation = "Obstacle detected"
        self.node._record_safety_violation(violation)
        
        # 6. Stop simulation
        success = self.node._stop_simulation()
        self.assertTrue(success)
        self.assertEqual(self.node.simulation_state, SimulationState.COMPLETED)

    def test_multiple_scenario_execution(self):
        """Test execution of multiple scenarios"""
        scenarios = [
            'collision_avoidance',
            'human_proximity',
            'velocity_limits',
            'workspace_boundary',
            'emergency_stop'
        ]
        
        for scenario_name in scenarios:
            # Generate and start scenario
            scenario = self.node._generate_scenario(scenario_name)
            self.assertIsNotNone(scenario)
            
            success = self.node._start_simulation(scenario)
            self.assertTrue(success)
            
            # Process task plan
            task_plan = TaskPlan()
            task_plan.goal_description = f"Execute {scenario_name} scenario"
            task_plan.steps = [Mock(action_type="execute", estimated_duration=10.0)]
            
            safety_score = self.node._validate_task_plan_safety(task_plan)
            self.node._record_task_plan_metrics(task_plan, safety_score)
            
            # Stop simulation
            success = self.node._stop_simulation()
            self.assertTrue(success)
        
        # Check that all scenarios were recorded
        self.assertEqual(len(self.node.metrics_collector), len(scenarios))


if __name__ == '__main__':
    unittest.main() 