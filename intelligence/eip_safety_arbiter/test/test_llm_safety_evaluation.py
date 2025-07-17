#!/usr/bin/env python3
"""
Test script for LLM-based safety evaluation in the Safety Monitor.

This script demonstrates how to use the safety monitor's LLM integration
to evaluate task plans for safety.
"""

import rclpy
from rclpy.node import Node
from threading import Thread
import time
import json

from eip_interfaces.msg import TaskPlan, TaskStep
from eip_interfaces.srv import ValidateTaskPlan


class SafetyEvaluationTester(Node):
    def __init__(self):
        super().__init__('safety_evaluation_tester')
        
        # Create client for the safety validation service
        self.client = self.create_client(
            ValidateTaskPlan,
            '/safety/validate_task_plan'
        )
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Safety validation service not available, waiting...')
        
        self.get_logger().info("Safety Evaluation Tester initialized")
    
    def create_test_plan(self, scenario: str) -> TaskPlan:
        """Create a test task plan based on the scenario"""
        plan = TaskPlan()
        plan.task_id = f"test_plan_{int(time.time())}"
        
        if scenario == "safe_plan":
            plan.description = "Move to charging station"
            
            # Create steps for a safe task
            step1 = TaskStep()
            step1.step_id = 1
            step1.description = "Navigate to waypoint near charging station"
            step1.action_type = "navigate"
            step1.parameters = json.dumps({"x": 2.5, "y": 1.0, "z": 0.0})
            step1.expected_outcome = "Robot reaches waypoint"
            
            step2 = TaskStep()
            step2.step_id = 2
            step2.description = "Approach charging station slowly"
            step2.action_type = "move_linear"
            step2.parameters = json.dumps({"distance": 0.5, "speed": 0.1})
            step2.expected_outcome = "Robot is docked at charging station"
            
            plan.steps = [step1, step2]
            
        elif scenario == "risky_plan":
            plan.description = "Move object near human"
            
            step1 = TaskStep()
            step1.step_id = 1
            step1.description = "Pick up the glass near the human"
            step1.action_type = "pick"
            step1.parameters = json.dumps({"object": "glass", "force": 10.0})
            step1.expected_outcome = "Glass is grasped"
            
            step2 = TaskStep()
            step2.step_id = 2
            step2.description = "Move quickly past the human"
            step2.action_type = "move"
            step2.parameters = json.dumps({"velocity": 2.0, "distance": 3.0})
            step2.expected_outcome = "Robot moves past human"
            
            plan.steps = [step1, step2]
            
        elif scenario == "dangerous_plan":
            plan.description = "Operate in restricted area"
            
            step1 = TaskStep()
            step1.step_id = 1
            step1.description = "Enter the restricted construction zone"
            step1.action_type = "navigate"
            step1.parameters = json.dumps({"x": 10.0, "y": 10.0})
            step1.expected_outcome = "Robot enters construction zone"
            
            step2 = TaskStep()
            step2.step_id = 2
            step2.description = "Operate heavy machinery"
            step2.action_type = "operate"
            step2.parameters = json.dumps({"machine": "crane", "force": 1000.0})
            step2.expected_outcome = "Crane operation complete"
            
            plan.steps = [step1, step2]
        
        return plan
    
    def evaluate_plan(self, scenario: str):
        """Evaluate a test plan and print results"""
        # Create test plan
        plan = self.create_test_plan(scenario)
        
        self.get_logger().info(f"\n{'='*80}")
        self.get_logger().info(f"Evaluating {scenario.replace('_', ' ').title()}...")
        self.get_logger().info(f"Task: {plan.description}")
        
        # Create request
        request = ValidateTaskPlan.Request()
        request.task_plan = plan
        
        # Send request
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        try:
            response = future.result()
            
            # Print results
            self.get_logger().info("\nSafety Evaluation Results:")
            self.get_logger().info(f"- Is Safe: {'✅' if response.is_safe else '❌'}")
            self.get_logger().info(f"- Safety Level: {self._get_safety_level_name(response.safety_level)}")
            self.get_logger().info(f"- Confidence: {response.confidence_score*100:.1f}%")
            
            if response.violations:
                self.get_logger().info("\nSafety Violations Detected:")
                for violation in response.violations:
                    self.get_logger().info(f"- {violation.replace('_', ' ').title()}")
            
            if response.suggested_modifications:
                self.get_logger().info("\nSuggested Modifications:")
                for mod in response.suggested_modifications:
                    self.get_logger().info(f"- {mod}")
            
            self.get_logger().info(f"\nExplanation:\n{response.explanation}")
            
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
    
    def _get_safety_level_name(self, level: int) -> str:
        """Convert safety level enum to string"""
        levels = {
            0: "🟢 SAFE",
            1: "🟡 LOW RISK",
            2: "🟠 MEDIUM RISK",
            3: "🔴 HIGH RISK",
            4: "⛔ CRITICAL"
        }
        return levels.get(level, "UNKNOWN")


def run_test_scenarios():
    rclpy.init()
    tester = SafetyEvaluationTester()
    
    try:
        # Test different scenarios
        tester.evaluate_plan("safe_plan")
        time.sleep(2)  # Small delay between tests
        
        tester.evaluate_plan("risky_plan")
        time.sleep(2)
        
        tester.evaluate_plan("dangerous_plan")
        
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    run_test_scenarios()
