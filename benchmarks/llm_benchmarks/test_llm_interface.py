#!/usr/bin/env python3
"""
LLM Interface Tests

Tests for the LLM interface node functionality including:
- Task plan generation
- Safety verification
- Response parsing
- Error handling
"""

import pytest
import json
import time
from unittest.mock import Mock, patch

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from eip_interfaces.msg import TaskPlan, TaskStep
from eip_interfaces.srv import ValidateTaskPlan


class TestLLMInterface:
    """Test suite for LLM Interface functionality"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        rclpy.init()
        yield
        rclpy.shutdown()
    
    def test_task_plan_generation(self):
        """Test basic task plan generation from natural language"""
        # This would test the LLM interface node's ability to generate
        # valid task plans from natural language commands
        
        # Mock LLM response
        mock_response = '''
{
    "goal_description": "Navigate to the red chair",
    "steps": [
        {
            "action_type": "navigation",
            "description": "Move to target location",
            "target_pose": {"x": 2.0, "y": 1.0, "z": 0.0, "w": 1.0},
            "parameters": ["target_x", "target_y"],
            "estimated_duration": 10.0,
            "preconditions": ["path_clear"],
            "postconditions": ["at_target_location"]
        }
    ],
    "estimated_duration_seconds": 10,
    "required_capabilities": ["navigation"],
    "safety_considerations": ["Avoid obstacles", "Maintain safe speed"]
}
'''
        
        # Test parsing
        from eip_llm_interface.llm_interface_node import LLMInterfaceNode
        node = LLMInterfaceNode()
        
        task_plan = node._parse_task_plan_response(mock_response, "Navigate to the red chair")
        
        assert task_plan is not None
        assert task_plan.goal_description == "Navigate to the red chair"
        assert len(task_plan.steps) == 1
        assert task_plan.steps[0].action_type == "navigation"
        assert task_plan.estimated_duration_seconds == 10
        
        node.destroy_node()
    
    def test_safety_verification(self):
        """Test safety verification of task plans"""
        from eip_llm_interface.llm_interface_node import LLMInterfaceNode
        node = LLMInterfaceNode()
        
        # Test safe task plan
        safe_plan = TaskPlan()
        safe_plan.goal_description = "Safe navigation task"
        safe_step = TaskStep()
        safe_step.action_type = "navigation"
        safe_step.estimated_duration = 10.0
        safe_plan.steps.append(safe_step)
        
        safety_response = node._verify_task_plan_safety(safe_plan)
        assert safety_response.is_safe == True
        
        # Test unsafe task plan
        unsafe_plan = TaskPlan()
        unsafe_plan.goal_description = "Unsafe high-speed task"
        unsafe_step = TaskStep()
        unsafe_step.action_type = "high_speed"
        unsafe_step.estimated_duration = 600.0  # 10 minutes
        unsafe_plan.steps.append(unsafe_step)
        
        safety_response = node._verify_task_plan_safety(unsafe_plan)
        assert safety_response.is_safe == False
        
        node.destroy_node()
    
    def test_prompt_engineering(self):
        """Test prompt engineering for different task types"""
        from eip_llm_interface.llm_interface_node import LLMInterfaceNode
        node = LLMInterfaceNode()
        
        # Test navigation prompt
        nav_prompt = node._create_task_planning_prompt("Go to the kitchen")
        assert "navigation" in nav_prompt.lower()
        assert "JSON" in nav_prompt
        assert "steps" in nav_prompt
        
        # Test manipulation prompt
        manip_prompt = node._create_task_planning_prompt("Pick up the cup")
        assert "manipulation" in manip_prompt.lower()
        
        node.destroy_node()
    
    def test_error_handling(self):
        """Test error handling for malformed responses"""
        from eip_llm_interface.llm_interface_node import LLMInterfaceNode
        node = LLMInterfaceNode()
        
        # Test invalid JSON
        invalid_response = "This is not valid JSON"
        task_plan = node._parse_task_plan_response(invalid_response, "test command")
        assert task_plan is None
        
        # Test missing required fields
        incomplete_response = '{"goal_description": "test"}'
        task_plan = node._parse_task_plan_response(incomplete_response, "test command")
        # Should handle gracefully
        assert task_plan is not None
        
        node.destroy_node()
    
    def test_mock_llm_response(self):
        """Test mock LLM response generation"""
        from eip_llm_interface.llm_interface_node import LLMInterfaceNode
        node = LLMInterfaceNode()
        
        # Test navigation prompt
        nav_prompt = "Generate a navigation plan"
        mock_response = node._generate_mock_response(nav_prompt)
        
        # Parse the response
        json_start = mock_response.find('{')
        json_end = mock_response.rfind('}') + 1
        json_str = mock_response[json_start:json_end]
        response_data = json.loads(json_str)
        
        assert "goal_description" in response_data
        assert "steps" in response_data
        assert len(response_data["steps"]) > 0
        
        node.destroy_node()


class TestLLMIntegration:
    """Integration tests for LLM interface with other components"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        rclpy.init()
        yield
        rclpy.shutdown()
    
    def test_safety_arbiter_integration(self):
        """Test integration with safety arbiter"""
        # This would test the full integration between LLM interface
        # and safety arbiter
        
        # Mock safety arbiter service
        from eip_llm_interface.llm_interface_node import LLMInterfaceNode
        node = LLMInterfaceNode()
        
        # Create a test task plan
        task_plan = TaskPlan()
        task_plan.goal_description = "Test integration"
        task_plan.estimated_duration_seconds = 30
        
        # Test safety verification
        safety_response = node._verify_task_plan_safety(task_plan)
        assert hasattr(safety_response, 'is_safe')
        assert hasattr(safety_response, 'confidence_score')
        assert hasattr(safety_response, 'explanation')
        
        node.destroy_node()
    
    def test_slam_integration(self):
        """Test integration with SLAM system"""
        # This would test how LLM interface works with SLAM data
        
        # Mock scene description
        scene_desc = "Room contains a red chair at position (2, 1) and a table at (3, 2)"
        
        from eip_llm_interface.llm_interface_node import LLMInterfaceNode
        node = LLMInterfaceNode()
        
        # Test scene description handling
        node._handle_scene_description(String(data=scene_desc))
        
        # Verify scene context is stored
        assert hasattr(node, 'current_scene_context')
        
        node.destroy_node()


if __name__ == '__main__':
    pytest.main([__file__]) 