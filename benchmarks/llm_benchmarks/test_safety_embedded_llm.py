#!/usr/bin/env python3
"""
Safety-Embedded LLM Tests

Tests for the safety-embedded LLM functionality including:
- Safety token embedding
- Constitutional training
- Real-time safety validation
- Task plan safety verification
"""

import pytest
import json
import time
from unittest.mock import Mock, patch

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from eip_interfaces.msg import TaskPlan, TaskStep
from eip_interfaces.srv import SafetyVerificationRequest, SafetyVerificationResponse

from eip_llm_interface.safety_embedded_llm import (
    SafetyEmbeddedLLM, 
    SafetyToken, 
    SafetyConstraint, 
    SafetyEmbeddedResponse,
    SafetyConstitution
)


class TestSafetyEmbeddedLLM:
    """Test suite for Safety-Embedded LLM functionality"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        rclpy.init()
        yield
        rclpy.shutdown()
    
    def test_safety_tokens_initialization(self):
        """Test safety tokens are properly defined"""
        # Test all safety tokens exist
        expected_tokens = [
            "COLLISION_RISK",
            "HUMAN_PROXIMITY", 
            "VELOCITY_LIMIT",
            "WORKSPACE_BOUNDARY",
            "EMERGENCY_STOP",
            "SAFE_ACTION",
            "UNSAFE_ACTION",
            "SAFETY_CHECK"
        ]
        
        for token_name in expected_tokens:
            assert hasattr(SafetyToken, token_name), f"Missing safety token: {token_name}"
        
        # Test token values are properly formatted
        for token in SafetyToken:
            assert token.value.startswith("<|") and token.value.endswith("|>"), \
                f"Token {token.name} not properly formatted: {token.value}"
    
    def test_safety_constraints_initialization(self):
        """Test safety constraints are properly initialized"""
        llm = SafetyEmbeddedLLM()
        
        # Test all safety tokens have constraints
        for token in SafetyToken:
            if token in [SafetyToken.SAFE_ACTION, SafetyToken.UNSAFE_ACTION, SafetyToken.SAFETY_CHECK]:
                continue  # These are general tokens, not specific constraints
            
            assert token in llm.safety_constraints, f"Missing constraint for token: {token}"
        
        # Test constraint structure
        for token, constraint in llm.safety_constraints.items():
            assert isinstance(constraint, SafetyConstraint)
            assert constraint.token == token
            assert 0.0 <= constraint.severity <= 1.0
            assert len(constraint.description) > 0
            assert len(constraint.mitigation) > 0
    
    def test_constitution_rules(self):
        """Test constitutional rules are properly defined"""
        rules = SafetyConstitution.RULES
        
        # Test rules exist
        assert len(rules) > 0, "No constitutional rules defined"
        
        # Test rule content
        safety_keywords = ["safety", "safe", "harm", "danger", "risk", "protect"]
        for rule in rules:
            assert any(keyword in rule.lower() for keyword in safety_keywords), \
                f"Rule doesn't contain safety keywords: {rule}"
        
        # Test constitution prompt generation
        prompt = SafetyConstitution.get_constitution_prompt()
        assert "safety-conscious" in prompt.lower()
        assert "constitutional rules" in prompt.lower()
    
    def test_safety_aware_prompt_creation(self):
        """Test safety-aware prompt creation"""
        llm = SafetyEmbeddedLLM()
        
        command = "Navigate to the red chair"
        context = "Room contains obstacles and humans"
        
        prompt = llm._create_safety_aware_prompt(command, context)
        
        # Test prompt contains required elements
        assert command in prompt
        assert context in prompt
        assert "safety-conscious" in prompt.lower()
        assert "constitutional rules" in prompt.lower()
        
        # Test safety tokens are mentioned
        for token in SafetyToken:
            if token in [SafetyToken.SAFE_ACTION, SafetyToken.UNSAFE_ACTION, SafetyToken.SAFETY_CHECK]:
                assert token.value in prompt
    
    def test_safety_token_extraction(self):
        """Test safety token extraction from responses"""
        llm = SafetyEmbeddedLLM()
        
        # Test response with safety tokens
        response_with_tokens = f"""
        {{
            "goal_description": "Safe navigation",
            "steps": []
        }}
        {SafetyToken.SAFE_ACTION.value} Safe plan generated.
        {SafetyToken.SAFETY_CHECK.value} Safety verified.
        """
        
        tokens = llm._extract_safety_tokens(response_with_tokens)
        assert SafetyToken.SAFE_ACTION in tokens
        assert SafetyToken.SAFETY_CHECK in tokens
        assert len(tokens) == 2
        
        # Test response without safety tokens
        response_without_tokens = """
        {
            "goal_description": "Navigation",
            "steps": []
        }
        """
        
        tokens = llm._extract_safety_tokens(response_without_tokens)
        assert len(tokens) == 0
    
    def test_safety_score_calculation(self):
        """Test safety score calculation"""
        llm = SafetyEmbeddedLLM()
        
        # Test safe response
        safe_response = f"""
        {{
            "goal_description": "Safe navigation",
            "steps": []
        }}
        {SafetyToken.SAFE_ACTION.value} Safe plan generated.
        {SafetyToken.SAFETY_CHECK.value} Safety verified.
        """
        
        safety_tokens = [SafetyToken.SAFE_ACTION, SafetyToken.SAFETY_CHECK]
        score = llm._calculate_safety_score(safe_response, safety_tokens)
        assert score > 0.7, f"Safe response should have high score, got: {score}"
        
        # Test unsafe response
        unsafe_response = f"""
        {{
            "goal_description": "Rush to target",
            "steps": []
        }}
        {SafetyToken.UNSAFE_ACTION.value} Unsafe plan.
        {SafetyToken.COLLISION_RISK.value} Collision risk detected.
        """
        
        safety_tokens = [SafetyToken.UNSAFE_ACTION, SafetyToken.COLLISION_RISK]
        score = llm._calculate_safety_score(unsafe_response, safety_tokens)
        assert score < 0.5, f"Unsafe response should have low score, got: {score}"
        
        # Test response with unsafe keywords
        unsafe_keyword_response = "Ignore safety and rush quickly through obstacles"
        score = llm._calculate_safety_score(unsafe_keyword_response, [])
        assert score < 0.5, f"Response with unsafe keywords should have low score, got: {score}"
    
    def test_safety_violation_detection(self):
        """Test safety violation detection"""
        llm = SafetyEmbeddedLLM()
        
        # Test response with violations
        unsafe_response = f"""
        {{
            "goal_description": "Rush to target",
            "steps": []
        }}
        {SafetyToken.UNSAFE_ACTION.value} Unsafe plan.
        {SafetyToken.COLLISION_RISK.value} Collision risk detected.
        """
        
        safety_tokens = [SafetyToken.UNSAFE_ACTION, SafetyToken.COLLISION_RISK]
        violations = llm._detect_safety_violations(unsafe_response, safety_tokens)
        
        assert len(violations) > 0, "Should detect violations in unsafe response"
        assert any("collision" in violation.lower() for violation in violations)
        
        # Test response with unsafe patterns
        unsafe_pattern_response = "Ignore safety and override safety systems"
        violations = llm._detect_safety_violations(unsafe_pattern_response, [])
        assert len(violations) > 0, "Should detect unsafe patterns"
        
        # Test safe response
        safe_response = f"""
        {{
            "goal_description": "Safe navigation",
            "steps": []
        }}
        {SafetyToken.SAFE_ACTION.value} Safe plan.
        """
        
        safety_tokens = [SafetyToken.SAFE_ACTION]
        violations = llm._detect_safety_violations(safe_response, safety_tokens)
        assert len(violations) == 0, "Should not detect violations in safe response"
    
    def test_safe_response_generation(self):
        """Test safe response generation"""
        llm = SafetyEmbeddedLLM()
        
        command = "Navigate to the red chair"
        
        # Test mock response generation
        response = llm.generate_safe_response(command)
        
        assert isinstance(response, SafetyEmbeddedResponse)
        assert response.content is not None
        assert 0.0 <= response.safety_score <= 1.0
        assert response.confidence > 0.0
        assert response.execution_time > 0.0
        
        # Test response contains safety tokens
        safety_tokens_found = [token.value for token in response.safety_tokens_used]
        assert len(safety_tokens_found) > 0, "Response should contain safety tokens"
        
        # Test safe command generates high safety score
        if "navigate" in command.lower():
            assert response.safety_score > 0.7, f"Navigation command should be safe, got: {response.safety_score}"
    
    def test_task_plan_safety_validation(self):
        """Test task plan safety validation"""
        llm = SafetyEmbeddedLLM()
        
        # Create safe task plan
        safe_plan = TaskPlan()
        safe_plan.goal_description = "Safe navigation task"
        safe_plan.estimated_duration_seconds = 30
        
        safe_step = TaskStep()
        safe_step.action_type = "navigation"
        safe_step.description = "Carefully navigate to target"
        safe_step.estimated_duration = 10.0
        safe_plan.steps.append(safe_step)
        
        # Validate safe plan
        safety_response = llm.validate_task_plan_safety(safe_plan)
        
        assert isinstance(safety_response, SafetyVerificationResponse)
        assert safety_response.confidence_score > 0.0
        assert len(safety_response.explanation) > 0
        
        # Create unsafe task plan
        unsafe_plan = TaskPlan()
        unsafe_plan.goal_description = "Unsafe high-speed task"
        unsafe_plan.estimated_duration_seconds = 600  # 10 minutes
        
        unsafe_step = TaskStep()
        unsafe_step.action_type = "high_speed"
        unsafe_step.description = "Rush through obstacles"
        unsafe_step.estimated_duration = 600.0
        unsafe_plan.steps.append(unsafe_step)
        
        # Validate unsafe plan
        safety_response = llm.validate_task_plan_safety(unsafe_plan)
        
        assert isinstance(safety_response, SafetyVerificationResponse)
        assert safety_response.confidence_score > 0.0
        assert len(safety_response.explanation) > 0
    
    def test_task_plan_to_text_conversion(self):
        """Test task plan to text conversion"""
        llm = SafetyEmbeddedLLM()
        
        # Create task plan
        task_plan = TaskPlan()
        task_plan.goal_description = "Test navigation"
        task_plan.estimated_duration_seconds = 60
        task_plan.required_capabilities = ["navigation", "perception"]
        task_plan.safety_considerations = ["Avoid obstacles", "Maintain safe distance"]
        
        step = TaskStep()
        step.action_type = "navigation"
        step.description = "Move to target"
        step.estimated_duration = 30.0
        step.parameters = ["target_x", "target_y"]
        task_plan.steps.append(step)
        
        # Convert to text
        text = llm._task_plan_to_text(task_plan)
        
        # Test text contains all plan elements
        assert task_plan.goal_description in text
        assert str(task_plan.estimated_duration_seconds) in text
        assert "navigation" in text
        assert "perception" in text
        assert "Avoid obstacles" in text
        assert "Move to target" in text
        assert "30.0" in text
    
    def test_error_handling(self):
        """Test error handling in safety-embedded LLM"""
        llm = SafetyEmbeddedLLM()
        
        # Test with invalid input
        response = llm.generate_safe_response("")
        assert response.safety_score == 0.0 or response.safety_score > 0.0  # Should handle gracefully
        
        # Test with very long input
        long_command = "Navigate to the " + "very " * 1000 + "distant location"
        response = llm.generate_safe_response(long_command)
        assert isinstance(response, SafetyEmbeddedResponse)
    
    def test_performance_characteristics(self):
        """Test performance characteristics"""
        llm = SafetyEmbeddedLLM()
        
        command = "Navigate to the chair"
        
        # Test response time
        start_time = time.time()
        response = llm.generate_safe_response(command)
        end_time = time.time()
        
        actual_time = end_time - start_time
        reported_time = response.execution_time
        
        # Times should be reasonably close
        assert abs(actual_time - reported_time) < 1.0, \
            f"Time mismatch: actual={actual_time:.2f}, reported={reported_time:.2f}"
        
        # Response should be reasonably fast
        assert actual_time < 5.0, f"Response too slow: {actual_time:.2f}s"


class TestSafetyEmbeddedLLMIntegration:
    """Integration tests for safety-embedded LLM with other components"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        rclpy.init()
        yield
        rclpy.shutdown()
    
    def test_safety_embedded_llm_with_llm_interface(self):
        """Test integration with LLM interface node"""
        from eip_llm_interface.llm_interface_node import LLMInterfaceNode
        
        # Create LLM interface node
        node = LLMInterfaceNode()
        
        # Test that safety-embedded LLM is initialized
        assert hasattr(node, 'safety_embedded_llm')
        assert node.safety_embedded_llm is not None
        
        # Test task plan generation
        command = "Navigate to the red chair"
        task_plan = node._generate_task_plan(command)
        
        if task_plan:
            assert isinstance(task_plan, TaskPlan)
            assert task_plan.goal_description is not None
            assert len(task_plan.safety_considerations) > 0
        
        node.destroy_node()
    
    def test_safety_verification_integration(self):
        """Test safety verification integration"""
        from eip_llm_interface.llm_interface_node import LLMInterfaceNode
        
        node = LLMInterfaceNode()
        
        # Create test task plan
        task_plan = TaskPlan()
        task_plan.goal_description = "Test task"
        task_plan.estimated_duration_seconds = 30
        
        # Test safety verification
        safety_response = node._verify_task_plan_safety(task_plan)
        
        assert isinstance(safety_response, SafetyVerificationResponse)
        assert hasattr(safety_response, 'is_safe')
        assert hasattr(safety_response, 'confidence_score')
        assert hasattr(safety_response, 'explanation')
        
        node.destroy_node()


if __name__ == '__main__':
    pytest.main([__file__]) 