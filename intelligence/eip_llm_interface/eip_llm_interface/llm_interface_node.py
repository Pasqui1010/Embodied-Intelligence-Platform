#!/usr/bin/env python3
"""
LLM Interface Node

This node provides the interface between LLMs and the robotics system.
It handles prompt engineering, response parsing, and safety validation.
"""

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image, PointCloud2
from eip_interfaces.msg import TaskPlan, TaskStep, SafetyVerificationRequest, SafetyVerificationResponse
from eip_interfaces.srv import ValidateTaskPlan
from .safety_embedded_llm import SafetyEmbeddedLLM, SafetyEmbeddedResponse


class LLMProvider(Enum):
    """Supported LLM providers"""
    LOCAL_MISTRAL = "local_mistral"
    LOCAL_PHI = "local_phi"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMResponse:
    """Structured LLM response"""
    success: bool
    content: str
    confidence: float
    safety_score: float
    execution_time: float
    tokens_used: int
    model_used: str


class LLMInterfaceNode(Node):
    """
    LLM Interface Node for Embodied Intelligence Platform
    
    This node provides:
    - Prompt engineering for robotics tasks
    - Response validation and parsing
    - Safety verification integration
    - Local model deployment support
    """

    def __init__(self):
        super().__init__('llm_interface_node')
        
        # Node parameters
        self.declare_parameter('llm_provider', LLMProvider.LOCAL_MISTRAL.value)
        self.declare_parameter('model_path', '/opt/models/mistral-7b-instruct')
        self.declare_parameter('max_tokens', 2048)
        self.declare_parameter('temperature', 0.1)
        self.declare_parameter('enable_safety_verification', True)
        self.declare_parameter('safety_threshold', 0.8)
        
        # Get parameters
        self.llm_provider = LLMProvider(self.get_parameter('llm_provider').value)
        self.model_path = self.get_parameter('model_path').value
        self.max_tokens = self.get_parameter('max_tokens').value
        self.temperature = self.get_parameter('temperature').value
        self.enable_safety_verification = self.get_parameter('enable_safety_verification').value
        self.safety_threshold = self.get_parameter('safety_threshold').value
        
        # Initialize LLM components
        self.llm_model = None
        self.tokenizer = None
        self.safety_embedded_llm = None
        self._initialize_llm()
        
        # Callback group for async operations
        self.callback_group = ReentrantCallbackGroup()
        
        # QoS profiles
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Publishers
        self.task_plan_pub = self.create_publisher(
            TaskPlan, 
            '/eip/task_plan', 
            qos_profile,
            callback_group=self.callback_group
        )
        
        self.llm_status_pub = self.create_publisher(
            String, 
            '/eip/llm/status', 
            qos_profile,
            callback_group=self.callback_group
        )
        
        # Subscribers
        self.natural_language_sub = self.create_subscription(
            String,
            '/eip/natural_language_command',
            self._handle_natural_language_command,
            qos_profile,
            callback_group=self.callback_group
        )
        
        self.scene_description_sub = self.create_subscription(
            String,
            '/eip/scene_description',
            self._handle_scene_description,
            qos_profile,
            callback_group=self.callback_group
        )
        
        # Services
        self.validate_task_plan_srv = self.create_service(
            ValidateTaskPlan,
            '/eip/validate_task_plan',
            self._handle_validate_task_plan,
            callback_group=self.callback_group
        )
        
        # Timers
        self.status_timer = self.create_timer(
            5.0,  # 5 seconds
            self._publish_status,
            callback_group=self.callback_group
        )
        
        self.get_logger().info(f'LLM Interface Node initialized with provider: {self.llm_provider.value}')
        self.get_logger().info(f'Model path: {self.model_path}')
        self.get_logger().info(f'Safety verification: {self.enable_safety_verification}')

    def _initialize_llm(self):
        """Initialize the LLM model based on provider"""
        try:
            # Initialize safety-embedded LLM
            self.safety_embedded_llm = SafetyEmbeddedLLM(
                model_name="microsoft/DialoGPT-medium" if self.llm_provider in [LLMProvider.LOCAL_MISTRAL, LLMProvider.LOCAL_PHI] else "microsoft/DialoGPT-medium",
                device="auto"
            )
            
            # Keep legacy initialization for backward compatibility
            if self.llm_provider in [LLMProvider.LOCAL_MISTRAL, LLMProvider.LOCAL_PHI]:
                self._initialize_local_model()
            else:
                self._initialize_api_model()
                
            self.get_logger().info(f'Safety-embedded LLM initialized successfully: {self.llm_provider.value}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize LLM: {e}')
            # Fallback to mock mode for development
            self._initialize_mock_model()

    def _initialize_local_model(self):
        """Initialize local model (Mistral 7B or Phi-3)"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # For now, use a smaller model for development
            model_name = "microsoft/DialoGPT-medium" if self.llm_provider == LLMProvider.LOCAL_PHI else "microsoft/DialoGPT-medium"
            
            self.get_logger().info(f'Loading local model: {model_name}')
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except ImportError:
            self.get_logger().warn('Transformers not available, using mock model')
            self._initialize_mock_model()

    def _initialize_api_model(self):
        """Initialize API-based model (OpenAI, Anthropic)"""
        # Placeholder for API integration
        self.get_logger().info('API model initialization (placeholder)')
        self._initialize_mock_model()

    def _initialize_mock_model(self):
        """Initialize mock model for development/testing"""
        self.get_logger().info('Initializing mock LLM model for development')
        self.llm_model = "mock_model"
        self.tokenizer = "mock_tokenizer"

    def _handle_natural_language_command(self, msg: String):
        """Handle natural language commands from users"""
        try:
            command = msg.data
            self.get_logger().info(f'Received natural language command: {command}')
            
            # Generate task plan from command
            task_plan = self._generate_task_plan(command)
            
            if task_plan:
                # Validate task plan safety
                if self.enable_safety_verification:
                    safety_response = self._verify_task_plan_safety(task_plan)
                    if not safety_response.is_safe:
                        self.get_logger().warn(f'Task plan rejected for safety: {safety_response.explanation}')
                        return
                
                # Publish validated task plan
                self.task_plan_pub.publish(task_plan)
                self.get_logger().info('Task plan published successfully')
            else:
                self.get_logger().warn('Failed to generate task plan from command')
                
        except Exception as e:
            self.get_logger().error(f'Error handling natural language command: {e}')

    def _handle_scene_description(self, msg: String):
        """Handle scene descriptions for context-aware planning"""
        try:
            scene_desc = msg.data
            self.get_logger().info(f'Received scene description: {scene_desc[:100]}...')
            
            # Store scene context for future planning
            self.current_scene_context = scene_desc
            
        except Exception as e:
            self.get_logger().error(f'Error handling scene description: {e}')

    def _handle_validate_task_plan(self, request: ValidateTaskPlan.Request, response: ValidateTaskPlan.Response) -> ValidateTaskPlan.Response:
        """Handle task plan validation requests"""
        try:
            task_plan = request.task_plan
            
            # Perform safety verification
            safety_response = self._verify_task_plan_safety(task_plan)
            
            response.is_valid = safety_response.is_safe
            response.explanation = safety_response.explanation
            response.confidence = safety_response.confidence_score
            
            self.get_logger().info(f'Task plan validation: {response.is_valid} (confidence: {response.confidence:.2f})')
            
        except Exception as e:
            self.get_logger().error(f'Error validating task plan: {e}')
            response.is_valid = False
            response.explanation = f'Validation error: {e}'
            response.confidence = 0.0
            
        return response

    def _generate_task_plan(self, command: str) -> Optional[TaskPlan]:
        """Generate a task plan from natural language command using safety-embedded LLM"""
        try:
            # Use safety-embedded LLM for generation
            if self.safety_embedded_llm:
                safety_response = self.safety_embedded_llm.generate_safe_response(command)
                
                # Log safety analysis
                self.get_logger().info(f'Safety score: {safety_response.safety_score:.2f}')
                if safety_response.violations_detected:
                    self.get_logger().warn(f'Safety violations: {safety_response.violations_detected}')
                
                # Parse response into task plan
                task_plan = self._parse_task_plan_response(safety_response.content, command)
                
                # Add safety metadata to task plan
                if task_plan:
                    task_plan.safety_considerations.extend([
                        f"Safety score: {safety_response.safety_score:.2f}",
                        f"Safety tokens: {[token.value for token in safety_response.safety_tokens_used]}"
                    ])
                
                return task_plan
            else:
                # Fallback to legacy method
                prompt = self._create_task_planning_prompt(command)
                llm_response = self._query_llm(prompt)
                
                if not llm_response.success:
                    self.get_logger().error(f'LLM query failed: {llm_response.content}')
                    return None
                
                task_plan = self._parse_task_plan_response(llm_response.content, command)
                return task_plan
            
        except Exception as e:
            self.get_logger().error(f'Error generating task plan: {e}')
            return None

    def _create_task_planning_prompt(self, command: str) -> str:
        """Create a prompt for task planning"""
        base_prompt = f"""
You are a robotics task planner. Given a natural language command, generate a structured task plan.

Command: "{command}"

Generate a JSON response with the following structure:
{{
    "goal_description": "Clear description of the goal",
    "steps": [
        {{
            "action_type": "navigation|manipulation|perception|communication",
            "description": "What this step does",
            "target_pose": {{"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
            "parameters": ["param1", "param2"],
            "estimated_duration": 5.0,
            "preconditions": ["precondition1"],
            "postconditions": ["postcondition1"]
        }}
    ],
    "estimated_duration_seconds": 30,
    "required_capabilities": ["navigation", "manipulation"],
    "safety_considerations": ["Keep safe distance from humans", "Avoid obstacles"]
}}

Focus on safety and feasibility. Keep steps simple and clear.
"""
        return base_prompt

    def _query_llm(self, prompt: str) -> LLMResponse:
        """Query the LLM with a prompt"""
        start_time = time.time()
        
        try:
            if self.llm_model == "mock_model":
                # Mock response for development
                mock_response = self._generate_mock_response(prompt)
                return LLMResponse(
                    success=True,
                    content=mock_response,
                    confidence=0.9,
                    safety_score=0.95,
                    execution_time=time.time() - start_time,
                    tokens_used=100,
                    model_used="mock_model"
                )
            
            # Real LLM query (placeholder for actual implementation)
            # This would use the loaded model and tokenizer
            self.get_logger().info('LLM query (placeholder for real implementation)')
            
            # For now, return mock response
            mock_response = self._generate_mock_response(prompt)
            return LLMResponse(
                success=True,
                content=mock_response,
                confidence=0.9,
                safety_score=0.95,
                execution_time=time.time() - start_time,
                tokens_used=100,
                model_used=self.llm_provider.value
            )
            
        except Exception as e:
            self.get_logger().error(f'LLM query error: {e}')
            return LLMResponse(
                success=False,
                content=str(e),
                confidence=0.0,
                safety_score=0.0,
                execution_time=time.time() - start_time,
                tokens_used=0,
                model_used=self.llm_provider.value
            )

    def _generate_mock_response(self, prompt: str) -> str:
        """Generate a mock response for development"""
        # Simple mock response based on prompt content
        if "navigation" in prompt.lower():
            return '''
{
    "goal_description": "Navigate to specified location",
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
        else:
            return '''
{
    "goal_description": "Execute basic task",
    "steps": [
        {
            "action_type": "perception",
            "description": "Scan environment",
            "target_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "parameters": ["scan_radius"],
            "estimated_duration": 2.0,
            "preconditions": ["sensors_ready"],
            "postconditions": ["environment_mapped"]
        }
    ],
    "estimated_duration_seconds": 2,
    "required_capabilities": ["perception"],
    "safety_considerations": ["Stay in safe area"]
}
'''

    def _parse_task_plan_response(self, response_content: str, original_command: str) -> Optional[TaskPlan]:
        """Parse LLM response into TaskPlan message"""
        try:
            # Extract JSON from response
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                self.get_logger().error('No JSON found in LLM response')
                return None
            
            json_str = response_content[json_start:json_end]
            plan_data = json.loads(json_str)
            
            # Create TaskPlan message
            task_plan = TaskPlan()
            task_plan.timestamp = self.get_clock().now().to_msg()
            task_plan.plan_id = f"plan_{int(time.time())}"
            task_plan.goal_description = plan_data.get('goal_description', original_command)
            task_plan.estimated_duration_seconds = plan_data.get('estimated_duration_seconds', 30)
            task_plan.required_capabilities = plan_data.get('required_capabilities', [])
            task_plan.safety_considerations = plan_data.get('safety_considerations', [])
            
            # Parse steps
            for step_data in plan_data.get('steps', []):
                step = TaskStep()
                step.action_type = step_data.get('action_type', 'unknown')
                step.description = step_data.get('description', '')
                step.estimated_duration = step_data.get('estimated_duration', 5.0)
                step.parameters = step_data.get('parameters', [])
                step.preconditions = step_data.get('preconditions', [])
                step.postconditions = step_data.get('postconditions', [])
                
                # Parse target pose
                pose_data = step_data.get('target_pose', {})
                step.target_pose.position.x = pose_data.get('x', 0.0)
                step.target_pose.position.y = pose_data.get('y', 0.0)
                step.target_pose.position.z = pose_data.get('z', 0.0)
                step.target_pose.orientation.w = pose_data.get('w', 1.0)
                
                task_plan.steps.append(step)
            
            return task_plan
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse JSON response: {e}')
            return None
        except Exception as e:
            self.get_logger().error(f'Error parsing task plan response: {e}')
            return None

    def _verify_task_plan_safety(self, task_plan: TaskPlan) -> SafetyVerificationResponse:
        """Verify the safety of a task plan using safety-embedded LLM"""
        try:
            # Use safety-embedded LLM for verification
            if self.safety_embedded_llm:
                safety_response = self.safety_embedded_llm.validate_task_plan_safety(task_plan)
                
                # Log detailed safety analysis
                self.get_logger().info(f'Safety-embedded verification: {safety_response.explanation}')
                
                return safety_response
            else:
                # Fallback to simple rule-based safety checking
                is_safe = self._simple_safety_check(task_plan)
                
                response = SafetyVerificationResponse()
                response.is_safe = is_safe
                response.confidence_score = 0.9 if is_safe else 0.1
                response.explanation = "Task plan passed basic safety checks" if is_safe else "Task plan contains potential safety risks"
                
                return response
            
        except Exception as e:
            self.get_logger().error(f'Error verifying task plan safety: {e}')
            response = SafetyVerificationResponse()
            response.is_safe = False
            response.confidence_score = 0.0
            response.explanation = f'Safety verification error: {e}'
            return response

    def _simple_safety_check(self, task_plan: TaskPlan) -> bool:
        """Simple rule-based safety checking"""
        # Check for dangerous action types
        dangerous_actions = ['high_speed', 'aggressive', 'unsafe']
        
        for step in task_plan.steps:
            if any(dangerous in step.action_type.lower() for dangerous in dangerous_actions):
                return False
            
            # Check for reasonable duration
            if step.estimated_duration > 300:  # 5 minutes
                return False
        
        return True

    def _publish_status(self):
        """Publish LLM interface status"""
        try:
            status_msg = String()
            status_msg.data = json.dumps({
                'node': 'llm_interface',
                'provider': self.llm_provider.value,
                'model_loaded': self.llm_model is not None,
                'safety_verification': self.enable_safety_verification,
                'timestamp': time.time()
            })
            
            self.llm_status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing status: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    node = LLMInterfaceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 