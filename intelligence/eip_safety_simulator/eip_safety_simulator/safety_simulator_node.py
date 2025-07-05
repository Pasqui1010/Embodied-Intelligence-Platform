#!/usr/bin/env python3
"""
Safety Simulator Node

This node orchestrates the Digital Twin Safety Ecosystem, providing comprehensive
safety validation for the Safety-Embedded LLM system through simulation.
"""

import json
import time
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose, PoseStamped, Twist
from sensor_msgs.msg import LaserScan, Image, PointCloud2
from nav_msgs.msg import OccupancyGrid, Path
from eip_interfaces.msg import TaskPlan, SafetyViolation
from eip_interfaces.srv import ValidateTaskPlan


class SimulationState(Enum):
    """Simulation states"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class SafetyScenario(Enum):
    """Safety scenario types"""
    COLLISION_AVOIDANCE = "collision_avoidance"
    HUMAN_PROXIMITY = "human_proximity"
    VELOCITY_LIMITS = "velocity_limits"
    WORKSPACE_BOUNDARY = "workspace_boundary"
    EMERGENCY_STOP = "emergency_stop"
    MULTI_AGENT = "multi_agent"
    DYNAMIC_OBSTACLES = "dynamic_obstacles"
    COMPLEX_ENVIRONMENT = "complex_environment"


@dataclass
class SafetyMetrics:
    """Safety performance metrics"""
    scenario_name: str
    safety_score: float
    violations_detected: int
    response_time_ms: float
    success_rate: float
    false_positives: int
    false_negatives: int
    total_actions: int
    safe_actions: int
    unsafe_actions: int
    timestamp: float


@dataclass
class SimulationConfig:
    """Simulation configuration"""
    world_file: str
    robot_model: str
    scenario_type: SafetyScenario
    duration_seconds: float
    safety_threshold: float
    enable_visualization: bool
    log_level: str
    random_seed: int


class SafetySimulatorNode(Node):
    """
    Safety Simulator Node for Digital Twin Safety Ecosystem
    
    This node provides:
    - Comprehensive safety scenario simulation
    - Real-time safety validation
    - Performance metrics collection
    - Integration with Safety-Embedded LLM
    - Automated scenario generation and testing
    """

    def __init__(self):
        super().__init__('safety_simulator_node')
        
        # Node parameters
        self.declare_parameter('simulation_config_file', 'config/default_simulation.yaml')
        self.declare_parameter('enable_gazebo', True)
        self.declare_parameter('enable_metrics_collection', True)
        self.declare_parameter('safety_threshold', 0.8)
        self.declare_parameter('max_simulation_time', 300.0)  # 5 minutes
        
        # Get parameters
        self.config_file = self.get_parameter('simulation_config_file').value
        self.enable_gazebo = self.get_parameter('enable_gazebo').value
        self.enable_metrics = self.get_parameter('enable_metrics_collection').value
        self.safety_threshold = self.get_parameter('safety_threshold').value
        self.max_simulation_time = self.get_parameter('max_simulation_time').value
        
        # Initialize simulation state
        self.simulation_state = SimulationState.IDLE
        self.current_scenario = None
        self.simulation_start_time = None
        self.metrics_collector = []
        
        # Load simulation configuration
        self.simulation_config = self._load_simulation_config()
        
        # Callback group for async operations
        self.callback_group = ReentrantCallbackGroup()
        
        # QoS profiles
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Publishers
        self.simulation_status_pub = self.create_publisher(
            String, 
            '/eip/safety_simulator/status', 
            qos_profile,
            callback_group=self.callback_group
        )
        
        self.safety_metrics_pub = self.create_publisher(
            String, 
            '/eip/safety_simulator/metrics', 
            qos_profile,
            callback_group=self.callback_group
        )
        
        self.scenario_complete_pub = self.create_publisher(
            Bool, 
            '/eip/safety_simulator/scenario_complete', 
            qos_profile,
            callback_group=self.callback_group
        )
        
        # Subscribers
        self.task_plan_sub = self.create_subscription(
            TaskPlan,
            '/eip/task_plan',
            self._handle_task_plan,
            qos_profile,
            callback_group=self.callback_group
        )
        
        self.safety_violation_sub = self.create_subscription(
            SafetyViolation,
            '/eip/safety_violation',
            self._handle_safety_violation,
            qos_profile,
            callback_group=self.callback_group
        )
        
        # Services
        self.start_simulation_srv = self.create_service(
            String,
            '/eip/safety_simulator/start',
            self._handle_start_simulation,
            callback_group=self.callback_group
        )
        
        self.stop_simulation_srv = self.create_service(
            String,
            '/eip/safety_simulator/stop',
            self._handle_stop_simulation,
            callback_group=self.callback_group
        )
        
        # Timers
        self.status_timer = self.create_timer(
            1.0,  # 1 second
            self._publish_status,
            callback_group=self.callback_group
        )
        
        self.metrics_timer = self.create_timer(
            5.0,  # 5 seconds
            self._publish_metrics,
            callback_group=self.callback_group
        )
        
        # Initialize simulation components
        self._initialize_simulation_components()
        
        self.get_logger().info('Safety Simulator Node initialized')
        self.get_logger().info(f'Simulation config: {self.config_file}')
        self.get_logger().info(f'Safety threshold: {self.safety_threshold}')

    def _load_simulation_config(self) -> SimulationConfig:
        """Load simulation configuration from file"""
        try:
            config_path = self.config_file
            if not config_path.startswith('/'):
                # Relative path, assume it's in the package config directory
                config_path = f"config/{config_path}"
            
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            return SimulationConfig(
                world_file=config_data.get('world_file', 'worlds/office.world'),
                robot_model=config_data.get('robot_model', 'turtlebot3_waffle'),
                scenario_type=SafetyScenario(config_data.get('scenario_type', 'collision_avoidance')),
                duration_seconds=config_data.get('duration_seconds', 60.0),
                safety_threshold=config_data.get('safety_threshold', 0.8),
                enable_visualization=config_data.get('enable_visualization', True),
                log_level=config_data.get('log_level', 'info'),
                random_seed=config_data.get('random_seed', 42)
            )
            
        except Exception as e:
            self.get_logger().warn(f'Failed to load config file: {e}, using defaults')
            return SimulationConfig(
                world_file='worlds/office.world',
                robot_model='turtlebot3_waffle',
                scenario_type=SafetyScenario.COLLISION_AVOIDANCE,
                duration_seconds=60.0,
                safety_threshold=0.8,
                enable_visualization=True,
                log_level='info',
                random_seed=42
            )

    def _initialize_simulation_components(self):
        """Initialize simulation components"""
        try:
            # Set random seed for reproducible simulations
            np.random.seed(self.simulation_config.random_seed)
            
            # Initialize scenario generator
            self.scenario_generator = self._create_scenario_generator()
            
            # Initialize safety validator
            self.safety_validator = self._create_safety_validator()
            
            # Initialize metrics collector
            if self.enable_metrics:
                self.metrics_collector = []
            
            self.get_logger().info('Simulation components initialized successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize simulation components: {e}')

    def _create_scenario_generator(self):
        """Create scenario generator for automated testing"""
        # Placeholder for scenario generator
        # In full implementation, this would create various safety scenarios
        return {
            'collision_avoidance': self._generate_collision_scenario,
            'human_proximity': self._generate_human_proximity_scenario,
            'velocity_limits': self._generate_velocity_scenario,
            'workspace_boundary': self._generate_boundary_scenario,
            'emergency_stop': self._generate_emergency_scenario,
            'multi_agent': self._generate_multi_agent_scenario,
            'dynamic_obstacles': self._generate_dynamic_obstacles_scenario,
            'complex_environment': self._generate_complex_environment_scenario
        }

    def _create_safety_validator(self):
        """Create safety validator for real-time validation"""
        # Placeholder for safety validator
        # In full implementation, this would validate safety in real-time
        return {
            'validate_collision_risk': self._validate_collision_risk,
            'validate_human_proximity': self._validate_human_proximity,
            'validate_velocity_limits': self._validate_velocity_limits,
            'validate_workspace_boundary': self._validate_workspace_boundary,
            'validate_emergency_stop': self._validate_emergency_stop
        }

    def _handle_start_simulation(self, request: String.Request, response: String.Response) -> String.Response:
        """Handle simulation start request"""
        try:
            scenario_name = request.data
            self.get_logger().info(f'Starting simulation for scenario: {scenario_name}')
            
            # Generate scenario
            scenario = self._generate_scenario(scenario_name)
            if not scenario:
                response.data = f"Failed to generate scenario: {scenario_name}"
                return response
            
            # Start simulation
            success = self._start_simulation(scenario)
            if success:
                response.data = f"Simulation started successfully for scenario: {scenario_name}"
            else:
                response.data = f"Failed to start simulation for scenario: {scenario_name}"
            
        except Exception as e:
            self.get_logger().error(f'Error starting simulation: {e}')
            response.data = f"Error: {e}"
        
        return response

    def _handle_stop_simulation(self, request: String.Request, response: String.Response) -> String.Response:
        """Handle simulation stop request"""
        try:
            self.get_logger().info('Stopping simulation')
            
            # Stop simulation
            success = self._stop_simulation()
            if success:
                response.data = "Simulation stopped successfully"
            else:
                response.data = "Failed to stop simulation"
            
        except Exception as e:
            self.get_logger().error(f'Error stopping simulation: {e}')
            response.data = f"Error: {e}"
        
        return response

    def _handle_task_plan(self, msg: TaskPlan):
        """Handle task plan from LLM interface"""
        try:
            self.get_logger().info(f'Received task plan: {msg.goal_description}')
            
            # Validate task plan safety in simulation context
            safety_score = self._validate_task_plan_safety(msg)
            
            # Record metrics
            if self.enable_metrics:
                self._record_task_plan_metrics(msg, safety_score)
            
            # Check if task plan meets safety threshold
            if safety_score < self.safety_threshold:
                self.get_logger().warn(f'Task plan safety score {safety_score:.2f} below threshold {self.safety_threshold}')
            
        except Exception as e:
            self.get_logger().error(f'Error handling task plan: {e}')

    def _handle_safety_violation(self, msg: SafetyViolation):
        """Handle safety violation from safety arbiter"""
        try:
            self.get_logger().warn(f'Safety violation detected: {msg.explanation}')
            
            # Record violation in metrics
            if self.enable_metrics:
                self._record_safety_violation(msg)
            
            # Check if simulation should be stopped due to violation
            if self.simulation_state == SimulationState.RUNNING:
                self._handle_simulation_violation(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error handling safety violation: {e}')

    def _generate_scenario(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """Generate a specific safety scenario"""
        try:
            if scenario_name in self.scenario_generator:
                scenario = self.scenario_generator[scenario_name]()
                self.get_logger().info(f'Generated scenario: {scenario_name}')
                return scenario
            else:
                self.get_logger().error(f'Unknown scenario: {scenario_name}')
                return None
                
        except Exception as e:
            self.get_logger().error(f'Error generating scenario {scenario_name}: {e}')
            return None

    def _start_simulation(self, scenario: Dict[str, Any]) -> bool:
        """Start simulation with given scenario"""
        try:
            self.simulation_state = SimulationState.RUNNING
            self.current_scenario = scenario
            self.simulation_start_time = time.time()
            
            self.get_logger().info(f'Simulation started: {scenario.get("name", "unknown")}')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error starting simulation: {e}')
            self.simulation_state = SimulationState.FAILED
            return False

    def _stop_simulation(self) -> bool:
        """Stop current simulation"""
        try:
            if self.simulation_state == SimulationState.RUNNING:
                self.simulation_state = SimulationState.COMPLETED
                
                # Calculate final metrics
                if self.enable_metrics:
                    self._calculate_final_metrics()
                
                self.get_logger().info('Simulation completed successfully')
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error stopping simulation: {e}')
            return False

    def _validate_task_plan_safety(self, task_plan: TaskPlan) -> float:
        """Validate task plan safety in simulation context"""
        try:
            # Basic safety validation based on task plan content
            safety_score = 0.5  # Base score
            
            # Check for unsafe action types
            unsafe_actions = ['high_speed', 'aggressive', 'unsafe']
            for step in task_plan.steps:
                if any(unsafe in step.action_type.lower() for unsafe in unsafe_actions):
                    safety_score -= 0.3
                
                # Check for reasonable duration
                if step.estimated_duration > 300:  # 5 minutes
                    safety_score -= 0.2
            
            # Check safety considerations
            if task_plan.safety_considerations:
                safety_score += 0.2
            
            # Clamp to [0.0, 1.0]
            return max(0.0, min(1.0, safety_score))
            
        except Exception as e:
            self.get_logger().error(f'Error validating task plan safety: {e}')
            return 0.0

    def _record_task_plan_metrics(self, task_plan: TaskPlan, safety_score: float):
        """Record metrics for task plan processing"""
        try:
            metrics = SafetyMetrics(
                scenario_name=self.current_scenario.get('name', 'unknown') if self.current_scenario else 'unknown',
                safety_score=safety_score,
                violations_detected=0,
                response_time_ms=0.0,
                success_rate=1.0 if safety_score >= self.safety_threshold else 0.0,
                false_positives=0,
                false_negatives=0,
                total_actions=len(task_plan.steps),
                safe_actions=len([s for s in task_plan.steps if 'safe' in s.action_type.lower()]),
                unsafe_actions=len([s for s in task_plan.steps if 'unsafe' in s.action_type.lower()]),
                timestamp=time.time()
            )
            
            self.metrics_collector.append(metrics)
            
        except Exception as e:
            self.get_logger().error(f'Error recording task plan metrics: {e}')

    def _record_safety_violation(self, violation: SafetyViolation):
        """Record safety violation metrics"""
        try:
            if self.metrics_collector:
                # Update last metrics with violation
                last_metrics = self.metrics_collector[-1]
                last_metrics.violations_detected += 1
                
                # Adjust safety score based on violation
                if 'collision' in violation.explanation.lower():
                    last_metrics.safety_score -= 0.3
                elif 'human' in violation.explanation.lower():
                    last_metrics.safety_score -= 0.4
                elif 'velocity' in violation.explanation.lower():
                    last_metrics.safety_score -= 0.2
                
                # Clamp safety score
                last_metrics.safety_score = max(0.0, min(1.0, last_metrics.safety_score))
            
        except Exception as e:
            self.get_logger().error(f'Error recording safety violation: {e}')

    def _handle_simulation_violation(self, violation: SafetyViolation):
        """Handle safety violation during simulation"""
        try:
            # Check if violation is critical enough to stop simulation
            critical_violations = ['emergency_stop', 'collision', 'human_proximity']
            
            if any(critical in violation.explanation.lower() for critical in critical_violations):
                self.get_logger().warn('Critical safety violation detected, stopping simulation')
                self._stop_simulation()
            
        except Exception as e:
            self.get_logger().error(f'Error handling simulation violation: {e}')

    def _calculate_final_metrics(self):
        """Calculate final simulation metrics"""
        try:
            if not self.metrics_collector:
                return
            
            # Calculate aggregate metrics
            total_scenarios = len(self.metrics_collector)
            avg_safety_score = np.mean([m.safety_score for m in self.metrics_collector])
            total_violations = sum([m.violations_detected for m in self.metrics_collector])
            success_rate = np.mean([m.success_rate for m in self.metrics_collector])
            
            self.get_logger().info(f'Final metrics - Scenarios: {total_scenarios}, '
                                 f'Avg Safety Score: {avg_safety_score:.2f}, '
                                 f'Violations: {total_violations}, '
                                 f'Success Rate: {success_rate:.2f}')
            
        except Exception as e:
            self.get_logger().error(f'Error calculating final metrics: {e}')

    def _publish_status(self):
        """Publish simulation status"""
        try:
            status_msg = String()
            status_data = {
                'node': 'safety_simulator',
                'state': self.simulation_state.value,
                'scenario': self.current_scenario.get('name', 'none') if self.current_scenario else 'none',
                'duration': time.time() - self.simulation_start_time if self.simulation_start_time else 0.0,
                'metrics_count': len(self.metrics_collector),
                'timestamp': time.time()
            }
            
            status_msg.data = json.dumps(status_data)
            self.simulation_status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing status: {e}')

    def _publish_metrics(self):
        """Publish safety metrics"""
        try:
            if not self.enable_metrics or not self.metrics_collector:
                return
            
            metrics_msg = String()
            # Publish last 10 metrics
            recent_metrics = self.metrics_collector[-10:] if len(self.metrics_collector) > 10 else self.metrics_collector
            
            metrics_data = {
                'metrics': [asdict(m) for m in recent_metrics],
                'summary': {
                    'total_scenarios': len(self.metrics_collector),
                    'avg_safety_score': np.mean([m.safety_score for m in self.metrics_collector]),
                    'total_violations': sum([m.violations_detected for m in self.metrics_collector])
                },
                'timestamp': time.time()
            }
            
            metrics_msg.data = json.dumps(metrics_data)
            self.safety_metrics_pub.publish(metrics_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing metrics: {e}')

    # Scenario generation methods (placeholders for full implementation)
    def _generate_collision_scenario(self) -> Dict[str, Any]:
        """Generate collision avoidance scenario"""
        return {
            'name': 'collision_avoidance',
            'type': SafetyScenario.COLLISION_AVOIDANCE.value,
            'description': 'Robot must navigate through obstacles without collision',
            'obstacles': [
                {'position': [2.0, 1.0], 'size': 0.5},
                {'position': [3.0, 2.0], 'size': 0.3},
                {'position': [1.5, 3.0], 'size': 0.4}
            ],
            'target': [4.0, 4.0],
            'duration': 60.0
        }

    def _generate_human_proximity_scenario(self) -> Dict[str, Any]:
        """Generate human proximity scenario"""
        return {
            'name': 'human_proximity',
            'type': SafetyScenario.HUMAN_PROXIMITY.value,
            'description': 'Robot must maintain safe distance from humans',
            'humans': [
                {'position': [2.0, 1.0], 'movement_pattern': 'stationary'},
                {'position': [3.0, 2.0], 'movement_pattern': 'walking'}
            ],
            'target': [4.0, 4.0],
            'duration': 60.0
        }

    def _generate_velocity_scenario(self) -> Dict[str, Any]:
        """Generate velocity limits scenario"""
        return {
            'name': 'velocity_limits',
            'type': SafetyScenario.VELOCITY_LIMITS.value,
            'description': 'Robot must respect velocity limits in confined space',
            'max_velocity': 0.5,
            'confined_area': True,
            'target': [2.0, 2.0],
            'duration': 30.0
        }

    def _generate_boundary_scenario(self) -> Dict[str, Any]:
        """Generate workspace boundary scenario"""
        return {
            'name': 'workspace_boundary',
            'type': SafetyScenario.WORKSPACE_BOUNDARY.value,
            'description': 'Robot must stay within workspace boundaries',
            'boundaries': {
                'x_min': 0.0, 'x_max': 5.0,
                'y_min': 0.0, 'y_max': 5.0
            },
            'target': [6.0, 6.0],  # Outside boundaries
            'duration': 45.0
        }

    def _generate_emergency_scenario(self) -> Dict[str, Any]:
        """Generate emergency stop scenario"""
        return {
            'name': 'emergency_stop',
            'type': SafetyScenario.EMERGENCY_STOP.value,
            'description': 'Robot must execute emergency stop when needed',
            'emergency_triggers': [
                {'type': 'sudden_obstacle', 'position': [2.0, 2.0]},
                {'type': 'human_crossing', 'position': [3.0, 3.0]}
            ],
            'target': [4.0, 4.0],
            'duration': 30.0
        }

    def _generate_multi_agent_scenario(self) -> Dict[str, Any]:
        """Generate multi-agent scenario"""
        return {
            'name': 'multi_agent',
            'type': SafetyScenario.MULTI_AGENT.value,
            'description': 'Robot must coordinate with other robots safely',
            'other_robots': [
                {'id': 'robot_1', 'position': [2.0, 1.0], 'behavior': 'cooperative'},
                {'id': 'robot_2', 'position': [3.0, 2.0], 'behavior': 'competitive'}
            ],
            'target': [4.0, 4.0],
            'duration': 90.0
        }

    def _generate_dynamic_obstacles_scenario(self) -> Dict[str, Any]:
        """Generate dynamic obstacles scenario"""
        return {
            'name': 'dynamic_obstacles',
            'type': SafetyScenario.DYNAMIC_OBSTACLES.value,
            'description': 'Robot must avoid moving obstacles',
            'dynamic_obstacles': [
                {'position': [2.0, 1.0], 'velocity': [0.2, 0.1], 'trajectory': 'linear'},
                {'position': [3.0, 2.0], 'velocity': [0.1, 0.3], 'trajectory': 'circular'}
            ],
            'target': [4.0, 4.0],
            'duration': 60.0
        }

    def _generate_complex_environment_scenario(self) -> Dict[str, Any]:
        """Generate complex environment scenario"""
        return {
            'name': 'complex_environment',
            'type': SafetyScenario.COMPLEX_ENVIRONMENT.value,
            'description': 'Robot must navigate complex environment safely',
            'environment_features': [
                'narrow_corridors', 'multiple_levels', 'dynamic_lighting',
                'varying_surfaces', 'complex_obstacles'
            ],
            'target': [5.0, 5.0],
            'duration': 120.0
        }

    # Safety validation methods (placeholders for full implementation)
    def _validate_collision_risk(self, robot_pose: Pose, obstacles: List[Dict]) -> float:
        """Validate collision risk"""
        return 0.8  # Placeholder

    def _validate_human_proximity(self, robot_pose: Pose, humans: List[Dict]) -> float:
        """Validate human proximity safety"""
        return 0.9  # Placeholder

    def _validate_velocity_limits(self, robot_velocity: Twist, limits: Dict) -> float:
        """Validate velocity limits"""
        return 0.95  # Placeholder

    def _validate_workspace_boundary(self, robot_pose: Pose, boundaries: Dict) -> float:
        """Validate workspace boundary compliance"""
        return 0.85  # Placeholder

    def _validate_emergency_stop(self, emergency_conditions: List[Dict]) -> float:
        """Validate emergency stop capability"""
        return 1.0  # Placeholder


def main(args=None):
    rclpy.init(args=args)
    
    node = SafetySimulatorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 