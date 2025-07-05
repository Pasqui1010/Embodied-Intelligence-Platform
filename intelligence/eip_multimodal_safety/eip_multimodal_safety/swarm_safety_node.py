#!/usr/bin/env python3
"""
Swarm Safety Intelligence Node

Implements distributed safety validation across multiple nodes with bio-mimetic evolution.
Each node acts as a "safety cell" that can detect, learn, and adapt to safety patterns.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import threading
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import queue
import uuid

# ROS 2 imports
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import Image, PointCloud2, Imu
from geometry_msgs.msg import Twist, PoseStamped
from eip_interfaces.msg import SafetyVerificationRequest, SafetyVerificationResponse, SafetyViolation
from eip_interfaces.srv import ValidateTaskPlan

# Custom imports
from .sensor_fusion import SensorFusionEngine
from .bio_mimetic_learning import BioMimeticSafetyLearner
from .swarm_coordination import SwarmCoordinator


class SafetyCellType(Enum):
    """Types of safety cells in the swarm"""
    VISION = "vision"
    AUDIO = "audio"
    TACTILE = "tactile"
    PROPRIOCEPTIVE = "proprioceptive"
    FUSION = "fusion"
    COORDINATOR = "coordinator"


@dataclass
class SafetyPattern:
    """Represents a learned safety pattern"""
    pattern_id: str
    cell_type: SafetyCellType
    features: np.ndarray
    confidence: float
    timestamp: float
    violation_count: int = 0
    adaptation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmDecision:
    """Represents a swarm safety decision"""
    decision_id: str
    timestamp: float
    safety_level: float  # 0.0 (unsafe) to 1.0 (safe)
    confidence: float
    contributing_cells: List[str]
    consensus_level: float
    evolution_stage: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class SwarmSafetyNode(Node):
    """
    Swarm Safety Intelligence Node
    
    Implements bio-mimetic safety evolution with distributed validation.
    Each node acts as a specialized safety cell that can learn and adapt.
    """
    
    def __init__(self):
        super().__init__('swarm_safety_node')
        
        # Node configuration
        self.node_id = str(uuid.uuid4())[:8]
        self.cell_type = self.declare_parameter('cell_type', 'fusion').value
        self.swarm_size = self.declare_parameter('swarm_size', 5).value
        self.learning_rate = self.declare_parameter('learning_rate', 0.001).value
        self.evolution_threshold = self.declare_parameter('evolution_threshold', 0.8).value
        
        # Initialize components
        self.sensor_fusion = SensorFusionEngine()
        self.bio_learner = BioMimeticSafetyLearner(
            learning_rate=self.learning_rate,
            evolution_threshold=self.evolution_threshold
        )
        self.swarm_coordinator = SwarmCoordinator(
            node_id=self.node_id,
            swarm_size=self.swarm_size
        )
        
        # State management
        self.safety_patterns: Dict[str, SafetyPattern] = {}
        self.swarm_decisions: List[SwarmDecision] = []
        self.evolution_stage = 0
        self.adaptation_count = 0
        
        # Thread safety
        self.lock = threading.RLock()
        self.decision_queue = queue.Queue(maxsize=100)
        
        # Setup QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Setup callback groups
        self.callback_group = ReentrantCallbackGroup()
        
        # Initialize publishers and subscribers
        self._setup_communications(qos_profile)
        
        # Start background threads
        self._start_background_threads()
        
        self.get_logger().info(f"Swarm Safety Node {self.node_id} initialized as {self.cell_type} cell")
    
    def _setup_communications(self, qos_profile: QoSProfile):
        """Setup ROS 2 communications"""
        
        # Publishers
        self.safety_response_pub = self.create_publisher(
            SafetyVerificationResponse,
            '/swarm/safety/response',
            10
        )
        
        self.safety_violation_pub = self.create_publisher(
            SafetyViolation,
            '/swarm/safety/violation',
            10
        )
        
        self.swarm_decision_pub = self.create_publisher(
            String,
            '/swarm/decision',
            10
        )
        
        self.evolution_status_pub = self.create_publisher(
            String,
            '/swarm/evolution/status',
            10
        )
        
        # Subscribers
        self.safety_request_sub = self.create_subscription(
            SafetyVerificationRequest,
            '/swarm/safety/request',
            self._handle_safety_request,
            10,
            callback_group=self.callback_group
        )
        
        self.swarm_decision_sub = self.create_subscription(
            String,
            '/swarm/decision',
            self._handle_swarm_decision,
            10,
            callback_group=self.callback_group
        )
        
        # Services
        self.validate_task_service = self.create_service(
            ValidateTaskPlan,
            '/swarm/validate_task',
            self._handle_validate_task,
            callback_group=self.callback_group
        )
        
        # Timers
        self.evolution_timer = self.create_timer(
            5.0,  # Check for evolution every 5 seconds
            self._check_evolution_opportunity,
            callback_group=self.callback_group
        )
        
        self.pattern_cleanup_timer = self.create_timer(
            30.0,  # Cleanup old patterns every 30 seconds
            self._cleanup_old_patterns,
            callback_group=self.callback_group
        )
    
    def _start_background_threads(self):
        """Start background processing threads"""
        
        # Decision processing thread
        self.decision_thread = threading.Thread(
            target=self._process_decision_queue,
            daemon=True
        )
        self.decision_thread.start()
        
        # Pattern learning thread
        self.learning_thread = threading.Thread(
            target=self._continuous_learning_loop,
            daemon=True
        )
        self.learning_thread.start()
        
        # Swarm coordination thread
        self.coordination_thread = threading.Thread(
            target=self._swarm_coordination_loop,
            daemon=True
        )
        self.coordination_thread.start()
    
    def _handle_safety_request(self, request: SafetyVerificationRequest):
        """Handle incoming safety verification requests"""
        try:
            with self.lock:
                # Process the request through sensor fusion
                sensor_data = self._extract_sensor_data(request)
                fusion_result = self.sensor_fusion.process(sensor_data)
                
                # Apply bio-mimetic learning
                safety_assessment = self.bio_learner.assess_safety(
                    fusion_result, 
                    self.safety_patterns
                )
                
                # Generate swarm decision
                swarm_decision = self._generate_swarm_decision(
                    safety_assessment, 
                    fusion_result
                )
                
                # Publish response
                response = SafetyVerificationResponse()
                response.request_id = request.request_id
                response.is_safe = swarm_decision.safety_level > 0.7
                response.confidence = swarm_decision.confidence
                response.safety_level = swarm_decision.safety_level
                response.metadata = json.dumps(swarm_decision.metadata)
                
                self.safety_response_pub.publish(response)
                
                # Add to decision queue for swarm coordination
                self.decision_queue.put(swarm_decision, timeout=1.0)
                
                self.get_logger().debug(f"Processed safety request {request.request_id}")
                
        except Exception as e:
            self.get_logger().error(f"Error processing safety request: {e}")
            # Publish failure response
            response = SafetyVerificationResponse()
            response.request_id = request.request_id
            response.is_safe = False
            response.confidence = 0.0
            response.safety_level = 0.0
            response.metadata = json.dumps({"error": str(e)})
            self.safety_response_pub.publish(response)
    
    def _handle_swarm_decision(self, msg: String):
        """Handle decisions from other swarm nodes"""
        try:
            decision_data = json.loads(msg.data)
            decision = SwarmDecision(**decision_data)
            
            with self.lock:
                # Integrate external decision into local patterns
                self._integrate_external_decision(decision)
                
                # Check for consensus opportunities
                self._check_consensus(decision)
                
        except Exception as e:
            self.get_logger().error(f"Error processing swarm decision: {e}")
    
    def _handle_validate_task(self, request, response):
        """Handle task plan validation requests"""
        try:
            with self.lock:
                # Validate task plan through swarm intelligence
                validation_result = self._validate_task_plan(request.task_plan)
                
                response.is_valid = validation_result['is_valid']
                response.confidence = validation_result['confidence']
                response.safety_level = validation_result['safety_level']
                response.recommendations = validation_result['recommendations']
                
                return response
                
        except Exception as e:
            self.get_logger().error(f"Error validating task plan: {e}")
            response.is_valid = False
            response.confidence = 0.0
            response.safety_level = 0.0
            response.recommendations = [f"Validation error: {str(e)}"]
            return response
    
    def _extract_sensor_data(self, request: SafetyVerificationRequest) -> Dict[str, Any]:
        """Extract sensor data from safety request"""
        sensor_data = {
            'timestamp': time.time(),
            'cell_type': self.cell_type,
            'node_id': self.node_id
        }
        
        # Parse metadata for sensor information
        if request.metadata:
            try:
                metadata = json.loads(request.metadata)
                sensor_data.update(metadata)
            except json.JSONDecodeError:
                self.get_logger().warning("Invalid metadata format in safety request")
        
        return sensor_data
    
    def _generate_swarm_decision(self, safety_assessment: Dict, fusion_result: Dict) -> SwarmDecision:
        """Generate a swarm safety decision"""
        
        decision = SwarmDecision(
            decision_id=str(uuid.uuid4()),
            timestamp=time.time(),
            safety_level=safety_assessment['safety_level'],
            confidence=safety_assessment['confidence'],
            contributing_cells=[self.node_id],
            consensus_level=1.0,  # Will be updated by swarm coordinator
            evolution_stage=self.evolution_stage,
            metadata={
                'cell_type': self.cell_type,
                'fusion_result': fusion_result,
                'patterns_used': len(self.safety_patterns),
                'adaptation_count': self.adaptation_count
            }
        )
        
        # Add to local history
        self.swarm_decisions.append(decision)
        
        # Keep only recent decisions
        if len(self.swarm_decisions) > 100:
            self.swarm_decisions = self.swarm_decisions[-100:]
        
        return decision
    
    def _integrate_external_decision(self, decision: SwarmDecision):
        """Integrate external swarm decision into local patterns"""
        
        # Learn from external decision if it's from a different cell type
        if decision.metadata.get('cell_type') != self.cell_type:
            # Create a new pattern from external decision
            pattern = SafetyPattern(
                pattern_id=f"external_{decision.decision_id}",
                cell_type=SafetyCellType(decision.metadata.get('cell_type', 'fusion')),
                features=np.array([decision.safety_level, decision.confidence]),
                confidence=decision.confidence,
                timestamp=decision.timestamp,
                metadata=decision.metadata
            )
            
            self.safety_patterns[pattern.pattern_id] = pattern
            
            # Trigger adaptation if significant difference
            if abs(decision.safety_level - self._get_current_safety_level()) > 0.2:
                self._trigger_adaptation(decision)
    
    def _check_consensus(self, decision: SwarmDecision):
        """Check for consensus opportunities with other nodes"""
        
        # Find similar recent decisions
        similar_decisions = [
            d for d in self.swarm_decisions[-10:]
            if abs(d.safety_level - decision.safety_level) < 0.1
            and d.timestamp > time.time() - 30.0
        ]
        
        if len(similar_decisions) >= 3:
            # Consensus detected - trigger evolution
            self._trigger_evolution(similar_decisions + [decision])
    
    def _validate_task_plan(self, task_plan: str) -> Dict[str, Any]:
        """Validate a task plan using swarm intelligence"""
        
        # Parse task plan
        try:
            plan_data = json.loads(task_plan)
        except json.JSONDecodeError:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'safety_level': 0.0,
                'recommendations': ['Invalid task plan format']
            }
        
        # Assess safety using bio-mimetic learning
        safety_assessment = self.bio_learner.assess_task_safety(plan_data)
        
        # Generate recommendations
        recommendations = self._generate_safety_recommendations(safety_assessment)
        
        return {
            'is_valid': safety_assessment['is_safe'],
            'confidence': safety_assessment['confidence'],
            'safety_level': safety_assessment['safety_level'],
            'recommendations': recommendations
        }
    
    def _generate_safety_recommendations(self, assessment: Dict) -> List[str]:
        """Generate safety recommendations based on assessment"""
        recommendations = []
        
        if assessment['safety_level'] < 0.5:
            recommendations.append("Task has significant safety risks")
        
        if assessment['confidence'] < 0.7:
            recommendations.append("Low confidence in safety assessment - recommend human supervision")
        
        # Add specific recommendations based on patterns
        for pattern in self.safety_patterns.values():
            if pattern.violation_count > 5:
                recommendations.append(f"Pattern {pattern.pattern_id} has high violation rate")
        
        return recommendations
    
    def _trigger_adaptation(self, trigger_decision: SwarmDecision):
        """Trigger bio-mimetic adaptation"""
        with self.lock:
            self.adaptation_count += 1
            
            # Update learning parameters
            self.bio_learner.adapt(trigger_decision)
            
            # Publish adaptation status
            status_msg = String()
            status_msg.data = json.dumps({
                'node_id': self.node_id,
                'adaptation_count': self.adaptation_count,
                'trigger_decision': trigger_decision.decision_id,
                'timestamp': time.time()
            })
            self.evolution_status_pub.publish(status_msg)
    
    def _trigger_evolution(self, consensus_decisions: List[SwarmDecision]):
        """Trigger bio-mimetic evolution"""
        with self.lock:
            self.evolution_stage += 1
            
            # Evolve the bio-mimetic learner
            self.bio_learner.evolve(consensus_decisions)
            
            # Update swarm coordination
            self.swarm_coordinator.evolve(self.evolution_stage)
            
            # Publish evolution status
            status_msg = String()
            status_msg.data = json.dumps({
                'node_id': self.node_id,
                'evolution_stage': self.evolution_stage,
                'consensus_size': len(consensus_decisions),
                'timestamp': time.time()
            })
            self.evolution_status_pub.publish(status_msg)
            
            self.get_logger().info(f"Evolution triggered - stage {self.evolution_stage}")
    
    def _get_current_safety_level(self) -> float:
        """Get current safety level based on recent decisions"""
        if not self.swarm_decisions:
            return 0.5  # Neutral default
        
        recent_decisions = [
            d for d in self.swarm_decisions[-5:]
            if d.timestamp > time.time() - 10.0
        ]
        
        if not recent_decisions:
            return 0.5
        
        return np.mean([d.safety_level for d in recent_decisions])
    
    def _check_evolution_opportunity(self):
        """Check for evolution opportunities"""
        try:
            with self.lock:
                # Check if we have enough patterns for evolution
                if len(self.safety_patterns) >= 10:
                    # Check for pattern convergence
                    pattern_confidences = [p.confidence for p in self.safety_patterns.values()]
                    if np.std(pattern_confidences) < 0.1:  # Low variance indicates convergence
                        self._trigger_evolution([])
                        
        except Exception as e:
            self.get_logger().error(f"Error checking evolution opportunity: {e}")
    
    def _cleanup_old_patterns(self):
        """Clean up old safety patterns"""
        try:
            with self.lock:
                current_time = time.time()
                old_patterns = [
                    pattern_id for pattern_id, pattern in self.safety_patterns.items()
                    if current_time - pattern.timestamp > 3600  # 1 hour
                ]
                
                for pattern_id in old_patterns:
                    del self.safety_patterns[pattern_id]
                
                if old_patterns:
                    self.get_logger().debug(f"Cleaned up {len(old_patterns)} old patterns")
                    
        except Exception as e:
            self.get_logger().error(f"Error cleaning up old patterns: {e}")
    
    def _process_decision_queue(self):
        """Background thread for processing decision queue"""
        while rclpy.ok():
            try:
                decision = self.decision_queue.get(timeout=1.0)
                
                # Publish decision to swarm
                decision_msg = String()
                decision_msg.data = json.dumps(decision.__dict__)
                self.swarm_decision_pub.publish(decision_msg)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error processing decision queue: {e}")
    
    def _continuous_learning_loop(self):
        """Background thread for continuous learning"""
        while rclpy.ok():
            try:
                with self.lock:
                    # Update patterns based on recent decisions
                    self._update_patterns_from_decisions()
                    
                    # Trigger learning if needed
                    if len(self.swarm_decisions) >= 5:
                        self.bio_learner.learn_from_decisions(self.swarm_decisions[-5:])
                
                time.sleep(2.0)  # Learn every 2 seconds
                
            except Exception as e:
                self.get_logger().error(f"Error in continuous learning: {e}")
                time.sleep(5.0)
    
    def _swarm_coordination_loop(self):
        """Background thread for swarm coordination"""
        while rclpy.ok():
            try:
                # Update swarm coordination
                self.swarm_coordinator.update()
                
                # Check for coordination events
                events = self.swarm_coordinator.get_events()
                for event in events:
                    self._handle_coordination_event(event)
                
                time.sleep(1.0)  # Coordinate every second
                
            except Exception as e:
                self.get_logger().error(f"Error in swarm coordination: {e}")
                time.sleep(5.0)
    
    def _update_patterns_from_decisions(self):
        """Update patterns based on recent decisions"""
        if not self.swarm_decisions:
            return
        
        recent_decisions = self.swarm_decisions[-10:]
        
        for decision in recent_decisions:
            # Create or update pattern
            pattern_id = f"decision_{decision.decision_id}"
            
            if pattern_id not in self.safety_patterns:
                pattern = SafetyPattern(
                    pattern_id=pattern_id,
                    cell_type=SafetyCellType(self.cell_type),
                    features=np.array([decision.safety_level, decision.confidence]),
                    confidence=decision.confidence,
                    timestamp=decision.timestamp,
                    metadata=decision.metadata
                )
                self.safety_patterns[pattern_id] = pattern
            else:
                # Update existing pattern
                pattern = self.safety_patterns[pattern_id]
                pattern.confidence = (pattern.confidence + decision.confidence) / 2
                pattern.adaptation_count += 1
    
    def _handle_coordination_event(self, event: Dict):
        """Handle coordination events from swarm coordinator"""
        event_type = event.get('type')
        
        if event_type == 'consensus_reached':
            self.get_logger().info(f"Swarm consensus reached: {event}")
        elif event_type == 'conflict_detected':
            self.get_logger().warning(f"Swarm conflict detected: {event}")
        elif event_type == 'evolution_triggered':
            self.get_logger().info(f"Swarm evolution triggered: {event}")


def main(args=None):
    rclpy.init(args=args)
    
    # Create node
    node = SwarmSafetyNode()
    
    # Create executor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        # Spin the node
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 