#!/usr/bin/env python3
"""
Multi-Modal Reasoner

This module implements advanced reasoning capabilities that combine visual perception,
natural language understanding, spatial awareness, and safety constraints for
autonomous robotic systems.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from eip_interfaces.msg import TaskPlan, TaskStep
from eip_interfaces.srv import SafetyVerificationRequest, SafetyVerificationResponse

from .spatial_reasoner import SpatialReasoner
from .temporal_reasoner import TemporalReasoner
from .causal_reasoner import CausalReasoner


class ReasoningType(Enum):
    """Types of reasoning capabilities"""
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    SOCIAL = "social"
    SAFETY = "safety"


@dataclass
class VisualContext:
    """Visual context information"""
    objects: List[Dict[str, Any]]
    scene_description: str
    spatial_relationships: Dict[str, List[str]]
    affordances: Dict[str, List[str]]
    confidence: float


@dataclass
class SpatialContext:
    """Spatial context information"""
    robot_pose: Dict[str, float]
    object_positions: Dict[str, Dict[str, float]]
    workspace_boundaries: Dict[str, float]
    navigation_graph: Dict[str, List[str]]
    occupancy_grid: Optional[np.ndarray] = None


@dataclass
class SafetyConstraints:
    """Safety constraints for reasoning"""
    collision_threshold: float
    human_proximity_threshold: float
    velocity_limits: Dict[str, float]
    workspace_boundaries: Dict[str, float]
    emergency_stop_conditions: List[str]


@dataclass
class ReasoningResult:
    """Result of multi-modal reasoning"""
    plan: TaskPlan
    confidence: float
    safety_score: float
    reasoning_steps: List[str]
    alternative_plans: List[TaskPlan]
    execution_time: float


class MultiModalReasoner:
    """
    Advanced Multi-Modal Reasoner that combines visual, language, spatial, and safety reasoning
    """
    
    def __init__(self):
        """Initialize the multi-modal reasoner"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize specialized reasoners
        self.spatial_reasoner = SpatialReasoner()
        self.temporal_reasoner = TemporalReasoner()
        self.causal_reasoner = CausalReasoner()
        
        # Reasoning capabilities
        self.reasoning_capabilities = {
            ReasoningType.SPATIAL: True,
            ReasoningType.TEMPORAL: True,
            ReasoningType.CAUSAL: True,
            ReasoningType.SOCIAL: True,
            ReasoningType.SAFETY: True
        }
        
        # Performance tracking
        self.reasoning_times = {}
        self.confidence_scores = {}
        
        self.logger.info("Multi-Modal Reasoner initialized successfully")
    
    def reason_about_scene(self, 
                          visual_context: VisualContext,
                          language_command: str,
                          spatial_context: SpatialContext,
                          safety_constraints: SafetyConstraints) -> ReasoningResult:
        """
        Perform multi-modal reasoning about a scene and command
        
        Args:
            visual_context: Current visual understanding
            language_command: Natural language command
            spatial_context: Current spatial awareness
            safety_constraints: Active safety constraints
            
        Returns:
            ReasoningResult with plan, confidence, and safety validation
        """
        start_time = time.time()
        reasoning_steps = []
        
        try:
            # 1. Spatial reasoning about object relationships
            self.logger.info("Performing spatial reasoning...")
            spatial_understanding = self.spatial_reasoner.analyze_scene(
                visual_context, spatial_context
            )
            reasoning_steps.append(f"Spatial analysis: {spatial_understanding.summary}")
            
            # 2. Language understanding with visual grounding
            self.logger.info("Performing language grounding...")
            language_understanding = self._ground_language_command(
                language_command, visual_context, spatial_context
            )
            reasoning_steps.append(f"Language grounding: {language_understanding['action']}")
            
            # 3. Temporal reasoning for sequence planning
            self.logger.info("Performing temporal reasoning...")
            temporal_plan = self.temporal_reasoner.plan_sequence(
                language_understanding, spatial_understanding
            )
            reasoning_steps.append(f"Temporal planning: {len(temporal_plan.steps)} steps")
            
            # 4. Causal reasoning about action consequences
            self.logger.info("Performing causal reasoning...")
            causal_analysis = self.causal_reasoner.analyze_effects(
                temporal_plan, spatial_understanding, safety_constraints
            )
            reasoning_steps.append(f"Causal analysis: {causal_analysis.risk_level}")
            
            # 5. Safety-aware plan generation
            self.logger.info("Generating safety-aware plan...")
            safe_plan = self._generate_safe_plan(
                temporal_plan, causal_analysis, safety_constraints
            )
            reasoning_steps.append(f"Safety validation: {safe_plan.safety_score}")
            
            # 6. Calculate confidence and alternatives
            confidence = self._calculate_confidence(
                spatial_understanding, language_understanding, causal_analysis
            )
            
            alternative_plans = self._generate_alternatives(
                safe_plan, spatial_understanding, safety_constraints
            )
            
            execution_time = time.time() - start_time
            
            result = ReasoningResult(
                plan=safe_plan,
                confidence=confidence,
                safety_score=safe_plan.safety_score,
                reasoning_steps=reasoning_steps,
                alternative_plans=alternative_plans,
                execution_time=execution_time
            )
            
            # Track performance
            self._track_performance(execution_time, confidence)
            
            self.logger.info(f"Multi-modal reasoning completed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in multi-modal reasoning: {e}")
            # Return fallback plan
            return self._generate_fallback_plan(language_command, safety_constraints)
    
    def _ground_language_command(self, 
                                command: str, 
                                visual_context: VisualContext,
                                spatial_context: SpatialContext) -> Dict[str, Any]:
        """Ground natural language command with visual and spatial context"""
        grounded_command = {
            'action': 'unknown',
            'objects': [],
            'spatial_references': [],
            'parameters': {},
            'confidence': 0.0
        }
        
        # Simple keyword-based grounding (in production, use VLM)
        command_lower = command.lower()
        
        # Extract action
        if 'move' in command_lower or 'go' in command_lower:
            grounded_command['action'] = 'move'
        elif 'pick' in command_lower or 'grasp' in command_lower:
            grounded_command['action'] = 'pick'
        elif 'place' in command_lower or 'put' in command_lower:
            grounded_command['action'] = 'place'
        elif 'look' in command_lower or 'observe' in command_lower:
            grounded_command['action'] = 'observe'
        
        # Extract objects from visual context
        for obj in visual_context.objects:
            if obj['name'].lower() in command_lower:
                grounded_command['objects'].append(obj)
        
        # Extract spatial references
        spatial_keywords = ['left', 'right', 'front', 'back', 'above', 'below', 'near', 'far']
        for keyword in spatial_keywords:
            if keyword in command_lower:
                grounded_command['spatial_references'].append(keyword)
        
        # Calculate confidence based on grounding quality
        confidence = 0.5  # Base confidence
        if grounded_command['action'] != 'unknown':
            confidence += 0.2
        if grounded_command['objects']:
            confidence += 0.2
        if grounded_command['spatial_references']:
            confidence += 0.1
        
        grounded_command['confidence'] = min(confidence, 1.0)
        
        return grounded_command
    
    def _generate_safe_plan(self, 
                           temporal_plan: TaskPlan,
                           causal_analysis: Any,
                           safety_constraints: SafetyConstraints) -> TaskPlan:
        """Generate a safety-aware plan"""
        safe_plan = TaskPlan()
        safe_plan.plan_id = f"safe_plan_{int(time.time())}"
        safe_plan.goal_description = temporal_plan.goal_description
        safe_plan.steps = []
        safe_plan.estimated_duration_seconds = 0
        safe_plan.required_capabilities = temporal_plan.required_capabilities
        safe_plan.safety_considerations = []
        
        # Add safety considerations
        if causal_analysis.risk_level == 'high':
            safe_plan.safety_considerations.append('High risk operation - proceed with caution')
        if causal_analysis.risk_level == 'medium':
            safe_plan.safety_considerations.append('Medium risk - safety monitoring required')
        
        # Add safety steps
        safety_step = TaskStep()
        safety_step.action_type = 'safety_check'
        safety_step.description = 'Verify safety conditions before execution'
        safety_step.estimated_duration = 1.0
        safety_step.preconditions = ['workspace_clear', 'safety_systems_online']
        safety_step.postconditions = ['safety_validated']
        safe_plan.steps.append(safety_step)
        
        # Add original steps with safety modifications
        for step in temporal_plan.steps:
            # Add safety preconditions
            step.preconditions.extend(['safety_validated'])
            safe_plan.steps.append(step)
            safe_plan.estimated_duration_seconds += step.estimated_duration
        
        # Calculate safety score
        safety_score = 1.0 - (causal_analysis.risk_score * 0.5)  # Reduce risk impact
        safe_plan.safety_score = max(safety_score, 0.1)  # Minimum safety score
        
        return safe_plan
    
    def _calculate_confidence(self, 
                            spatial_understanding: Any,
                            language_understanding: Dict[str, Any],
                            causal_analysis: Any) -> float:
        """Calculate confidence in reasoning result"""
        confidence_factors = []
        
        # Spatial confidence
        if hasattr(spatial_understanding, 'confidence'):
            confidence_factors.append(spatial_understanding.confidence)
        
        # Language grounding confidence
        confidence_factors.append(language_understanding.get('confidence', 0.5))
        
        # Causal analysis confidence
        if hasattr(causal_analysis, 'confidence'):
            confidence_factors.append(causal_analysis.confidence)
        
        # Average confidence
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5  # Default confidence
    
    def _generate_alternatives(self, 
                             primary_plan: TaskPlan,
                             spatial_understanding: Any,
                             safety_constraints: SafetyConstraints) -> List[TaskPlan]:
        """Generate alternative plans"""
        alternatives = []
        
        # Generate alternative with different approach
        alt_plan = TaskPlan()
        alt_plan.plan_id = f"alt_plan_{int(time.time())}"
        alt_plan.goal_description = primary_plan.goal_description
        alt_plan.steps = primary_plan.steps.copy()
        alt_plan.estimated_duration_seconds = primary_plan.estimated_duration_seconds * 1.2
        alt_plan.required_capabilities = primary_plan.required_capabilities
        alt_plan.safety_considerations = primary_plan.safety_considerations + ['Alternative approach']
        
        alternatives.append(alt_plan)
        
        return alternatives
    
    def _generate_fallback_plan(self, 
                               command: str,
                               safety_constraints: SafetyConstraints) -> ReasoningResult:
        """Generate a fallback plan when reasoning fails"""
        fallback_plan = TaskPlan()
        fallback_plan.plan_id = f"fallback_plan_{int(time.time())}"
        fallback_plan.goal_description = f"Fallback for: {command}"
        fallback_plan.steps = []
        fallback_plan.estimated_duration_seconds = 5.0
        fallback_plan.required_capabilities = ['basic_movement']
        fallback_plan.safety_considerations = ['Fallback mode - proceed with extreme caution']
        
        # Add safety stop step
        safety_step = TaskStep()
        safety_step.action_type = 'stop'
        safety_step.description = 'Emergency stop due to reasoning failure'
        safety_step.estimated_duration = 1.0
        safety_step.preconditions = []
        safety_step.postconditions = ['robot_stopped']
        fallback_plan.steps.append(safety_step)
        
        return ReasoningResult(
            plan=fallback_plan,
            confidence=0.1,
            safety_score=0.9,  # High safety for fallback
            reasoning_steps=['Fallback plan generated due to reasoning failure'],
            alternative_plans=[],
            execution_time=0.1
        )
    
    def _track_performance(self, execution_time: float, confidence: float):
        """Track reasoning performance metrics"""
        self.reasoning_times[time.time()] = execution_time
        self.confidence_scores[time.time()] = confidence
        
        # Keep only recent metrics (last 100)
        if len(self.reasoning_times) > 100:
            oldest_time = min(self.reasoning_times.keys())
            del self.reasoning_times[oldest_time]
            del self.confidence_scores[oldest_time]
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.reasoning_times:
            return {}
        
        times = list(self.reasoning_times.values())
        confidences = list(self.confidence_scores.values())
        
        return {
            'avg_reasoning_time': np.mean(times),
            'max_reasoning_time': np.max(times),
            'min_reasoning_time': np.min(times),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        } 