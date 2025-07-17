#!/usr/bin/env python3
"""
Temporal Reasoner

This module implements temporal reasoning capabilities for planning sequences,
understanding time constraints, and optimizing task execution order.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from eip_interfaces.msg import TaskPlan, TaskStep

from .spatial_reasoner import SpatialUnderstanding


class TemporalRelation(Enum):
    """Types of temporal relationships"""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    MEETS = "meets"
    EQUALS = "equals"


@dataclass
class TemporalConstraint:
    """Temporal constraint for task planning"""
    step_id: str
    constraint_type: TemporalRelation
    related_step_id: str
    time_offset: float  # seconds
    priority: int  # 1-10, higher is more important


@dataclass
class TemporalPlan:
    """Temporal plan with sequence and timing"""
    steps: List[TaskStep]
    execution_order: List[int]
    time_estimates: Dict[str, float]
    dependencies: Dict[str, List[str]]
    constraints: List[TemporalConstraint]
    total_duration: float
    critical_path: List[str]


class TemporalReasoner:
    """
    Temporal reasoning engine for planning sequences and understanding time constraints
    """
    
    def __init__(self):
        """Initialize the temporal reasoner"""
        self.logger = logging.getLogger(__name__)
        
        # Temporal reasoning parameters
        self.max_planning_time = 5.0  # seconds
        self.min_step_duration = 0.1  # seconds
        self.max_step_duration = 60.0  # seconds
        
        # Performance tracking
        self.planning_times = []
        
        self.logger.info("Temporal Reasoner initialized successfully")
    
    def plan_sequence(self, 
                     language_understanding: Dict[str, Any],
                     spatial_understanding: SpatialUnderstanding) -> TaskPlan:
        """
        Plan temporal sequence for task execution
        
        Args:
            language_understanding: Grounded language command
            spatial_understanding: Spatial understanding of the scene
            
        Returns:
            TaskPlan with optimized sequence
        """
        start_time = time.time()
        
        try:
            # 1. Extract action requirements
            action_requirements = self._extract_action_requirements(
                language_understanding, spatial_understanding
            )
            
            # 2. Generate initial steps
            initial_steps = self._generate_initial_steps(action_requirements)
            
            # 3. Analyze dependencies
            dependencies = self._analyze_step_dependencies(initial_steps, spatial_understanding)
            
            # 4. Optimize execution order
            execution_order = self._optimize_execution_order(initial_steps, dependencies)
            
            # 5. Estimate timing
            time_estimates = self._estimate_step_timing(initial_steps, spatial_understanding)
            
            # 6. Create temporal plan
            temporal_plan = self._create_temporal_plan(
                initial_steps, execution_order, dependencies, time_estimates
            )
            
            # 7. Convert to TaskPlan
            task_plan = self._convert_to_task_plan(temporal_plan, language_understanding)
            
            execution_time = time.time() - start_time
            self.planning_times.append(execution_time)
            
            self.logger.info(f"Temporal planning completed in {execution_time:.3f}s")
            return task_plan
            
        except Exception as e:
            self.logger.error(f"Error in temporal reasoning: {e}")
            return self._generate_fallback_plan(language_understanding)
    
    def _extract_action_requirements(self, 
                                   language_understanding: Dict[str, Any],
                                   spatial_understanding: SpatialUnderstanding) -> Dict[str, Any]:
        """Extract requirements for action execution"""
        requirements = {
            'action_type': language_understanding.get('action', 'unknown'),
            'target_objects': language_understanding.get('objects', []),
            'spatial_references': language_understanding.get('spatial_references', []),
            'parameters': language_understanding.get('parameters', {}),
            'preconditions': [],
            'postconditions': []
        }
        
        # Add spatial constraints as preconditions
        for constraint in spatial_understanding.spatial_constraints:
            requirements['preconditions'].append(f"spatial_constraint_satisfied: {constraint}")
        
        # Add object availability as preconditions
        for obj in requirements['target_objects']:
            if isinstance(obj, dict) and 'name' in obj:
                requirements['preconditions'].append(f"object_available: {obj['name']}")
        
        # Add action-specific postconditions
        action_type = requirements['action_type']
        if action_type == 'move':
            requirements['postconditions'].append('robot_at_target_position')
        elif action_type == 'pick':
            requirements['postconditions'].append('object_grasped')
        elif action_type == 'place':
            requirements['postconditions'].append('object_placed')
        elif action_type == 'observe':
            requirements['postconditions'].append('scene_observed')
        
        return requirements
    
    def _generate_initial_steps(self, requirements: Dict[str, Any]) -> List[TaskStep]:
        """Generate initial task steps based on requirements"""
        steps = []
        step_id = 0
        
        # Add safety check step
        safety_step = TaskStep()
        safety_step.action_type = 'safety_check'
        safety_step.description = 'Verify safety conditions before execution'
        safety_step.estimated_duration = 1.0
        safety_step.preconditions = requirements['preconditions']
        safety_step.postconditions = ['safety_validated']
        steps.append(safety_step)
        step_id += 1
        
        # Add main action step
        main_step = TaskStep()
        main_step.action_type = requirements['action_type']
        main_step.description = f"Execute {requirements['action_type']} action"
        
        # Estimate duration based on action type
        duration = self._estimate_action_duration(requirements['action_type'])
        main_step.estimated_duration = duration
        
        main_step.preconditions = ['safety_validated']
        main_step.postconditions = requirements['postconditions']
        
        # Add parameters
        if requirements['parameters']:
            main_step.parameters = [f"{k}={v}" for k, v in requirements['parameters'].items()]
        
        steps.append(main_step)
        step_id += 1
        
        # Add verification step
        verification_step = TaskStep()
        verification_step.action_type = 'verify_completion'
        verification_step.description = 'Verify task completion'
        verification_step.estimated_duration = 0.5
        verification_step.preconditions = requirements['postconditions']
        verification_step.postconditions = ['task_completed']
        steps.append(verification_step)
        
        return steps
    
    def _estimate_action_duration(self, action_type: str) -> float:
        """Estimate duration for different action types"""
        duration_estimates = {
            'move': 3.0,
            'pick': 2.0,
            'place': 2.0,
            'observe': 1.0,
            'grasp': 1.5,
            'release': 0.5,
            'rotate': 1.0,
            'unknown': 2.0
        }
        
        return duration_estimates.get(action_type, 2.0)
    
    def _analyze_step_dependencies(self, 
                                 steps: List[TaskStep],
                                 spatial_understanding: SpatialUnderstanding) -> Dict[str, List[str]]:
        """Analyze dependencies between steps"""
        dependencies = {}
        
        for i, step in enumerate(steps):
            step_id = f"step_{i}"
            dependencies[step_id] = []
            
            # Check preconditions against previous steps' postconditions
            for j in range(i):
                prev_step = steps[j]
                prev_step_id = f"step_{j}"
                
                # Check if current step depends on previous step
                if self._steps_have_dependency(step, prev_step):
                    dependencies[step_id].append(prev_step_id)
        
        return dependencies
    
    def _steps_have_dependency(self, current_step: TaskStep, previous_step: TaskStep) -> bool:
        """Check if current step depends on previous step"""
        # Check if current step's preconditions are satisfied by previous step's postconditions
        for precondition in current_step.preconditions:
            if precondition in previous_step.postconditions:
                return True
        
        # Check for resource dependencies
        if current_step.action_type == 'pick' and previous_step.action_type == 'move':
            return True  # Pick depends on move to object
        
        if current_step.action_type == 'place' and previous_step.action_type == 'pick':
            return True  # Place depends on pick
        
        return False
    
    def _optimize_execution_order(self, 
                                steps: List[TaskStep],
                                dependencies: Dict[str, List[str]]) -> List[int]:
        """Optimize execution order based on dependencies"""
        # Simple topological sort
        execution_order = []
        visited = set()
        temp_visited = set()
        
        def visit(step_id: str):
            if step_id in temp_visited:
                raise ValueError("Circular dependency detected")
            if step_id in visited:
                return
            
            temp_visited.add(step_id)
            
            # Visit dependencies first
            for dep_id in dependencies.get(step_id, []):
                visit(dep_id)
            
            temp_visited.remove(step_id)
            visited.add(step_id)
            execution_order.append(int(step_id.split('_')[1]))
        
        # Visit all steps
        for step_id in dependencies.keys():
            if step_id not in visited:
                visit(step_id)
        
        return execution_order
    
    def _estimate_step_timing(self, 
                            steps: List[TaskStep],
                            spatial_understanding: SpatialUnderstanding) -> Dict[str, float]:
        """Estimate timing for each step"""
        time_estimates = {}
        
        for i, step in enumerate(steps):
            step_id = f"step_{i}"
            base_duration = step.estimated_duration
            
            # Adjust based on spatial complexity
            spatial_factor = self._calculate_spatial_complexity_factor(spatial_understanding)
            adjusted_duration = base_duration * spatial_factor
            
            # Ensure reasonable bounds
            adjusted_duration = max(adjusted_duration, self.min_step_duration)
            adjusted_duration = min(adjusted_duration, self.max_step_duration)
            
            time_estimates[step_id] = adjusted_duration
        
        return time_estimates
    
    def _calculate_spatial_complexity_factor(self, spatial_understanding: SpatialUnderstanding) -> float:
        """Calculate factor based on spatial complexity"""
        # Base factor
        factor = 1.0
        
        # Adjust based on number of objects
        num_objects = len(spatial_understanding.object_relationships)
        if num_objects > 5:
            factor *= 1.2
        elif num_objects > 10:
            factor *= 1.5
        
        # Adjust based on spatial constraints
        num_constraints = len(spatial_understanding.spatial_constraints)
        factor *= (1.0 + num_constraints * 0.1)
        
        # Adjust based on confidence
        factor *= (2.0 - spatial_understanding.confidence)  # Lower confidence = longer time
        
        return factor
    
    def _create_temporal_plan(self, 
                            steps: List[TaskStep],
                            execution_order: List[int],
                            dependencies: Dict[str, List[str]],
                            time_estimates: Dict[str, float]) -> TemporalPlan:
        """Create temporal plan with all timing information"""
        # Calculate total duration
        total_duration = sum(time_estimates.values())
        
        # Identify critical path
        critical_path = self._identify_critical_path(dependencies, time_estimates)
        
        # Create temporal constraints
        constraints = self._create_temporal_constraints(dependencies, time_estimates)
        
        return TemporalPlan(
            steps=steps,
            execution_order=execution_order,
            time_estimates=time_estimates,
            dependencies=dependencies,
            constraints=constraints,
            total_duration=total_duration,
            critical_path=critical_path
        )
    
    def _identify_critical_path(self, 
                              dependencies: Dict[str, List[str]],
                              time_estimates: Dict[str, float]) -> List[str]:
        """Identify critical path in the task"""
        # Simple critical path analysis
        step_durations = {}
        for step_id, duration in time_estimates.items():
            step_durations[step_id] = duration
        
        # Find longest path
        max_duration = 0
        critical_path = []
        
        for step_id in dependencies.keys():
            path_duration = self._calculate_path_duration(step_id, dependencies, step_durations)
            if path_duration > max_duration:
                max_duration = path_duration
                critical_path = self._get_path_steps(step_id, dependencies)
        
        return critical_path
    
    def _calculate_path_duration(self, 
                               step_id: str,
                               dependencies: Dict[str, List[str]],
                               step_durations: Dict[str, float]) -> float:
        """Calculate duration of path ending at step_id"""
        duration = step_durations.get(step_id, 0)
        
        # Add duration of dependencies
        for dep_id in dependencies.get(step_id, []):
            duration += self._calculate_path_duration(dep_id, dependencies, step_durations)
        
        return duration
    
    def _get_path_steps(self, 
                       step_id: str,
                       dependencies: Dict[str, List[str]]) -> List[str]:
        """Get all steps in path ending at step_id"""
        path = [step_id]
        
        for dep_id in dependencies.get(step_id, []):
            path.extend(self._get_path_steps(dep_id, dependencies))
        
        return path
    
    def _create_temporal_constraints(self, 
                                   dependencies: Dict[str, List[str]],
                                   time_estimates: Dict[str, float]) -> List[TemporalConstraint]:
        """Create temporal constraints between steps"""
        constraints = []
        
        for step_id, deps in dependencies.items():
            for dep_id in deps:
                constraint = TemporalConstraint(
                    step_id=step_id,
                    constraint_type=TemporalRelation.AFTER,
                    related_step_id=dep_id,
                    time_offset=time_estimates.get(dep_id, 0.0),
                    priority=5
                )
                constraints.append(constraint)
        
        return constraints
    
    def _convert_to_task_plan(self, 
                            temporal_plan: TemporalPlan,
                            language_understanding: Dict[str, Any]) -> TaskPlan:
        """Convert temporal plan to TaskPlan message"""
        task_plan = TaskPlan()
        task_plan.plan_id = f"temporal_plan_{int(time.time())}"
        task_plan.goal_description = f"Execute {language_understanding.get('action', 'unknown')} action"
        task_plan.steps = temporal_plan.steps
        task_plan.estimated_duration_seconds = temporal_plan.total_duration
        task_plan.required_capabilities = self._extract_required_capabilities(temporal_plan.steps)
        task_plan.safety_considerations = ['Temporal constraints may affect safety']
        
        return task_plan
    
    def _extract_required_capabilities(self, steps: List[TaskStep]) -> List[str]:
        """Extract required capabilities from steps"""
        capabilities = set()
        
        for step in steps:
            action_type = step.action_type
            if action_type == 'move':
                capabilities.add('navigation')
            elif action_type in ['pick', 'place', 'grasp', 'release']:
                capabilities.add('manipulation')
            elif action_type == 'observe':
                capabilities.add('perception')
            elif action_type == 'safety_check':
                capabilities.add('safety_monitoring')
        
        return list(capabilities)
    
    def _generate_fallback_plan(self, language_understanding: Dict[str, Any]) -> TaskPlan:
        """Generate fallback plan when temporal reasoning fails"""
        fallback_plan = TaskPlan()
        fallback_plan.plan_id = f"fallback_temporal_{int(time.time())}"
        fallback_plan.goal_description = f"Fallback for {language_understanding.get('action', 'unknown')}"
        fallback_plan.steps = []
        fallback_plan.estimated_duration_seconds = 5.0
        fallback_plan.required_capabilities = ['basic_movement']
        fallback_plan.safety_considerations = ['Temporal reasoning failed - proceed with caution']
        
        # Add simple fallback step
        fallback_step = TaskStep()
        fallback_step.action_type = 'stop'
        fallback_step.description = 'Stop due to temporal planning failure'
        fallback_step.estimated_duration = 1.0
        fallback_step.preconditions = []
        fallback_step.postconditions = ['robot_stopped']
        fallback_plan.steps.append(fallback_step)
        
        return fallback_plan
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.planning_times:
            return {}
        
        times = np.array(self.planning_times)
        return {
            'avg_planning_time': np.mean(times),
            'max_planning_time': np.max(times),
            'min_planning_time': np.min(times),
            'std_planning_time': np.std(times)
        } 