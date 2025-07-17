#!/usr/bin/env python3
"""
Reasoning Orchestrator

This module orchestrates the coordination between different reasoning components
and manages the overall reasoning pipeline for the Embodied Intelligence Platform.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import numpy as np

from eip_interfaces.msg import TaskPlan, TaskStep
from eip_interfaces.srv import SafetyVerificationRequest, SafetyVerificationResponse

from .multi_modal_reasoner import (
    MultiModalReasoner, VisualContext, SpatialContext, 
    SafetyConstraints, ReasoningResult
)
from .spatial_reasoner import SpatialReasoner
from .temporal_reasoner import TemporalReasoner
from .causal_reasoner import CausalReasoner


class ReasoningMode(Enum):
    """Reasoning modes for different scenarios"""
    FAST = "fast"           # Quick reasoning for real-time decisions
    BALANCED = "balanced"   # Balanced speed and accuracy
    THOROUGH = "thorough"   # Comprehensive reasoning for complex tasks
    SAFETY_CRITICAL = "safety_critical"  # Maximum safety focus


@dataclass
class ReasoningRequest:
    """Request for reasoning"""
    visual_context: VisualContext
    language_command: str
    spatial_context: SpatialContext
    safety_constraints: SafetyConstraints
    mode: ReasoningMode
    priority: int  # 1-10, higher is more important
    timeout: float  # seconds


@dataclass
class ReasoningResponse:
    """Response from reasoning orchestration"""
    result: ReasoningResult
    mode_used: ReasoningMode
    execution_time: float
    component_times: Dict[str, float]
    success: bool
    error_message: Optional[str] = None


class ReasoningOrchestrator:
    """
    Orchestrator that coordinates all reasoning components and manages the reasoning pipeline
    """
    
    def __init__(self):
        """Initialize the reasoning orchestrator"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize reasoning components
        self.multi_modal_reasoner = MultiModalReasoner()
        self.spatial_reasoner = SpatialReasoner()
        self.temporal_reasoner = TemporalReasoner()
        self.causal_reasoner = CausalReasoner()
        
        # Reasoning pipeline configuration
        self.reasoning_configs = self._initialize_reasoning_configs()
        
        # Performance tracking
        self.request_times = {}
        self.success_rates = {}
        self.component_performance = {}
        
        # Async processing infrastructure
        self.request_queue = queue.PriorityQueue()
        self.response_queue = queue.Queue()
        self.processing_thread = None
        self.processing_running = False
        
        # Start processing thread
        self._start_processing_thread()
        
        self.logger.info("Reasoning Orchestrator initialized successfully")
    
    def _initialize_reasoning_configs(self) -> Dict[ReasoningMode, Dict[str, Any]]:
        """Initialize reasoning configurations for different modes"""
        return {
            ReasoningMode.FAST: {
                'max_reasoning_time': 0.2,
                'spatial_detail_level': 'basic',
                'temporal_optimization': 'simple',
                'causal_analysis_depth': 'shallow',
                'safety_check_level': 'basic'
            },
            ReasoningMode.BALANCED: {
                'max_reasoning_time': 0.5,
                'spatial_detail_level': 'medium',
                'temporal_optimization': 'balanced',
                'causal_analysis_depth': 'medium',
                'safety_check_level': 'standard'
            },
            ReasoningMode.THOROUGH: {
                'max_reasoning_time': 2.0,
                'spatial_detail_level': 'detailed',
                'temporal_optimization': 'advanced',
                'causal_analysis_depth': 'deep',
                'safety_check_level': 'comprehensive'
            },
            ReasoningMode.SAFETY_CRITICAL: {
                'max_reasoning_time': 1.0,
                'spatial_detail_level': 'detailed',
                'temporal_optimization': 'balanced',
                'causal_analysis_depth': 'deep',
                'safety_check_level': 'maximum'
            }
        }
    
    def _start_processing_thread(self):
        """Start the async processing thread"""
        self.processing_running = True
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        self.logger.info("Reasoning processing thread started")
    
    def _processing_worker(self):
        """Worker thread for processing reasoning requests"""
        while self.processing_running:
            try:
                # Get request with timeout
                priority, timestamp, request = self.request_queue.get(timeout=1.0)
                
                # Process request
                response = self._process_reasoning_request(request)
                
                # Put response in queue
                self.response_queue.put((timestamp, response))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing worker: {e}")
    
    def reason_about_scene(self, 
                          visual_context: VisualContext,
                          language_command: str,
                          spatial_context: SpatialContext,
                          safety_constraints: SafetyConstraints,
                          mode: ReasoningMode = ReasoningMode.BALANCED,
                          priority: int = 5,
                          timeout: float = 5.0) -> ReasoningResponse:
        """
        Perform reasoning about a scene with specified mode and priority
        
        Args:
            visual_context: Visual understanding of the scene
            language_command: Natural language command
            spatial_context: Spatial context information
            safety_constraints: Safety constraints to consider
            mode: Reasoning mode to use
            priority: Request priority (1-10)
            timeout: Maximum time to wait for response
            
        Returns:
            ReasoningResponse with results and performance metrics
        """
        start_time = time.time()
        
        try:
            # Create reasoning request
            request = ReasoningRequest(
                visual_context=visual_context,
                language_command=language_command,
                spatial_context=spatial_context,
                safety_constraints=safety_constraints,
                mode=mode,
                priority=priority,
                timeout=timeout
            )
            
            # Add to processing queue
            timestamp = time.time()
            self.request_queue.put((priority, timestamp, request))
            
            # Wait for response
            response = self._wait_for_response(timestamp, timeout)
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_performance(mode, execution_time, response.success)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in reasoning orchestration: {e}")
            return self._generate_error_response(str(e), time.time() - start_time)
    
    def _process_reasoning_request(self, request: ReasoningRequest) -> ReasoningResponse:
        """Process a reasoning request with the specified mode"""
        start_time = time.time()
        component_times = {}
        
        try:
            # Get configuration for the mode
            config = self.reasoning_configs[request.mode]
            
            # Check if we have enough time
            if config['max_reasoning_time'] < 0.1:
                # Use fast fallback
                result = self._fast_reasoning_fallback(request)
                component_times['fast_fallback'] = time.time() - start_time
            else:
                # Use full reasoning pipeline
                result = self._full_reasoning_pipeline(request, config, component_times)
            
            execution_time = time.time() - start_time
            
            return ReasoningResponse(
                result=result,
                mode_used=request.mode,
                execution_time=execution_time,
                component_times=component_times,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error processing reasoning request: {e}")
            return ReasoningResponse(
                result=self._generate_fallback_result(request),
                mode_used=request.mode,
                execution_time=time.time() - start_time,
                component_times=component_times,
                success=False,
                error_message=str(e)
            )
    
    def _full_reasoning_pipeline(self, 
                               request: ReasoningRequest,
                               config: Dict[str, Any],
                               component_times: Dict[str, float]) -> ReasoningResult:
        """Execute full reasoning pipeline"""
        # 1. Multi-modal reasoning
        mm_start = time.time()
        result = self.multi_modal_reasoner.reason_about_scene(
            request.visual_context,
            request.language_command,
            request.spatial_context,
            request.safety_constraints
        )
        component_times['multi_modal'] = time.time() - mm_start
        
        # 2. Enhanced spatial reasoning (if needed)
        if config['spatial_detail_level'] == 'detailed':
            spatial_start = time.time()
            enhanced_spatial = self.spatial_reasoner.analyze_scene(
                request.visual_context,
                request.spatial_context
            )
            component_times['enhanced_spatial'] = time.time() - spatial_start
            
            # Update result with enhanced spatial understanding
            result.reasoning_steps.append(f"Enhanced spatial analysis: {enhanced_spatial.summary}")
        
        # 3. Enhanced temporal reasoning (if needed)
        if config['temporal_optimization'] == 'advanced':
            temporal_start = time.time()
            enhanced_temporal = self.temporal_reasoner.plan_sequence(
                {'action': 'unknown'},  # Simplified for this example
                enhanced_spatial if 'enhanced_spatial' in component_times else None
            )
            component_times['enhanced_temporal'] = time.time() - temporal_start
            
            # Update result with enhanced temporal planning
            result.reasoning_steps.append(f"Enhanced temporal planning: {len(enhanced_temporal.steps)} steps")
        
        # 4. Enhanced causal reasoning (if needed)
        if config['causal_analysis_depth'] == 'deep':
            causal_start = time.time()
            enhanced_causal = self.causal_reasoner.analyze_effects(
                result.plan,
                enhanced_spatial if 'enhanced_spatial' in component_times else None,
                request.safety_constraints
            )
            component_times['enhanced_causal'] = time.time() - causal_start
            
            # Update result with enhanced causal analysis
            result.reasoning_steps.append(f"Enhanced causal analysis: {enhanced_causal.risk_assessment.risk_level.value}")
        
        # 5. Safety validation (if needed)
        if config['safety_check_level'] == 'maximum':
            safety_start = time.time()
            safety_validated = self._validate_safety_maximum(result, request.safety_constraints)
            component_times['safety_validation'] = time.time() - safety_start
            
            if not safety_validated:
                result.safety_score *= 0.5  # Reduce safety score if validation fails
                result.reasoning_steps.append("Maximum safety validation failed")
        
        return result
    
    def _fast_reasoning_fallback(self, request: ReasoningRequest) -> ReasoningResult:
        """Fast reasoning fallback for time-critical situations"""
        # Use only basic multi-modal reasoning
        result = self.multi_modal_reasoner.reason_about_scene(
            request.visual_context,
            request.language_command,
            request.spatial_context,
            request.safety_constraints
        )
        
        # Add fast reasoning indicator
        result.reasoning_steps.append("Fast reasoning mode used")
        
        return result
    
    def _validate_safety_maximum(self, 
                               result: ReasoningResult,
                               safety_constraints: SafetyConstraints) -> bool:
        """Maximum safety validation"""
        # Check all safety constraints
        safety_checks = []
        
        # Check collision threshold
        if result.safety_score < safety_constraints.collision_threshold:
            safety_checks.append(False)
        else:
            safety_checks.append(True)
        
        # Check human proximity
        if result.safety_score < safety_constraints.human_proximity_threshold:
            safety_checks.append(False)
        else:
            safety_checks.append(True)
        
        # Check velocity limits
        velocity_check = True  # Simplified - would check actual velocity
        safety_checks.append(velocity_check)
        
        # Check workspace boundaries
        boundary_check = True  # Simplified - would check actual boundaries
        safety_checks.append(boundary_check)
        
        return all(safety_checks)
    
    def _wait_for_response(self, timestamp: float, timeout: float) -> ReasoningResponse:
        """Wait for response from processing thread"""
        start_wait = time.time()
        
        while time.time() - start_wait < timeout:
            try:
                # Check for response with short timeout
                response_timestamp, response = self.response_queue.get(timeout=0.1)
                
                if response_timestamp == timestamp:
                    return response
                else:
                    # Put back other responses
                    self.response_queue.put((response_timestamp, response))
                    
            except queue.Empty:
                continue
        
        # Timeout occurred
        raise TimeoutError("Reasoning request timed out")
    
    def _generate_fallback_result(self, request: ReasoningRequest) -> ReasoningResult:
        """Generate fallback result when reasoning fails"""
        from .multi_modal_reasoner import ReasoningResult
        
        fallback_plan = TaskPlan()
        fallback_plan.plan_id = f"fallback_orchestrator_{int(time.time())}"
        fallback_plan.goal_description = f"Fallback for: {request.language_command}"
        fallback_plan.steps = []
        fallback_plan.estimated_duration_seconds = 5.0
        fallback_plan.required_capabilities = ['basic_movement']
        fallback_plan.safety_considerations = ['Orchestrator fallback - proceed with extreme caution']
        
        # Add emergency stop step
        emergency_step = TaskStep()
        emergency_step.action_type = 'emergency_stop'
        emergency_step.description = 'Emergency stop due to orchestrator failure'
        emergency_step.estimated_duration = 1.0
        emergency_step.preconditions = []
        emergency_step.postconditions = ['robot_stopped']
        fallback_plan.steps.append(emergency_step)
        
        return ReasoningResult(
            plan=fallback_plan,
            confidence=0.1,
            safety_score=0.9,
            reasoning_steps=['Orchestrator fallback plan generated'],
            alternative_plans=[],
            execution_time=0.1
        )
    
    def _generate_error_response(self, error_message: str, execution_time: float) -> ReasoningResponse:
        """Generate error response"""
        return ReasoningResponse(
            result=self._generate_fallback_result(None),
            mode_used=ReasoningMode.FAST,
            execution_time=execution_time,
            component_times={},
            success=False,
            error_message=error_message
        )
    
    def _track_performance(self, mode: ReasoningMode, execution_time: float, success: bool):
        """Track performance metrics"""
        # Track request times
        if mode not in self.request_times:
            self.request_times[mode] = []
        self.request_times[mode].append(execution_time)
        
        # Track success rates
        if mode not in self.success_rates:
            self.success_rates[mode] = {'success': 0, 'total': 0}
        
        self.success_rates[mode]['total'] += 1
        if success:
            self.success_rates[mode]['success'] += 1
        
        # Keep only recent metrics (last 100)
        if len(self.request_times[mode]) > 100:
            self.request_times[mode] = self.request_times[mode][-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'modes': {},
            'components': {},
            'overall': {}
        }
        
        # Mode-specific stats
        for mode in ReasoningMode:
            if mode in self.request_times:
                times = np.array(self.request_times[mode])
                stats['modes'][mode.value] = {
                    'avg_time': np.mean(times),
                    'max_time': np.max(times),
                    'min_time': np.min(times),
                    'std_time': np.std(times),
                    'success_rate': self.success_rates[mode]['success'] / self.success_rates[mode]['total'] if self.success_rates[mode]['total'] > 0 else 0.0
                }
        
        # Component stats
        stats['components'] = {
            'multi_modal': self.multi_modal_reasoner.get_performance_stats(),
            'spatial': self.spatial_reasoner.get_performance_stats(),
            'temporal': self.temporal_reasoner.get_performance_stats(),
            'causal': self.causal_reasoner.get_performance_stats()
        }
        
        # Overall stats
        all_times = []
        for mode_times in self.request_times.values():
            all_times.extend(mode_times)
        
        if all_times:
            all_times_array = np.array(all_times)
            stats['overall'] = {
                'avg_time': np.mean(all_times_array),
                'max_time': np.max(all_times_array),
                'min_time': np.min(all_times_array),
                'std_time': np.std(all_times_array),
                'total_requests': len(all_times)
            }
        
        return stats
    
    def shutdown(self):
        """Shutdown the orchestrator"""
        self.processing_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        self.logger.info("Reasoning Orchestrator shutdown complete") 