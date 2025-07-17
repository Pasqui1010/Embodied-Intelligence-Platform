#!/usr/bin/env python3
"""
Causal Reasoner

This module implements causal reasoning capabilities for understanding cause-effect
relationships, analyzing action consequences, and assessing risks in robotic environments.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from eip_interfaces.msg import TaskPlan, TaskStep

from .spatial_reasoner import SpatialUnderstanding
from .multi_modal_reasoner import SafetyConstraints


class RiskLevel(Enum):
    """Risk levels for causal analysis"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EffectType(Enum):
    """Types of causal effects"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass
class CausalEffect:
    """Causal effect of an action"""
    action: str
    effect_type: EffectType
    description: str
    probability: float
    severity: float
    affected_objects: List[str]
    time_delay: float  # seconds


@dataclass
class RiskAssessment:
    """Risk assessment result"""
    risk_level: RiskLevel
    risk_score: float  # 0.0 to 1.0
    risk_factors: List[str]
    mitigation_strategies: List[str]
    confidence: float


@dataclass
class CausalAnalysis:
    """Result of causal reasoning"""
    effects: List[CausalEffect]
    risk_assessment: RiskAssessment
    causal_chains: List[List[str]]
    safety_implications: List[str]
    confidence: float


class CausalReasoner:
    """
    Causal reasoning engine for understanding cause-effect relationships and risks
    """
    
    def __init__(self):
        """Initialize the causal reasoner"""
        self.logger = logging.getLogger(__name__)
        
        # Causal reasoning parameters
        self.max_chain_length = 5
        self.min_effect_probability = 0.1
        self.max_analysis_time = 3.0  # seconds
        
        # Risk thresholds
        self.risk_thresholds = {
            RiskLevel.LOW: 0.3,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 0.9
        }
        
        # Causal knowledge base
        self.causal_knowledge = self._initialize_causal_knowledge()
        
        # Performance tracking
        self.analysis_times = []
        
        self.logger.info("Causal Reasoner initialized successfully")
    
    def analyze_effects(self, 
                       task_plan: TaskPlan,
                       spatial_understanding: SpatialUnderstanding,
                       safety_constraints: SafetyConstraints) -> CausalAnalysis:
        """
        Analyze causal effects of a task plan
        
        Args:
            task_plan: Task plan to analyze
            spatial_understanding: Spatial understanding of the scene
            safety_constraints: Safety constraints to consider
            
        Returns:
            CausalAnalysis with effects and risk assessment
        """
        start_time = time.time()
        
        try:
            # 1. Extract causal factors
            causal_factors = self._extract_causal_factors(
                task_plan, spatial_understanding
            )
            
            # 2. Predict effects
            effects = self._predict_effects(causal_factors, spatial_understanding)
            
            # 3. Assess risks
            risk_assessment = self._assess_risks(
                effects, safety_constraints, spatial_understanding
            )
            
            # 4. Identify causal chains
            causal_chains = self._identify_causal_chains(effects)
            
            # 5. Analyze safety implications
            safety_implications = self._analyze_safety_implications(
                effects, risk_assessment, safety_constraints
            )
            
            # 6. Calculate confidence
            confidence = self._calculate_causal_confidence(
                effects, spatial_understanding
            )
            
            execution_time = time.time() - start_time
            self.analysis_times.append(execution_time)
            
            result = CausalAnalysis(
                effects=effects,
                risk_assessment=risk_assessment,
                causal_chains=causal_chains,
                safety_implications=safety_implications,
                confidence=confidence
            )
            
            self.logger.info(f"Causal analysis completed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in causal reasoning: {e}")
            return self._generate_fallback_analysis()
    
    def _initialize_causal_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """Initialize causal knowledge base"""
        return {
            'move': {
                'effects': [
                    {'type': EffectType.NEUTRAL, 'description': 'Robot position changes', 'probability': 1.0, 'severity': 0.1},
                    {'type': EffectType.NEGATIVE, 'description': 'Risk of collision', 'probability': 0.3, 'severity': 0.8},
                    {'type': EffectType.NEGATIVE, 'description': 'Energy consumption', 'probability': 1.0, 'severity': 0.2}
                ],
                'risk_factors': ['obstacles_present', 'high_velocity', 'confined_space'],
                'mitigation': ['path_planning', 'velocity_control', 'obstacle_avoidance']
            },
            'pick': {
                'effects': [
                    {'type': EffectType.POSITIVE, 'description': 'Object grasped', 'probability': 0.9, 'severity': 0.3},
                    {'type': EffectType.NEGATIVE, 'description': 'Risk of dropping', 'probability': 0.2, 'severity': 0.6},
                    {'type': EffectType.NEGATIVE, 'description': 'Object damage', 'probability': 0.1, 'severity': 0.7}
                ],
                'risk_factors': ['fragile_object', 'poor_grasp_pose', 'object_movement'],
                'mitigation': ['grasp_planning', 'force_control', 'stability_check']
            },
            'place': {
                'effects': [
                    {'type': EffectType.POSITIVE, 'description': 'Object placed', 'probability': 0.9, 'severity': 0.3},
                    {'type': EffectType.NEGATIVE, 'description': 'Risk of knocking over', 'probability': 0.3, 'severity': 0.5},
                    {'type': EffectType.NEGATIVE, 'description': 'Unstable placement', 'probability': 0.2, 'severity': 0.6}
                ],
                'risk_factors': ['unstable_surface', 'crowded_area', 'poor_placement_pose'],
                'mitigation': ['surface_analysis', 'placement_planning', 'stability_verification']
            },
            'observe': {
                'effects': [
                    {'type': EffectType.POSITIVE, 'description': 'Scene understanding improved', 'probability': 1.0, 'severity': 0.2},
                    {'type': EffectType.NEUTRAL, 'description': 'Time consumption', 'probability': 1.0, 'severity': 0.1}
                ],
                'risk_factors': ['poor_lighting', 'occlusions', 'moving_objects'],
                'mitigation': ['multiple_viewpoints', 'adaptive_lighting', 'motion_compensation']
            }
        }
    
    def _extract_causal_factors(self, 
                              task_plan: TaskPlan,
                              spatial_understanding: SpatialUnderstanding) -> Dict[str, Any]:
        """Extract causal factors from task plan and spatial context"""
        factors = {
            'actions': [],
            'objects': [],
            'spatial_constraints': [],
            'environmental_factors': []
        }
        
        # Extract actions from task plan
        for step in task_plan.steps:
            factors['actions'].append({
                'type': step.action_type,
                'description': step.description,
                'duration': step.estimated_duration,
                'parameters': step.parameters
            })
        
        # Extract objects from spatial understanding
        factors['objects'] = list(spatial_understanding.object_relationships.keys())
        
        # Extract spatial constraints
        factors['spatial_constraints'] = spatial_understanding.spatial_constraints
        
        # Extract environmental factors
        factors['environmental_factors'] = self._extract_environmental_factors(spatial_understanding)
        
        return factors
    
    def _extract_environmental_factors(self, spatial_understanding: SpatialUnderstanding) -> List[str]:
        """Extract environmental factors from spatial understanding"""
        factors = []
        
        # Analyze object relationships for environmental factors
        for obj_name, relationships in spatial_understanding.object_relationships.items():
            for rel_obj, relation, confidence in relationships:
                if relation.value in ['near', 'above', 'below'] and confidence > 0.7:
                    factors.append(f"{obj_name}_{relation.value}_{rel_obj}")
        
        # Add spatial constraint factors
        for constraint in spatial_understanding.spatial_constraints:
            factors.append(f"constraint_{constraint.lower().replace(' ', '_')}")
        
        return factors
    
    def _predict_effects(self, 
                        causal_factors: Dict[str, Any],
                        spatial_understanding: SpatialUnderstanding) -> List[CausalEffect]:
        """Predict causal effects based on factors"""
        effects = []
        
        # Predict effects for each action
        for action_info in causal_factors['actions']:
            action_type = action_info['type']
            
            if action_type in self.causal_knowledge:
                knowledge = self.causal_knowledge[action_type]
                
                for effect_info in knowledge['effects']:
                    # Adjust probability based on spatial context
                    adjusted_probability = self._adjust_effect_probability(
                        effect_info['probability'], action_type, spatial_understanding
                    )
                    
                    # Adjust severity based on context
                    adjusted_severity = self._adjust_effect_severity(
                        effect_info['severity'], action_type, spatial_understanding
                    )
                    
                    # Only include effects above threshold
                    if adjusted_probability >= self.min_effect_probability:
                        effect = CausalEffect(
                            action=action_type,
                            effect_type=effect_info['type'],
                            description=effect_info['description'],
                            probability=adjusted_probability,
                            severity=adjusted_severity,
                            affected_objects=self._identify_affected_objects(
                                action_type, causal_factors['objects']
                            ),
                            time_delay=self._estimate_effect_delay(action_type, effect_info['description'])
                        )
                        effects.append(effect)
        
        return effects
    
    def _adjust_effect_probability(self, 
                                 base_probability: float,
                                 action_type: str,
                                 spatial_understanding: SpatialUnderstanding) -> float:
        """Adjust effect probability based on spatial context"""
        adjusted_prob = base_probability
        
        # Adjust based on spatial constraints
        num_constraints = len(spatial_understanding.spatial_constraints)
        if num_constraints > 3:
            adjusted_prob *= 1.2  # More constraints = higher risk
        elif num_constraints > 5:
            adjusted_prob *= 1.5
        
        # Adjust based on object relationships
        num_objects = len(spatial_understanding.object_relationships)
        if num_objects > 5:
            adjusted_prob *= 1.1  # More objects = higher complexity
        
        # Adjust based on spatial confidence
        adjusted_prob *= (2.0 - spatial_understanding.confidence)  # Lower confidence = higher uncertainty
        
        return min(adjusted_prob, 1.0)
    
    def _adjust_effect_severity(self, 
                              base_severity: float,
                              action_type: str,
                              spatial_understanding: SpatialUnderstanding) -> float:
        """Adjust effect severity based on spatial context"""
        adjusted_severity = base_severity
        
        # Adjust based on spatial constraints
        for constraint in spatial_understanding.spatial_constraints:
            if 'boundary' in constraint.lower():
                adjusted_severity *= 1.3  # Boundary violations are more severe
            elif 'proximity' in constraint.lower():
                adjusted_severity *= 1.2  # Proximity issues are more severe
        
        # Adjust based on object relationships
        for obj_name, relationships in spatial_understanding.object_relationships.items():
            for rel_obj, relation, confidence in relationships:
                if relation.value == 'near' and confidence > 0.8:
                    adjusted_severity *= 1.1  # Close objects increase severity
        
        return min(adjusted_severity, 1.0)
    
    def _identify_affected_objects(self, 
                                 action_type: str,
                                 objects: List[str]) -> List[str]:
        """Identify objects that might be affected by an action"""
        affected = []
        
        if action_type in ['pick', 'place']:
            # These actions directly affect objects
            affected.extend(objects)
        elif action_type == 'move':
            # Movement might affect nearby objects
            affected.extend(objects)  # Simplified - in practice would check proximity
        
        return affected
    
    def _estimate_effect_delay(self, action_type: str, effect_description: str) -> float:
        """Estimate time delay for effect to occur"""
        if 'immediate' in effect_description.lower():
            return 0.0
        elif 'delayed' in effect_description.lower():
            return 2.0
        elif action_type in ['pick', 'place']:
            return 0.5
        elif action_type == 'move':
            return 0.1
        else:
            return 1.0
    
    def _assess_risks(self, 
                     effects: List[CausalEffect],
                     safety_constraints: SafetyConstraints,
                     spatial_understanding: SpatialUnderstanding) -> RiskAssessment:
        """Assess overall risks based on effects"""
        risk_factors = []
        risk_score = 0.0
        mitigation_strategies = []
        
        # Calculate risk score from effects
        for effect in effects:
            if effect.effect_type == EffectType.NEGATIVE:
                effect_risk = effect.probability * effect.severity
                risk_score = max(risk_score, effect_risk)
                
                if effect_risk > 0.5:
                    risk_factors.append(f"High risk effect: {effect.description}")
        
        # Add risk factors from safety constraints
        if safety_constraints.collision_threshold < 0.5:
            risk_factors.append("Low collision threshold")
            risk_score = max(risk_score, 0.7)
        
        if safety_constraints.human_proximity_threshold < 0.5:
            risk_factors.append("Low human proximity threshold")
            risk_score = max(risk_score, 0.8)
        
        # Add spatial constraint risks
        for constraint in spatial_understanding.spatial_constraints:
            if 'boundary' in constraint.lower():
                risk_factors.append(f"Spatial constraint: {constraint}")
                risk_score = max(risk_score, 0.6)
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(
            effects, risk_factors, safety_constraints
        )
        
        # Calculate confidence
        confidence = self._calculate_risk_confidence(effects, spatial_understanding)
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            risk_factors=risk_factors,
            mitigation_strategies=mitigation_strategies,
            confidence=confidence
        )
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on risk score"""
        if risk_score >= self.risk_thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif risk_score >= self.risk_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_mitigation_strategies(self, 
                                      effects: List[CausalEffect],
                                      risk_factors: List[str],
                                      safety_constraints: SafetyConstraints) -> List[str]:
        """Generate mitigation strategies for identified risks"""
        strategies = []
        
        # Add strategies for high-risk effects
        for effect in effects:
            if effect.effect_type == EffectType.NEGATIVE and effect.probability * effect.severity > 0.5:
                if 'collision' in effect.description.lower():
                    strategies.append("Implement enhanced obstacle avoidance")
                elif 'dropping' in effect.description.lower():
                    strategies.append("Use robust grasp planning and force control")
                elif 'damage' in effect.description.lower():
                    strategies.append("Implement gentle manipulation protocols")
        
        # Add general safety strategies
        strategies.extend([
            "Increase safety monitoring frequency",
            "Reduce execution velocity",
            "Add intermediate safety checks"
        ])
        
        return strategies
    
    def _calculate_risk_confidence(self, 
                                 effects: List[CausalEffect],
                                 spatial_understanding: SpatialUnderstanding) -> float:
        """Calculate confidence in risk assessment"""
        confidence_factors = []
        
        # Effect confidence
        if effects:
            avg_effect_confidence = np.mean([
                effect.probability for effect in effects
            ])
            confidence_factors.append(avg_effect_confidence)
        
        # Spatial confidence
        confidence_factors.append(spatial_understanding.confidence)
        
        # Number of effects (more effects = more comprehensive analysis)
        if len(effects) > 3:
            confidence_factors.append(0.8)
        elif len(effects) > 1:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _identify_causal_chains(self, effects: List[CausalEffect]) -> List[List[str]]:
        """Identify causal chains from effects"""
        chains = []
        
        # Simple causal chain identification
        for effect in effects:
            if effect.effect_type == EffectType.NEGATIVE:
                chain = [effect.action, effect.description]
                chains.append(chain)
        
        return chains
    
    def _analyze_safety_implications(self, 
                                   effects: List[CausalEffect],
                                   risk_assessment: RiskAssessment,
                                   safety_constraints: SafetyConstraints) -> List[str]:
        """Analyze safety implications of effects"""
        implications = []
        
        # High-risk implications
        if risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            implications.append("High risk operation requires enhanced safety monitoring")
            implications.append("Consider emergency stop procedures")
        
        # Effect-specific implications
        for effect in effects:
            if effect.effect_type == EffectType.NEGATIVE and effect.severity > 0.7:
                implications.append(f"High severity effect: {effect.description}")
        
        # Constraint implications
        if safety_constraints.collision_threshold < 0.5:
            implications.append("Low collision threshold requires careful navigation")
        
        if safety_constraints.human_proximity_threshold < 0.5:
            implications.append("Low human proximity threshold requires human monitoring")
        
        return implications
    
    def _calculate_causal_confidence(self, 
                                   effects: List[CausalEffect],
                                   spatial_understanding: SpatialUnderstanding) -> float:
        """Calculate confidence in causal analysis"""
        confidence_factors = []
        
        # Effect confidence
        if effects:
            avg_effect_confidence = np.mean([
                effect.probability for effect in effects
            ])
            confidence_factors.append(avg_effect_confidence)
        
        # Spatial confidence
        confidence_factors.append(spatial_understanding.confidence)
        
        # Analysis completeness
        if len(effects) > 2:
            confidence_factors.append(0.8)
        elif len(effects) > 0:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _generate_fallback_analysis(self) -> CausalAnalysis:
        """Generate fallback analysis when causal reasoning fails"""
        fallback_effect = CausalEffect(
            action='unknown',
            effect_type=EffectType.UNKNOWN,
            description='Causal analysis failed - proceed with caution',
            probability=0.5,
            severity=0.5,
            affected_objects=[],
            time_delay=0.0
        )
        
        fallback_risk = RiskAssessment(
            risk_level=RiskLevel.MEDIUM,
            risk_score=0.5,
            risk_factors=['Causal reasoning failure'],
            mitigation_strategies=['Enhanced safety monitoring', 'Reduced execution speed'],
            confidence=0.1
        )
        
        return CausalAnalysis(
            effects=[fallback_effect],
            risk_assessment=fallback_risk,
            causal_chains=[],
            safety_implications=['Causal analysis failed - proceed with extreme caution'],
            confidence=0.1
        )
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.analysis_times:
            return {}
        
        times = np.array(self.analysis_times)
        return {
            'avg_analysis_time': np.mean(times),
            'max_analysis_time': np.max(times),
            'min_analysis_time': np.min(times),
            'std_analysis_time': np.std(times)
        } 