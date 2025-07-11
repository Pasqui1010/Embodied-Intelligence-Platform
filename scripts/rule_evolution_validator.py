#!/usr/bin/env python3
"""
Rule Evolution Validation Script

Implements the Rule Evolution Validation Prompt for adaptive safety systems.
"""

import json
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import numpy as np

@dataclass
class RuleEvolutionData:
    rule_id: str
    original_condition: str
    current_condition: str
    original_threshold: float
    current_threshold: float
    adaptation_count: int
    success_rate_trend: List[float]
    evolution_triggers: List[str]
    last_adaptation_time: float

@dataclass
class RuleEvolutionReport:
    rule_id: str
    evolution_validity: int
    status: str
    adaptation_analysis: Dict[str, Any]
    safety_impact: Dict[str, Any]
    stability_metrics: Dict[str, Any]
    evolution_justification: Dict[str, Any]
    recommendations: List[str]
    risk_level: str

class RuleEvolutionValidator:
    def __init__(self):
        # Thresholds from the prompt
        self.evolution_validity_min = 70
        self.adaptation_quality_min = 0.7
        self.stability_threshold = 0.8

    def analyze_adaptation_quality(self, original_threshold: float, current_threshold: float, 
                                 adaptation_count: int, evolution_triggers: List[str]) -> Dict[str, Any]:
        """Analyze the quality of adaptations made"""
        # Calculate threshold change magnitude
        threshold_change = abs(current_threshold - original_threshold)
        threshold_change_ratio = threshold_change / max(original_threshold, 0.1)
        
        # Assess adaptation frequency
        if adaptation_count <= 2:
            adaptation_frequency = "low"
        elif adaptation_count <= 5:
            adaptation_frequency = "moderate"
        else:
            adaptation_frequency = "high"
        
        # Evaluate trigger quality
        trigger_quality = self._assess_trigger_quality(evolution_triggers)
        
        # Calculate quality score
        quality_score = min(1.0, (0.4 * (1 - threshold_change_ratio) + 
                                  0.3 * trigger_quality + 
                                  0.3 * (1 - min(adaptation_count / 10, 1))))
        
        # Determine justification
        if trigger_quality > 0.8:
            justification = "Adaptations based on consistent near-miss patterns"
        elif trigger_quality > 0.6:
            justification = "Adaptations based on performance trends"
        else:
            justification = "Adaptations may be reactive to noise"
        
        # Risk assessment
        if quality_score > 0.8:
            risk_assessment = "low"
        elif quality_score > 0.6:
            risk_assessment = "medium"
        else:
            risk_assessment = "high"
        
        return {
            "quality_score": round(quality_score, 2),
            "justification": justification,
            "risk_assessment": risk_assessment,
            "threshold_change_ratio": round(threshold_change_ratio, 3),
            "adaptation_frequency": adaptation_frequency
        }

    def _assess_trigger_quality(self, evolution_triggers: List[str]) -> float:
        """Assess the quality of evolution triggers"""
        if not evolution_triggers:
            return 0.0
        
        # Define high-quality triggers
        high_quality_triggers = [
            "near_miss_incident", "safety_violation", "performance_degradation",
            "statistical_anomaly", "expert_review", "regulatory_change"
        ]
        
        # Define low-quality triggers
        low_quality_triggers = [
            "noise", "temporary_fluctuation", "single_incident", "unverified_data"
        ]
        
        quality_score = 0.0
        for trigger in evolution_triggers:
            if trigger in high_quality_triggers:
                quality_score += 0.2
            elif trigger in low_quality_triggers:
                quality_score -= 0.1
            else:
                quality_score += 0.1
        
        return max(0.0, min(1.0, quality_score))

    def assess_safety_impact(self, success_rate_trend: List[float], 
                           original_threshold: float, current_threshold: float) -> Dict[str, Any]:
        """Assess the impact of rule evolution on safety"""
        if len(success_rate_trend) < 2:
            return {
                "improvement_score": 0.0,
                "safety_level_change": "unknown",
                "confidence_impact": "unknown"
            }
        
        # Calculate improvement
        recent_success = np.mean(success_rate_trend[-3:]) if len(success_rate_trend) >= 3 else success_rate_trend[-1]
        earlier_success = np.mean(success_rate_trend[:3]) if len(success_rate_trend) >= 3 else success_rate_trend[0]
        
        improvement_score = max(0, recent_success - earlier_success)
        
        # Determine safety level change
        if improvement_score > 0.05:
            safety_level_change = "improved"
        elif improvement_score > -0.05:
            safety_level_change = "maintained"
        else:
            safety_level_change = "degraded"
        
        # Assess confidence impact
        threshold_stability = 1 - abs(current_threshold - original_threshold) / max(original_threshold, 0.1)
        if threshold_stability > 0.8 and improvement_score > 0:
            confidence_impact = "positive"
        elif threshold_stability > 0.6:
            confidence_impact = "neutral"
        else:
            confidence_impact = "negative"
        
        return {
            "improvement_score": round(improvement_score, 3),
            "safety_level_change": safety_level_change,
            "confidence_impact": confidence_impact,
            "recent_success_rate": round(recent_success, 3),
            "earlier_success_rate": round(earlier_success, 3)
        }

    def assess_stability(self, adaptation_count: int, success_rate_trend: List[float],
                        last_adaptation_time: float) -> Dict[str, Any]:
        """Assess rule stability"""
        current_time = time.time()
        time_since_adaptation = current_time - last_adaptation_time
        
        # Adaptation frequency
        if adaptation_count == 0:
            adaptation_frequency = "none"
        elif time_since_adaptation > 86400 * 30:  # 30 days
            adaptation_frequency = "decreasing"
        elif time_since_adaptation > 86400 * 7:   # 7 days
            adaptation_frequency = "moderate"
        else:
            adaptation_frequency = "high"
        
        # Threshold variance (simulated)
        if len(success_rate_trend) > 1:
            variance = np.var(success_rate_trend)
            threshold_variance = "low" if variance < 0.01 else "medium" if variance < 0.05 else "high"
        else:
            threshold_variance = "unknown"
        
        # Convergence indicator
        if adaptation_frequency == "decreasing" and threshold_variance == "low":
            convergence_indicator = "stable"
        elif adaptation_frequency == "moderate":
            convergence_indicator = "converging"
        else:
            convergence_indicator = "unstable"
        
        return {
            "adaptation_frequency": adaptation_frequency,
            "threshold_variance": threshold_variance,
            "convergence_indicator": convergence_indicator,
            "days_since_last_adaptation": round(time_since_adaptation / 86400, 1)
        }

    def justify_evolution(self, evolution_triggers: List[str], success_rate_trend: List[float],
                         adaptation_count: int) -> Dict[str, Any]:
        """Provide evolution justification"""
        # Trigger analysis
        if "near_miss_incident" in evolution_triggers:
            trigger_analysis = "Multiple near-miss incidents in similar conditions"
        elif "safety_violation" in evolution_triggers:
            trigger_analysis = "Safety violations requiring immediate response"
        elif "performance_degradation" in evolution_triggers:
            trigger_analysis = "Performance degradation over time"
        else:
            trigger_analysis = "Various operational factors"
        
        # Data support
        if len(success_rate_trend) >= 10:
            data_support = "Strong statistical evidence"
        elif len(success_rate_trend) >= 5:
            data_support = "Moderate statistical evidence"
        else:
            data_support = "Limited statistical evidence"
        
        # Expert validation
        if adaptation_count > 5:
            expert_validation = "Required"
        elif adaptation_count > 2:
            expert_validation = "Recommended"
        else:
            expert_validation = "Not required"
        
        return {
            "trigger_analysis": trigger_analysis,
            "data_support": data_support,
            "expert_validation": expert_validation,
            "trigger_count": len(evolution_triggers)
        }

    def compute_evolution_validity(self, adaptation_analysis: Dict[str, Any],
                                 safety_impact: Dict[str, Any],
                                 stability_metrics: Dict[str, Any]) -> int:
        """Compute evolution validity score (0-100)"""
        score = 0
        
        # Adaptation quality (40 points)
        quality_score = adaptation_analysis.get("quality_score", 0)
        score += int(quality_score * 40)
        
        # Safety impact (30 points)
        if safety_impact.get("safety_level_change") == "improved":
            score += 30
        elif safety_impact.get("safety_level_change") == "maintained":
            score += 20
        else:
            score += 10
        
        # Stability (30 points)
        if stability_metrics.get("convergence_indicator") == "stable":
            score += 30
        elif stability_metrics.get("convergence_indicator") == "converging":
            score += 20
        else:
            score += 10
        
        return min(100, max(0, score))

    def determine_status(self, evolution_validity: int, stability_metrics: Dict[str, Any]) -> str:
        """Determine rule status"""
        if stability_metrics.get("convergence_indicator") == "stable":
            return "stable"
        elif evolution_validity > 70:
            return "evolving"
        else:
            return "unstable"

    def determine_risk_level(self, evolution_validity: int, adaptation_analysis: Dict[str, Any]) -> str:
        """Determine risk level"""
        if evolution_validity < 50 or adaptation_analysis.get("risk_assessment") == "high":
            return "high"
        elif evolution_validity < 70 or adaptation_analysis.get("risk_assessment") == "medium":
            return "medium"
        else:
            return "low"

    def generate_recommendations(self, status: str, evolution_validity: int,
                               stability_metrics: Dict[str, Any],
                               safety_impact: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if status == "stable":
            recommendations.append("Rule has reached stable state")
            recommendations.append("Monitor for new edge cases")
            recommendations.append("Consider rule consolidation with similar patterns")
        elif status == "evolving":
            recommendations.append("Continue monitoring evolution progress")
            recommendations.append("Validate adaptations with additional data")
            if safety_impact.get("safety_level_change") == "degraded":
                recommendations.append("Review recent adaptations for safety impact")
        else:  # unstable
            recommendations.append("Immediate review of rule evolution required")
            recommendations.append("Consider reverting to previous stable version")
            recommendations.append("Implement additional validation checks")
        
        if evolution_validity < 70:
            recommendations.append("Increase validation rigor for future adaptations")
        
        if stability_metrics.get("adaptation_frequency") == "high":
            recommendations.append("Reduce adaptation frequency to prevent overfitting")
        
        return recommendations

    def validate_rule_evolution(self, rule_data: RuleEvolutionData) -> RuleEvolutionReport:
        """Perform comprehensive rule evolution validation"""
        # Analyze adaptation quality
        adaptation_analysis = self.analyze_adaptation_quality(
            rule_data.original_threshold, rule_data.current_threshold,
            rule_data.adaptation_count, rule_data.evolution_triggers
        )
        
        # Assess safety impact
        safety_impact = self.assess_safety_impact(
            rule_data.success_rate_trend, rule_data.original_threshold, rule_data.current_threshold
        )
        
        # Assess stability
        stability_metrics = self.assess_stability(
            rule_data.adaptation_count, rule_data.success_rate_trend, rule_data.last_adaptation_time
        )
        
        # Justify evolution
        evolution_justification = self.justify_evolution(
            rule_data.evolution_triggers, rule_data.success_rate_trend, rule_data.adaptation_count
        )
        
        # Compute evolution validity
        evolution_validity = self.compute_evolution_validity(adaptation_analysis, safety_impact, stability_metrics)
        
        # Determine status and risk level
        status = self.determine_status(evolution_validity, stability_metrics)
        risk_level = self.determine_risk_level(evolution_validity, adaptation_analysis)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            status, evolution_validity, stability_metrics, safety_impact
        )
        
        return RuleEvolutionReport(
            rule_id=rule_data.rule_id,
            evolution_validity=evolution_validity,
            status=status,
            adaptation_analysis=adaptation_analysis,
            safety_impact=safety_impact,
            stability_metrics=stability_metrics,
            evolution_justification=evolution_justification,
            recommendations=recommendations,
            risk_level=risk_level
        )

def main():
    parser = argparse.ArgumentParser(description="Validate rule evolution for adaptive safety system.")
    parser.add_argument('--rule_id', type=str, required=True)
    parser.add_argument('--original_condition', type=str, required=True)
    parser.add_argument('--current_condition', type=str, required=True)
    parser.add_argument('--original_threshold', type=float, required=True)
    parser.add_argument('--current_threshold', type=float, required=True)
    parser.add_argument('--adaptation_count', type=int, required=True)
    parser.add_argument('--success_rate_trend', type=str, required=True, 
                       help='Comma-separated list of success rates over time')
    parser.add_argument('--evolution_triggers', type=str, required=True,
                       help='Comma-separated list of evolution triggers')
    parser.add_argument('--last_adaptation_time', type=float, default=None)
    
    args = parser.parse_args()
    
    # Parse inputs
    success_rate_trend = [float(x) for x in args.success_rate_trend.split(',')]
    evolution_triggers = [x.strip() for x in args.evolution_triggers.split(',')]
    last_adaptation_time = args.last_adaptation_time if args.last_adaptation_time is not None else time.time()
    
    rule_data = RuleEvolutionData(
        rule_id=args.rule_id,
        original_condition=args.original_condition,
        current_condition=args.current_condition,
        original_threshold=args.original_threshold,
        current_threshold=args.current_threshold,
        adaptation_count=args.adaptation_count,
        success_rate_trend=success_rate_trend,
        evolution_triggers=evolution_triggers,
        last_adaptation_time=last_adaptation_time
    )
    
    validator = RuleEvolutionValidator()
    report = validator.validate_rule_evolution(rule_data)
    print(json.dumps(asdict(report), indent=2))

if __name__ == "__main__":
    main() 