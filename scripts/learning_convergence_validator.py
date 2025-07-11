#!/usr/bin/env python3
"""
Learning Convergence Validation Script

Implements the Learning Convergence Validation Prompt for adaptive safety learning systems.
"""

import json
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import numpy as np

@dataclass
class ConvergenceMetrics:
    learning_rounds: int
    pattern_stability_trend: List[float]
    rule_evolution_frequency: float
    threshold_variance: float
    confidence_convergence: float
    performance_plateau: bool
    adaptation_rate: float
    learning_curve_slope: float
    pattern_maturity: float
    rule_maturity: float

@dataclass
class ConvergenceReport:
    convergence_score: int
    status: str
    stability_analysis: Dict[str, Any]
    learning_maturity: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    overfitting_check: Dict[str, Any]
    convergence_indicators: Dict[str, Any]
    future_learning: Dict[str, Any]
    recommendations: List[str]
    convergence_confidence: str

class LearningConvergenceValidator:
    def __init__(self):
        # Convergence thresholds from the prompt
        self.convergence_score_min = 80
        self.stability_min = 0.85
        self.performance_plateau_threshold = 0.02
        self.adaptation_rate_max = 0.1

    def analyze_stability(self, pattern_stability_trend: List[float], 
                         threshold_variance: float, rule_evolution_frequency: float) -> Dict[str, Any]:
        """Analyze pattern and rule stability"""
        if not pattern_stability_trend:
            return {
                "pattern_stability": 0.0,
                "rule_stability": 0.0,
                "threshold_stability": 0.0,
                "stability_trend": "unknown"
            }
        
        # Calculate pattern stability
        if len(pattern_stability_trend) >= 3:
            recent_stability = np.mean(pattern_stability_trend[-3:])
            earlier_stability = np.mean(pattern_stability_trend[:3])
            pattern_stability = min(1.0, recent_stability)
            stability_trend = "improving" if recent_stability > earlier_stability else "stable" if abs(recent_stability - earlier_stability) < 0.05 else "degrading"
        else:
            pattern_stability = pattern_stability_trend[-1] if pattern_stability_trend else 0.0
            stability_trend = "unknown"
        
        # Calculate rule stability
        rule_stability = max(0.0, 1.0 - rule_evolution_frequency)
        
        # Calculate threshold stability
        threshold_stability = max(0.0, 1.0 - threshold_variance)
        
        return {
            "pattern_stability": round(pattern_stability, 3),
            "rule_stability": round(rule_stability, 3),
            "threshold_stability": round(threshold_stability, 3),
            "stability_trend": stability_trend,
            "pattern_variance": round(np.var(pattern_stability_trend) if len(pattern_stability_trend) > 1 else 0.0, 3)
        }

    def assess_learning_maturity(self, learning_rounds: int, pattern_maturity: float,
                                rule_maturity: float) -> Dict[str, Any]:
        """Assess system learning maturity level"""
        # Determine maturity level
        if learning_rounds >= 1000 and pattern_maturity >= 0.9 and rule_maturity >= 0.9:
            maturity_level = "advanced"
        elif learning_rounds >= 500 and pattern_maturity >= 0.8 and rule_maturity >= 0.8:
            maturity_level = "intermediate"
        elif learning_rounds >= 100 and pattern_maturity >= 0.7 and rule_maturity >= 0.7:
            maturity_level = "developing"
        else:
            maturity_level = "early"
        
        # Determine learning stage
        if maturity_level == "advanced":
            learning_stage = "stable"
        elif maturity_level == "intermediate":
            learning_stage = "converging"
        elif maturity_level == "developing":
            learning_stage = "learning"
        else:
            learning_stage = "initializing"
        
        # Assess experience sufficiency
        if learning_rounds >= 1000:
            experience_sufficiency = "adequate"
        elif learning_rounds >= 500:
            experience_sufficiency = "moderate"
        elif learning_rounds >= 100:
            experience_sufficiency = "limited"
        else:
            experience_sufficiency = "insufficient"
        
        return {
            "maturity_level": maturity_level,
            "learning_stage": learning_stage,
            "experience_sufficiency": experience_sufficiency,
            "learning_rounds": learning_rounds,
            "pattern_maturity": round(pattern_maturity, 3),
            "rule_maturity": round(rule_maturity, 3)
        }

    def analyze_performance(self, performance_plateau: bool, learning_curve_slope: float,
                           confidence_convergence: float) -> Dict[str, Any]:
        """Analyze performance stabilization"""
        # Determine if plateau is reached
        plateau_reached = performance_plateau or abs(learning_curve_slope) < self.performance_plateau_threshold
        
        # Calculate performance variance (simulated)
        if plateau_reached:
            performance_variance = "low"
        elif abs(learning_curve_slope) < 0.05:
            performance_variance = "medium"
        else:
            performance_variance = "high"
        
        # Determine improvement rate
        if abs(learning_curve_slope) < 0.01:
            improvement_rate = "minimal"
        elif abs(learning_curve_slope) < 0.05:
            improvement_rate = "slow"
        else:
            improvement_rate = "significant"
        
        return {
            "plateau_reached": plateau_reached,
            "performance_variance": performance_variance,
            "improvement_rate": improvement_rate,
            "learning_curve_slope": round(learning_curve_slope, 4),
            "confidence_convergence": round(confidence_convergence, 3)
        }

    def check_overfitting(self, adaptation_rate: float, pattern_stability_trend: List[float],
                         rule_evolution_frequency: float) -> Dict[str, Any]:
        """Detect overfitting or over-adaptation"""
        # Check for overfitting indicators
        overfitting_indicators = []
        
        if adaptation_rate > self.adaptation_rate_max:
            overfitting_indicators.append("High adaptation rate")
        
        if rule_evolution_frequency > 0.2:
            overfitting_indicators.append("Frequent rule changes")
        
        if len(pattern_stability_trend) >= 5:
            recent_variance = np.var(pattern_stability_trend[-5:])
            if recent_variance > 0.1:
                overfitting_indicators.append("High pattern variance")
        
        # Determine overfitting status
        overfitting_detected = len(overfitting_indicators) > 1
        
        # Calculate generalization score
        if overfitting_detected:
            generalization_score = max(0.5, 1.0 - len(overfitting_indicators) * 0.1)
        else:
            generalization_score = 0.9 + (1.0 - adaptation_rate) * 0.1
        
        # Assess validation performance
        if generalization_score >= 0.9:
            validation_performance = "excellent"
        elif generalization_score >= 0.8:
            validation_performance = "good"
        elif generalization_score >= 0.7:
            validation_performance = "adequate"
        else:
            validation_performance = "poor"
        
        return {
            "overfitting_detected": overfitting_detected,
            "generalization_score": round(generalization_score, 3),
            "validation_performance": validation_performance,
            "overfitting_indicators": overfitting_indicators,
            "adaptation_rate": round(adaptation_rate, 3)
        }

    def assess_convergence_indicators(self, adaptation_rate: float, rule_evolution_frequency: float,
                                    pattern_stability_trend: List[float], confidence_convergence: float) -> Dict[str, Any]:
        """Assess convergence indicators"""
        # Adaptation frequency
        if adaptation_rate < 0.05:
            adaptation_frequency = "decreasing"
        elif adaptation_rate < 0.1:
            adaptation_frequency = "moderate"
        else:
            adaptation_frequency = "high"
        
        # Rule changes
        if rule_evolution_frequency < 0.1:
            rule_changes = "minimal"
        elif rule_evolution_frequency < 0.2:
            rule_changes = "occasional"
        else:
            rule_changes = "frequent"
        
        # Pattern consistency
        if len(pattern_stability_trend) >= 3:
            pattern_variance = np.var(pattern_stability_trend[-3:])
            if pattern_variance < 0.01:
                pattern_consistency = "high"
            elif pattern_variance < 0.05:
                pattern_consistency = "medium"
            else:
                pattern_consistency = "low"
        else:
            pattern_consistency = "unknown"
        
        # Confidence stability
        if confidence_convergence > 0.9:
            confidence_stability = "stable"
        elif confidence_convergence > 0.8:
            confidence_stability = "converging"
        else:
            confidence_stability = "unstable"
        
        return {
            "adaptation_frequency": adaptation_frequency,
            "rule_changes": rule_changes,
            "pattern_consistency": pattern_consistency,
            "confidence_stability": confidence_stability,
            "convergence_strength": round(min(1.0, (1 - adaptation_rate) * confidence_convergence), 3)
        }

    def assess_future_learning(self, learning_maturity: Dict[str, Any],
                              convergence_indicators: Dict[str, Any],
                              overfitting_check: Dict[str, Any]) -> Dict[str, Any]:
        """Assess capacity for continued learning"""
        # Learning capacity
        maturity_level = learning_maturity.get("maturity_level", "early")
        if maturity_level == "advanced":
            learning_capacity = "maintained"
        elif maturity_level == "intermediate":
            learning_capacity = "high"
        else:
            learning_capacity = "developing"
        
        # Adaptation readiness
        adaptation_frequency = convergence_indicators.get("adaptation_frequency", "high")
        if adaptation_frequency == "decreasing":
            adaptation_readiness = "high"
        elif adaptation_frequency == "moderate":
            adaptation_readiness = "medium"
        else:
            adaptation_readiness = "low"
        
        # New scenario handling
        generalization_score = overfitting_check.get("generalization_score", 0.5)
        if generalization_score >= 0.8:
            new_scenario_handling = "capable"
        elif generalization_score >= 0.6:
            new_scenario_handling = "limited"
        else:
            new_scenario_handling = "constrained"
        
        return {
            "learning_capacity": learning_capacity,
            "adaptation_readiness": adaptation_readiness,
            "new_scenario_handling": new_scenario_handling,
            "future_learning_potential": round(generalization_score, 3)
        }

    def compute_convergence_score(self, stability_analysis: Dict[str, Any],
                                learning_maturity: Dict[str, Any],
                                performance_analysis: Dict[str, Any],
                                overfitting_check: Dict[str, Any]) -> int:
        """Compute overall convergence score (0-100)"""
        score = 0
        
        # Stability analysis (30 points)
        pattern_stability = stability_analysis.get("pattern_stability", 0)
        rule_stability = stability_analysis.get("rule_stability", 0)
        threshold_stability = stability_analysis.get("threshold_stability", 0)
        avg_stability = (pattern_stability + rule_stability + threshold_stability) / 3
        score += int(avg_stability * 30)
        
        # Learning maturity (25 points)
        maturity_level = learning_maturity.get("maturity_level", "early")
        if maturity_level == "advanced":
            score += 25
        elif maturity_level == "intermediate":
            score += 20
        elif maturity_level == "developing":
            score += 15
        else:
            score += 10
        
        # Performance analysis (25 points)
        if performance_analysis.get("plateau_reached", False):
            score += 25
        elif performance_analysis.get("improvement_rate") == "minimal":
            score += 20
        elif performance_analysis.get("improvement_rate") == "slow":
            score += 15
        else:
            score += 10
        
        # Overfitting check (20 points)
        if not overfitting_check.get("overfitting_detected", False):
            generalization_score = overfitting_check.get("generalization_score", 0)
            score += int(generalization_score * 20)
        
        return min(100, max(0, score))

    def determine_status(self, convergence_score: int, stability_analysis: Dict[str, Any]) -> str:
        """Determine convergence status"""
        avg_stability = (stability_analysis.get("pattern_stability", 0) + 
                        stability_analysis.get("rule_stability", 0) + 
                        stability_analysis.get("threshold_stability", 0)) / 3
        
        if convergence_score >= self.convergence_score_min and avg_stability >= self.stability_min:
            return "converged"
        elif convergence_score >= 60:
            return "converging"
        else:
            return "unstable"

    def determine_convergence_confidence(self, convergence_score: int, overfitting_check: Dict[str, Any],
                                       stability_analysis: Dict[str, Any]) -> str:
        """Determine convergence confidence"""
        if convergence_score >= 90 and not overfitting_check.get("overfitting_detected", False):
            return "high"
        elif convergence_score >= 70 and stability_analysis.get("stability_trend") == "improving":
            return "medium"
        else:
            return "low"

    def generate_recommendations(self, status: str, convergence_score: int,
                               overfitting_check: Dict[str, Any],
                               future_learning: Dict[str, Any]) -> List[str]:
        """Generate convergence recommendations"""
        recommendations = []
        
        if status == "converged":
            recommendations.append("System has converged to stable state")
            recommendations.append("Maintain monitoring for new scenarios")
            recommendations.append("Consider periodic retraining for long-term stability")
        elif status == "converging":
            recommendations.append("System is converging well")
            recommendations.append("Continue monitoring convergence progress")
            recommendations.append("Validate stability with additional data")
        else:  # unstable
            recommendations.append("System convergence requires attention")
            recommendations.append("Review learning parameters and data quality")
            recommendations.append("Consider resetting and retraining if necessary")
        
        # Add overfitting recommendations
        if overfitting_check.get("overfitting_detected", False):
            recommendations.append("Address overfitting through regularization")
            recommendations.append("Increase validation data diversity")
        
        # Add future learning recommendations
        if future_learning.get("new_scenario_handling") == "constrained":
            recommendations.append("Improve generalization for new scenarios")
        
        if convergence_score < self.convergence_score_min:
            recommendations.append("Increase convergence validation rigor")
        
        return recommendations

    def validate_convergence(self, metrics: ConvergenceMetrics) -> ConvergenceReport:
        """Perform comprehensive learning convergence validation"""
        # Analyze stability
        stability_analysis = self.analyze_stability(
            metrics.pattern_stability_trend, metrics.threshold_variance, metrics.rule_evolution_frequency
        )
        
        # Assess learning maturity
        learning_maturity = self.assess_learning_maturity(
            metrics.learning_rounds, metrics.pattern_maturity, metrics.rule_maturity
        )
        
        # Analyze performance
        performance_analysis = self.analyze_performance(
            metrics.performance_plateau, metrics.learning_curve_slope, metrics.confidence_convergence
        )
        
        # Check for overfitting
        overfitting_check = self.check_overfitting(
            metrics.adaptation_rate, metrics.pattern_stability_trend, metrics.rule_evolution_frequency
        )
        
        # Assess convergence indicators
        convergence_indicators = self.assess_convergence_indicators(
            metrics.adaptation_rate, metrics.rule_evolution_frequency,
            metrics.pattern_stability_trend, metrics.confidence_convergence
        )
        
        # Assess future learning
        future_learning = self.assess_future_learning(
            learning_maturity, convergence_indicators, overfitting_check
        )
        
        # Compute convergence score
        convergence_score = self.compute_convergence_score(
            stability_analysis, learning_maturity, performance_analysis, overfitting_check
        )
        
        # Determine status and confidence
        status = self.determine_status(convergence_score, stability_analysis)
        convergence_confidence = self.determine_convergence_confidence(
            convergence_score, overfitting_check, stability_analysis
        )
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            status, convergence_score, overfitting_check, future_learning
        )
        
        return ConvergenceReport(
            convergence_score=convergence_score,
            status=status,
            stability_analysis=stability_analysis,
            learning_maturity=learning_maturity,
            performance_analysis=performance_analysis,
            overfitting_check=overfitting_check,
            convergence_indicators=convergence_indicators,
            future_learning=future_learning,
            recommendations=recommendations,
            convergence_confidence=convergence_confidence
        )

def main():
    parser = argparse.ArgumentParser(description="Validate learning convergence for adaptive safety system.")
    parser.add_argument('--learning_rounds', type=int, required=True, help='Number of learning rounds')
    parser.add_argument('--pattern_stability_trend', type=str, required=True, 
                       help='Comma-separated list of pattern stability values')
    parser.add_argument('--rule_evolution_frequency', type=float, required=True, help='Rule evolution frequency (0.0-1.0)')
    parser.add_argument('--threshold_variance', type=float, required=True, help='Threshold variance (0.0-1.0)')
    parser.add_argument('--confidence_convergence', type=float, required=True, help='Confidence convergence (0.0-1.0)')
    parser.add_argument('--performance_plateau', type=bool, required=True, help='Performance plateau reached (True/False)')
    parser.add_argument('--adaptation_rate', type=float, required=True, help='Adaptation rate (0.0-1.0)')
    parser.add_argument('--learning_curve_slope', type=float, required=True, help='Learning curve slope')
    parser.add_argument('--pattern_maturity', type=float, required=True, help='Pattern maturity (0.0-1.0)')
    parser.add_argument('--rule_maturity', type=float, required=True, help='Rule maturity (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Parse pattern stability trend
    pattern_stability_trend = [float(x) for x in args.pattern_stability_trend.split(',')]
    
    metrics = ConvergenceMetrics(
        learning_rounds=args.learning_rounds,
        pattern_stability_trend=pattern_stability_trend,
        rule_evolution_frequency=args.rule_evolution_frequency,
        threshold_variance=args.threshold_variance,
        confidence_convergence=args.confidence_convergence,
        performance_plateau=args.performance_plateau,
        adaptation_rate=args.adaptation_rate,
        learning_curve_slope=args.learning_curve_slope,
        pattern_maturity=args.pattern_maturity,
        rule_maturity=args.rule_maturity
    )
    
    validator = LearningConvergenceValidator()
    report = validator.validate_convergence(metrics)
    print(json.dumps(asdict(report), indent=2))

if __name__ == "__main__":
    main() 