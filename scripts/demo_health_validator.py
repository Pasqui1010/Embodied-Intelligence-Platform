#!/usr/bin/env python3
"""
Demo script for Adaptive Safety Health Validator

Demonstrates the implementation of the System Health Validation Prompt
without complex import dependencies.
"""

import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class SystemMetrics:
    """System metrics for health validation"""
    learning_rounds: int
    pattern_discoveries: int
    rule_evolutions: int
    adaptation_count: int
    safety_level: float
    confidence: float
    memory_usage: float
    processing_latency: float
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class HealthAssessment:
    """Health assessment results"""
    health_score: float
    status: str
    learning_effectiveness: Dict[str, Any]
    safety_reliability: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    recommendations: List[str]
    risk_level: str
    timestamp: float
    validation_duration: float


class SimpleHealthValidator:
    """
    Simplified Health Validator
    
    Demonstrates the System Health Validation Prompt implementation
    without complex dependencies.
    """
    
    def __init__(self):
        """Initialize the health validator"""
        self.alert_thresholds = {
            'health_score_critical': 50,
            'health_score_warning': 70,
            'safety_reliability_min': 0.8,
            'learning_effectiveness_min': 0.6,
            'memory_usage_max': 0.9,
            'latency_max': 100.0
        }
        
        self.scoring_weights = {
            'learning_effectiveness': 0.25,
            'safety_reliability': 0.35,
            'performance': 0.25,
            'stability': 0.15
        }
    
    def calculate_learning_effectiveness(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Calculate learning effectiveness score and assessment"""
        learning_rate = metrics.pattern_discoveries / max(metrics.learning_rounds, 1)
        adaptation_efficiency = metrics.adaptation_count / max(metrics.rule_evolutions, 1)
        
        learning_rate_score = min(learning_rate * 100, 1.0)
        adaptation_efficiency_score = min(adaptation_efficiency * 10, 1.0)
        
        effectiveness_score = (learning_rate_score * 0.6 + adaptation_efficiency_score * 0.4)
        
        if effectiveness_score > 0.8:
            assessment = "System is learning effectively with good pattern recognition"
            concerns = []
        elif effectiveness_score > 0.6:
            assessment = "System is learning but could benefit from more diverse experiences"
            concerns = ["Consider adding more training scenarios"]
        else:
            assessment = "System learning effectiveness is below optimal levels"
            concerns = [
                "Learning rate is too low",
                "Pattern discovery rate needs improvement",
                "Consider adjusting learning parameters"
            ]
        
        return {
            "score": round(effectiveness_score, 3),
            "assessment": assessment,
            "concerns": concerns,
            "learning_rate": round(learning_rate, 4),
            "adaptation_efficiency": round(adaptation_efficiency, 3)
        }
    
    def calculate_safety_reliability(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Calculate safety reliability score and assessment"""
        safety_score = metrics.safety_level
        confidence_score = metrics.confidence
        
        reliability_score = (safety_score * 0.7 + confidence_score * 0.3)
        
        if reliability_score > 0.9:
            assessment = "High safety assurance with robust rule evolution"
            concerns = []
        elif reliability_score > 0.8:
            assessment = "Good safety assurance with room for improvement"
            concerns = ["Monitor confidence levels"]
        else:
            assessment = "Safety reliability needs attention"
            concerns = [
                "Safety level is below optimal",
                "Confidence score needs improvement",
                "Review safety rule effectiveness"
            ]
        
        return {
            "score": round(reliability_score, 3),
            "assessment": assessment,
            "concerns": concerns,
            "safety_level": round(safety_score, 3),
            "confidence_level": round(confidence_score, 3)
        }
    
    def analyze_performance(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Analyze system performance and identify bottlenecks"""
        bottlenecks = []
        optimization_opportunities = []
        
        if metrics.memory_usage > 0.8:
            bottlenecks.append("High memory usage")
            optimization_opportunities.append("Consider memory optimization or cleanup")
        
        if metrics.processing_latency > 50:
            bottlenecks.append("High processing latency")
            optimization_opportunities.append("Optimize processing pipeline")
        
        if metrics.learning_rounds > 0 and metrics.pattern_discoveries / metrics.learning_rounds < 0.05:
            bottlenecks.append("Low pattern discovery rate")
            optimization_opportunities.append("Review pattern recognition algorithms")
        
        performance_score = max(0, 1.0 - len(bottlenecks) * 0.2)
        
        return {
            "bottlenecks": bottlenecks,
            "optimization_opportunities": optimization_opportunities,
            "performance_score": round(performance_score, 3),
            "memory_usage": round(metrics.memory_usage, 3),
            "processing_latency": round(metrics.processing_latency, 3)
        }
    
    def calculate_health_score(self, learning_effectiveness: Dict[str, Any], 
                             safety_reliability: Dict[str, Any], 
                             performance_analysis: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        health_score = (
            learning_effectiveness['score'] * self.scoring_weights['learning_effectiveness'] +
            safety_reliability['score'] * self.scoring_weights['safety_reliability'] +
            performance_analysis['performance_score'] * self.scoring_weights['performance'] +
            (1.0 - len(performance_analysis['bottlenecks']) * 0.1) * self.scoring_weights['stability']
        ) * 100
        
        return round(max(0, min(100, health_score)), 1)
    
    def determine_status(self, health_score: float) -> str:
        """Determine system status based on health score"""
        if health_score >= 80:
            return "healthy"
        elif health_score >= 60:
            return "degraded"
        else:
            return "critical"
    
    def determine_risk_level(self, health_score: float, safety_reliability: Dict[str, Any]) -> str:
        """Determine risk level based on health score and safety reliability"""
        if health_score < 50 or safety_reliability['score'] < 0.7:
            return "high"
        elif health_score < 70 or safety_reliability['score'] < 0.8:
            return "medium"
        else:
            return "low"
    
    def generate_recommendations(self, learning_effectiveness: Dict[str, Any],
                               safety_reliability: Dict[str, Any],
                               performance_analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on assessment results"""
        recommendations = []
        
        if learning_effectiveness['score'] < 0.8:
            recommendations.append("Increase pattern confidence threshold to 0.75")
            recommendations.append("Add more diverse training scenarios")
        
        if safety_reliability['score'] < 0.9:
            recommendations.append("Review and optimize safety rule thresholds")
            recommendations.append("Implement additional safety validation layers")
        
        if performance_analysis['bottlenecks']:
            recommendations.extend(performance_analysis['optimization_opportunities'])
        
        if not recommendations:
            recommendations.append("Continue monitoring system performance")
            recommendations.append("Schedule periodic safety rule review")
        
        return recommendations
    
    def validate_health(self, metrics: SystemMetrics) -> HealthAssessment:
        """Perform comprehensive health validation using the System Health Validation Prompt"""
        start_time = time.time()
        
        # Calculate learning effectiveness
        learning_effectiveness = self.calculate_learning_effectiveness(metrics)
        
        # Calculate safety reliability
        safety_reliability = self.calculate_safety_reliability(metrics)
        
        # Analyze performance
        performance_analysis = self.analyze_performance(metrics)
        
        # Calculate overall health score
        health_score = self.calculate_health_score(
            learning_effectiveness, safety_reliability, performance_analysis
        )
        
        # Determine status and risk level
        status = self.determine_status(health_score)
        risk_level = self.determine_risk_level(health_score, safety_reliability)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            learning_effectiveness, safety_reliability, performance_analysis
        )
        
        # Create health assessment
        assessment = HealthAssessment(
            health_score=health_score,
            status=status,
            learning_effectiveness=learning_effectiveness,
            safety_reliability=safety_reliability,
            performance_analysis=performance_analysis,
            recommendations=recommendations,
            risk_level=risk_level,
            timestamp=time.time(),
            validation_duration=time.time() - start_time
        )
        
        return assessment
    
    def get_validation_report(self, assessment: HealthAssessment) -> Dict[str, Any]:
        """Generate validation report in the required JSON format"""
        return {
            "health_score": assessment.health_score,
            "status": assessment.status,
            "learning_effectiveness": assessment.learning_effectiveness,
            "safety_reliability": assessment.safety_reliability,
            "performance_analysis": assessment.performance_analysis,
            "recommendations": assessment.recommendations,
            "risk_level": assessment.risk_level,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(assessment.timestamp)),
            "validation_duration": round(assessment.validation_duration, 3)
        }
    
    def check_alerts(self, assessment: HealthAssessment) -> List[Dict[str, Any]]:
        """Check for alerts based on assessment results"""
        alerts = []
        
        if assessment.health_score < self.alert_thresholds['health_score_critical']:
            alerts.append({
                "level": "critical",
                "message": f"System health score is critically low: {assessment.health_score}",
                "timestamp": assessment.timestamp
            })
        elif assessment.health_score < self.alert_thresholds['health_score_warning']:
            alerts.append({
                "level": "warning",
                "message": f"System health score is below optimal: {assessment.health_score}",
                "timestamp": assessment.timestamp
            })
        
        if assessment.safety_reliability['score'] < self.alert_thresholds['safety_reliability_min']:
            alerts.append({
                "level": "critical",
                "message": f"Safety reliability below minimum threshold: {assessment.safety_reliability['score']}",
                "timestamp": assessment.timestamp
            })
        
        if assessment.learning_effectiveness['score'] < self.alert_thresholds['learning_effectiveness_min']:
            alerts.append({
                "level": "warning",
                "message": f"Learning effectiveness below optimal: {assessment.learning_effectiveness['score']}",
                "timestamp": assessment.timestamp
            })
        
        return alerts


def demo_healthy_system():
    """Demonstrate healthy system validation"""
    print("=" * 60)
    print("DEMO 1: Healthy System")
    print("=" * 60)
    
    validator = SimpleHealthValidator()
    
    # Create healthy metrics
    metrics = SystemMetrics(
        learning_rounds=500,
        pattern_discoveries=45,
        rule_evolutions=12,
        adaptation_count=85,
        safety_level=0.92,
        confidence=0.88,
        memory_usage=0.65,
        processing_latency=35.0
    )
    
    # Perform validation
    assessment = validator.validate_health(metrics)
    report = validator.get_validation_report(assessment)
    
    print("Input Metrics:")
    print(f"  Learning Rounds: {metrics.learning_rounds}")
    print(f"  Pattern Discoveries: {metrics.pattern_discoveries}")
    print(f"  Rule Evolutions: {metrics.rule_evolutions}")
    print(f"  Adaptation Count: {metrics.adaptation_count}")
    print(f"  Safety Level: {metrics.safety_level}")
    print(f"  Confidence: {metrics.confidence}")
    print(f"  Memory Usage: {metrics.memory_usage}")
    print(f"  Processing Latency: {metrics.processing_latency}ms")
    
    print("\nValidation Report:")
    print(json.dumps(report, indent=2))
    
    # Check alerts
    alerts = validator.check_alerts(assessment)
    if alerts:
        print("\nAlerts:")
        for alert in alerts:
            print(f"  [{alert['level'].upper()}] {alert['message']}")
    else:
        print("\nNo alerts generated - system is healthy!")


def demo_degraded_system():
    """Demonstrate degraded system validation"""
    print("\n" + "=" * 60)
    print("DEMO 2: Degraded System")
    print("=" * 60)
    
    validator = SimpleHealthValidator()
    
    # Create degraded metrics
    metrics = SystemMetrics(
        learning_rounds=800,
        pattern_discoveries=25,
        rule_evolutions=8,
        adaptation_count=60,
        safety_level=0.75,
        confidence=0.65,
        memory_usage=0.85,
        processing_latency=75.0
    )
    
    # Perform validation
    assessment = validator.validate_health(metrics)
    report = validator.get_validation_report(assessment)
    
    print("Input Metrics:")
    print(f"  Learning Rounds: {metrics.learning_rounds}")
    print(f"  Pattern Discoveries: {metrics.pattern_discoveries}")
    print(f"  Rule Evolutions: {metrics.rule_evolutions}")
    print(f"  Adaptation Count: {metrics.adaptation_count}")
    print(f"  Safety Level: {metrics.safety_level}")
    print(f"  Confidence: {metrics.confidence}")
    print(f"  Memory Usage: {metrics.memory_usage}")
    print(f"  Processing Latency: {metrics.processing_latency}ms")
    
    print("\nValidation Report:")
    print(json.dumps(report, indent=2))
    
    # Check alerts
    alerts = validator.check_alerts(assessment)
    if alerts:
        print("\nAlerts:")
        for alert in alerts:
            print(f"  [{alert['level'].upper()}] {alert['message']}")


def demo_critical_system():
    """Demonstrate critical system validation"""
    print("\n" + "=" * 60)
    print("DEMO 3: Critical System")
    print("=" * 60)
    
    validator = SimpleHealthValidator()
    
    # Create critical metrics
    metrics = SystemMetrics(
        learning_rounds=1000,
        pattern_discoveries=5,
        rule_evolutions=2,
        adaptation_count=15,
        safety_level=0.55,
        confidence=0.45,
        memory_usage=0.95,
        processing_latency=150.0
    )
    
    # Perform validation
    assessment = validator.validate_health(metrics)
    report = validator.get_validation_report(assessment)
    
    print("Input Metrics:")
    print(f"  Learning Rounds: {metrics.learning_rounds}")
    print(f"  Pattern Discoveries: {metrics.pattern_discoveries}")
    print(f"  Rule Evolutions: {metrics.rule_evolutions}")
    print(f"  Adaptation Count: {metrics.adaptation_count}")
    print(f"  Safety Level: {metrics.safety_level}")
    print(f"  Confidence: {metrics.confidence}")
    print(f"  Memory Usage: {metrics.memory_usage}")
    print(f"  Processing Latency: {metrics.processing_latency}ms")
    
    print("\nValidation Report:")
    print(json.dumps(report, indent=2))
    
    # Check alerts
    alerts = validator.check_alerts(assessment)
    if alerts:
        print("\nAlerts:")
        for alert in alerts:
            print(f"  [{alert['level'].upper()}] {alert['message']}")


def main():
    """Run demonstration"""
    print("Adaptive Safety Health Validator - Implementation Demo")
    print("Demonstrating the System Health Validation Prompt")
    print("=" * 80)
    
    try:
        # Run demonstrations
        demo_healthy_system()
        demo_degraded_system()
        demo_critical_system()
        
        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("The System Health Validation Prompt implementation is working correctly.")
        print("\nKey Features Demonstrated:")
        print("✅ Automated health assessment with scoring algorithm")
        print("✅ Learning effectiveness analysis")
        print("✅ Safety reliability evaluation")
        print("✅ Performance bottleneck identification")
        print("✅ Intelligent recommendation generation")
        print("✅ Alert system for critical issues")
        print("✅ JSON-formatted output matching prompt requirements")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 