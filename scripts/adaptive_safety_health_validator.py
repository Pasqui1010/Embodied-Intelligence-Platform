#!/usr/bin/env python3
"""
Adaptive Safety Health Validator

Implements the System Health Validation Prompt for automated health assessment
of the adaptive safety learning system in the embodied intelligence platform.
"""

import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../intelligence/eip_adaptive_safety'))

try:
    from eip_adaptive_safety.adaptive_learning_engine import AdaptiveLearningEngine
    from eip_adaptive_safety.adaptive_safety_node import AdaptiveSafetyNode
except ImportError:
    print("Warning: Could not import adaptive safety modules. Running in standalone mode.")
    # Define placeholder classes for standalone mode
    class AdaptiveLearningEngine:
        def get_status(self):
            return {}
    
    class AdaptiveSafetyNode:
        pass


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


class SystemHealthValidator:
    """
    System Health Validator
    
    Implements the System Health Validation Prompt for automated health assessment
    of the adaptive safety learning system.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the health validator"""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_file)
        self.validation_history = []
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'health_score_critical': 50,
            'health_score_warning': 70,
            'safety_reliability_min': 0.8,
            'learning_effectiveness_min': 0.6,
            'memory_usage_max': 0.9,
            'latency_max': 100.0
        })
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('adaptive_safety_health.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'validation_interval': 60.0,  # seconds
            'history_retention_hours': 24,
            'alert_thresholds': {
                'health_score_critical': 50,
                'health_score_warning': 70,
                'safety_reliability_min': 0.8,
                'learning_effectiveness_min': 0.6,
                'memory_usage_max': 0.9,
                'latency_max': 100.0
            },
            'scoring_weights': {
                'learning_effectiveness': 0.25,
                'safety_reliability': 0.35,
                'performance': 0.25,
                'stability': 0.15
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
                self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load config file: {e}")
        
        return default_config
    
    def collect_system_metrics(self, learning_engine: Optional[AdaptiveLearningEngine] = None) -> SystemMetrics:
        """
        Collect system metrics from the adaptive safety learning system
        
        Args:
            learning_engine: Optional learning engine instance for direct metrics
            
        Returns:
            SystemMetrics object with current system state
        """
        try:
            if learning_engine:
                # Get metrics directly from learning engine
                status = learning_engine.get_status()
                
                metrics = SystemMetrics(
                    learning_rounds=status.get('learning_rounds', 0),
                    pattern_discoveries=status.get('pattern_discoveries', 0),
                    rule_evolutions=status.get('rule_evolutions', 0),
                    adaptation_count=status.get('adaptation_count', 0),
                    safety_level=status.get('safety_level', 0.0),
                    confidence=status.get('confidence', 0.0),
                    memory_usage=status.get('memory_usage', 0.0),
                    processing_latency=status.get('processing_latency', 0.0)
                )
            else:
                # Mock metrics for testing or when engine not available
                metrics = SystemMetrics(
                    learning_rounds=np.random.randint(100, 1000),
                    pattern_discoveries=np.random.randint(10, 50),
                    rule_evolutions=np.random.randint(5, 20),
                    adaptation_count=np.random.randint(20, 100),
                    safety_level=np.random.uniform(0.7, 0.95),
                    confidence=np.random.uniform(0.6, 0.9),
                    memory_usage=np.random.uniform(0.3, 0.8),
                    processing_latency=np.random.uniform(10, 80)
                )
            
            self.logger.debug(f"Collected metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            # Return default metrics on error
            return SystemMetrics(
                learning_rounds=0,
                pattern_discoveries=0,
                rule_evolutions=0,
                adaptation_count=0,
                safety_level=0.5,
                confidence=0.5,
                memory_usage=0.5,
                processing_latency=100.0
            )
    
    def calculate_learning_effectiveness(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """
        Calculate learning effectiveness score and assessment
        
        Args:
            metrics: System metrics
            
        Returns:
            Learning effectiveness assessment
        """
        # Calculate learning effectiveness based on various factors
        learning_rate = metrics.pattern_discoveries / max(metrics.learning_rounds, 1)
        adaptation_efficiency = metrics.adaptation_count / max(metrics.rule_evolutions, 1)
        
        # Normalize scores
        learning_rate_score = min(learning_rate * 100, 1.0)
        adaptation_efficiency_score = min(adaptation_efficiency * 10, 1.0)
        
        # Overall learning effectiveness score
        effectiveness_score = (learning_rate_score * 0.6 + adaptation_efficiency_score * 0.4)
        
        # Determine assessment
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
        """
        Calculate safety reliability score and assessment
        
        Args:
            metrics: System metrics
            
        Returns:
            Safety reliability assessment
        """
        # Calculate safety reliability based on safety level and confidence
        safety_score = metrics.safety_level
        confidence_score = metrics.confidence
        
        # Overall safety reliability score
        reliability_score = (safety_score * 0.7 + confidence_score * 0.3)
        
        # Determine assessment
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
        """
        Analyze system performance and identify bottlenecks
        
        Args:
            metrics: System metrics
            
        Returns:
            Performance analysis
        """
        bottlenecks = []
        optimization_opportunities = []
        
        # Check memory usage
        if metrics.memory_usage > 0.8:
            bottlenecks.append("High memory usage")
            optimization_opportunities.append("Consider memory optimization or cleanup")
        
        # Check processing latency
        if metrics.processing_latency > 50:
            bottlenecks.append("High processing latency")
            optimization_opportunities.append("Optimize processing pipeline")
        
        # Check learning efficiency
        if metrics.learning_rounds > 0 and metrics.pattern_discoveries / metrics.learning_rounds < 0.05:
            bottlenecks.append("Low pattern discovery rate")
            optimization_opportunities.append("Review pattern recognition algorithms")
        
        # Performance score based on bottlenecks
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
        """
        Calculate overall system health score
        
        Args:
            learning_effectiveness: Learning effectiveness assessment
            safety_reliability: Safety reliability assessment
            performance_analysis: Performance analysis
            
        Returns:
            Overall health score (0-100)
        """
        weights = self.config['scoring_weights']
        
        # Calculate weighted score
        health_score = (
            learning_effectiveness['score'] * weights['learning_effectiveness'] +
            safety_reliability['score'] * weights['safety_reliability'] +
            performance_analysis['performance_score'] * weights['performance'] +
            (1.0 - len(performance_analysis['bottlenecks']) * 0.1) * weights['stability']
        ) * 100
        
        return round(max(0, min(100, health_score)), 1)
    
    def determine_status(self, health_score: float) -> str:
        """
        Determine system status based on health score
        
        Args:
            health_score: Overall health score
            
        Returns:
            Status string
        """
        if health_score >= 80:
            return "healthy"
        elif health_score >= 60:
            return "degraded"
        else:
            return "critical"
    
    def determine_risk_level(self, health_score: float, safety_reliability: Dict[str, Any]) -> str:
        """
        Determine risk level based on health score and safety reliability
        
        Args:
            health_score: Overall health score
            safety_reliability: Safety reliability assessment
            
        Returns:
            Risk level string
        """
        if health_score < 50 or safety_reliability['score'] < 0.7:
            return "high"
        elif health_score < 70 or safety_reliability['score'] < 0.8:
            return "medium"
        else:
            return "low"
    
    def generate_recommendations(self, learning_effectiveness: Dict[str, Any],
                               safety_reliability: Dict[str, Any],
                               performance_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate specific recommendations based on assessment results
        
        Args:
            learning_effectiveness: Learning effectiveness assessment
            safety_reliability: Safety reliability assessment
            performance_analysis: Performance analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Learning effectiveness recommendations
        if learning_effectiveness['score'] < 0.8:
            recommendations.append("Increase pattern confidence threshold to 0.75")
            recommendations.append("Add more diverse training scenarios")
        
        # Safety reliability recommendations
        if safety_reliability['score'] < 0.9:
            recommendations.append("Review and optimize safety rule thresholds")
            recommendations.append("Implement additional safety validation layers")
        
        # Performance recommendations
        if performance_analysis['bottlenecks']:
            recommendations.extend(performance_analysis['optimization_opportunities'])
        
        # General recommendations
        if not recommendations:
            recommendations.append("Continue monitoring system performance")
            recommendations.append("Schedule periodic safety rule review")
        
        return recommendations
    
    def validate_health(self, metrics: SystemMetrics) -> HealthAssessment:
        """
        Perform comprehensive health validation using the System Health Validation Prompt
        
        Args:
            metrics: System metrics to validate
            
        Returns:
            HealthAssessment object with validation results
        """
        start_time = time.time()
        
        try:
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
            
            # Store in history
            self.validation_history.append(assessment)
            
            # Clean up old history
            self._cleanup_history()
            
            # Log results
            self.logger.info(f"Health validation completed: Score={health_score}, Status={status}, Risk={risk_level}")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error during health validation: {e}")
            # Return error assessment
            return HealthAssessment(
                health_score=0.0,
                status="error",
                learning_effectiveness={"score": 0.0, "assessment": "Error occurred", "concerns": [str(e)]},
                safety_reliability={"score": 0.0, "assessment": "Error occurred", "concerns": [str(e)]},
                performance_analysis={"bottlenecks": ["Validation error"], "optimization_opportunities": []},
                recommendations=["Investigate validation system errors"],
                risk_level="high",
                timestamp=time.time(),
                validation_duration=time.time() - start_time
            )
    
    def _cleanup_history(self):
        """Clean up old validation history"""
        retention_hours = self.config.get('history_retention_hours', 24)
        cutoff_time = time.time() - (retention_hours * 3600)
        
        self.validation_history = [
            assessment for assessment in self.validation_history
            if assessment.timestamp > cutoff_time
        ]
    
    def get_validation_report(self, assessment: HealthAssessment) -> Dict[str, Any]:
        """
        Generate validation report in the required JSON format
        
        Args:
            assessment: Health assessment results
            
        Returns:
            JSON-formatted validation report
        """
        return {
            "health_score": assessment.health_score,
            "status": assessment.status,
            "learning_effectiveness": assessment.learning_effectiveness,
            "safety_reliability": assessment.safety_reliability,
            "performance_analysis": assessment.performance_analysis,
            "recommendations": assessment.recommendations,
            "risk_level": assessment.risk_level,
            "timestamp": datetime.fromtimestamp(assessment.timestamp).isoformat(),
            "validation_duration": round(assessment.validation_duration, 3)
        }
    
    def check_alerts(self, assessment: HealthAssessment) -> List[Dict[str, Any]]:
        """
        Check for alerts based on assessment results
        
        Args:
            assessment: Health assessment results
            
        Returns:
            List of alerts
        """
        alerts = []
        
        # Health score alerts
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
        
        # Safety reliability alerts
        if assessment.safety_reliability['score'] < self.alert_thresholds['safety_reliability_min']:
            alerts.append({
                "level": "critical",
                "message": f"Safety reliability below minimum threshold: {assessment.safety_reliability['score']}",
                "timestamp": assessment.timestamp
            })
        
        # Learning effectiveness alerts
        if assessment.learning_effectiveness['score'] < self.alert_thresholds['learning_effectiveness_min']:
            alerts.append({
                "level": "warning",
                "message": f"Learning effectiveness below optimal: {assessment.learning_effectiveness['score']}",
                "timestamp": assessment.timestamp
            })
        
        return alerts
    
    def run_continuous_validation(self, interval: Optional[float] = None, max_iterations: Optional[int] = None):
        """
        Run continuous health validation
        
        Args:
            interval: Validation interval in seconds
            max_iterations: Maximum number of iterations (None for infinite)
        """
        if interval is None:
            interval = self.config.get('validation_interval', 60.0)
        
        iteration = 0
        self.logger.info(f"Starting continuous health validation with {interval}s interval")
        
        try:
            while max_iterations is None or iteration < max_iterations:
                # Collect metrics
                metrics = self.collect_system_metrics()
                
                # Perform validation
                assessment = self.validate_health(metrics)
                
                # Generate report
                report = self.get_validation_report(assessment)
                
                # Check for alerts
                alerts = self.check_alerts(assessment)
                
                # Log results
                if alerts:
                    for alert in alerts:
                        self.logger.warning(f"ALERT [{alert['level'].upper()}]: {alert['message']}")
                
                # Print report
                print(json.dumps(report, indent=2))
                
                iteration += 1
                
                if max_iterations is None or iteration < max_iterations:
                    time.sleep(float(interval))
                    
        except KeyboardInterrupt:
            self.logger.info("Continuous validation stopped by user")
        except Exception as e:
            self.logger.error(f"Error in continuous validation: {e}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Adaptive Safety Health Validator')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--continuous', action='store_true', help='Run continuous validation')
    parser.add_argument('--interval', type=float, default=60.0, help='Validation interval in seconds')
    parser.add_argument('--iterations', type=int, help='Maximum number of iterations')
    parser.add_argument('--mock', action='store_true', help='Use mock metrics for testing')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = SystemHealthValidator(args.config)
    
    if args.continuous:
        # Run continuous validation
        validator.run_continuous_validation(args.interval, args.iterations)
    else:
        # Run single validation
        metrics = validator.collect_system_metrics()
        assessment = validator.validate_health(metrics)
        report = validator.get_validation_report(assessment)
        
        # Print results
        print("Adaptive Safety Health Validation Report")
        print("=" * 50)
        print(json.dumps(report, indent=2))
        
        # Check for alerts
        alerts = validator.check_alerts(assessment)
        if alerts:
            print("\nAlerts:")
            for alert in alerts:
                print(f"[{alert['level'].upper()}] {alert['message']}")


if __name__ == "__main__":
    main() 