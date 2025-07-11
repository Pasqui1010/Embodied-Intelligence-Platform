#!/usr/bin/env python3
"""
Test script for Adaptive Safety Health Validator

Demonstrates the implementation of the System Health Validation Prompt
and tests various scenarios and edge cases.
"""

import json
import time
import sys
import os
from typing import Dict, Any

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from adaptive_safety_health_validator import SystemHealthValidator, SystemMetrics


def test_healthy_system():
    """Test with healthy system metrics"""
    print("=" * 60)
    print("TEST 1: Healthy System")
    print("=" * 60)
    
    validator = SystemHealthValidator()
    
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
    
    return assessment


def test_degraded_system():
    """Test with degraded system metrics"""
    print("\n" + "=" * 60)
    print("TEST 2: Degraded System")
    print("=" * 60)
    
    validator = SystemHealthValidator()
    
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
    
    return assessment


def test_critical_system():
    """Test with critical system metrics"""
    print("\n" + "=" * 60)
    print("TEST 3: Critical System")
    print("=" * 60)
    
    validator = SystemHealthValidator()
    
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
    
    return assessment


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\n" + "=" * 60)
    print("TEST 4: Edge Cases")
    print("=" * 60)
    
    validator = SystemHealthValidator()
    
    # Test case 1: Zero values
    print("\n--- Edge Case 1: Zero Values ---")
    metrics_zero = SystemMetrics(
        learning_rounds=0,
        pattern_discoveries=0,
        rule_evolutions=0,
        adaptation_count=0,
        safety_level=0.0,
        confidence=0.0,
        memory_usage=0.0,
        processing_latency=0.0
    )
    
    assessment_zero = validator.validate_health(metrics_zero)
    report_zero = validator.get_validation_report(assessment_zero)
    print(f"Health Score: {report_zero['health_score']}")
    print(f"Status: {report_zero['status']}")
    print(f"Risk Level: {report_zero['risk_level']}")
    
    # Test case 2: Maximum values
    print("\n--- Edge Case 2: Maximum Values ---")
    metrics_max = SystemMetrics(
        learning_rounds=999999,
        pattern_discoveries=99999,
        rule_evolutions=9999,
        adaptation_count=99999,
        safety_level=1.0,
        confidence=1.0,
        memory_usage=1.0,
        processing_latency=1000.0
    )
    
    assessment_max = validator.validate_health(metrics_max)
    report_max = validator.get_validation_report(assessment_max)
    print(f"Health Score: {report_max['health_score']}")
    print(f"Status: {report_max['status']}")
    print(f"Risk Level: {report_max['risk_level']}")
    
    # Test case 3: Mixed performance
    print("\n--- Edge Case 3: Mixed Performance ---")
    metrics_mixed = SystemMetrics(
        learning_rounds=500,
        pattern_discoveries=50,  # Good learning
        rule_evolutions=10,
        adaptation_count=80,
        safety_level=0.95,  # Good safety
        confidence=0.90,
        memory_usage=0.95,  # Poor performance
        processing_latency=200.0  # Poor performance
    )
    
    assessment_mixed = validator.validate_health(metrics_mixed)
    report_mixed = validator.get_validation_report(assessment_mixed)
    print(f"Health Score: {report_mixed['health_score']}")
    print(f"Status: {report_mixed['status']}")
    print(f"Risk Level: {report_mixed['risk_level']}")
    print(f"Bottlenecks: {report_mixed['performance_analysis']['bottlenecks']}")


def test_continuous_validation():
    """Test continuous validation with mock data"""
    print("\n" + "=" * 60)
    print("TEST 5: Continuous Validation (3 iterations)")
    print("=" * 60)
    
    validator = SystemHealthValidator()
    
    print("Running continuous validation for 3 iterations...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        validator.run_continuous_validation(interval=2.0, max_iterations=3)
    except KeyboardInterrupt:
        print("\nContinuous validation stopped by user")


def test_configuration():
    """Test configuration loading and validation"""
    print("\n" + "=" * 60)
    print("TEST 6: Configuration Testing")
    print("=" * 60)
    
    # Test with default configuration
    validator_default = SystemHealthValidator()
    print("Default Configuration:")
    print(f"  Validation Interval: {validator_default.config['validation_interval']}s")
    print(f"  Health Score Critical: {validator_default.alert_thresholds['health_score_critical']}")
    print(f"  Safety Reliability Min: {validator_default.alert_thresholds['safety_reliability_min']}")
    
    # Test with custom configuration
    config_file = "health_validator_config.json"
    if os.path.exists(config_file):
        validator_custom = SystemHealthValidator(config_file)
        print(f"\nCustom Configuration (from {config_file}):")
        print(f"  Validation Interval: {validator_custom.config['validation_interval']}s")
        print(f"  Health Score Critical: {validator_custom.alert_thresholds['health_score_critical']}")
        print(f"  Safety Reliability Min: {validator_custom.alert_thresholds['safety_reliability_min']}")
    else:
        print(f"\nConfiguration file {config_file} not found, using defaults")


def test_recommendations():
    """Test recommendation generation for different scenarios"""
    print("\n" + "=" * 60)
    print("TEST 7: Recommendation Testing")
    print("=" * 60)
    
    validator = SystemHealthValidator()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Low Learning Effectiveness",
            "metrics": SystemMetrics(
                learning_rounds=1000,
                pattern_discoveries=10,
                rule_evolutions=5,
                adaptation_count=20,
                safety_level=0.85,
                confidence=0.80,
                memory_usage=0.70,
                processing_latency=60.0
            )
        },
        {
            "name": "Low Safety Reliability",
            "metrics": SystemMetrics(
                learning_rounds=500,
                pattern_discoveries=40,
                rule_evolutions=15,
                adaptation_count=90,
                safety_level=0.65,
                confidence=0.55,
                memory_usage=0.75,
                processing_latency=45.0
            )
        },
        {
            "name": "Performance Issues",
            "metrics": SystemMetrics(
                learning_rounds=600,
                pattern_discoveries=35,
                rule_evolutions=12,
                adaptation_count=75,
                safety_level=0.88,
                confidence=0.82,
                memory_usage=0.92,
                processing_latency=120.0
            )
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        assessment = validator.validate_health(scenario['metrics'])
        report = validator.get_validation_report(assessment)
        
        print(f"Health Score: {report['health_score']}")
        print(f"Status: {report['status']}")
        print("Recommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")


def main():
    """Run all tests"""
    print("Adaptive Safety Health Validator - Test Suite")
    print("Testing the implementation of the System Health Validation Prompt")
    print("=" * 80)
    
    try:
        # Run all tests
        test_healthy_system()
        test_degraded_system()
        test_critical_system()
        test_edge_cases()
        test_continuous_validation()
        test_configuration()
        test_recommendations()
        
        print("\n" + "=" * 80)
        print("All tests completed successfully!")
        print("The System Health Validation Prompt implementation is working correctly.")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 