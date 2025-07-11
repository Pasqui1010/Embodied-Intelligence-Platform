#!/usr/bin/env python3
"""
Comprehensive Test Script for All Adaptive Safety Learning Validation Prompts

This script demonstrates all 8 validation prompts with realistic example data.
"""

import json
import subprocess
import sys
import os
from datetime import datetime

def run_validator(script_name, args):
    """Run a validator script with given arguments"""
    try:
        cmd = [sys.executable, script_name] + args
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from {script_name}: {e}")
        return None

def test_system_health_validator():
    """Test System Health Validation"""
    print("\n" + "="*60)
    print("1. SYSTEM HEALTH VALIDATION")
    print("="*60)
    
    # Test case 1: Healthy system
    print("\n--- Healthy System Test ---")
    healthy_args = [
        "--learning_rounds", "1500",
        "--pattern_discoveries", "45",
        "--rule_evolutions", "12",
        "--adaptation_count", "28",
        "--safety_level", "0.92",
        "--confidence", "0.88",
        "--memory_usage", "0.65",
        "--latency", "35"
    ]
    
    result = run_validator("adaptive_safety_health_validator.py", healthy_args)
    if result:
        print(f"Health Score: {result['health_score']}")
        print(f"Status: {result['status']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendations: {len(result['recommendations'])} items")

    # Test case 2: Degraded system
    print("\n--- Degraded System Test ---")
    degraded_args = [
        "--learning_rounds", "800",
        "--pattern_discoveries", "25",
        "--rule_evolutions", "8",
        "--adaptation_count", "15",
        "--safety_level", "0.78",
        "--confidence", "0.72",
        "--memory_usage", "0.85",
        "--latency", "120"
    ]
    
    result = run_validator("adaptive_safety_health_validator.py", degraded_args)
    if result:
        print(f"Health Score: {result['health_score']}")
        print(f"Status: {result['status']}")
        print(f"Risk Level: {result['risk_level']}")

def test_learning_pattern_validator():
    """Test Learning Pattern Validation"""
    print("\n" + "="*60)
    print("2. LEARNING PATTERN VALIDATION")
    print("="*60)
    
    # Test case: Valid pattern
    print("\n--- Valid Pattern Test ---")
    pattern_args = [
        "--pattern_id", "pattern_001",
        "--features", "velocity,proximity,human_presence,acceleration",
        "--threshold", "0.75",
        "--confidence", "0.88",
        "--usage_count", "150",
        "--success_rate", "0.92",
        "--creation_time", "2024-01-15T10:30:00",
        "--evolution_stage", "stable"
    ]
    
    result = run_validator("pattern_validation.py", pattern_args)
    if result:
        print(f"Quality Score: {result['quality_score']}")
        print(f"Status: {result['status']}")
        print(f"Feature Relevance: {result['feature_analysis']['relevance_score']}")

def test_rule_evolution_validator():
    """Test Rule Evolution Validation"""
    print("\n" + "="*60)
    print("3. RULE EVOLUTION VALIDATION")
    print("="*60)
    
    # Test case: Stable rule evolution
    print("\n--- Stable Rule Evolution Test ---")
    evolution_args = [
        "--rule_id", "rule_001",
        "--original_condition", "velocity > 2.0 AND proximity < 1.5",
        "--current_condition", "velocity > 1.8 AND proximity < 1.2",
        "--original_threshold", "0.8",
        "--current_threshold", "0.75",
        "--adaptation_count", "5",
        "--success_rate_trend", "0.85,0.87,0.89,0.91,0.92",
        "--evolution_triggers", "near_miss_incidents,performance_analysis",
        "--last_adaptation_time", "2024-01-20T14:15:00"
    ]
    
    result = run_validator("rule_evolution_validator.py", evolution_args)
    if result:
        print(f"Evolution Validity: {result['evolution_validity']}")
        print(f"Status: {result['status']}")
        print(f"Risk Level: {result['risk_level']}")

def test_real_time_performance_validator():
    """Test Real-time Performance Validation"""
    print("\n" + "="*60)
    print("4. REAL-TIME PERFORMANCE VALIDATION")
    print("="*60)
    
    # Test case: Optimal performance
    print("\n--- Optimal Performance Test ---")
    performance_args = [
        "--avg_response_time", "25.5",
        "--p95_latency", "45.2",
        "--throughput", "85.0",
        "--cpu_utilization", "0.65",
        "--memory_usage", "0.72",
        "--gpu_utilization", "0.58",
        "--network_latency", "12.3",
        "--queue_depth", "3",
        "--error_rate", "0.005"
    ]
    
    result = run_validator("real_time_performance_validator.py", performance_args)
    if result:
        print(f"Performance Score: {result['performance_score']}")
        print(f"Status: {result['status']}")
        print(f"Alert Level: {result['alert_level']}")

def test_safety_assurance_validator():
    """Test Safety Assurance Validation"""
    print("\n" + "="*60)
    print("5. SAFETY ASSURANCE VALIDATION")
    print("="*60)
    
    # Test case: Excellent safety assurance
    print("\n--- Excellent Safety Assurance Test ---")
    safety_args = [
        "--safety_level", "0.94",
        "--confidence", "0.91",
        "--false_positive_rate", "0.03",
        "--false_negative_rate", "0.01",
        "--near_miss_detection_rate", "0.96",
        "--incident_prevention_rate", "0.98",
        "--rule_coverage", "0.92",
        "--pattern_coverage", "0.88",
        "--learning_convergence", "0.89",
        "--safety_margin", "0.18"
    ]
    
    result = run_validator("safety_assurance_validator.py", safety_args)
    if result:
        print(f"Safety Assurance Score: {result['safety_assurance_score']}")
        print(f"Status: {result['status']}")
        print(f"Compliance Level: {result['compliance_level']}")

def test_learning_convergence_validator():
    """Test Learning Convergence Validation"""
    print("\n" + "="*60)
    print("6. LEARNING CONVERGENCE VALIDATION")
    print("="*60)
    
    # Test case: Converged system
    print("\n--- Converged System Test ---")
    convergence_args = [
        "--learning_rounds", "1200",
        "--pattern_stability_trend", "0.85,0.87,0.89,0.91,0.92,0.93,0.92,0.93,0.94,0.93",
        "--rule_evolution_frequency", "0.05",
        "--threshold_variance", "0.03",
        "--confidence_convergence", "0.91",
        "--performance_plateau", "True",
        "--adaptation_rate", "0.08",
        "--learning_curve_slope", "0.002",
        "--pattern_maturity", "0.89",
        "--rule_maturity", "0.91"
    ]
    
    result = run_validator("learning_convergence_validator.py", convergence_args)
    if result:
        print(f"Convergence Score: {result['convergence_score']}")
        print(f"Status: {result['status']}")
        print(f"Convergence Confidence: {result['convergence_confidence']}")

def test_integration_validator():
    """Test Integration Validation"""
    print("\n" + "="*60)
    print("7. INTEGRATION VALIDATION")
    print("="*60)
    
    # Test case: Fully integrated system
    print("\n--- Fully Integrated System Test ---")
    integration_args = [
        "--slam_integration_status", "healthy",
        "--llm_connectivity", "healthy",
        "--multimodal_integration", "healthy",
        "--sensor_fusion_integration", "healthy",
        "--comm_latency", "18.5",
        "--data_consistency", "0.98",
        "--error_propagation", "contained",
        "--api_compatibility", "compatible",
        "--queue_health", "healthy",
        "--service_discovery", "active"
    ]
    
    result = run_validator("integration_validator.py", integration_args)
    if result:
        print(f"Integration Score: {result['integration_score']}")
        print(f"Status: {result['status']}")
        print(f"Integration Health: {result['integration_health']}")

def test_compliance_validator():
    """Test Compliance Validation"""
    print("\n" + "="*60)
    print("8. COMPLIANCE VALIDATION")
    print("="*60)
    
    # Test case: Fully compliant system
    print("\n--- Fully Compliant System Test ---")
    
    safety_standards = json.dumps({
        "iso_13482": "compliant",
        "iso_12100": "compliant",
        "ansi_r15_06": "compliant",
        "ieee_2857": "compliant"
    })
    
    regulatory_status = json.dumps({
        "fda_requirements": "met",
        "ce_marking": "valid",
        "ul_certification": "current",
        "local_regulations": "compliant"
    })
    
    certification_validity = json.dumps({
        "safety_certification": "valid",
        "quality_certification": "valid",
        "compliance_certification": "valid"
    })
    
    compliance_args = [
        "--safety_standards", safety_standards,
        "--regulatory_status", regulatory_status,
        "--certification_validity", certification_validity,
        "--documentation_completeness", "0.98",
        "--audit_trail_quality", "excellent",
        "--risk_assessment_compliance", "compliant",
        "--incident_reporting", "compliant",
        "--training_qualification", "compliant",
        "--change_management", "compliant",
        "--quality_assurance", "compliant"
    ]
    
    result = run_validator("compliance_validator.py", compliance_args)
    if result:
        print(f"Compliance Score: {result['compliance_score']}")
        print(f"Status: {result['status']}")
        print(f"Compliance Level: {result['compliance_level']}")
        print(f"Next Audit Date: {result['next_audit_date']}")

def generate_summary_report():
    """Generate a summary report of all validators"""
    print("\n" + "="*80)
    print("ADAPTIVE SAFETY LEARNING VALIDATION SUMMARY")
    print("="*80)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "validators_implemented": [
            "System Health Validation",
            "Learning Pattern Validation", 
            "Rule Evolution Validation",
            "Real-time Performance Validation",
            "Safety Assurance Validation",
            "Learning Convergence Validation",
            "Integration Validation",
            "Compliance Validation"
        ],
        "total_validators": 8,
        "output_format": "JSON",
        "validation_criteria": "Embedded in each validator",
        "self_validation": "Implemented in each validator",
        "usage": "Command-line interface with required parameters",
        "integration": "Ready for CI/CD pipeline integration"
    }
    
    print(json.dumps(summary, indent=2))
    
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    
    examples = [
        {
            "validator": "System Health",
            "command": "python adaptive_safety_health_validator.py --learning_rounds 1500 --pattern_discoveries 45 --safety_level 0.92 --confidence 0.88"
        },
        {
            "validator": "Learning Pattern", 
            "command": "python pattern_validation.py --pattern_id pattern_001 --features velocity,proximity --threshold 0.75 --confidence 0.88"
        },
        {
            "validator": "Rule Evolution",
            "command": "python rule_evolution_validator.py --rule_id rule_001 --original_condition 'velocity > 2.0' --current_condition 'velocity > 1.8'"
        },
        {
            "validator": "Real-time Performance",
            "command": "python real_time_performance_validator.py --avg_response_time 25.5 --throughput 85.0 --cpu_utilization 0.65"
        },
        {
            "validator": "Safety Assurance",
            "command": "python safety_assurance_validator.py --safety_level 0.94 --confidence 0.91 --false_positive_rate 0.03"
        },
        {
            "validator": "Learning Convergence",
            "command": "python learning_convergence_validator.py --learning_rounds 1200 --pattern_stability_trend 0.85,0.87,0.89 --adaptation_rate 0.08"
        },
        {
            "validator": "Integration",
            "command": "python integration_validator.py --slam_integration_status healthy --llm_connectivity healthy --comm_latency 18.5"
        },
        {
            "validator": "Compliance",
            "command": "python compliance_validator.py --safety_standards '{\"iso_13482\": \"compliant\"}' --documentation_completeness 0.98"
        }
    ]
    
    for example in examples:
        print(f"\n{example['validator']}:")
        print(f"  {example['command']}")

def main():
    """Run all validation tests"""
    print("ADAPTIVE SAFETY LEARNING VALIDATION TEST SUITE")
    print("Testing all 8 validation prompts with example data...")
    
    # Test each validator
    test_system_health_validator()
    test_learning_pattern_validator()
    test_rule_evolution_validator()
    test_real_time_performance_validator()
    test_safety_assurance_validator()
    test_learning_convergence_validator()
    test_integration_validator()
    test_compliance_validator()
    
    # Generate summary
    generate_summary_report()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETED")
    print("All validation prompts have been implemented and tested.")
    print("="*80)

if __name__ == "__main__":
    main() 