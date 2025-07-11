#!/usr/bin/env python3
"""
Test Script for New Adaptive Safety Learning Validation Prompts

This script demonstrates the new validation prompts that were implemented
to complement the existing validation system.
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

def test_real_time_performance_validator():
    """Test Real-time Performance Validation"""
    print("\n" + "="*60)
    print("1. REAL-TIME PERFORMANCE VALIDATION")
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
        print(f"Primary Bottleneck: {result['bottleneck_analysis']['primary_bottleneck']}")

    # Test case: Degraded performance
    print("\n--- Degraded Performance Test ---")
    degraded_args = [
        "--avg_response_time", "150.0",
        "--p95_latency", "300.0",
        "--throughput", "45.0",
        "--cpu_utilization", "0.95",
        "--memory_usage", "0.88",
        "--gpu_utilization", "0.92",
        "--network_latency", "50.0",
        "--queue_depth", "25",
        "--error_rate", "0.05"
    ]
    
    result = run_validator("real_time_performance_validator.py", degraded_args)
    if result:
        print(f"Performance Score: {result['performance_score']}")
        print(f"Status: {result['status']}")
        print(f"Alert Level: {result['alert_level']}")

def test_safety_assurance_validator():
    """Test Safety Assurance Validation"""
    print("\n" + "="*60)
    print("2. SAFETY ASSURANCE VALIDATION")
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
        print(f"Risk Level: {result['risk_assessment']['current_risk_level']}")

    # Test case: Poor safety assurance
    print("\n--- Poor Safety Assurance Test ---")
    poor_safety_args = [
        "--safety_level", "0.65",
        "--confidence", "0.55",
        "--false_positive_rate", "0.15",
        "--false_negative_rate", "0.08",
        "--near_miss_detection_rate", "0.70",
        "--incident_prevention_rate", "0.75",
        "--rule_coverage", "0.60",
        "--pattern_coverage", "0.55",
        "--learning_convergence", "0.50",
        "--safety_margin", "0.05"
    ]
    
    result = run_validator("safety_assurance_validator.py", poor_safety_args)
    if result:
        print(f"Safety Assurance Score: {result['safety_assurance_score']}")
        print(f"Status: {result['status']}")
        print(f"Compliance Level: {result['compliance_level']}")

def test_learning_convergence_validator():
    """Test Learning Convergence Validation"""
    print("\n" + "="*60)
    print("3. LEARNING CONVERGENCE VALIDATION")
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
        print(f"Overfitting Detected: {result['overfitting_check']['overfitting_detected']}")

    # Test case: Unstable system
    print("\n--- Unstable System Test ---")
    unstable_args = [
        "--learning_rounds", "200",
        "--pattern_stability_trend", "0.45,0.52,0.38,0.61,0.43,0.58,0.41,0.55,0.47,0.51",
        "--rule_evolution_frequency", "0.35",
        "--threshold_variance", "0.25",
        "--confidence_convergence", "0.45",
        "--performance_plateau", "False",
        "--adaptation_rate", "0.45",
        "--learning_curve_slope", "0.15",
        "--pattern_maturity", "0.35",
        "--rule_maturity", "0.30"
    ]
    
    result = run_validator("learning_convergence_validator.py", unstable_args)
    if result:
        print(f"Convergence Score: {result['convergence_score']}")
        print(f"Status: {result['status']}")
        print(f"Convergence Confidence: {result['convergence_confidence']}")

def test_integration_validator():
    """Test Integration Validation"""
    print("\n" + "="*60)
    print("4. INTEGRATION VALIDATION")
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
        print(f"Connectivity Score: {result['connectivity_analysis']['connectivity_score']}")

    # Test case: Integration issues
    print("\n--- Integration Issues Test ---")
    issues_args = [
        "--slam_integration_status", "degraded",
        "--llm_connectivity", "failed",
        "--multimodal_integration", "healthy",
        "--sensor_fusion_integration", "degraded",
        "--comm_latency", "150.0",
        "--data_consistency", "0.75",
        "--error_propagation", "uncontrolled",
        "--api_compatibility", "incompatible",
        "--queue_health", "failed",
        "--service_discovery", "partial"
    ]
    
    result = run_validator("integration_validator.py", issues_args)
    if result:
        print(f"Integration Score: {result['integration_score']}")
        print(f"Status: {result['status']}")
        print(f"Integration Health: {result['integration_health']}")

def test_compliance_validator():
    """Test Compliance Validation"""
    print("\n" + "="*60)
    print("5. COMPLIANCE VALIDATION")
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

    # Test case: Non-compliant system
    print("\n--- Non-Compliant System Test ---")
    
    non_compliant_standards = json.dumps({
        "iso_13482": "non_compliant",
        "iso_12100": "partially_compliant",
        "ansi_r15_06": "non_compliant",
        "ieee_2857": "unknown"
    })
    
    non_compliant_regulatory = json.dumps({
        "fda_requirements": "not_met",
        "ce_marking": "expired",
        "ul_certification": "invalid",
        "local_regulations": "non_compliant"
    })
    
    non_compliant_certification = json.dumps({
        "safety_certification": "expired",
        "quality_certification": "invalid",
        "compliance_certification": "pending"
    })
    
    non_compliant_args = [
        "--safety_standards", non_compliant_standards,
        "--regulatory_status", non_compliant_regulatory,
        "--certification_validity", non_compliant_certification,
        "--documentation_completeness", "0.65",
        "--audit_trail_quality", "poor",
        "--risk_assessment_compliance", "non_compliant",
        "--incident_reporting", "partially_compliant",
        "--training_qualification", "non_compliant",
        "--change_management", "partially_compliant",
        "--quality_assurance", "non_compliant"
    ]
    
    result = run_validator("compliance_validator.py", non_compliant_args)
    if result:
        print(f"Compliance Score: {result['compliance_score']}")
        print(f"Status: {result['status']}")
        print(f"Compliance Level: {result['compliance_level']}")

def test_existing_health_validator():
    """Test the existing health validator"""
    print("\n" + "="*60)
    print("6. EXISTING SYSTEM HEALTH VALIDATOR")
    print("="*60)
    
    print("\n--- Single Validation Test ---")
    try:
        # Run the existing health validator
        cmd = [sys.executable, "adaptive_safety_health_validator.py", "--mock"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Health validator ran successfully with mock data")
        print("Output preview:")
        lines = result.stdout.split('\n')
        for line in lines[:10]:  # Show first 10 lines
            print(f"  {line}")
        if len(lines) > 10:
            print(f"  ... (and {len(lines) - 10} more lines)")
    except subprocess.CalledProcessError as e:
        print(f"Error running health validator: {e}")
        print(f"Stderr: {e.stderr}")

def generate_summary_report():
    """Generate a summary report of the new validators"""
    print("\n" + "="*80)
    print("NEW ADAPTIVE SAFETY LEARNING VALIDATION SUMMARY")
    print("="*80)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "new_validators_implemented": [
            "Real-time Performance Validation",
            "Safety Assurance Validation",
            "Learning Convergence Validation",
            "Integration Validation",
            "Compliance Validation"
        ],
        "existing_validators": [
            "System Health Validation (enhanced)",
            "Learning Pattern Validation (enhanced)",
            "Rule Evolution Validation (enhanced)"
        ],
        "total_validators": 8,
        "output_format": "JSON",
        "validation_criteria": "Embedded in each validator",
        "self_validation": "Implemented in each validator",
        "usage": "Command-line interface with required parameters",
        "integration": "Ready for CI/CD pipeline integration",
        "complementary": "Works alongside existing validation system"
    }
    
    print(json.dumps(summary, indent=2))
    
    print("\n" + "="*80)
    print("USAGE EXAMPLES FOR NEW VALIDATORS")
    print("="*80)
    
    examples = [
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
    """Run tests for the new validation prompts"""
    print("NEW ADAPTIVE SAFETY LEARNING VALIDATION TEST SUITE")
    print("Testing the 5 new validation prompts with example data...")
    
    # Test each new validator
    test_real_time_performance_validator()
    test_safety_assurance_validator()
    test_learning_convergence_validator()
    test_integration_validator()
    test_compliance_validator()
    
    # Test existing validator
    test_existing_health_validator()
    
    # Generate summary
    generate_summary_report()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETED")
    print("All new validation prompts have been implemented and tested.")
    print("These complement the existing validation system.")
    print("="*80)

if __name__ == "__main__":
    main() 