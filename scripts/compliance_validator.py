#!/usr/bin/env python3
"""
Compliance and Regulatory Validation Script

Implements the Compliance and Regulatory Validation Prompt for adaptive safety learning systems.
"""

import json
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import numpy as np
from datetime import datetime, timedelta

@dataclass
class ComplianceMetrics:
    safety_standard_compliance: Dict[str, str]
    regulatory_status: Dict[str, str]
    certification_validity: Dict[str, str]
    documentation_completeness: float
    audit_trail_quality: str
    risk_assessment_compliance: str
    incident_reporting: str
    training_qualification: str
    change_management: str
    quality_assurance: str

@dataclass
class ComplianceReport:
    compliance_score: int
    status: str
    standard_compliance: Dict[str, Any]
    regulatory_status: Dict[str, Any]
    documentation_review: Dict[str, Any]
    process_compliance: Dict[str, Any]
    audit_trail: Dict[str, Any]
    risk_management: Dict[str, Any]
    recommendations: List[str]
    compliance_level: str
    next_audit_date: str

class ComplianceValidator:
    def __init__(self):
        # Compliance thresholds from the prompt
        self.compliance_score_min = 90
        self.documentation_completeness_min = 0.95
        self.audit_trail_quality_min = 0.95

    def assess_standard_compliance(self, safety_standard_compliance: Dict[str, str]) -> Dict[str, Any]:
        """Assess specific standard compliance status"""
        # Define expected standards
        expected_standards = {
            "iso_13482": "Personal care robots",
            "iso_12100": "Risk assessment",
            "ansi_r15_06": "Industrial robots",
            "ieee_2857": "AI safety"
        }
        
        # Assess each standard
        compliance_results = {}
        compliant_count = 0
        total_standards = len(expected_standards)
        
        for standard, description in expected_standards.items():
            status = safety_standard_compliance.get(standard, "unknown")
            compliance_results[standard] = {
                "status": status,
                "description": description,
                "compliant": status == "compliant"
            }
            if status == "compliant":
                compliant_count += 1
        
        # Calculate compliance rate
        compliance_rate = compliant_count / total_standards if total_standards > 0 else 0
        
        # Determine overall compliance
        if compliance_rate >= 0.9:
            overall_compliance = "fully_compliant"
        elif compliance_rate >= 0.7:
            overall_compliance = "mostly_compliant"
        else:
            overall_compliance = "non_compliant"
        
        return {
            "iso_13482": compliance_results.get("iso_13482", {}).get("status", "unknown"),
            "iso_12100": compliance_results.get("iso_12100", {}).get("status", "unknown"),
            "ansi_r15_06": compliance_results.get("ansi_r15_06", {}).get("status", "unknown"),
            "ieee_2857": compliance_results.get("ieee_2857", {}).get("status", "unknown"),
            "compliance_rate": round(compliance_rate, 3),
            "overall_compliance": overall_compliance,
            "compliant_standards": compliant_count,
            "total_standards": total_standards
        }

    def assess_regulatory_status(self, regulatory_status: Dict[str, str]) -> Dict[str, Any]:
        """Assess regulatory requirement fulfillment"""
        # Define expected regulations
        expected_regulations = {
            "fda_requirements": "Food and Drug Administration",
            "ce_marking": "European Conformity",
            "ul_certification": "Underwriters Laboratories",
            "local_regulations": "Local jurisdiction requirements"
        }
        
        # Assess each regulation
        regulatory_results = {}
        met_count = 0
        total_regulations = len(expected_regulations)
        
        for regulation, description in expected_regulations.items():
            status = regulatory_status.get(regulation, "unknown")
            regulatory_results[regulation] = {
                "status": status,
                "description": description,
                "met": status == "met"
            }
            if status == "met":
                met_count += 1
        
        # Calculate fulfillment rate
        fulfillment_rate = met_count / total_regulations if total_regulations > 0 else 0
        
        # Determine overall status
        if fulfillment_rate >= 0.9:
            overall_status = "fully_met"
        elif fulfillment_rate >= 0.7:
            overall_status = "mostly_met"
        else:
            overall_status = "not_met"
        
        return {
            "fda_requirements": regulatory_results.get("fda_requirements", {}).get("status", "unknown"),
            "ce_marking": regulatory_results.get("ce_marking", {}).get("status", "unknown"),
            "ul_certification": regulatory_results.get("ul_certification", {}).get("status", "unknown"),
            "local_regulations": regulatory_results.get("local_regulations", {}).get("status", "unknown"),
            "fulfillment_rate": round(fulfillment_rate, 3),
            "overall_status": overall_status,
            "met_regulations": met_count,
            "total_regulations": total_regulations
        }

    def review_documentation(self, documentation_completeness: float) -> Dict[str, Any]:
        """Review documentation completeness and quality"""
        # Assess completeness
        if documentation_completeness >= 0.98:
            completeness_level = "excellent"
        elif documentation_completeness >= 0.95:
            completeness_level = "good"
        elif documentation_completeness >= 0.9:
            completeness_level = "adequate"
        else:
            completeness_level = "poor"
        
        # Assess quality (simulated based on completeness)
        if documentation_completeness >= 0.95:
            quality = "excellent"
        elif documentation_completeness >= 0.9:
            quality = "good"
        elif documentation_completeness >= 0.8:
            quality = "adequate"
        else:
            quality = "poor"
        
        # Assess accessibility
        if documentation_completeness >= 0.9:
            accessibility = "good"
        elif documentation_completeness >= 0.8:
            accessibility = "adequate"
        else:
            accessibility = "poor"
        
        # Assess maintenance
        if documentation_completeness >= 0.95:
            maintenance = "current"
        elif documentation_completeness >= 0.9:
            maintenance = "mostly_current"
        else:
            maintenance = "needs_update"
        
        return {
            "completeness": round(documentation_completeness, 3),
            "quality": quality,
            "accessibility": accessibility,
            "maintenance": maintenance,
            "completeness_level": completeness_level
        }

    def assess_process_compliance(self, risk_assessment_compliance: str, incident_reporting: str,
                                change_management: str, quality_assurance: str) -> Dict[str, Any]:
        """Assess process and procedure compliance"""
        # Assess each process
        processes = {
            "risk_assessment": risk_assessment_compliance,
            "incident_reporting": incident_reporting,
            "change_management": change_management,
            "quality_assurance": quality_assurance
        }
        
        # Count compliant processes
        compliant_count = sum(1 for status in processes.values() if status == "compliant")
        total_processes = len(processes)
        compliance_rate = compliant_count / total_processes if total_processes > 0 else 0
        
        # Determine overall compliance
        if compliance_rate >= 0.9:
            overall_compliance = "fully_compliant"
        elif compliance_rate >= 0.7:
            overall_compliance = "mostly_compliant"
        else:
            overall_compliance = "non_compliant"
        
        return {
            "risk_assessment": risk_assessment_compliance,
            "incident_reporting": incident_reporting,
            "change_management": change_management,
            "quality_assurance": quality_assurance,
            "compliance_rate": round(compliance_rate, 3),
            "overall_compliance": overall_compliance,
            "compliant_processes": compliant_count,
            "total_processes": total_processes
        }

    def assess_audit_trail(self, audit_trail_quality: str) -> Dict[str, Any]:
        """Assess audit trail completeness and quality"""
        # Parse audit trail quality
        if audit_trail_quality == "excellent":
            completeness = 0.98
            traceability = "excellent"
            retention = "compliant"
            accessibility = "good"
        elif audit_trail_quality == "good":
            completeness = 0.95
            traceability = "good"
            retention = "compliant"
            accessibility = "adequate"
        elif audit_trail_quality == "adequate":
            completeness = 0.9
            traceability = "adequate"
            retention = "mostly_compliant"
            accessibility = "adequate"
        else:
            completeness = 0.7
            traceability = "poor"
            retention = "non_compliant"
            accessibility = "poor"
        
        return {
            "completeness": round(completeness, 3),
            "traceability": traceability,
            "retention": retention,
            "accessibility": accessibility,
            "quality_level": audit_trail_quality
        }

    def assess_risk_management(self, training_qualification: str) -> Dict[str, Any]:
        """Assess risk assessment and management compliance"""
        # Assess training and qualification
        if training_qualification == "compliant":
            training_status = "adequate"
            assessment_frequency = "adequate"
            mitigation_effectiveness = "high"
            monitoring_continuous = "yes"
            review_cycle = "compliant"
        elif training_qualification == "partially_compliant":
            training_status = "moderate"
            assessment_frequency = "moderate"
            mitigation_effectiveness = "medium"
            monitoring_continuous = "partial"
            review_cycle = "mostly_compliant"
        else:
            training_status = "inadequate"
            assessment_frequency = "inadequate"
            mitigation_effectiveness = "low"
            monitoring_continuous = "no"
            review_cycle = "non_compliant"
        
        return {
            "assessment_frequency": assessment_frequency,
            "mitigation_effectiveness": mitigation_effectiveness,
            "monitoring_continuous": monitoring_continuous,
            "review_cycle": review_cycle,
            "training_status": training_status
        }

    def compute_compliance_score(self, standard_compliance: Dict[str, Any],
                               regulatory_status: Dict[str, Any],
                               documentation_review: Dict[str, Any],
                               process_compliance: Dict[str, Any],
                               audit_trail: Dict[str, Any]) -> int:
        """Compute overall compliance score (0-100)"""
        score = 0
        
        # Standard compliance (25 points)
        compliance_rate = standard_compliance.get("compliance_rate", 0)
        score += int(compliance_rate * 25)
        
        # Regulatory status (25 points)
        fulfillment_rate = regulatory_status.get("fulfillment_rate", 0)
        score += int(fulfillment_rate * 25)
        
        # Documentation review (20 points)
        documentation_completeness = documentation_review.get("completeness", 0)
        score += int(documentation_completeness * 20)
        
        # Process compliance (20 points)
        process_compliance_rate = process_compliance.get("compliance_rate", 0)
        score += int(process_compliance_rate * 20)
        
        # Audit trail (10 points)
        audit_completeness = audit_trail.get("completeness", 0)
        score += int(audit_completeness * 10)
        
        return min(100, max(0, score))

    def determine_status(self, compliance_score: int, standard_compliance: Dict[str, Any],
                        regulatory_status: Dict[str, Any]) -> str:
        """Determine compliance status"""
        standard_rate = standard_compliance.get("compliance_rate", 0)
        regulatory_rate = regulatory_status.get("fulfillment_rate", 0)
        
        if compliance_score >= self.compliance_score_min and standard_rate >= 0.9 and regulatory_rate >= 0.9:
            return "fully_compliant"
        elif compliance_score >= 70 and standard_rate >= 0.7 and regulatory_rate >= 0.7:
            return "mostly_compliant"
        else:
            return "non_compliant"

    def determine_compliance_level(self, status: str, compliance_score: int) -> str:
        """Determine compliance level"""
        if status == "fully_compliant" and compliance_score >= 95:
            return "full"
        elif status == "mostly_compliant" and compliance_score >= 80:
            return "partial"
        else:
            return "non-compliant"

    def calculate_next_audit_date(self, compliance_score: int) -> str:
        """Calculate next audit date based on compliance score"""
        base_date = datetime.now()
        
        if compliance_score >= 95:
            # Excellent compliance - audit every 18 months
            next_date = base_date + timedelta(days=18*30)
        elif compliance_score >= 85:
            # Good compliance - audit every 12 months
            next_date = base_date + timedelta(days=12*30)
        elif compliance_score >= 75:
            # Adequate compliance - audit every 6 months
            next_date = base_date + timedelta(days=6*30)
        else:
            # Poor compliance - audit every 3 months
            next_date = base_date + timedelta(days=3*30)
        
        return next_date.strftime("%Y-%m-%d")

    def generate_recommendations(self, status: str, compliance_score: int,
                               standard_compliance: Dict[str, Any],
                               regulatory_status: Dict[str, Any],
                               documentation_review: Dict[str, Any]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if status == "fully_compliant":
            recommendations.append("Maintain current compliance status")
            recommendations.append("Schedule next compliance review")
            recommendations.append("Continue monitoring for regulatory changes")
        elif status == "mostly_compliant":
            recommendations.append("Address remaining compliance gaps")
            recommendations.append("Strengthen documentation and processes")
            recommendations.append("Schedule follow-up compliance review")
        else:  # non_compliant
            recommendations.append("Immediate compliance intervention required")
            recommendations.append("Develop comprehensive compliance plan")
            recommendations.append("Prioritize critical compliance gaps")
        
        # Add standard-specific recommendations
        non_compliant_standards = []
        for standard, status in standard_compliance.items():
            if isinstance(status, str) and status != "compliant" and not standard.startswith("compliance_rate"):
                non_compliant_standards.append(standard)
        
        if non_compliant_standards:
            recommendations.append(f"Address compliance gaps in: {', '.join(non_compliant_standards)}")
        
        # Add regulatory recommendations
        non_met_regulations = []
        for regulation, status in regulatory_status.items():
            if isinstance(status, str) and status != "met" and not regulation.startswith("fulfillment_rate"):
                non_met_regulations.append(regulation)
        
        if non_met_regulations:
            recommendations.append(f"Fulfill regulatory requirements for: {', '.join(non_met_regulations)}")
        
        # Add documentation recommendations
        if documentation_review.get("completeness", 0) < self.documentation_completeness_min:
            recommendations.append("Improve documentation completeness")
        
        if documentation_review.get("maintenance") == "needs_update":
            recommendations.append("Update documentation for new features")
        
        # Add general recommendations
        if compliance_score < self.compliance_score_min:
            recommendations.append("Implement comprehensive compliance improvement plan")
        
        return recommendations

    def validate_compliance(self, metrics: ComplianceMetrics) -> ComplianceReport:
        """Perform comprehensive compliance validation"""
        # Assess standard compliance
        standard_compliance = self.assess_standard_compliance(metrics.safety_standard_compliance)
        
        # Assess regulatory status
        regulatory_status = self.assess_regulatory_status(metrics.regulatory_status)
        
        # Review documentation
        documentation_review = self.review_documentation(metrics.documentation_completeness)
        
        # Assess process compliance
        process_compliance = self.assess_process_compliance(
            metrics.risk_assessment_compliance, metrics.incident_reporting,
            metrics.change_management, metrics.quality_assurance
        )
        
        # Assess audit trail
        audit_trail = self.assess_audit_trail(metrics.audit_trail_quality)
        
        # Assess risk management
        risk_management = self.assess_risk_management(metrics.training_qualification)
        
        # Compute compliance score
        compliance_score = self.compute_compliance_score(
            standard_compliance, regulatory_status, documentation_review,
            process_compliance, audit_trail
        )
        
        # Determine status and level
        status = self.determine_status(compliance_score, standard_compliance, regulatory_status)
        compliance_level = self.determine_compliance_level(status, compliance_score)
        
        # Calculate next audit date
        next_audit_date = self.calculate_next_audit_date(compliance_score)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            status, compliance_score, standard_compliance, regulatory_status, documentation_review
        )
        
        return ComplianceReport(
            compliance_score=compliance_score,
            status=status,
            standard_compliance=standard_compliance,
            regulatory_status=regulatory_status,
            documentation_review=documentation_review,
            process_compliance=process_compliance,
            audit_trail=audit_trail,
            risk_management=risk_management,
            recommendations=recommendations,
            compliance_level=compliance_level,
            next_audit_date=next_audit_date
        )

def main():
    parser = argparse.ArgumentParser(description="Validate compliance for adaptive safety system.")
    parser.add_argument('--safety_standards', type=str, required=True,
                       help='JSON string of safety standard compliance status')
    parser.add_argument('--regulatory_status', type=str, required=True,
                       help='JSON string of regulatory status')
    parser.add_argument('--certification_validity', type=str, required=True,
                       help='JSON string of certification validity')
    parser.add_argument('--documentation_completeness', type=float, required=True,
                       help='Documentation completeness (0.0-1.0)')
    parser.add_argument('--audit_trail_quality', type=str, required=True,
                       choices=['excellent', 'good', 'adequate', 'poor'], help='Audit trail quality')
    parser.add_argument('--risk_assessment_compliance', type=str, required=True,
                       choices=['compliant', 'partially_compliant', 'non_compliant'], help='Risk assessment compliance')
    parser.add_argument('--incident_reporting', type=str, required=True,
                       choices=['compliant', 'partially_compliant', 'non_compliant'], help='Incident reporting compliance')
    parser.add_argument('--training_qualification', type=str, required=True,
                       choices=['compliant', 'partially_compliant', 'non_compliant'], help='Training qualification compliance')
    parser.add_argument('--change_management', type=str, required=True,
                       choices=['compliant', 'partially_compliant', 'non_compliant'], help='Change management compliance')
    parser.add_argument('--quality_assurance', type=str, required=True,
                       choices=['compliant', 'partially_compliant', 'non_compliant'], help='Quality assurance compliance')
    
    args = parser.parse_args()
    
    # Parse JSON strings
    safety_standards = json.loads(args.safety_standards)
    regulatory_status = json.loads(args.regulatory_status)
    certification_validity = json.loads(args.certification_validity)
    
    metrics = ComplianceMetrics(
        safety_standard_compliance=safety_standards,
        regulatory_status=regulatory_status,
        certification_validity=certification_validity,
        documentation_completeness=args.documentation_completeness,
        audit_trail_quality=args.audit_trail_quality,
        risk_assessment_compliance=args.risk_assessment_compliance,
        incident_reporting=args.incident_reporting,
        training_qualification=args.training_qualification,
        change_management=args.change_management,
        quality_assurance=args.quality_assurance
    )
    
    validator = ComplianceValidator()
    report = validator.validate_compliance(metrics)
    print(json.dumps(asdict(report), indent=2))

if __name__ == "__main__":
    main() 