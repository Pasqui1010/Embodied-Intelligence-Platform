#!/usr/bin/env python3
"""
Integration Validation Script

Implements the Integration Validation Prompt for adaptive safety learning systems.
"""

import json
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import numpy as np

@dataclass
class IntegrationMetrics:
    slam_integration_status: str
    llm_connectivity: str
    multimodal_integration: str
    sensor_fusion_integration: str
    comm_latency: float
    data_consistency: float
    error_propagation: str
    api_compatibility: str
    queue_health: str
    service_discovery: str

@dataclass
class IntegrationReport:
    integration_score: int
    status: str
    connectivity_analysis: Dict[str, Any]
    data_flow_analysis: Dict[str, Any]
    performance_impact: Dict[str, Any]
    error_handling: Dict[str, Any]
    api_compatibility: Dict[str, Any]
    scalability_assessment: Dict[str, Any]
    recommendations: List[str]
    integration_health: str

class IntegrationValidator:
    def __init__(self):
        # Integration thresholds from the prompt
        self.integration_score_min = 85
        self.data_consistency_min = 0.95
        self.comm_latency_max = 50.0  # ms
        self.performance_overhead_max = 0.1  # 10%

    def analyze_connectivity(self, slam_integration_status: str, llm_connectivity: str,
                           multimodal_integration: str, sensor_fusion_integration: str) -> Dict[str, Any]:
        """Analyze component connectivity status"""
        # Assess connection health
        connections = {
            "slam_connection": slam_integration_status,
            "llm_connection": llm_connectivity,
            "multimodal_connection": multimodal_integration,
            "sensor_connection": sensor_fusion_integration
        }
        
        # Count healthy connections
        healthy_count = sum(1 for status in connections.values() if status == "healthy")
        total_connections = len(connections)
        connectivity_score = healthy_count / total_connections
        
        # Determine overall connectivity status
        if connectivity_score == 1.0:
            connectivity_status = "excellent"
        elif connectivity_score >= 0.75:
            connectivity_status = "good"
        elif connectivity_score >= 0.5:
            connectivity_status = "fair"
        else:
            connectivity_status = "poor"
        
        # Identify connection issues
        connection_issues = [name for name, status in connections.items() if status != "healthy"]
        
        return {
            "slam_connection": slam_integration_status,
            "llm_connection": llm_connectivity,
            "multimodal_connection": multimodal_integration,
            "sensor_connection": sensor_fusion_integration,
            "connectivity_score": round(connectivity_score, 3),
            "connectivity_status": connectivity_status,
            "connection_issues": connection_issues,
            "healthy_connections": healthy_count,
            "total_connections": total_connections
        }

    def analyze_data_flow(self, data_consistency: float, comm_latency: float,
                         queue_health: str, service_discovery: str) -> Dict[str, Any]:
        """Analyze data consistency and flow"""
        # Assess data consistency
        if data_consistency >= 0.98:
            data_quality = "excellent"
        elif data_consistency >= 0.95:
            data_quality = "good"
        elif data_consistency >= 0.9:
            data_quality = "adequate"
        else:
            data_quality = "poor"
        
        # Assess data latency
        if comm_latency <= 20:
            data_latency = "excellent"
        elif comm_latency <= 50:
            data_latency = "acceptable"
        elif comm_latency <= 100:
            data_latency = "high"
        else:
            data_latency = "unacceptable"
        
        # Assess synchronization
        if queue_health == "healthy" and service_discovery == "active":
            synchronization = "proper"
        elif queue_health == "healthy" or service_discovery == "active":
            synchronization = "partial"
        else:
            synchronization = "poor"
        
        return {
            "data_consistency": round(data_consistency, 3),
            "data_latency": data_latency,
            "data_quality": data_quality,
            "synchronization": synchronization,
            "comm_latency_ms": round(comm_latency, 1),
            "queue_health": queue_health,
            "service_discovery": service_discovery
        }

    def assess_performance_impact(self, comm_latency: float, data_consistency: float,
                                connectivity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess integration performance effects"""
        # Calculate overhead
        base_latency = 10.0  # assumed base latency
        overhead = max(0, comm_latency - base_latency)
        overhead_percentage = (overhead / base_latency) * 100 if base_latency > 0 else 0
        
        # Assess overhead level
        if overhead_percentage <= 5:
            overhead_level = "minimal"
        elif overhead_percentage <= 10:
            overhead_level = "low"
        elif overhead_percentage <= 20:
            overhead_level = "moderate"
        else:
            overhead_level = "high"
        
        # Calculate latency impact
        latency_impact = f"+{overhead:.1f}ms" if overhead > 0 else "none"
        
        # Estimate throughput impact
        connectivity_score = connectivity_analysis.get("connectivity_score", 0)
        throughput_impact = f"-{(1 - connectivity_score) * 10:.1f}%" if connectivity_score < 1.0 else "none"
        
        # Assess resource impact
        if overhead_percentage <= 10 and data_consistency >= 0.95:
            resource_impact = "acceptable"
        elif overhead_percentage <= 20 and data_consistency >= 0.9:
            resource_impact = "moderate"
        else:
            resource_impact = "high"
        
        return {
            "overhead": overhead_level,
            "latency_impact": latency_impact,
            "throughput_impact": throughput_impact,
            "resource_impact": resource_impact,
            "overhead_percentage": round(overhead_percentage, 1),
            "base_latency_ms": base_latency,
            "total_latency_ms": round(comm_latency, 1)
        }

    def assess_error_handling(self, error_propagation: str, queue_health: str,
                            connectivity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess error propagation and handling"""
        # Assess error propagation
        if error_propagation == "contained":
            propagation_level = "contained"
        elif error_propagation == "limited":
            propagation_level = "limited"
        else:
            propagation_level = "uncontrolled"
        
        # Assess fault tolerance
        healthy_connections = connectivity_analysis.get("healthy_connections", 0)
        total_connections = connectivity_analysis.get("total_connections", 1)
        fault_tolerance_score = healthy_connections / total_connections
        
        if fault_tolerance_score >= 0.9:
            fault_tolerance = "high"
        elif fault_tolerance_score >= 0.7:
            fault_tolerance = "medium"
        else:
            fault_tolerance = "low"
        
        # Assess recovery time
        if queue_health == "healthy" and fault_tolerance_score >= 0.8:
            recovery_time = "fast"
        elif queue_health == "healthy" or fault_tolerance_score >= 0.6:
            recovery_time = "moderate"
        else:
            recovery_time = "slow"
        
        # Assess error isolation
        if error_propagation == "contained" and fault_tolerance_score >= 0.8:
            error_isolation = "effective"
        elif error_propagation == "limited" and fault_tolerance_score >= 0.6:
            error_isolation = "partial"
        else:
            error_isolation = "ineffective"
        
        return {
            "error_propagation": propagation_level,
            "fault_tolerance": fault_tolerance,
            "recovery_time": recovery_time,
            "error_isolation": error_isolation,
            "fault_tolerance_score": round(fault_tolerance_score, 3),
            "queue_health": queue_health
        }

    def assess_api_compatibility(self, api_compatibility: str, service_discovery: str,
                               connectivity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess API compatibility and interface consistency"""
        # Assess version compatibility
        if api_compatibility == "compatible":
            version_compatibility = "compatible"
        elif api_compatibility == "partially_compatible":
            version_compatibility = "partially_compatible"
        else:
            version_compatibility = "incompatible"
        
        # Assess interface consistency
        connection_issues = connectivity_analysis.get("connection_issues", [])
        if len(connection_issues) == 0:
            interface_consistency = "consistent"
        elif len(connection_issues) <= 1:
            interface_consistency = "mostly_consistent"
        else:
            interface_consistency = "inconsistent"
        
        # Assess protocol compliance
        if service_discovery == "active" and version_compatibility == "compatible":
            protocol_compliance = "compliant"
        elif service_discovery == "active" or version_compatibility == "compatible":
            protocol_compliance = "partially_compliant"
        else:
            protocol_compliance = "non_compliant"
        
        return {
            "version_compatibility": version_compatibility,
            "interface_consistency": interface_consistency,
            "protocol_compliance": protocol_compliance,
            "api_compatibility": api_compatibility,
            "service_discovery": service_discovery
        }

    def assess_scalability(self, connectivity_analysis: Dict[str, Any],
                          performance_impact: Dict[str, Any],
                          error_handling: Dict[str, Any]) -> Dict[str, Any]:
        """Assess integration scalability"""
        # Assess scaling readiness
        connectivity_score = connectivity_analysis.get("connectivity_score", 0)
        overhead_percentage = performance_impact.get("overhead_percentage", 0)
        fault_tolerance_score = error_handling.get("fault_tolerance_score", 0)
        
        if connectivity_score >= 0.9 and overhead_percentage <= 10 and fault_tolerance_score >= 0.8:
            scaling_readiness = "ready"
        elif connectivity_score >= 0.7 and overhead_percentage <= 20 and fault_tolerance_score >= 0.6:
            scaling_readiness = "limited"
        else:
            scaling_readiness = "constrained"
        
        # Identify scaling bottlenecks
        bottleneck_identification = []
        if connectivity_score < 0.9:
            bottleneck_identification.append("Connectivity limitations")
        if overhead_percentage > 10:
            bottleneck_identification.append("Performance overhead")
        if fault_tolerance_score < 0.8:
            bottleneck_identification.append("Fault tolerance issues")
        
        # Assess capacity planning
        if scaling_readiness == "ready":
            capacity_planning = "adequate"
        elif scaling_readiness == "limited":
            capacity_planning = "needs_improvement"
        else:
            capacity_planning = "insufficient"
        
        # Calculate scaling factor
        scaling_factor = min(3.0, 1.0 / max(overhead_percentage / 100, 0.1))
        
        return {
            "scaling_readiness": scaling_readiness,
            "bottleneck_identification": bottleneck_identification,
            "capacity_planning": capacity_planning,
            "scaling_factor": round(scaling_factor, 1),
            "recommended_improvements": bottleneck_identification
        }

    def compute_integration_score(self, connectivity_analysis: Dict[str, Any],
                                data_flow_analysis: Dict[str, Any],
                                performance_impact: Dict[str, Any],
                                error_handling: Dict[str, Any],
                                api_compatibility: Dict[str, Any]) -> int:
        """Compute overall integration score (0-100)"""
        score = 0
        
        # Connectivity analysis (25 points)
        connectivity_score = connectivity_analysis.get("connectivity_score", 0)
        score += int(connectivity_score * 25)
        
        # Data flow analysis (25 points)
        data_consistency = data_flow_analysis.get("data_consistency", 0)
        score += int(data_consistency * 25)
        
        # Performance impact (20 points)
        overhead_percentage = performance_impact.get("overhead_percentage", 0)
        performance_score = max(0, 1.0 - (overhead_percentage / 100))
        score += int(performance_score * 20)
        
        # Error handling (20 points)
        fault_tolerance_score = error_handling.get("fault_tolerance_score", 0)
        score += int(fault_tolerance_score * 20)
        
        # API compatibility (10 points)
        version_compatibility = api_compatibility.get("version_compatibility", "incompatible")
        if version_compatibility == "compatible":
            score += 10
        elif version_compatibility == "partially_compatible":
            score += 5
        else:
            score += 0
        
        return min(100, max(0, score))

    def determine_status(self, integration_score: int, connectivity_analysis: Dict[str, Any]) -> str:
        """Determine integration status"""
        connectivity_score = connectivity_analysis.get("connectivity_score", 0)
        
        if integration_score >= self.integration_score_min and connectivity_score >= 0.9:
            return "fully_integrated"
        elif integration_score >= 70 and connectivity_score >= 0.7:
            return "partially_integrated"
        else:
            return "integration_issues"

    def determine_integration_health(self, integration_score: int, connectivity_analysis: Dict[str, Any],
                                   performance_impact: Dict[str, Any]) -> str:
        """Determine integration health"""
        connectivity_score = connectivity_analysis.get("connectivity_score", 0)
        overhead_percentage = performance_impact.get("overhead_percentage", 0)
        
        if integration_score >= 90 and connectivity_score >= 0.95 and overhead_percentage <= 5:
            return "excellent"
        elif integration_score >= 80 and connectivity_score >= 0.8 and overhead_percentage <= 10:
            return "good"
        elif integration_score >= 70 and connectivity_score >= 0.7 and overhead_percentage <= 20:
            return "fair"
        else:
            return "poor"

    def generate_recommendations(self, status: str, connectivity_analysis: Dict[str, Any],
                               performance_impact: Dict[str, Any],
                               scalability_assessment: Dict[str, Any]) -> List[str]:
        """Generate integration recommendations"""
        recommendations = []
        
        if status == "fully_integrated":
            recommendations.append("Integration is functioning well")
            recommendations.append("Continue monitoring for performance optimization")
        elif status == "partially_integrated":
            recommendations.append("Integration needs improvement")
            recommendations.append("Address connectivity and performance issues")
        else:  # integration_issues
            recommendations.append("Integration requires immediate attention")
            recommendations.append("Review system architecture and connectivity")
        
        # Add connectivity recommendations
        connection_issues = connectivity_analysis.get("connection_issues", [])
        for issue in connection_issues:
            recommendations.append(f"Fix {issue} connectivity")
        
        # Add performance recommendations
        overhead_percentage = performance_impact.get("overhead_percentage", 0)
        if overhead_percentage > 10:
            recommendations.append("Optimize communication latency")
        
        # Add scalability recommendations
        if scalability_assessment.get("scaling_readiness") == "constrained":
            recommendations.append("Improve scalability before expansion")
        
        # Add general recommendations
        if connectivity_analysis.get("connectivity_score", 0) < 0.9:
            recommendations.append("Monitor communication latency for optimization")
        
        if performance_impact.get("overhead_percentage", 0) > 5:
            recommendations.append("Consider API versioning for future compatibility")
        
        return recommendations

    def validate_integration(self, metrics: IntegrationMetrics) -> IntegrationReport:
        """Perform comprehensive integration validation"""
        # Analyze connectivity
        connectivity_analysis = self.analyze_connectivity(
            metrics.slam_integration_status, metrics.llm_connectivity,
            metrics.multimodal_integration, metrics.sensor_fusion_integration
        )
        
        # Analyze data flow
        data_flow_analysis = self.analyze_data_flow(
            metrics.data_consistency, metrics.comm_latency,
            metrics.queue_health, metrics.service_discovery
        )
        
        # Assess performance impact
        performance_impact = self.assess_performance_impact(
            metrics.comm_latency, metrics.data_consistency, connectivity_analysis
        )
        
        # Assess error handling
        error_handling = self.assess_error_handling(
            metrics.error_propagation, metrics.queue_health, connectivity_analysis
        )
        
        # Assess API compatibility
        api_compatibility = self.assess_api_compatibility(
            metrics.api_compatibility, metrics.service_discovery, connectivity_analysis
        )
        
        # Assess scalability
        scalability_assessment = self.assess_scalability(
            connectivity_analysis, performance_impact, error_handling
        )
        
        # Compute integration score
        integration_score = self.compute_integration_score(
            connectivity_analysis, data_flow_analysis, performance_impact,
            error_handling, api_compatibility
        )
        
        # Determine status and health
        status = self.determine_status(integration_score, connectivity_analysis)
        integration_health = self.determine_integration_health(
            integration_score, connectivity_analysis, performance_impact
        )
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            status, connectivity_analysis, performance_impact, scalability_assessment
        )
        
        return IntegrationReport(
            integration_score=integration_score,
            status=status,
            connectivity_analysis=connectivity_analysis,
            data_flow_analysis=data_flow_analysis,
            performance_impact=performance_impact,
            error_handling=error_handling,
            api_compatibility=api_compatibility,
            scalability_assessment=scalability_assessment,
            recommendations=recommendations,
            integration_health=integration_health
        )

def main():
    parser = argparse.ArgumentParser(description="Validate integration for adaptive safety system.")
    parser.add_argument('--slam_integration_status', type=str, required=True, 
                       choices=['healthy', 'degraded', 'failed'], help='SLAM integration status')
    parser.add_argument('--llm_connectivity', type=str, required=True,
                       choices=['healthy', 'degraded', 'failed'], help='LLM connectivity status')
    parser.add_argument('--multimodal_integration', type=str, required=True,
                       choices=['healthy', 'degraded', 'failed'], help='Multimodal integration status')
    parser.add_argument('--sensor_fusion_integration', type=str, required=True,
                       choices=['healthy', 'degraded', 'failed'], help='Sensor fusion integration status')
    parser.add_argument('--comm_latency', type=float, required=True, help='Communication latency in ms')
    parser.add_argument('--data_consistency', type=float, required=True, help='Data consistency (0.0-1.0)')
    parser.add_argument('--error_propagation', type=str, required=True,
                       choices=['contained', 'limited', 'uncontrolled'], help='Error propagation level')
    parser.add_argument('--api_compatibility', type=str, required=True,
                       choices=['compatible', 'partially_compatible', 'incompatible'], help='API compatibility')
    parser.add_argument('--queue_health', type=str, required=True,
                       choices=['healthy', 'degraded', 'failed'], help='Queue health status')
    parser.add_argument('--service_discovery', type=str, required=True,
                       choices=['active', 'partial', 'failed'], help='Service discovery status')
    
    args = parser.parse_args()
    
    metrics = IntegrationMetrics(
        slam_integration_status=args.slam_integration_status,
        llm_connectivity=args.llm_connectivity,
        multimodal_integration=args.multimodal_integration,
        sensor_fusion_integration=args.sensor_fusion_integration,
        comm_latency=args.comm_latency,
        data_consistency=args.data_consistency,
        error_propagation=args.error_propagation,
        api_compatibility=args.api_compatibility,
        queue_health=args.queue_health,
        service_discovery=args.service_discovery
    )
    
    validator = IntegrationValidator()
    report = validator.validate_integration(metrics)
    print(json.dumps(asdict(report), indent=2))

if __name__ == "__main__":
    main() 