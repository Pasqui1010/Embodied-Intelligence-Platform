#!/usr/bin/env python3
"""
Real-time Performance Validation Script

Implements the Real-time Performance Validation Prompt for adaptive safety learning systems.
"""

import json
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import numpy as np

@dataclass
class PerformanceMetrics:
    avg_response_time: float
    p95_latency: float
    throughput: float
    cpu_utilization: float
    memory_usage: float
    gpu_utilization: float
    network_latency: float
    queue_depth: int
    error_rate: float

@dataclass
class PerformanceReport:
    performance_score: int
    status: str
    latency_analysis: Dict[str, Any]
    throughput_analysis: Dict[str, Any]
    resource_analysis: Dict[str, Any]
    bottleneck_analysis: Dict[str, Any]
    scalability_assessment: Dict[str, Any]
    recommendations: List[str]
    alert_level: str

class RealTimePerformanceValidator:
    def __init__(self):
        # Performance thresholds from the prompt
        self.response_time_max = 100.0  # ms
        self.throughput_headroom = 0.5  # 50% headroom
        self.resource_utilization_max = 0.8  # 80%
        self.error_rate_max = 0.01  # 1%

    def analyze_latency(self, avg_response_time: float, p95_latency: float) -> Dict[str, Any]:
        """Analyze response time performance"""
        # Calculate response time score
        if avg_response_time <= 50:
            response_time_score = 1.0
        elif avg_response_time <= 100:
            response_time_score = 0.8
        elif avg_response_time <= 200:
            response_time_score = 0.5
        else:
            response_time_score = 0.2
        
        # Determine latency distribution
        if p95_latency <= avg_response_time * 1.5:
            latency_distribution = "normal"
        elif p95_latency <= avg_response_time * 2.0:
            latency_distribution = "slightly_skewed"
        else:
            latency_distribution = "highly_skewed"
        
        # Identify bottlenecks
        bottlenecks = []
        if avg_response_time > self.response_time_max:
            bottlenecks.append("Response time exceeds real-time requirements")
        if p95_latency > avg_response_time * 2.0:
            bottlenecks.append("High latency variance indicates processing issues")
        
        return {
            "response_time_score": round(response_time_score, 2),
            "latency_distribution": latency_distribution,
            "bottlenecks": bottlenecks,
            "avg_response_time_ms": round(avg_response_time, 1),
            "p95_latency_ms": round(p95_latency, 1)
        }

    def analyze_throughput(self, throughput: float, queue_depth: int) -> Dict[str, Any]:
        """Analyze system capacity and efficiency"""
        # Estimate capacity utilization (assuming target throughput of 100 req/sec)
        target_throughput = 100.0
        capacity_utilization = min(1.0, throughput / target_throughput)
        
        # Calculate efficiency score
        if capacity_utilization <= 0.5:
            efficiency_score = 1.0  # Under-utilized
        elif capacity_utilization <= 0.7:
            efficiency_score = 0.9  # Good utilization
        elif capacity_utilization <= 0.8:
            efficiency_score = 0.7  # High utilization
        else:
            efficiency_score = 0.5  # Over-utilized
        
        # Assess scaling headroom
        if capacity_utilization <= 0.5:
            scaling_headroom = "excessive"
        elif capacity_utilization <= 0.7:
            scaling_headroom = "adequate"
        elif capacity_utilization <= 0.8:
            scaling_headroom = "limited"
        else:
            scaling_headroom = "insufficient"
        
        return {
            "capacity_utilization": round(capacity_utilization, 2),
            "efficiency_score": round(efficiency_score, 2),
            "scaling_headroom": scaling_headroom,
            "target_throughput": target_throughput,
            "current_throughput": round(throughput, 1)
        }

    def analyze_resources(self, cpu_utilization: float, memory_usage: float, 
                         gpu_utilization: float) -> Dict[str, Any]:
        """Analyze resource usage optimization"""
        # Calculate efficiency scores
        cpu_efficiency = 1.0 - (cpu_utilization / self.resource_utilization_max)
        memory_efficiency = 1.0 - (memory_usage / self.resource_utilization_max)
        gpu_efficiency = 1.0 - (gpu_utilization / self.resource_utilization_max)
        
        # Identify optimization opportunities
        optimization_opportunities = []
        if cpu_utilization > self.resource_utilization_max:
            optimization_opportunities.append("CPU utilization exceeds optimal levels")
        if memory_usage > self.resource_utilization_max:
            optimization_opportunities.append("Memory usage requires optimization")
        if gpu_utilization > self.resource_utilization_max:
            optimization_opportunities.append("GPU utilization needs monitoring")
        
        return {
            "cpu_efficiency": round(cpu_efficiency, 2),
            "memory_efficiency": round(memory_efficiency, 2),
            "gpu_efficiency": round(gpu_efficiency, 2),
            "optimization_opportunities": optimization_opportunities,
            "cpu_utilization": round(cpu_utilization, 2),
            "memory_usage": round(memory_usage, 2),
            "gpu_utilization": round(gpu_utilization, 2)
        }

    def identify_bottlenecks(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Identify performance limiting factors"""
        bottlenecks = []
        mitigation_strategies = []
        
        # Check response time bottlenecks
        if metrics.avg_response_time > self.response_time_max:
            bottlenecks.append("High response time")
            mitigation_strategies.append("Optimize processing pipeline")
        
        # Check throughput bottlenecks
        if metrics.queue_depth > 10:
            bottlenecks.append("Queue depth too high")
            mitigation_strategies.append("Increase processing capacity")
        
        # Check resource bottlenecks
        if metrics.cpu_utilization > self.resource_utilization_max:
            bottlenecks.append("CPU bottleneck")
            mitigation_strategies.append("Optimize CPU-intensive operations")
        
        if metrics.memory_usage > self.resource_utilization_max:
            bottlenecks.append("Memory bottleneck")
            mitigation_strategies.append("Implement memory optimization")
        
        if metrics.gpu_utilization > self.resource_utilization_max:
            bottlenecks.append("GPU bottleneck")
            mitigation_strategies.append("Optimize GPU utilization")
        
        # Check error rate
        if metrics.error_rate > self.error_rate_max:
            bottlenecks.append("High error rate")
            mitigation_strategies.append("Investigate error sources")
        
        return {
            "primary_bottleneck": bottlenecks[0] if bottlenecks else None,
            "secondary_bottlenecks": bottlenecks[1:] if len(bottlenecks) > 1 else [],
            "mitigation_strategies": mitigation_strategies,
            "bottleneck_count": len(bottlenecks)
        }

    def assess_scalability(self, metrics: PerformanceMetrics, 
                          throughput_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess system scaling capabilities"""
        # Calculate scaling factor based on current utilization
        capacity_utilization = throughput_analysis.get("capacity_utilization", 0.5)
        scaling_factor = 1.0 / max(capacity_utilization, 0.1)
        
        # Determine current capacity
        if capacity_utilization <= 0.5:
            current_capacity = "excessive"
        elif capacity_utilization <= 0.7:
            current_capacity = "sufficient"
        elif capacity_utilization <= 0.8:
            current_capacity = "adequate"
        else:
            current_capacity = "insufficient"
        
        # Identify scaling bottlenecks
        scaling_bottlenecks = []
        if metrics.cpu_utilization > 0.8:
            scaling_bottlenecks.append("CPU scaling limit")
        if metrics.memory_usage > 0.8:
            scaling_bottlenecks.append("Memory scaling limit")
        if metrics.gpu_utilization > 0.8:
            scaling_bottlenecks.append("GPU scaling limit")
        
        # Assess scaling readiness
        if len(scaling_bottlenecks) == 0 and capacity_utilization <= 0.7:
            scaling_readiness = "ready"
        elif len(scaling_bottlenecks) <= 1:
            scaling_readiness = "limited"
        else:
            scaling_readiness = "constrained"
        
        return {
            "scaling_readiness": scaling_readiness,
            "bottleneck_identification": scaling_bottlenecks,
            "capacity_planning": "adequate" if scaling_readiness != "constrained" else "insufficient",
            "current_capacity": current_capacity,
            "scaling_factor": round(scaling_factor, 1),
            "recommended_improvements": scaling_bottlenecks
        }

    def compute_performance_score(self, latency_analysis: Dict[str, Any],
                                throughput_analysis: Dict[str, Any],
                                resource_analysis: Dict[str, Any],
                                bottleneck_analysis: Dict[str, Any]) -> int:
        """Compute overall performance score (0-100)"""
        score = 0
        
        # Latency score (30 points)
        latency_score = latency_analysis.get("response_time_score", 0)
        score += int(latency_score * 30)
        
        # Throughput score (25 points)
        throughput_score = throughput_analysis.get("efficiency_score", 0)
        score += int(throughput_score * 25)
        
        # Resource efficiency (25 points)
        resource_scores = [
            resource_analysis.get("cpu_efficiency", 0),
            resource_analysis.get("memory_efficiency", 0),
            resource_analysis.get("gpu_efficiency", 0)
        ]
        avg_resource_efficiency = sum(resource_scores) / len(resource_scores)
        score += int(avg_resource_efficiency * 25)
        
        # Bottleneck penalty (20 points)
        bottleneck_count = bottleneck_analysis.get("bottleneck_count", 0)
        bottleneck_penalty = max(0, bottleneck_count * 5)
        score = max(0, score - bottleneck_penalty)
        
        return min(100, max(0, score))

    def determine_status(self, performance_score: int) -> str:
        """Determine performance status"""
        if performance_score >= 90:
            return "optimal"
        elif performance_score >= 70:
            return "acceptable"
        elif performance_score >= 50:
            return "degraded"
        else:
            return "critical"

    def determine_alert_level(self, performance_score: int, bottleneck_count: int) -> str:
        """Determine alert level"""
        if performance_score < 50 or bottleneck_count > 3:
            return "critical"
        elif performance_score < 70 or bottleneck_count > 1:
            return "warning"
        else:
            return "none"

    def generate_recommendations(self, status: str, bottleneck_analysis: Dict[str, Any],
                               resource_analysis: Dict[str, Any],
                               scalability_assessment: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if status == "optimal":
            recommendations.append("Performance is within acceptable limits")
            recommendations.append("Continue monitoring for trends")
        elif status == "acceptable":
            recommendations.append("Performance is acceptable but could be optimized")
            recommendations.append("Monitor resource utilization trends")
        elif status == "degraded":
            recommendations.append("Performance requires attention")
            recommendations.append("Implement optimization strategies")
        else:  # critical
            recommendations.append("Immediate performance intervention required")
            recommendations.append("Review system architecture")
        
        # Add specific recommendations based on bottlenecks
        mitigation_strategies = bottleneck_analysis.get("mitigation_strategies", [])
        recommendations.extend(mitigation_strategies)
        
        # Add resource-specific recommendations
        optimization_opportunities = resource_analysis.get("optimization_opportunities", [])
        recommendations.extend(optimization_opportunities)
        
        # Add scalability recommendations
        if scalability_assessment.get("scaling_readiness") == "constrained":
            recommendations.append("Consider infrastructure scaling")
        
        return recommendations

    def validate_performance(self, metrics: PerformanceMetrics) -> PerformanceReport:
        """Perform comprehensive performance validation"""
        # Analyze latency
        latency_analysis = self.analyze_latency(metrics.avg_response_time, metrics.p95_latency)
        
        # Analyze throughput
        throughput_analysis = self.analyze_throughput(metrics.throughput, metrics.queue_depth)
        
        # Analyze resources
        resource_analysis = self.analyze_resources(
            metrics.cpu_utilization, metrics.memory_usage, metrics.gpu_utilization
        )
        
        # Identify bottlenecks
        bottleneck_analysis = self.identify_bottlenecks(metrics)
        
        # Assess scalability
        scalability_assessment = self.assess_scalability(metrics, throughput_analysis)
        
        # Compute performance score
        performance_score = self.compute_performance_score(
            latency_analysis, throughput_analysis, resource_analysis, bottleneck_analysis
        )
        
        # Determine status and alert level
        status = self.determine_status(performance_score)
        alert_level = self.determine_alert_level(performance_score, bottleneck_analysis.get("bottleneck_count", 0))
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            status, bottleneck_analysis, resource_analysis, scalability_assessment
        )
        
        return PerformanceReport(
            performance_score=performance_score,
            status=status,
            latency_analysis=latency_analysis,
            throughput_analysis=throughput_analysis,
            resource_analysis=resource_analysis,
            bottleneck_analysis=bottleneck_analysis,
            scalability_assessment=scalability_assessment,
            recommendations=recommendations,
            alert_level=alert_level
        )

def main():
    parser = argparse.ArgumentParser(description="Validate real-time performance for adaptive safety system.")
    parser.add_argument('--avg_response_time', type=float, required=True, help='Average response time in ms')
    parser.add_argument('--p95_latency', type=float, required=True, help='95th percentile latency in ms')
    parser.add_argument('--throughput', type=float, required=True, help='Throughput in requests/sec')
    parser.add_argument('--cpu_utilization', type=float, required=True, help='CPU utilization (0.0-1.0)')
    parser.add_argument('--memory_usage', type=float, required=True, help='Memory usage (0.0-1.0)')
    parser.add_argument('--gpu_utilization', type=float, required=True, help='GPU utilization (0.0-1.0)')
    parser.add_argument('--network_latency', type=float, required=True, help='Network latency in ms')
    parser.add_argument('--queue_depth', type=int, required=True, help='Queue depth')
    parser.add_argument('--error_rate', type=float, required=True, help='Error rate (0.0-1.0)')
    
    args = parser.parse_args()
    
    metrics = PerformanceMetrics(
        avg_response_time=args.avg_response_time,
        p95_latency=args.p95_latency,
        throughput=args.throughput,
        cpu_utilization=args.cpu_utilization,
        memory_usage=args.memory_usage,
        gpu_utilization=args.gpu_utilization,
        network_latency=args.network_latency,
        queue_depth=args.queue_depth,
        error_rate=args.error_rate
    )
    
    validator = RealTimePerformanceValidator()
    report = validator.validate_performance(metrics)
    print(json.dumps(asdict(report), indent=2))

if __name__ == "__main__":
    main() 