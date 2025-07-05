#!/usr/bin/env python3
"""
Performance Report Generator

Combines SLAM and safety performance results into a comprehensive report
for CI/CD pipeline and deployment decisions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class PerformanceReportGenerator:
    """Generates comprehensive performance reports"""
    
    def __init__(self):
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'summary': '',
            'safety_performance': {},
            'slam_performance': {},
            'safety_response_times': '',
            'slam_performance_metrics': '',
            'recommendations': [],
            'deployment_ready': True,
            'critical_issues': []
        }
    
    def load_slam_results(self, slam_results_path: str):
        """Load SLAM performance results"""
        try:
            with open(slam_results_path, 'r') as f:
                slam_data = json.load(f)
            
            self.report['slam_performance'] = slam_data
            
            # Generate SLAM performance summary
            if 'benchmarks' in slam_data:
                slam_summary = []
                for benchmark in slam_data['benchmarks']:
                    name = benchmark.get('name', 'Unknown')
                    mean_time = benchmark.get('stats', {}).get('mean', 0)
                    slam_summary.append(f"- {name}: {mean_time:.3f}s")
                
                self.report['slam_performance_metrics'] = '\n'.join(slam_summary)
            
        except Exception as e:
            print(f"Warning: Could not load SLAM results from {slam_results_path}: {e}")
            self.report['slam_performance'] = {'error': str(e)}
    
    def load_safety_results(self, safety_results_path: str):
        """Load safety performance results"""
        try:
            with open(safety_results_path, 'r') as f:
                safety_data = json.load(f)
            
            self.report['safety_performance'] = safety_data
            
            # Generate safety response time summary
            if 'regression_analysis' in safety_data:
                analysis = safety_data['regression_analysis']
                
                if analysis.get('critical_regressions'):
                    self.report['deployment_ready'] = False
                    self.report['critical_issues'].extend([
                        f"Critical regression in {reg['metric']}: {reg['degradation_percent']:.1f}% degradation"
                        for reg in analysis['critical_regressions']
                    ])
                
                # Format safety response times
                safety_times = []
                if 'current' in safety_data:
                    current = safety_data['current']
                    for metric, value in current.items():
                        if 'response_time' in metric:
                            safety_times.append(f"- {metric}: {value:.3f}s")
                
                self.report['safety_response_times'] = '\n'.join(safety_times)
            
        except Exception as e:
            print(f"Warning: Could not load safety results from {safety_results_path}: {e}")
            self.report['safety_performance'] = {'error': str(e)}
    
    def load_response_time_results(self, response_time_path: str):
        """Load safety response time benchmark results"""
        try:
            with open(response_time_path, 'r') as f:
                response_data = json.load(f)
            
            # Extract response time metrics
            if 'benchmarks' in response_data:
                response_times = []
                for benchmark in response_data['benchmarks']:
                    name = benchmark.get('name', 'Unknown')
                    mean_time = benchmark.get('stats', {}).get('mean', 0)
                    response_times.append(f"- {name}: {mean_time:.3f}s")
                
                self.report['safety_response_times'] = '\n'.join(response_times)
            
        except Exception as e:
            print(f"Warning: Could not load response time results from {response_time_path}: {e}")
    
    def generate_summary(self):
        """Generate overall performance summary"""
        summary_parts = []
        
        # Safety performance summary
        if self.report['safety_performance'] and 'regression_analysis' in self.report['safety_performance']:
            analysis = self.report['safety_performance']['regression_analysis']
            summary_parts.append(f"Safety: {analysis.get('summary', 'No analysis available')}")
        
        # SLAM performance summary
        if self.report['slam_performance'] and 'benchmarks' in self.report['slam_performance']:
            slam_count = len(self.report['slam_performance']['benchmarks'])
            summary_parts.append(f"SLAM: {slam_count} benchmarks completed")
        
        # Deployment readiness
        if self.report['deployment_ready']:
            summary_parts.append("✅ Deployment Ready")
        else:
            summary_parts.append("❌ Deployment Blocked")
        
        self.report['summary'] = ' | '.join(summary_parts)
    
    def generate_recommendations(self):
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Safety performance recommendations
        if self.report['safety_performance'] and 'regression_analysis' in self.report['safety_performance']:
            analysis = self.report['safety_performance']['regression_analysis']
            
            if analysis.get('critical_regressions'):
                recommendations.append("🚨 INVESTIGATE: Critical safety performance regressions detected")
                recommendations.append("   - Review recent changes to safety-critical components")
                recommendations.append("   - Consider reverting changes if safety is compromised")
            
            if analysis.get('regressions_detected'):
                recommendations.append("⚠️  MONITOR: Performance regressions detected in safety components")
                recommendations.append("   - Monitor safety system performance in production")
                recommendations.append("   - Consider optimization of safety algorithms")
        
        # SLAM performance recommendations
        if self.report['slam_performance'] and 'benchmarks' in self.report['slam_performance']:
            for benchmark in self.report['slam_performance']['benchmarks']:
                mean_time = benchmark.get('stats', {}).get('mean', 0)
                name = benchmark.get('name', 'Unknown')
                
                if mean_time > 1.0:  # More than 1 second
                    recommendations.append(f"🐌 OPTIMIZE: {name} is slow ({mean_time:.3f}s)")
                elif mean_time < 0.1:  # Less than 100ms
                    recommendations.append(f"⚡ EXCELLENT: {name} is very fast ({mean_time:.3f}s)")
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ PERFORMANCE: All systems performing within acceptable bounds")
        
        self.report['recommendations'] = recommendations
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate complete performance report"""
        self.generate_summary()
        self.generate_recommendations()
        
        return self.report


def main():
    parser = argparse.ArgumentParser(description='Generate Performance Report')
    parser.add_argument('--slam-results', help='Path to SLAM performance results JSON')
    parser.add_argument('--safety-results', help='Path to safety performance results JSON')
    parser.add_argument('--response-time-results', help='Path to response time benchmark results JSON')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = PerformanceReportGenerator()
    
    # Load results
    if args.slam_results:
        generator.load_slam_results(args.slam_results)
    
    if args.safety_results:
        generator.load_safety_results(args.safety_results)
    
    if args.response_time_results:
        generator.load_response_time_results(args.response_time_results)
    
    # Generate report
    report = generator.generate_report()
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Performance report generated: {args.output}")
    print(f"Summary: {report['summary']}")
    
    # Print recommendations
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")
    
    # Exit with error if deployment is not ready
    if not report['deployment_ready']:
        print("\n❌ Deployment blocked due to critical issues!")
        sys.exit(1)
    
    print("\n✅ Performance report completed successfully")


if __name__ == '__main__':
    main() 