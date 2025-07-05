#!/usr/bin/env python3
"""
Safety Performance Regression Testing

Compares safety system performance between different branches/commits
to detect performance regressions in safety-critical components.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import statistics


class SafetyPerformanceTester:
    """Safety performance regression testing framework"""
    
    def __init__(self, baseline_branch: str, current_branch: str):
        self.baseline_branch = baseline_branch
        self.current_branch = current_branch
        self.results = {
            'baseline': {},
            'current': {},
            'regression_analysis': {}
        }
    
    def run_safety_benchmarks(self, branch: str) -> Dict[str, Any]:
        """Run safety benchmarks on specified branch"""
        print(f"Running safety benchmarks on {branch}...")
        
        # Checkout the branch
        subprocess.run(['git', 'checkout', branch], check=True)
        
        # Build safety packages
        subprocess.run([
            'colcon', 'build', '--packages-select', 
            'eip_interfaces', 'eip_safety_arbiter', 'eip_slam'
        ], check=True)
        
        # Run safety benchmarks with timing
        benchmark_results = {}
        
        # Test emergency stop response time
        emergency_stop_time = self._measure_emergency_stop_time()
        benchmark_results['emergency_stop_response_time'] = emergency_stop_time
        
        # Test collision detection response time
        collision_detection_time = self._measure_collision_detection_time()
        benchmark_results['collision_detection_response_time'] = collision_detection_time
        
        # Test human proximity detection response time
        human_proximity_time = self._measure_human_proximity_time()
        benchmark_results['human_proximity_response_time'] = human_proximity_time
        
        # Test workspace boundary detection response time
        workspace_boundary_time = self._measure_workspace_boundary_time()
        benchmark_results['workspace_boundary_response_time'] = workspace_boundary_time
        
        # Test velocity limit enforcement response time
        velocity_limit_time = self._measure_velocity_limit_time()
        benchmark_results['velocity_limit_response_time'] = velocity_limit_time
        
        # Test overall safety system throughput
        throughput = self._measure_safety_system_throughput()
        benchmark_results['safety_system_throughput'] = throughput
        
        return benchmark_results
    
    def _measure_emergency_stop_time(self) -> float:
        """Measure emergency stop response time"""
        times = []
        for _ in range(10):  # Run 10 times for statistical significance
            start_time = time.time()
            
            # Run emergency stop test
            result = subprocess.run([
                'python', '-m', 'pytest', 
                'benchmarks/safety_benchmarks/test_emergency_stop.py::test_emergency_stop_response_time',
                '-v', '--tb=no'
            ], capture_output=True, text=True)
            
            end_time = time.time()
            if result.returncode == 0:
                times.append(end_time - start_time)
        
        return statistics.mean(times) if times else float('inf')
    
    def _measure_collision_detection_time(self) -> float:
        """Measure collision detection response time"""
        times = []
        for _ in range(10):
            start_time = time.time()
            
            result = subprocess.run([
                'python', '-m', 'pytest', 
                'benchmarks/safety_benchmarks/test_collision_avoidance.py::test_collision_detection_response_time',
                '-v', '--tb=no'
            ], capture_output=True, text=True)
            
            end_time = time.time()
            if result.returncode == 0:
                times.append(end_time - start_time)
        
        return statistics.mean(times) if times else float('inf')
    
    def _measure_human_proximity_time(self) -> float:
        """Measure human proximity detection response time"""
        times = []
        for _ in range(10):
            start_time = time.time()
            
            result = subprocess.run([
                'python', '-m', 'pytest', 
                'benchmarks/safety_benchmarks/test_human_proximity.py::test_human_proximity_response_time',
                '-v', '--tb=no'
            ], capture_output=True, text=True)
            
            end_time = time.time()
            if result.returncode == 0:
                times.append(end_time - start_time)
        
        return statistics.mean(times) if times else float('inf')
    
    def _measure_workspace_boundary_time(self) -> float:
        """Measure workspace boundary detection response time"""
        times = []
        for _ in range(10):
            start_time = time.time()
            
            result = subprocess.run([
                'python', '-m', 'pytest', 
                'benchmarks/safety_benchmarks/test_workspace_boundary.py::test_workspace_boundary_response_time',
                '-v', '--tb=no'
            ], capture_output=True, text=True)
            
            end_time = time.time()
            if result.returncode == 0:
                times.append(end_time - start_time)
        
        return statistics.mean(times) if times else float('inf')
    
    def _measure_velocity_limit_time(self) -> float:
        """Measure velocity limit enforcement response time"""
        times = []
        for _ in range(10):
            start_time = time.time()
            
            result = subprocess.run([
                'python', '-m', 'pytest', 
                'benchmarks/safety_benchmarks/test_velocity_limits.py::test_velocity_limit_response_time',
                '-v', '--tb=no'
            ], capture_output=True, text=True)
            
            end_time = time.time()
            if result.returncode == 0:
                times.append(end_time - start_time)
        
        return statistics.mean(times) if times else float('inf')
    
    def _measure_safety_system_throughput(self) -> float:
        """Measure overall safety system throughput (operations per second)"""
        # Run a comprehensive safety test and measure throughput
        start_time = time.time()
        
        result = subprocess.run([
            'python', '-m', 'pytest', 
            'benchmarks/safety_benchmarks/',
            '-v', '--tb=no', '--durations=0'
        ], capture_output=True, text=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Count number of safety operations (test cases)
        operation_count = 20  # Approximate number of safety test cases
        
        return operation_count / total_time if total_time > 0 else 0
    
    def analyze_regressions(self) -> Dict[str, Any]:
        """Analyze performance regressions between baseline and current"""
        analysis = {
            'regressions_detected': [],
            'improvements_detected': [],
            'summary': '',
            'critical_regressions': []
        }
        
        baseline = self.results['baseline']
        current = self.results['current']
        
        for metric in baseline.keys():
            if metric in current:
                baseline_value = baseline[metric]
                current_value = current[metric]
                
                if baseline_value == 0 or baseline_value == float('inf'):
                    continue
                
                # Calculate percentage change
                if baseline_value > 0:
                    change_percent = ((current_value - baseline_value) / baseline_value) * 100
                else:
                    change_percent = 0
                
                # Define thresholds
                critical_threshold = 20  # 20% degradation is critical
                warning_threshold = 10   # 10% degradation is a warning
                
                if change_percent > critical_threshold:
                    analysis['critical_regressions'].append({
                        'metric': metric,
                        'baseline': baseline_value,
                        'current': current_value,
                        'degradation_percent': change_percent
                    })
                    analysis['regressions_detected'].append(metric)
                elif change_percent > warning_threshold:
                    analysis['regressions_detected'].append(metric)
                elif change_percent < -10:  # 10% improvement
                    analysis['improvements_detected'].append(metric)
        
        # Generate summary
        if analysis['critical_regressions']:
            analysis['summary'] = f"🚨 CRITICAL: {len(analysis['critical_regressions'])} critical performance regressions detected!"
        elif analysis['regressions_detected']:
            analysis['summary'] = f"⚠️  WARNING: {len(analysis['regressions_detected'])} performance regressions detected"
        elif analysis['improvements_detected']:
            analysis['summary'] = f"✅ IMPROVEMENT: {len(analysis['improvements_detected'])} performance improvements detected"
        else:
            analysis['summary'] = "✅ NO CHANGE: Performance is within acceptable bounds"
        
        return analysis
    
    def run_regression_test(self) -> Dict[str, Any]:
        """Run complete regression test"""
        print(f"Running safety performance regression test...")
        print(f"Baseline: {self.baseline_branch}")
        print(f"Current: {self.current_branch}")
        
        # Run benchmarks on baseline
        self.results['baseline'] = self.run_safety_benchmarks(self.baseline_branch)
        
        # Run benchmarks on current
        self.results['current'] = self.run_safety_benchmarks(self.current_branch)
        
        # Analyze regressions
        self.results['regression_analysis'] = self.analyze_regressions()
        
        return self.results


def main():
    parser = argparse.ArgumentParser(description='Safety Performance Regression Testing')
    parser.add_argument('--baseline-branch', required=True, help='Baseline branch for comparison')
    parser.add_argument('--current-branch', required=True, help='Current branch to test')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = SafetyPerformanceTester(args.baseline_branch, args.current_branch)
    
    try:
        # Run regression test
        results = tester.run_regression_test()
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {args.output}")
        print(f"Summary: {results['regression_analysis']['summary']}")
        
        # Exit with error code if critical regressions detected
        if results['regression_analysis']['critical_regressions']:
            print("Critical performance regressions detected!")
            sys.exit(1)
        
    except Exception as e:
        print(f"Error during regression testing: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 