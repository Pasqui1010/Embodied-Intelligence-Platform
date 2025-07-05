#!/usr/bin/env python3
"""
SLAM Performance Regression Testing

Compares SLAM system performance between different branches/commits
to detect performance regressions in SLAM components.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import statistics


class SLAMPerformanceTester:
    """SLAM performance regression testing framework"""
    
    def __init__(self, baseline_branch: str, current_branch: str):
        self.baseline_branch = baseline_branch
        self.current_branch = current_branch
        self.results = {
            'baseline': {},
            'current': {},
            'regression_analysis': {}
        }
    
    def run_slam_benchmarks(self, branch: str) -> Dict[str, Any]:
        """Run SLAM benchmarks on specified branch"""
        print(f"Running SLAM benchmarks on {branch}...")
        
        # Checkout the branch
        subprocess.run(['git', 'checkout', branch], check=True)
        
        # Build SLAM packages
        subprocess.run([
            'colcon', 'build', '--packages-select', 'eip_slam'
        ], check=True)
        
        # Run SLAM benchmarks with timing
        benchmark_results = {}
        
        # Test SLAM initialization time
        init_time = self._measure_slam_initialization_time()
        benchmark_results['slam_initialization_time'] = init_time
        
        # Test SLAM processing time
        processing_time = self._measure_slam_processing_time()
        benchmark_results['slam_processing_time'] = processing_time
        
        # Test SLAM memory usage
        memory_usage = self._measure_slam_memory_usage()
        benchmark_results['slam_memory_usage'] = memory_usage
        
        # Test SLAM accuracy (placeholder)
        accuracy = self._measure_slam_accuracy()
        benchmark_results['slam_accuracy'] = accuracy
        
        return benchmark_results
    
    def _measure_slam_initialization_time(self) -> float:
        """Measure SLAM system initialization time"""
        times = []
        for _ in range(5):  # Run 5 times for statistical significance
            start_time = time.time()
            
            # Run SLAM initialization test
            result = subprocess.run([
                'python', '-m', 'pytest', 
                'benchmarks/slam_benchmarks/test_slam_initialization.py',
                '-v', '--tb=no'
            ], capture_output=True, text=True)
            
            end_time = time.time()
            if result.returncode == 0:
                times.append(end_time - start_time)
        
        return statistics.mean(times) if times else float('inf')
    
    def _measure_slam_processing_time(self) -> float:
        """Measure SLAM processing time per frame"""
        times = []
        for _ in range(10):
            start_time = time.time()
            
            # Run SLAM processing test
            result = subprocess.run([
                'python', '-m', 'pytest', 
                'benchmarks/slam_benchmarks/test_slam_processing.py',
                '-v', '--tb=no'
            ], capture_output=True, text=True)
            
            end_time = time.time()
            if result.returncode == 0:
                times.append(end_time - start_time)
        
        return statistics.mean(times) if times else float('inf')
    
    def _measure_slam_memory_usage(self) -> float:
        """Measure SLAM memory usage"""
        # Placeholder for memory measurement
        # In real implementation, this would use psutil or similar
        return 128.0  # MB - placeholder value
    
    def _measure_slam_accuracy(self) -> float:
        """Measure SLAM accuracy"""
        # Placeholder for accuracy measurement
        # In real implementation, this would compare against ground truth
        return 0.95  # 95% accuracy - placeholder value
    
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
                critical_threshold = 50  # 50% degradation is critical for SLAM
                warning_threshold = 20   # 20% degradation is a warning
                
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
            analysis['summary'] = f"🚨 CRITICAL: {len(analysis['critical_regressions'])} critical SLAM performance regressions detected!"
        elif analysis['regressions_detected']:
            analysis['summary'] = f"⚠️  WARNING: {len(analysis['regressions_detected'])} SLAM performance regressions detected"
        elif analysis['improvements_detected']:
            analysis['summary'] = f"✅ IMPROVEMENT: {len(analysis['improvements_detected'])} SLAM performance improvements detected"
        else:
            analysis['summary'] = "✅ NO CHANGE: SLAM performance is within acceptable bounds"
        
        return analysis
    
    def run_regression_test(self) -> Dict[str, Any]:
        """Run complete regression test"""
        print(f"Running SLAM performance regression test...")
        print(f"Baseline: {self.baseline_branch}")
        print(f"Current: {self.current_branch}")
        
        # Run benchmarks on baseline
        self.results['baseline'] = self.run_slam_benchmarks(self.baseline_branch)
        
        # Run benchmarks on current
        self.results['current'] = self.run_slam_benchmarks(self.current_branch)
        
        # Analyze regressions
        self.results['regression_analysis'] = self.analyze_regressions()
        
        return self.results


def main():
    parser = argparse.ArgumentParser(description='SLAM Performance Regression Testing')
    parser.add_argument('--baseline-branch', required=True, help='Baseline branch for comparison')
    parser.add_argument('--current-branch', required=True, help='Current branch to test')
    parser.add_argument('--test-type', default='slam', help='Type of test to run')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = SLAMPerformanceTester(args.baseline_branch, args.current_branch)
    
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
            print("Critical SLAM performance regressions detected!")
            sys.exit(1)
        
    except Exception as e:
        print(f"Error during regression testing: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 