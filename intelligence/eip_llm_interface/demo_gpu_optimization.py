#!/usr/bin/env python3
"""
GPU Optimization Demo for Safety-Embedded LLM

This script demonstrates the GPU optimization features including:
- Performance comparison between CPU and GPU
- Memory management and optimization
- Real-time performance monitoring
- Benchmarking and metrics collection
"""

import time
import json
import argparse
import logging
from typing import List, Dict, Any

from eip_llm_interface.gpu_optimized_llm import GPUOptimizedSafetyLLM, GPUConfig
from eip_llm_interface.performance_monitor import PerformanceMonitor, PerformanceBenchmark


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_test_commands() -> List[str]:
    """Create test commands for benchmarking"""
    return [
        "move to the kitchen and avoid obstacles",
        "stop immediately if you detect a human nearby",
        "maintain safe velocity limits while navigating",
        "check workspace boundaries before moving",
        "perform emergency stop if safety is compromised",
        "navigate to the living room using the safest path",
        "monitor sensor data for potential hazards",
        "adjust speed based on environmental conditions",
        "verify safety constraints before executing movement",
        "report current safety status and any violations"
    ]


def run_performance_comparison(cpu_config: GPUConfig, gpu_config: GPUConfig, num_requests: int = 10):
    """Run performance comparison between CPU and GPU configurations"""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON: CPU vs GPU")
    print("="*60)
    
    test_commands = create_test_commands()
    
    # Test CPU configuration
    print("\n🔧 Testing CPU Configuration...")
    cpu_llm = GPUOptimizedSafetyLLM(gpu_config=cpu_config)
    cpu_monitor = PerformanceMonitor(cpu_llm)
    cpu_benchmark = PerformanceBenchmark(cpu_llm, cpu_monitor)
    
    cpu_start = time.time()
    cpu_results = cpu_benchmark.run_benchmark(num_requests, test_commands)
    cpu_total_time = time.time() - cpu_start
    
    print(f"CPU Results:")
    print(f"  - Total Time: {cpu_total_time:.3f}s")
    print(f"  - Requests/sec: {cpu_results['requests_per_second']:.2f}")
    print(f"  - Avg Processing Time: {cpu_results['average_processing_time']:.3f}s")
    print(f"  - Success Rate: {cpu_results['success_rate']:.1%}")
    
    # Test GPU configuration (if available)
    print("\n🚀 Testing GPU Configuration...")
    gpu_llm = GPUOptimizedSafetyLLM(gpu_config=gpu_config)
    gpu_monitor = PerformanceMonitor(gpu_llm)
    gpu_benchmark = PerformanceBenchmark(gpu_llm, gpu_monitor)
    
    gpu_start = time.time()
    gpu_results = gpu_benchmark.run_benchmark(num_requests, test_commands)
    gpu_total_time = time.time() - gpu_start
    
    print(f"GPU Results:")
    print(f"  - Total Time: {gpu_total_time:.3f}s")
    print(f"  - Requests/sec: {gpu_results['requests_per_second']:.2f}")
    print(f"  - Avg Processing Time: {gpu_results['average_processing_time']:.3f}s")
    print(f"  - Success Rate: {gpu_results['success_rate']:.1%}")
    
    # Calculate improvements
    if cpu_results['requests_per_second'] > 0:
        speedup = gpu_results['requests_per_second'] / cpu_results['requests_per_second']
        print(f"\n📊 Performance Improvement:")
        print(f"  - Speedup: {speedup:.2f}x")
        print(f"  - Time Reduction: {((cpu_total_time - gpu_total_time) / cpu_total_time * 100):.1f}%")
    
    # Cleanup
    cpu_monitor.stop_monitoring()
    gpu_monitor.stop_monitoring()
    cpu_llm.shutdown()
    gpu_llm.shutdown()
    
    return cpu_results, gpu_results


def demonstrate_memory_management():
    """Demonstrate memory management features"""
    print("\n" + "="*60)
    print("MEMORY MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    # Create GPU-optimized LLM
    config = GPUConfig(device="auto", batch_size=4)
    llm = GPUOptimizedSafetyLLM(gpu_config=config)
    monitor = PerformanceMonitor(llm)
    
    print(f"\n🔧 Device: {llm.device}")
    print(f"📊 Batch Size: {config.batch_size}")
    print(f"💾 Max Memory: {config.max_memory_mb}MB")
    
    # Generate multiple responses to test memory management
    print("\n📝 Generating responses to test memory management...")
    test_commands = create_test_commands()
    
    for i, command in enumerate(test_commands[:5]):
        print(f"  Processing request {i+1}/5: {command[:50]}...")
        
        start_time = time.time()
        response = llm.generate_safe_response(command)
        processing_time = time.time() - start_time
        
        print(f"    ✅ Completed in {processing_time:.3f}s")
        print(f"    🛡️ Safety Score: {response.safety_score:.2f}")
        
        # Show memory usage
        memory_usage = llm.memory_manager.get_memory_usage()
        print(f"    💾 GPU Memory: {memory_usage['allocated_mb']:.1f}MB")
    
    # Demonstrate memory optimization
    print("\n🧹 Running memory optimization...")
    llm.optimize_memory()
    
    # Show final memory state
    final_memory = llm.memory_manager.get_memory_usage()
    print(f"📊 Final Memory Usage: {final_memory['allocated_mb']:.1f}MB")
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print(f"📈 Performance Summary:")
    print(f"  - Total Requests: {summary['total_requests']}")
    print(f"  - Average Processing Time: {summary['average_processing_time']:.3f}s")
    print(f"  - Error Rate: {summary['error_rate']:.1%}")
    
    # Cleanup
    monitor.stop_monitoring()
    llm.shutdown()


def demonstrate_real_time_monitoring():
    """Demonstrate real-time performance monitoring"""
    print("\n" + "="*60)
    print("REAL-TIME PERFORMANCE MONITORING")
    print("="*60)
    
    # Create GPU-optimized LLM with monitoring
    config = GPUConfig(device="auto", batch_size=2)
    llm = GPUOptimizedSafetyLLM(gpu_config=config)
    monitor = PerformanceMonitor(llm)
    
    print(f"\n🔧 Monitoring active: {monitor.monitoring_active}")
    print(f"📊 Alert thresholds:")
    for threshold, value in monitor.thresholds.items():
        print(f"  - {threshold}: {value}")
    
    # Simulate real-time requests
    print("\n📝 Simulating real-time requests...")
    test_commands = create_test_commands()
    
    for i in range(8):
        command = test_commands[i % len(test_commands)]
        print(f"\n  Request {i+1}: {command[:40]}...")
        
        # Generate response
        response = llm.generate_safe_response(command)
        
        # Record metrics
        monitor.record_request(
            request_id=f"demo_{i}",
            processing_time=response.execution_time,
            safety_score=response.safety_score,
            success=True
        )
        
        print(f"    ⏱️ Processing Time: {response.execution_time:.3f}s")
        print(f"    🛡️ Safety Score: {response.safety_score:.2f}")
        
        # Show current metrics
        summary = monitor.get_performance_summary()
        print(f"    📊 Current RPS: {summary.get('requests_per_second', 0):.2f}")
        
        # Small delay to simulate real-time processing
        time.sleep(0.5)
    
    # Show final monitoring results
    print("\n📈 Final Monitoring Results:")
    final_summary = monitor.get_performance_summary()
    for key, value in final_summary.items():
        if isinstance(value, (int, float)):
            print(f"  - {key}: {value}")
    
    # Show alerts if any
    if monitor.alerts_history:
        print(f"\n⚠️ Alerts Generated: {len(monitor.alerts_history)}")
        for alert in monitor.alerts_history[-3:]:  # Show last 3 alerts
            print(f"  - [{alert.severity.upper()}] {alert.message}")
    
    # Cleanup
    monitor.stop_monitoring()
    llm.shutdown()


def run_comprehensive_benchmark(num_requests: int = 50):
    """Run comprehensive benchmark with detailed analysis"""
    print("\n" + "="*60)
    print("COMPREHENSIVE BENCHMARK")
    print("="*60)
    
    # Create configurations
    cpu_config = GPUConfig(device="cpu", batch_size=1)
    gpu_config = GPUConfig(device="auto", batch_size=4)
    
    # Run benchmark
    cpu_results, gpu_results = run_performance_comparison(cpu_config, gpu_config, num_requests)
    
    # Detailed analysis
    print("\n📊 Detailed Analysis:")
    
    # Performance metrics
    print(f"CPU Performance:")
    print(f"  - P95 Latency: {cpu_results.get('p95_processing_time', 0):.3f}s")
    print(f"  - P99 Latency: {cpu_results.get('p99_processing_time', 0):.3f}s")
    print(f"  - Memory Usage: {cpu_results.get('device_usage', {}).get('gpu_utilization', 0):.1f}MB")
    
    print(f"GPU Performance:")
    print(f"  - P95 Latency: {gpu_results.get('p95_processing_time', 0):.3f}s")
    print(f"  - P99 Latency: {gpu_results.get('p99_processing_time', 0):.3f}s")
    print(f"  - Memory Usage: {gpu_results.get('device_usage', {}).get('gpu_utilization', 0):.1f}MB")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if gpu_results['requests_per_second'] > cpu_results['requests_per_second'] * 1.5:
        print("  ✅ GPU optimization provides significant performance improvement")
    elif gpu_results['requests_per_second'] > cpu_results['requests_per_second']:
        print("  ⚠️ GPU optimization provides moderate improvement")
    else:
        print("  ❌ GPU optimization may not be beneficial for this workload")
    
    return cpu_results, gpu_results


def export_results(cpu_results: Dict[str, Any], gpu_results: Dict[str, Any], filename: str):
    """Export benchmark results to JSON file"""
    export_data = {
        'timestamp': time.time(),
        'cpu_results': cpu_results,
        'gpu_results': gpu_results,
        'comparison': {
            'speedup': gpu_results['requests_per_second'] / max(cpu_results['requests_per_second'], 1),
            'time_reduction': ((cpu_results.get('total_time', 0) - gpu_results.get('total_time', 0)) / 
                              max(cpu_results.get('total_time', 1), 1)) * 100
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"\n💾 Results exported to: {filename}")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="GPU Optimization Demo for Safety-Embedded LLM")
    parser.add_argument("--demo", choices=["performance", "memory", "monitoring", "benchmark", "all"], 
                       default="all", help="Demo to run")
    parser.add_argument("--requests", type=int, default=20, help="Number of requests for benchmark")
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print("🚀 GPU Optimization Demo for Safety-Embedded LLM")
    print("="*60)
    
    try:
        if args.demo in ["performance", "all"]:
            run_performance_comparison(
                GPUConfig(device="cpu", batch_size=1),
                GPUConfig(device="auto", batch_size=4),
                args.requests
            )
        
        if args.demo in ["memory", "all"]:
            demonstrate_memory_management()
        
        if args.demo in ["monitoring", "all"]:
            demonstrate_real_time_monitoring()
        
        if args.demo in ["benchmark", "all"]:
            cpu_results, gpu_results = run_comprehensive_benchmark(args.requests)
            
            if args.export:
                export_results(cpu_results, gpu_results, args.export)
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logging.error(f"Demo error: {e}", exc_info=True)


if __name__ == "__main__":
    main() 