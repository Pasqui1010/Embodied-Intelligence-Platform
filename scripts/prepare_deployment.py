#!/usr/bin/env python3
"""
Deployment Preparation Script for Adaptive Safety Orchestration

This script validates the ASO system and prepares it for production deployment
by checking all critical components and generating deployment reports.
"""

import os
import sys
import subprocess
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """Validates the ASO system for deployment"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.validation_results = {}
        self.deployment_status = {
            'overall_status': 'PENDING',
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
    
    def validate_system_architecture(self) -> bool:
        """Validate system architecture and dependencies"""
        logger.info("Validating system architecture...")
        
        try:
            # Check required packages
            required_packages = [
                'torch', 'numpy', 'scikit-learn', 'matplotlib', 'pandas',
                'rclpy', 'std_msgs', 'eip_interfaces'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                    logger.info(f"✓ {package} available")
                except ImportError:
                    missing_packages.append(package)
                    logger.error(f"✗ {package} missing")
            
            if missing_packages:
                self.deployment_status['critical_issues'].append(
                    f"Missing required packages: {', '.join(missing_packages)}"
                )
                return False
            
            # Check ROS 2 environment
            try:
                result = subprocess.run(
                    ['ros2', '--version'], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                if result.returncode == 0:
                    logger.info(f"✓ ROS 2 available: {result.stdout.strip()}")
                else:
                    raise Exception("ROS 2 not properly configured")
            except Exception as e:
                self.deployment_status['critical_issues'].append(
                    f"ROS 2 environment issue: {str(e)}"
                )
                return False
            
            # Check package structure
            required_files = [
                'intelligence/eip_adaptive_safety/package.xml',
                'intelligence/eip_adaptive_safety/CMakeLists.txt',
                'intelligence/eip_adaptive_safety/setup.py',
                'intelligence/eip_adaptive_safety/eip_adaptive_safety/adaptive_learning_engine.py',
                'intelligence/eip_adaptive_safety/eip_adaptive_safety/adaptive_safety_node.py',
                'intelligence/eip_adaptive_safety/eip_adaptive_safety/thread_safe_containers.py'
            ]
            
            missing_files = []
            for file_path in required_files:
                if not (self.project_root / file_path).exists():
                    missing_files.append(file_path)
                    logger.error(f"✗ Missing file: {file_path}")
                else:
                    logger.info(f"✓ Found file: {file_path}")
            
            if missing_files:
                self.deployment_status['critical_issues'].append(
                    f"Missing required files: {', '.join(missing_files)}"
                )
                return False
            
            self.validation_results['architecture'] = 'PASS'
            return True
            
        except Exception as e:
            logger.error(f"Architecture validation failed: {e}")
            self.deployment_status['critical_issues'].append(
                f"Architecture validation error: {str(e)}"
            )
            return False
    
    def validate_thread_safety(self) -> bool:
        """Validate thread safety implementation"""
        logger.info("Validating thread safety...")
        
        try:
            # Import thread-safe components
            sys.path.append(str(self.project_root / 'intelligence/eip_adaptive_safety'))
            
            from eip_adaptive_safety.thread_safe_containers import (
                ThreadSafeExperienceBuffer, ThreadSafeRuleRegistry,
                InputValidator, ErrorRecoveryManager
            )
            
            # Test thread-safe containers
            buffer = ThreadSafeExperienceBuffer(maxlen=100)
            registry = ThreadSafeRuleRegistry(max_rules=10)
            validator = InputValidator()
            recovery = ErrorRecoveryManager()
            
            # Test concurrent access
            import threading
            import time
            
            def test_buffer_operations():
                for i in range(50):
                    buffer.append(f"test_data_{i}")
                    time.sleep(0.001)
            
            def test_registry_operations():
                for i in range(10):
                    registry.add_rule(f"rule_{i}", {"test": i})
                    time.sleep(0.001)
            
            # Start concurrent threads
            threads = []
            for _ in range(4):
                thread = threading.Thread(target=test_buffer_operations)
                threads.append(thread)
                thread.start()
            
            for _ in range(2):
                thread = threading.Thread(target=test_registry_operations)
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify results
            buffer_size = len(buffer)
            registry_size = len(registry)
            
            logger.info(f"✓ Thread-safe buffer size: {buffer_size}")
            logger.info(f"✓ Thread-safe registry size: {registry_size}")
            
            if buffer_size > 0 and registry_size > 0:
                self.validation_results['thread_safety'] = 'PASS'
                return True
            else:
                raise Exception("Thread-safe containers not working properly")
                
        except Exception as e:
            logger.error(f"Thread safety validation failed: {e}")
            self.deployment_status['critical_issues'].append(
                f"Thread safety validation error: {str(e)}"
            )
            return False
    
    def validate_input_validation(self) -> bool:
        """Validate input validation implementation"""
        logger.info("Validating input validation...")
        
        try:
            from eip_adaptive_safety.thread_safe_containers import InputValidator
            
            validator = InputValidator()
            
            # Test malicious inputs
            malicious_inputs = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "'; DROP TABLE users; --",
                "A" * 2000,
                None,
                123,
                {"malicious": "object"}
            ]
            
            all_blocked = True
            for malicious_input in malicious_inputs:
                is_valid, _ = validator.validate_task_plan(malicious_input)
                if is_valid:
                    logger.error(f"✗ Malicious input not blocked: {malicious_input}")
                    all_blocked = False
            
            if all_blocked:
                logger.info("✓ All malicious inputs properly blocked")
                self.validation_results['input_validation'] = 'PASS'
                return True
            else:
                raise Exception("Input validation not working properly")
                
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            self.deployment_status['critical_issues'].append(
                f"Input validation error: {str(e)}"
            )
            return False
    
    def validate_error_recovery(self) -> bool:
        """Validate error recovery implementation"""
        logger.info("Validating error recovery...")
        
        try:
            from eip_adaptive_safety.thread_safe_containers import ErrorRecoveryManager
            
            recovery = ErrorRecoveryManager(max_retries=3)
            
            # Register recovery strategy
            def test_recovery(error, *args, **kwargs):
                return "recovered"
            
            recovery.register_recovery_strategy("test_error", test_recovery)
            
            # Test recovery
            def failing_operation():
                raise RuntimeError("Test error")
            
            success, result = recovery.execute_with_recovery(
                failing_operation, "test_error"
            )
            
            if success and result == "recovered":
                logger.info("✓ Error recovery working properly")
                self.validation_results['error_recovery'] = 'PASS'
                return True
            else:
                raise Exception("Error recovery not working properly")
                
        except Exception as e:
            logger.error(f"Error recovery validation failed: {e}")
            self.deployment_status['critical_issues'].append(
                f"Error recovery validation error: {str(e)}"
            )
            return False
    
    def validate_performance(self) -> bool:
        """Validate performance characteristics"""
        logger.info("Validating performance...")
        
        try:
            from eip_adaptive_safety.thread_safe_containers import ThreadSafeExperienceBuffer
            
            buffer = ThreadSafeExperienceBuffer(maxlen=1000)
            
            # Test performance
            start_time = time.time()
            
            # Add experiences rapidly
            for i in range(1000):
                buffer.append(f"experience_{i}")
            
            end_time = time.time()
            processing_time = end_time - start_time
            throughput = 1000 / processing_time
            
            logger.info(f"✓ Processing time: {processing_time:.3f}s")
            logger.info(f"✓ Throughput: {throughput:.1f} exp/s")
            
            # Performance requirements
            if throughput > 100:  # At least 100 exp/s
                self.validation_results['performance'] = 'PASS'
                return True
            else:
                self.deployment_status['warnings'].append(
                    f"Low throughput: {throughput:.1f} exp/s (target: >100 exp/s)"
                )
                return False
                
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            self.deployment_status['critical_issues'].append(
                f"Performance validation error: {str(e)}"
            )
            return False
    
    def validate_memory_management(self) -> bool:
        """Validate memory management"""
        logger.info("Validating memory management...")
        
        try:
            import psutil
            import gc
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            from eip_adaptive_safety.thread_safe_containers import ThreadSafeExperienceBuffer
            
            buffer = ThreadSafeExperienceBuffer(maxlen=1000)
            
            # Add many experiences
            for i in range(2000):
                buffer.append(f"large_experience_{i}" * 100)  # Large data
            
            # Force garbage collection
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = final_memory - initial_memory
            
            logger.info(f"✓ Initial memory: {initial_memory:.1f}MB")
            logger.info(f"✓ Final memory: {final_memory:.1f}MB")
            logger.info(f"✓ Memory growth: {memory_growth:.1f}MB")
            
            # Check buffer size is limited
            buffer_size = len(buffer)
            logger.info(f"✓ Buffer size: {buffer_size}")
            
            if memory_growth < 500 and buffer_size <= 1000:  # Reasonable limits
                self.validation_results['memory_management'] = 'PASS'
                return True
            else:
                self.deployment_status['warnings'].append(
                    f"High memory growth: {memory_growth:.1f}MB"
                )
                return False
                
        except Exception as e:
            logger.error(f"Memory management validation failed: {e}")
            self.deployment_status['critical_issues'].append(
                f"Memory management validation error: {str(e)}"
            )
            return False
    
    def run_all_validations(self) -> bool:
        """Run all validation checks"""
        logger.info("Starting comprehensive deployment validation...")
        
        validations = [
            ('architecture', self.validate_system_architecture),
            ('thread_safety', self.validate_thread_safety),
            ('input_validation', self.validate_input_validation),
            ('error_recovery', self.validate_error_recovery),
            ('performance', self.validate_performance),
            ('memory_management', self.validate_memory_management)
        ]
        
        all_passed = True
        
        for name, validation_func in validations:
            try:
                if validation_func():
                    logger.info(f"✓ {name.replace('_', ' ').title()} validation PASSED")
                else:
                    logger.error(f"✗ {name.replace('_', ' ').title()} validation FAILED")
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ {name.replace('_', ' ').title()} validation ERROR: {e}")
                all_passed = False
        
        # Update overall status
        if all_passed and not self.deployment_status['critical_issues']:
            self.deployment_status['overall_status'] = 'READY'
        elif self.deployment_status['critical_issues']:
            self.deployment_status['overall_status'] = 'BLOCKED'
        else:
            self.deployment_status['overall_status'] = 'WARNINGS'
        
        return all_passed
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report"""
        report = {
            'timestamp': time.time(),
            'project_root': str(self.project_root),
            'validation_results': self.validation_results,
            'deployment_status': self.deployment_status,
            'recommendations': []
        }
        
        # Generate recommendations
        if self.deployment_status['critical_issues']:
            report['recommendations'].append(
                "CRITICAL: Fix all critical issues before deployment"
            )
        
        if self.deployment_status['warnings']:
            report['recommendations'].append(
                "WARNING: Address warnings for optimal performance"
            )
        
        if self.deployment_status['overall_status'] == 'READY':
            report['recommendations'].extend([
                "Deploy with monitoring enabled",
                "Set up alerting for critical metrics",
                "Prepare rollback plan",
                "Document deployment procedures"
            ])
        
        return report
    
    def save_deployment_report(self, report: Dict[str, Any], filename: str = None):
        """Save deployment report to file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"deployment_report_{timestamp}.json"
        
        report_path = self.project_root / 'reports' / filename
        
        # Create reports directory if it doesn't exist
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Deployment report saved to: {report_path}")
        return report_path

def main():
    """Main deployment preparation function"""
    print("="*60)
    print("ADAPTIVE SAFETY ORCHESTRATION - DEPLOYMENT PREPARATION")
    print("="*60)
    
    # Get project root
    project_root = os.getcwd()
    logger.info(f"Project root: {project_root}")
    
    # Create validator
    validator = DeploymentValidator(project_root)
    
    # Run validations
    success = validator.run_all_validations()
    
    # Generate report
    report = validator.generate_deployment_report()
    
    # Print summary
    print("\n" + "="*60)
    print("DEPLOYMENT VALIDATION SUMMARY")
    print("="*60)
    print(f"Overall Status: {report['deployment_status']['overall_status']}")
    print(f"Validation Results: {report['validation_results']}")
    
    if report['deployment_status']['critical_issues']:
        print("\nCRITICAL ISSUES:")
        for issue in report['deployment_status']['critical_issues']:
            print(f"  - {issue}")
    
    if report['deployment_status']['warnings']:
        print("\nWARNINGS:")
        for warning in report['deployment_status']['warnings']:
            print(f"  - {warning}")
    
    if report['recommendations']:
        print("\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    # Save report
    report_path = validator.save_deployment_report(report)
    
    print(f"\nDeployment report saved to: {report_path}")
    
    if success:
        print("\n✓ Deployment preparation completed successfully!")
        return 0
    else:
        print("\n✗ Deployment preparation failed - check critical issues!")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 