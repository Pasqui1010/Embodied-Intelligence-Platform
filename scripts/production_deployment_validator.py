#!/usr/bin/env python3
"""
Production Deployment Validator for Embodied Intelligence Platform

This script implements the production deployment validation prompts to ensure
all safety, performance, and monitoring requirements are met before deployment.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import docker
import yaml


@dataclass
class ValidationResult:
    """Result of a validation check"""
    name: str
    status: str  # PASS, FAIL, WARNING
    message: str
    remediation: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class DeploymentValidationReport:
    """Comprehensive deployment validation report"""
    timestamp: str
    environment_validation: List[ValidationResult]
    safety_validation: List[ValidationResult]
    performance_validation: List[ValidationResult]
    monitoring_validation: List[ValidationResult]
    health_validation: List[ValidationResult]
    critical_issues: List[str]
    deployment_ready: bool
    summary: str
    recommendations: List[str]


class ProductionDeploymentValidator:
    """Implements production deployment validation prompts"""
    
    def __init__(self, config_path: str = "deployment_config.json"):
        """Initialize the validator"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            self.logger.info("Docker client initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")
            self.docker_client = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        default_config = {
            "deployment_mode": "production",
            "gpu_optimization": True,
            "monitoring_enabled": True,
            "safety_validation": True,
            "performance_benchmarking": True,
            "deployment_targets": ["demo-llm", "demo-full-stack"],
            "performance_thresholds": {
                "response_time_ms": 200,
                "throughput_req_per_sec": 10,
                "memory_gb": 2,
                "success_rate_percent": 95
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                self.logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        return default_config
    
    def validate_environment(self) -> List[ValidationResult]:
        """Prompt 1: Environment Validation"""
        self.logger.info("=== Environment Validation ===")
        
        results = []
        
        # Docker check
        docker_result = self._check_docker()
        results.append(docker_result)
        
        # Docker Compose check
        compose_result = self._check_docker_compose()
        results.append(compose_result)
        
        # GPU Support check
        gpu_result = self._check_gpu_support()
        results.append(gpu_result)
        
        # System Resources check
        resources_result = self._check_system_resources()
        results.append(resources_result)
        
        # Network Connectivity check
        network_result = self._check_network()
        results.append(network_result)
        
        # File Permissions check
        permissions_result = self._check_permissions()
        results.append(permissions_result)
        
        return results
    
    def _check_docker(self) -> ValidationResult:
        """Check Docker availability"""
        if self.docker_client is None:
            return ValidationResult(
                name="Docker",
                status="FAIL",
                message="Docker client not available",
                remediation="Install Docker and ensure it's running"
            )
        
        try:
            self.docker_client.ping()
            return ValidationResult(
                name="Docker",
                status="PASS",
                message="Docker is running and accessible"
            )
        except Exception as e:
            return ValidationResult(
                name="Docker",
                status="FAIL",
                message=f"Docker check failed: {e}",
                remediation="Start Docker service and ensure proper permissions"
            )
    
    def _check_docker_compose(self) -> ValidationResult:
        """Check Docker Compose availability"""
        try:
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return ValidationResult(
                    name="Docker Compose",
                    status="PASS",
                    message=f"Docker Compose available: {result.stdout.strip()}"
                )
            else:
                return ValidationResult(
                    name="Docker Compose",
                    status="FAIL",
                    message="Docker Compose not found",
                    remediation="Install Docker Compose"
                )
        except Exception as e:
            return ValidationResult(
                name="Docker Compose",
                status="FAIL",
                message=f"Docker Compose check failed: {e}",
                remediation="Install Docker Compose and ensure it's in PATH"
            )
    
    def _check_gpu_support(self) -> ValidationResult:
        """Check GPU support"""
        if not self.config.get("gpu_optimization", False):
            return ValidationResult(
                name="GPU Support",
                status="PASS",
                message="GPU optimization disabled, skipping check"
            )
        
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.0-base", "nvidia-smi"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return ValidationResult(
                    name="GPU Support",
                    status="PASS",
                    message="NVIDIA GPU support available"
                )
            else:
                return ValidationResult(
                    name="GPU Support",
                    status="FAIL",
                    message="NVIDIA driver not found",
                    remediation="Install NVIDIA drivers and restart the host"
                )
        except Exception as e:
            return ValidationResult(
                name="GPU Support",
                status="WARNING",
                message=f"GPU support check failed: {e}",
                remediation="Install NVIDIA Docker runtime or disable GPU optimization"
            )
    
    def _check_system_resources(self) -> ValidationResult:
        """Check system resources"""
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_gb = disk.free / (1024**3)
            
            details = {
                "memory_gb": round(memory_gb, 2),
                "disk_free_gb": round(disk_gb, 2)
            }
            
            if memory_gb < 8:
                return ValidationResult(
                    name="System Resources",
                    status="WARNING",
                    message=f"System has {memory_gb:.1f}GB RAM (recommended: 8GB+)",
                    details=details
                )
            elif disk_gb < 10:
                return ValidationResult(
                    name="System Resources",
                    status="WARNING",
                    message=f"Only {disk_gb:.1f}GB free disk space (recommended: 10GB+)",
                    details=details
                )
            else:
                return ValidationResult(
                    name="System Resources",
                    status="PASS",
                    message=f"System resources adequate: {memory_gb:.1f}GB RAM, {disk_gb:.1f}GB free disk",
                    details=details
                )
        except Exception as e:
            return ValidationResult(
                name="System Resources",
                status="WARNING",
                message=f"Could not check system resources: {e}"
            )
    
    def _check_network(self) -> ValidationResult:
        """Check network connectivity"""
        try:
            response = requests.get("https://www.google.com", timeout=5)
            if response.status_code == 200:
                return ValidationResult(
                    name="Network Connectivity",
                    status="PASS",
                    message="Network connectivity confirmed"
                )
            else:
                return ValidationResult(
                    name="Network Connectivity",
                    status="WARNING",
                    message=f"Network check returned status {response.status_code}"
                )
        except Exception as e:
            return ValidationResult(
                name="Network Connectivity",
                status="WARNING",
                message=f"Network connectivity check failed: {e}"
            )
    
    def _check_permissions(self) -> ValidationResult:
        """Check file permissions"""
        try:
            test_file = Path("deployment_test.tmp")
            test_file.write_text("test")
            test_file.unlink()
            return ValidationResult(
                name="File Permissions",
                status="PASS",
                message="File permissions adequate"
            )
        except Exception as e:
            return ValidationResult(
                name="File Permissions",
                status="FAIL",
                message=f"File permission check failed: {e}",
                remediation="Check directory permissions and user access"
            )
    
    def validate_safety(self) -> List[ValidationResult]:
        """Prompt 2: Safety Validation"""
        self.logger.info("=== Safety Validation ===")
        
        results = []
        
        if not self.config.get("safety_validation", True):
            results.append(ValidationResult(
                name="Safety Tests",
                status="PASS",
                message="Safety validation disabled in config"
            ))
            return results
        
        try:
            # Run safety benchmarks
            result = subprocess.run([
                "python", "-m", "pytest", "benchmarks/safety_benchmarks/", "-v"
            ], capture_output=True, text=True, timeout=300)
            
            # Parse test results
            test_results = self._parse_pytest_output(result.stdout)
            failed_tests = [test for test in test_results if test["status"] == "FAIL"]
            
            if result.returncode == 0:
                results.append(ValidationResult(
                    name="Safety Tests",
                    status="PASS",
                    message=f"All {len(test_results)} safety tests passed",
                    details={"test_count": len(test_results), "failed_count": 0}
                ))
            else:
                results.append(ValidationResult(
                    name="Safety Tests",
                    status="FAIL",
                    message=f"{len(failed_tests)} safety tests failed",
                    remediation="Review and fix failed safety tests",
                    details={"test_count": len(test_results), "failed_count": len(failed_tests), "failed_tests": failed_tests}
                ))
            
            # Calculate safety score
            safety_score = ((len(test_results) - len(failed_tests)) / len(test_results)) * 100 if test_results else 0
            results.append(ValidationResult(
                name="Safety Score",
                status="PASS" if safety_score >= 95 else "FAIL",
                message=f"Overall Safety Score: {safety_score:.1f}%",
                details={"safety_score": safety_score}
            ))
            
        except subprocess.TimeoutExpired:
            results.append(ValidationResult(
                name="Safety Tests",
                status="FAIL",
                message="Safety tests timed out after 5 minutes",
                remediation="Check for hanging tests or system issues"
            ))
        except Exception as e:
            results.append(ValidationResult(
                name="Safety Tests",
                status="FAIL",
                message=f"Failed to run safety tests: {e}",
                remediation="Check test environment and dependencies"
            ))
        
        return results
    
    def _parse_pytest_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse pytest output to extract test results"""
        tests = []
        lines = output.split('\n')
        
        for line in lines:
            if line.strip().startswith('test_') and ('PASSED' in line or 'FAILED' in line):
                parts = line.split()
                test_name = parts[0]
                status = "PASS" if "PASSED" in line else "FAIL"
                tests.append({
                    "name": test_name,
                    "status": status
                })
        
        return tests
    
    def validate_performance(self) -> List[ValidationResult]:
        """Prompt 3: Performance Benchmarking"""
        self.logger.info("=== Performance Benchmarking ===")
        
        results = []
        
        if not self.config.get("performance_benchmarking", True):
            results.append(ValidationResult(
                name="Performance Tests",
                status="PASS",
                message="Performance benchmarking disabled in config"
            ))
            return results
        
        try:
            # Run performance benchmarks
            result = subprocess.run([
                "python", "intelligence/eip_llm_interface/demo_gpu_optimization.py"
            ], capture_output=True, text=True, timeout=600)
            
            # Parse performance metrics
            metrics = self._parse_performance_output(result.stdout)
            thresholds = self.config.get("performance_thresholds", {})
            
            # Validate each metric
            for metric_name, value in metrics.items():
                threshold = thresholds.get(metric_name)
                if threshold is not None:
                    if metric_name == "response_time_ms":
                        status = "PASS" if value <= threshold else "FAIL"
                        message = f"Average Response Time: {value}ms"
                    elif metric_name == "throughput_req_per_sec":
                        status = "PASS" if value >= threshold else "FAIL"
                        message = f"Throughput: {value} req/s"
                    elif metric_name == "memory_gb":
                        status = "PASS" if value <= threshold else "FAIL"
                        message = f"Memory Usage: {value}GB"
                    elif metric_name == "success_rate_percent":
                        status = "PASS" if value >= threshold else "FAIL"
                        message = f"Success Rate: {value}%"
                    else:
                        status = "PASS"
                        message = f"{metric_name}: {value}"
                    
                    results.append(ValidationResult(
                        name=f"Performance - {metric_name}",
                        status=status,
                        message=message,
                        details={metric_name: value, "threshold": threshold}
                    ))
            
        except subprocess.TimeoutExpired:
            results.append(ValidationResult(
                name="Performance Tests",
                status="FAIL",
                message="Performance benchmarks timed out after 10 minutes",
                remediation="Check for performance issues or system overload"
            ))
        except Exception as e:
            results.append(ValidationResult(
                name="Performance Tests",
                status="FAIL",
                message=f"Failed to run performance benchmarks: {e}",
                remediation="Check benchmark environment and dependencies"
            ))
        
        return results
    
    def _parse_performance_output(self, output: str) -> Dict[str, float]:
        """Parse performance benchmark output to extract metrics"""
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if "Average Response Time:" in line:
                try:
                    value = float(line.split(":")[1].replace("ms", "").strip())
                    metrics["response_time_ms"] = value
                except:
                    pass
            elif "Throughput:" in line:
                try:
                    value = float(line.split(":")[1].replace("req/s", "").strip())
                    metrics["throughput_req_per_sec"] = value
                except:
                    pass
            elif "Memory Usage:" in line:
                try:
                    value = float(line.split(":")[1].replace("GB", "").strip())
                    metrics["memory_gb"] = value
                except:
                    pass
            elif "Success Rate:" in line:
                try:
                    value = float(line.split(":")[1].replace("%", "").strip())
                    metrics["success_rate_percent"] = value
                except:
                    pass
        
        return metrics
    
    def validate_monitoring(self) -> List[ValidationResult]:
        """Prompt 4: Monitoring & Alerting Validation"""
        self.logger.info("=== Monitoring & Alerting Validation ===")
        
        results = []
        
        # Check Prometheus
        prometheus_result = self._check_prometheus()
        results.append(prometheus_result)
        
        # Check Grafana
        grafana_result = self._check_grafana()
        results.append(grafana_result)
        
        # Check metrics collection
        metrics_result = self._check_metrics_collection()
        results.append(metrics_result)
        
        # Test alerting
        alerting_result = self._test_alerting()
        results.append(alerting_result)
        
        return results
    
    def _check_prometheus(self) -> ValidationResult:
        """Check Prometheus availability"""
        try:
            response = requests.get("http://localhost:9090/api/v1/status/config", timeout=5)
            if response.status_code == 200:
                return ValidationResult(
                    name="Prometheus",
                    status="PASS",
                    message="Prometheus is running and accessible"
                )
            else:
                return ValidationResult(
                    name="Prometheus",
                    status="FAIL",
                    message=f"Prometheus returned status {response.status_code}",
                    remediation="Start Prometheus service"
                )
        except Exception as e:
            return ValidationResult(
                name="Prometheus",
                status="FAIL",
                message=f"Prometheus check failed: {e}",
                remediation="Install and start Prometheus"
            )
    
    def _check_grafana(self) -> ValidationResult:
        """Check Grafana availability"""
        try:
            response = requests.get("http://localhost:3000/api/health", timeout=5)
            if response.status_code == 200:
                return ValidationResult(
                    name="Grafana",
                    status="PASS",
                    message="Grafana is running and accessible"
                )
            else:
                return ValidationResult(
                    name="Grafana",
                    status="FAIL",
                    message=f"Grafana returned status {response.status_code}",
                    remediation="Start Grafana service"
                )
        except Exception as e:
            return ValidationResult(
                name="Grafana",
                status="FAIL",
                message=f"Grafana check failed: {e}",
                remediation="Install and start Grafana"
            )
    
    def _check_metrics_collection(self) -> ValidationResult:
        """Check that key metrics are being collected"""
        try:
            response = requests.get("http://localhost:9090/api/v1/query?query=up", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("data", {}).get("result"):
                    return ValidationResult(
                        name="Metrics Collection",
                        status="PASS",
                        message="Key metrics are being collected"
                    )
                else:
                    return ValidationResult(
                        name="Metrics Collection",
                        status="WARNING",
                        message="No metrics data available",
                        remediation="Check metrics collection configuration"
                    )
            else:
                return ValidationResult(
                    name="Metrics Collection",
                    status="FAIL",
                    message="Could not query metrics",
                    remediation="Check Prometheus configuration"
                )
        except Exception as e:
            return ValidationResult(
                name="Metrics Collection",
                status="FAIL",
                message=f"Metrics collection check failed: {e}",
                remediation="Check Prometheus and metrics endpoints"
            )
    
    def _test_alerting(self) -> ValidationResult:
        """Test alerting functionality"""
        try:
            # This would typically test actual alerting rules
            # For now, we'll check if alertmanager is accessible
            response = requests.get("http://localhost:9093/api/v1/status", timeout=5)
            if response.status_code == 200:
                return ValidationResult(
                    name="Alerting",
                    status="PASS",
                    message="AlertManager is accessible"
                )
            else:
                return ValidationResult(
                    name="Alerting",
                    status="WARNING",
                    message="AlertManager not accessible",
                    remediation="Configure AlertManager for alerting"
                )
        except Exception as e:
            return ValidationResult(
                name="Alerting",
                status="WARNING",
                message=f"Alerting test failed: {e}",
                remediation="Set up AlertManager for production alerting"
            )
    
    def validate_health(self) -> List[ValidationResult]:
        """Prompt 5: Deployment Health Check"""
        self.logger.info("=== Deployment Health Check ===")
        
        results = []
        
        try:
            # Check service status
            result = subprocess.run([
                "docker-compose", "ps"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                services = self._parse_docker_compose_ps(result.stdout)
                
                for service_name, status in services.items():
                    if status == "Up":
                        results.append(ValidationResult(
                            name=f"Service - {service_name}",
                            status="PASS",
                            message=f"Service {service_name} is running"
                        ))
                    else:
                        results.append(ValidationResult(
                            name=f"Service - {service_name}",
                            status="FAIL",
                            message=f"Service {service_name} is not running (status: {status})",
                            remediation=f"Check logs for {service_name} and restart if needed"
                        ))
                
                # Check recent logs for errors
                log_result = self._check_service_logs()
                results.extend(log_result)
                
            else:
                results.append(ValidationResult(
                    name="Service Status",
                    status="FAIL",
                    message="Failed to check service status",
                    remediation="Check Docker Compose configuration"
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                name="Service Health",
                status="FAIL",
                message=f"Health check failed: {e}",
                remediation="Check Docker Compose and service configuration"
            ))
        
        return results
    
    def _parse_docker_compose_ps(self, output: str) -> Dict[str, str]:
        """Parse docker-compose ps output"""
        services = {}
        lines = output.split('\n')
        
        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    service_name = parts[0]
                    status = parts[1]
                    services[service_name] = status
        
        return services
    
    def _check_service_logs(self) -> List[ValidationResult]:
        """Check service logs for errors"""
        results = []
        
        try:
            # Check logs for each service
            services = self.config.get("deployment_targets", [])
            for service in services:
                result = subprocess.run([
                    "docker-compose", "logs", "--tail=10", service
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    log_output = result.stdout
                    if "ERROR" in log_output or "error" in log_output:
                        results.append(ValidationResult(
                            name=f"Logs - {service}",
                            status="WARNING",
                            message=f"Found errors in {service} logs",
                            remediation=f"Review {service} logs for issues"
                        ))
                    else:
                        results.append(ValidationResult(
                            name=f"Logs - {service}",
                            status="PASS",
                            message=f"No recent errors in {service} logs"
                        ))
                        
        except Exception as e:
            results.append(ValidationResult(
                name="Service Logs",
                status="WARNING",
                message=f"Could not check service logs: {e}"
            ))
        
        return results
    
    def generate_final_report(self, all_results: Dict[str, List[ValidationResult]]) -> DeploymentValidationReport:
        """Prompt 6: Final Validation & Report"""
        self.logger.info("=== Generating Final Validation Report ===")
        
        # Collect all results
        environment_results = all_results.get("environment", [])
        safety_results = all_results.get("safety", [])
        performance_results = all_results.get("performance", [])
        monitoring_results = all_results.get("monitoring", [])
        health_results = all_results.get("health", [])
        
        # Check for critical issues
        critical_issues = []
        deployment_ready = True
        
        for result_list in all_results.values():
            for result in result_list:
                if result.status == "FAIL":
                    critical_issues.append(f"{result.name}: {result.message}")
                    deployment_ready = False
        
        # Generate summary
        summary_parts = []
        
        env_passed = sum(1 for r in environment_results if r.status == "PASS")
        env_total = len(environment_results)
        summary_parts.append(f"Environment: {env_passed}/{env_total} checks passed")
        
        safety_passed = sum(1 for r in safety_results if r.status == "PASS")
        safety_total = len(safety_results)
        summary_parts.append(f"Safety: {safety_passed}/{safety_total} checks passed")
        
        perf_passed = sum(1 for r in performance_results if r.status == "PASS")
        perf_total = len(performance_results)
        summary_parts.append(f"Performance: {perf_passed}/{perf_total} checks passed")
        
        monitor_passed = sum(1 for r in monitoring_results if r.status == "PASS")
        monitor_total = len(monitoring_results)
        summary_parts.append(f"Monitoring: {monitor_passed}/{monitor_total} checks passed")
        
        health_passed = sum(1 for r in health_results if r.status == "PASS")
        health_total = len(health_results)
        summary_parts.append(f"Health: {health_passed}/{health_total} checks passed")
        
        if deployment_ready:
            summary_parts.append("✅ Deployment Ready")
        else:
            summary_parts.append("❌ Deployment Blocked")
        
        summary = " | ".join(summary_parts)
        
        # Generate recommendations
        recommendations = []
        if not deployment_ready:
            recommendations.append("🚨 CRITICAL: Fix all FAILED checks before deployment")
        
        for result_list in all_results.values():
            for result in result_list:
                if result.status == "WARNING" and result.remediation:
                    recommendations.append(f"⚠️  {result.name}: {result.remediation}")
        
        if not recommendations:
            recommendations.append("✅ All systems ready for production deployment")
        
        return DeploymentValidationReport(
            timestamp=datetime.now().isoformat(),
            environment_validation=environment_results,
            safety_validation=safety_results,
            performance_validation=performance_results,
            monitoring_validation=monitoring_results,
            health_validation=health_results,
            critical_issues=critical_issues,
            deployment_ready=deployment_ready,
            summary=summary,
            recommendations=recommendations
        )
    
    def run_full_validation(self) -> DeploymentValidationReport:
        """Run complete production deployment validation"""
        self.logger.info("Starting production deployment validation...")
        
        all_results = {}
        
        # Run all validation steps
        all_results["environment"] = self.validate_environment()
        all_results["safety"] = self.validate_safety()
        all_results["performance"] = self.validate_performance()
        all_results["monitoring"] = self.validate_monitoring()
        all_results["health"] = self.validate_health()
        
        # Generate final report
        report = self.generate_final_report(all_results)
        
        return report
    
    def save_report(self, report: DeploymentValidationReport, output_path: str):
        """Save validation report to file"""
        report_dict = {
            "timestamp": report.timestamp,
            "summary": report.summary,
            "deployment_ready": report.deployment_ready,
            "critical_issues": report.critical_issues,
            "recommendations": report.recommendations,
            "environment_validation": [
                {
                    "name": r.name,
                    "status": r.status,
                    "message": r.message,
                    "remediation": r.remediation,
                    "details": r.details
                } for r in report.environment_validation
            ],
            "safety_validation": [
                {
                    "name": r.name,
                    "status": r.status,
                    "message": r.message,
                    "remediation": r.remediation,
                    "details": r.details
                } for r in report.safety_validation
            ],
            "performance_validation": [
                {
                    "name": r.name,
                    "status": r.status,
                    "message": r.message,
                    "remediation": r.remediation,
                    "details": r.details
                } for r in report.performance_validation
            ],
            "monitoring_validation": [
                {
                    "name": r.name,
                    "status": r.status,
                    "message": r.message,
                    "remediation": r.remediation,
                    "details": r.details
                } for r in report.monitoring_validation
            ],
            "health_validation": [
                {
                    "name": r.name,
                    "status": r.status,
                    "message": r.message,
                    "remediation": r.remediation,
                    "details": r.details
                } for r in report.health_validation
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        self.logger.info(f"Validation report saved to {output_path}")


def main():
    """Main validation script"""
    parser = argparse.ArgumentParser(description="Production Deployment Validator")
    parser.add_argument("--config", default="deployment_config.json", 
                       help="Path to deployment configuration file")
    parser.add_argument("--output", default="validation_report.json",
                       help="Output JSON file path for validation report")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize validator
    validator = ProductionDeploymentValidator(args.config)
    
    # Run validation
    report = validator.run_full_validation()
    
    # Save report
    validator.save_report(report, args.output)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PRODUCTION DEPLOYMENT VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Summary: {report.summary}")
    print(f"Deployment Ready: {'✅ YES' if report.deployment_ready else '❌ NO'}")
    
    if report.critical_issues:
        print(f"\nCritical Issues:")
        for issue in report.critical_issues:
            print(f"  - {issue}")
    
    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")
    
    print(f"\nDetailed report saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if report.deployment_ready else 1)


if __name__ == "__main__":
    main() 