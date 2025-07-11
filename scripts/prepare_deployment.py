#!/usr/bin/env python3
"""
Production Deployment Script for Embodied Intelligence Platform

This script automates the deployment process for the EIP system including:
- Environment validation
- GPU optimization setup
- Performance monitoring configuration
- Production deployment
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import docker
import yaml


class DeploymentManager:
    """Manages production deployment of the Embodied Intelligence Platform"""
    
    def __init__(self, config_path: str = "deployment_config.json"):
        """
        Initialize deployment manager
        
        Args:
            config_path: Path to deployment configuration file
        """
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
            "docker_registry": None,
            "deployment_targets": ["demo-llm", "demo-full-stack"],
            "environment_variables": {
                "ROS_DOMAIN_ID": "42",
                "PYTHONPATH": "/workspace",
                "CUDA_VISIBLE_DEVICES": "0"
            },
            "resource_limits": {
                "memory": "8g",
                "cpus": "4.0",
                "gpu_memory": "6g"
            },
            "monitoring_config": {
                "prometheus_enabled": True,
                "grafana_enabled": True,
                "alerting_enabled": True
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
    
    def validate_environment(self) -> bool:
        """Validate deployment environment"""
        self.logger.info("Validating deployment environment...")
        
        checks = [
            ("Docker", self._check_docker),
            ("Docker Compose", self._check_docker_compose),
            ("GPU Support", self._check_gpu_support),
            ("System Resources", self._check_system_resources),
            ("Network Connectivity", self._check_network),
            ("File Permissions", self._check_permissions)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                if check_func():
                    self.logger.info(f"✅ {check_name} check passed")
                else:
                    self.logger.error(f"❌ {check_name} check failed")
                    all_passed = False
            except Exception as e:
                self.logger.error(f"❌ {check_name} check error: {e}")
                all_passed = False
        
        return all_passed
    
    def _check_docker(self) -> bool:
        """Check Docker availability"""
        if self.docker_client is None:
            return False
        
        try:
            self.docker_client.ping()
            return True
        except Exception:
            return False
    
    def _check_docker_compose(self) -> bool:
        """Check Docker Compose availability"""
        try:
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _check_gpu_support(self) -> bool:
        """Check GPU support"""
        if not self.config.get("gpu_optimization", False):
            return True
        
        try:
            # Check NVIDIA Docker runtime
            result = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.0-base", "nvidia-smi"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            self.logger.warning("GPU support not available, will use CPU fallback")
            return True
    
    def _check_system_resources(self) -> bool:
        """Check system resources"""
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.total < 8 * 1024 * 1024 * 1024:  # 8GB
                self.logger.warning("System has less than 8GB RAM")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.free < 10 * 1024 * 1024 * 1024:  # 10GB
                self.logger.warning("Less than 10GB free disk space")
            
            return True
        except Exception as e:
            self.logger.warning(f"Could not check system resources: {e}")
            return True
    
    def _check_network(self) -> bool:
        """Check network connectivity"""
        try:
            import requests
            response = requests.get("https://www.google.com", timeout=5)
            return response.status_code == 200
        except Exception:
            self.logger.warning("Network connectivity check failed")
            return True
    
    def _check_permissions(self) -> bool:
        """Check file permissions"""
        try:
            # Check if we can write to current directory
            test_file = Path("deployment_test.tmp")
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception:
            return False
    
    def build_images(self) -> bool:
        """Build Docker images for deployment"""
        self.logger.info("Building Docker images...")
        
        try:
            # Build development image
            self.logger.info("Building development image...")
            result = subprocess.run([
                "docker-compose", "build", "dev-env"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to build development image: {result.stderr}")
                return False
            
            # Build simulation image
            self.logger.info("Building simulation image...")
            result = subprocess.run([
                "docker-compose", "build", "demo-slam"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to build simulation image: {result.stderr}")
                return False
            
            self.logger.info("✅ All images built successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build images: {e}")
            return False
    
    def run_safety_tests(self) -> bool:
        """Run safety validation tests"""
        if not self.config.get("safety_validation", True):
            self.logger.info("Skipping safety tests (disabled in config)")
            return True
        
        self.logger.info("Running safety validation tests...")
        
        try:
            # Run safety benchmarks
            result = subprocess.run([
                "python", "-m", "pytest", "benchmarks/safety_benchmarks/", "-v"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Safety tests failed: {result.stderr}")
                return False
            
            self.logger.info("✅ Safety tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to run safety tests: {e}")
            return False
    
    def run_performance_benchmarks(self) -> bool:
        """Run performance benchmarks"""
        if not self.config.get("performance_benchmarking", True):
            self.logger.info("Skipping performance benchmarks (disabled in config)")
            return True
        
        self.logger.info("Running performance benchmarks...")
        
        try:
            # Run LLM benchmarks
            result = subprocess.run([
                "python", "intelligence/eip_llm_interface/demo_gpu_optimization.py"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.warning(f"Performance benchmarks had issues: {result.stderr}")
            
            self.logger.info("✅ Performance benchmarks completed")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to run performance benchmarks: {e}")
            return True  # Don't fail deployment for benchmark issues
    
    def deploy_services(self) -> bool:
        """Deploy the services"""
        self.logger.info("Deploying services...")
        
        try:
            # Start safety monitor first
            self.logger.info("Starting safety monitor...")
            result = subprocess.run([
                "docker-compose", "up", "-d", "safety-monitor"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to start safety monitor: {result.stderr}")
                return False
            
            # Wait for safety monitor to be ready
            time.sleep(10)
            
            # Deploy target services
            for target in self.config.get("deployment_targets", []):
                self.logger.info(f"Deploying {target}...")
                result = subprocess.run([
                    "docker-compose", "up", "-d", target
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.logger.error(f"Failed to deploy {target}: {result.stderr}")
                    return False
            
            self.logger.info("✅ All services deployed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deploy services: {e}")
            return False
    
    def setup_monitoring(self) -> bool:
        """Setup monitoring and alerting"""
        if not self.config.get("monitoring_enabled", True):
            self.logger.info("Skipping monitoring setup (disabled in config)")
            return True
        
        self.logger.info("Setting up monitoring...")
        
        try:
            # Create monitoring configuration
            monitoring_config = self.config.get("monitoring_config", {})
            
            if monitoring_config.get("prometheus_enabled", True):
                self._setup_prometheus()
            
            if monitoring_config.get("grafana_enabled", True):
                self._setup_grafana()
            
            if monitoring_config.get("alerting_enabled", True):
                self._setup_alerting()
            
            self.logger.info("✅ Monitoring setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup monitoring: {e}")
            return False
    
    def _setup_prometheus(self):
        """Setup Prometheus monitoring"""
        self.logger.info("Setting up Prometheus...")
        
        # Create prometheus.yml configuration
        prometheus_config = {
            "global": {
                "scrape_interval": "15s"
            },
            "scrape_configs": [
                {
                    "job_name": "eip-safety-monitor",
                    "static_configs": [{"targets": ["localhost:9090"]}]
                }
            ]
        }
        
        with open("monitoring/prometheus.yml", "w") as f:
            yaml.dump(prometheus_config, f)
    
    def _setup_grafana(self):
        """Setup Grafana dashboard"""
        self.logger.info("Setting up Grafana...")
        
        # Create basic dashboard configuration
        dashboard_config = {
            "dashboard": {
                "title": "EIP Performance Dashboard",
                "panels": [
                    {
                        "title": "Safety Score",
                        "type": "graph",
                        "targets": [{"expr": "eip_safety_score"}]
                    },
                    {
                        "title": "Processing Time",
                        "type": "graph", 
                        "targets": [{"expr": "eip_processing_time"}]
                    }
                ]
            }
        }
        
        os.makedirs("monitoring/grafana", exist_ok=True)
        with open("monitoring/grafana/dashboard.json", "w") as f:
            json.dump(dashboard_config, f, indent=2)
    
    def _setup_alerting(self):
        """Setup alerting rules"""
        self.logger.info("Setting up alerting...")
        
        alert_rules = [
            {
                "name": "HighSafetyViolationRate",
                "condition": "eip_safety_violations > 0.1",
                "severity": "critical"
            },
            {
                "name": "HighProcessingTime",
                "condition": "eip_processing_time > 5.0",
                "severity": "warning"
            }
        ]
        
        os.makedirs("monitoring/alerts", exist_ok=True)
        with open("monitoring/alerts/rules.yml", "w") as f:
            yaml.dump(alert_rules, f)
    
    def verify_deployment(self) -> bool:
        """Verify deployment is working correctly"""
        self.logger.info("Verifying deployment...")
        
        try:
            # Check if services are running
            result = subprocess.run([
                "docker-compose", "ps"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error("Failed to check service status")
                return False
            
            # Check service health
            services = ["safety-monitor"] + self.config.get("deployment_targets", [])
            for service in services:
                self.logger.info(f"Checking {service} health...")
                
                # Wait for service to be ready
                time.sleep(5)
                
                # Check logs for errors
                result = subprocess.run([
                    "docker-compose", "logs", "--tail=10", service
                ], capture_output=True, text=True)
                
                if "ERROR" in result.stdout or "error" in result.stdout:
                    self.logger.warning(f"Found errors in {service} logs")
            
            self.logger.info("✅ Deployment verification completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to verify deployment: {e}")
            return False
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate deployment report"""
        self.logger.info("Generating deployment report...")
        
        report = {
            "timestamp": time.time(),
            "deployment_config": self.config,
            "services": [],
            "performance_metrics": {},
            "safety_metrics": {},
            "recommendations": []
        }
        
        try:
            # Get service status
            result = subprocess.run([
                "docker-compose", "ps", "--format", "json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                services = json.loads(result.stdout)
                report["services"] = services
            
            # Add recommendations
            if self.config.get("gpu_optimization", False):
                report["recommendations"].append(
                    "GPU optimization enabled - monitor GPU memory usage"
                )
            
            if self.config.get("monitoring_enabled", True):
                report["recommendations"].append(
                    "Monitoring enabled - check Grafana dashboard for metrics"
                )
            
            # Save report
            report_path = f"reports/deployment_report_{int(time.time())}.json"
            os.makedirs("reports", exist_ok=True)
            
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"✅ Deployment report saved to {report_path}")
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate deployment report: {e}")
            return report
    
    def deploy(self) -> bool:
        """Complete deployment process"""
        self.logger.info("Starting production deployment...")
        
        # Run comprehensive validation first
        if not self._run_production_validation():
            self.logger.error("❌ Production validation failed - deployment blocked")
            return False
        
        steps = [
            ("Environment Validation", self.validate_environment),
            ("Build Images", self.build_images),
            ("Safety Tests", self.run_safety_tests),
            ("Performance Benchmarks", self.run_performance_benchmarks),
            ("Deploy Services", self.deploy_services),
            ("Setup Monitoring", self.setup_monitoring),
            ("Verify Deployment", self.verify_deployment)
        ]
        
        for step_name, step_func in steps:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Step: {step_name}")
            self.logger.info(f"{'='*50}")
            
            if not step_func():
                self.logger.error(f"❌ Deployment failed at step: {step_name}")
                return False
        
        # Generate deployment report
        self.generate_deployment_report()
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info("✅ DEPLOYMENT COMPLETED SUCCESSFULLY")
        self.logger.info(f"{'='*50}")
        
        return True
    
    def _run_production_validation(self) -> bool:
        """Run comprehensive production validation using the new validator"""
        try:
            from production_deployment_validator import ProductionDeploymentValidator
            
            self.logger.info("Running comprehensive production validation...")
            
            # Initialize validator with current config
            validator = ProductionDeploymentValidator(self.config_path)
            
            # Run full validation
            report = validator.run_full_validation()
            
            # Save validation report
            report_path = f"validation_report_{int(time.time())}.json"
            validator.save_report(report, report_path)
            
            # Check if deployment is ready
            if report.deployment_ready:
                self.logger.info("✅ Production validation passed")
                self.logger.info(f"Validation report saved to: {report_path}")
                return True
            else:
                self.logger.error("❌ Production validation failed")
                self.logger.error("Critical issues:")
                for issue in report.critical_issues:
                    self.logger.error(f"  - {issue}")
                self.logger.error("Recommendations:")
                for rec in report.recommendations:
                    self.logger.error(f"  - {rec}")
                return False
                
        except ImportError:
            self.logger.warning("Production validator not available, skipping comprehensive validation")
            return True
        except Exception as e:
            self.logger.error(f"Production validation failed: {e}")
            return False


def main():
    """Main deployment script"""
    parser = argparse.ArgumentParser(description="Deploy Embodied Intelligence Platform")
    parser.add_argument("--config", default="deployment_config.json", 
                       help="Path to deployment configuration file")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate environment, don't deploy")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip safety and performance tests")
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    manager = DeploymentManager(args.config)
    
    if args.validate_only:
        success = manager.validate_environment()
        sys.exit(0 if success else 1)
    
    # Update config based on arguments
    if args.skip_tests:
        manager.config["safety_validation"] = False
        manager.config["performance_benchmarking"] = False
    
    # Run deployment
    success = manager.deploy()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 