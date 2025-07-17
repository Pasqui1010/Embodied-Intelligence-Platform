from typing import Dict, Any, Optional, List
import torch
import psutil
import time
import logging
from dataclasses import dataclass
from datetime import datetime
import json

class GPUMetrics:
    """GPU performance metrics"""
    def __init__(self):
        self.timestamp = datetime.now()
        self.memory = {
            'total': 0.0,
            'allocated': 0.0,
            'cached': 0.0,
            'utilization': 0.0,
            'peak': 0.0
        }
        self.performance = {
            'throughput': 0.0,
            'latency': 0.0,
            'gpu_util': 0.0,
            'power_usage': 0.0,
            'temperature': 0.0
        }
        self.system = {
            'cpu_util': 0.0,
            'memory_util': 0.0,
            'load_avg': 0.0
        }
        self.process_metrics = []

class GPUMonitor:
    """
    Monitors GPU performance and resource usage
    """
    
    def __init__(
        self,
        monitoring_interval: float = 1.0,
        alert_thresholds: Optional[Dict[str, float]] = None,
        enable_real_time: bool = True,
        export_path: Optional[str] = None
    ):
        """
        Initialize GPU monitor
        
        Args:
            monitoring_interval: Interval between monitoring checks (seconds)
            alert_thresholds: Thresholds for triggering alerts
            enable_real_time: Enable real-time monitoring
            export_path: Path to export metrics
        """
        self.monitoring_interval = monitoring_interval
        self.alert_thresholds = alert_thresholds or {
            'memory': 0.9,
            'gpu_util': 0.9,
            'temperature': 85.0,
            'power_usage': 0.9
        }
        self.enable_real_time = enable_real_time
        self.export_path = export_path
        
        self.metrics_history = []
        self.alerts = []
        self.logger = logging.getLogger(__name__)
        
        # Real-time monitoring
        self.monitoring_thread = None
        self.running = False
        
    def start_monitoring(self) -> None:
        """
        Start real-time GPU monitoring
        """
        if not self.enable_real_time:
            return
            
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("GPU monitoring started")
    
    def stop_monitoring(self) -> None:
        """
        Stop real-time GPU monitoring
        """
        if not self.enable_real_time:
            return
            
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.logger.info("GPU monitoring stopped")
    
    def get_current_metrics(self) -> GPUMetrics:
        """
        Get current GPU metrics
        
        Returns:
            GPUMetrics instance with current metrics
        """
        metrics = GPUMetrics()
        
        # Get GPU metrics
        if torch.cuda.is_available():
            metrics.memory['total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            metrics.memory['allocated'] = torch.cuda.memory_allocated() / 1024**3
            metrics.memory['cached'] = torch.cuda.memory_reserved() / 1024**3
            metrics.memory['utilization'] = self._get_gpu_utilization()
            metrics.memory['peak'] = torch.cuda.max_memory_allocated() / 1024**3
            
            metrics.performance['gpu_util'] = metrics.memory['utilization']
            metrics.performance['temperature'] = self._get_gpu_temperature()
            metrics.performance['power_usage'] = self._get_gpu_power_usage()
            
        # Get system metrics
        metrics.system['cpu_util'] = psutil.cpu_percent()
        metrics.system['memory_util'] = psutil.virtual_memory().percent
        metrics.system['load_avg'] = psutil.getloadavg()[0]
        
        # Check for alerts
        self._check_alerts(metrics)
        
        return metrics
    
    def _monitor_loop(self) -> None:
        """
        Continuous monitoring loop
        """
        while self.running:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                
                if self.export_path:
                    self._export_metrics(metrics)
                    
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_alerts(self, metrics: GPUMetrics) -> None:
        """
        Check for alert conditions
        
        Args:
            metrics: Current metrics to check
        """
        alerts = []
        
        # Check memory usage
        if metrics.memory['utilization'] > self.alert_thresholds['memory']:
            alerts.append(f"High GPU memory usage: {metrics.memory['utilization']:.1%}")
            
        # Check GPU utilization
        if metrics.performance['gpu_util'] > self.alert_thresholds['gpu_util']:
            alerts.append(f"High GPU utilization: {metrics.performance['gpu_util']:.1%}")
            
        # Check temperature
        if metrics.performance['temperature'] > self.alert_thresholds['temperature']:
            alerts.append(f"High GPU temperature: {metrics.performance['temperature']}°C")
            
        # Check power usage
        if metrics.performance['power_usage'] > self.alert_thresholds['power_usage']:
            alerts.append(f"High GPU power usage: {metrics.performance['power_usage']:.1%}")
            
        if alerts:
            self.alerts.extend(alerts)
            self.logger.warning(f"GPU alerts: {', '.join(alerts)}")
    
    def _get_gpu_utilization(self) -> float:
        """
        Get GPU utilization as a percentage
        
        Returns:
            GPU utilization percentage
        """
        try:
            # This is a placeholder - actual implementation would depend on GPU monitoring library
            return psutil.cpu_percent()  # Using CPU as placeholder
        except:
            return 0.0
    
    def _get_gpu_temperature(self) -> float:
        """
        Get GPU temperature
        
        Returns:
            GPU temperature in Celsius
        """
        try:
            # This is a placeholder - actual implementation would depend on GPU monitoring library
            return psutil.sensors_temperatures()['coretemp'][0].current
        except:
            return 0.0
    
    def _get_gpu_power_usage(self) -> float:
        """
        Get GPU power usage as a percentage of maximum
        
        Returns:
            Power usage percentage
        """
        try:
            # This is a placeholder - actual implementation would depend on GPU monitoring library
            return psutil.cpu_percent() / 100.0
        except:
            return 0.0
    
    def _export_metrics(self, metrics: GPUMetrics) -> None:
        """
        Export metrics to file
        
        Args:
            metrics: Metrics to export
        """
        if not self.export_path:
            return
            
        try:
            metrics_dict = {
                'timestamp': metrics.timestamp.isoformat(),
                'memory': metrics.memory,
                'performance': metrics.performance,
                'system': metrics.system,
                'alerts': self.alerts
            }
            
            with open(self.export_path, 'a') as f:
                f.write(json.dumps(metrics_dict) + '\n')
                
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
    
    def get_metrics_history(self) -> List[GPUMetrics]:
        """
        Get history of collected metrics
        
        Returns:
            List of GPUMetrics instances
        """
        return self.metrics_history
    
    def get_alerts(self) -> List[str]:
        """
        Get list of alerts
        
        Returns:
            List of alert messages
        """
        return self.alerts
