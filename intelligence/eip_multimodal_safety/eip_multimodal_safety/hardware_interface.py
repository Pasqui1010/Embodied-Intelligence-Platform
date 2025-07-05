#!/usr/bin/env python3
"""
Hardware Interface Module

Provides hardware abstraction layer for multi-modal sensor integration
and real-time control in the swarm safety system.
"""

import numpy as np
import threading
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import serial
import struct
import queue


class SensorType(Enum):
    """Types of sensors supported"""
    VISION = "vision"
    AUDIO = "audio"
    TACTILE = "tactile"
    PROPRIOCEPTIVE = "proprioceptive"
    LIDAR = "lidar"
    IMU = "imu"


class CommunicationProtocol(Enum):
    """Communication protocols"""
    I2C = "i2c"
    SPI = "spi"
    UART = "uart"
    CAN = "can"
    ETHERNET = "ethernet"
    WIFI = "wifi"


@dataclass
class SensorConfig:
    """Configuration for a sensor"""
    sensor_id: str
    sensor_type: SensorType
    protocol: CommunicationProtocol
    address: str
    baud_rate: int = 115200
    timeout: float = 1.0
    retry_count: int = 3
    calibration_data: Dict[str, float] = None


@dataclass
class SensorReading:
    """Raw sensor reading"""
    sensor_id: str
    timestamp: float
    data: np.ndarray
    quality: float  # 0.0 to 1.0
    status: str  # 'ok', 'warning', 'error'
    metadata: Dict[str, Any] = None


class HardwareInterface:
    """
    Hardware interface for multi-modal sensor integration
    
    Provides abstraction layer for different sensor types and
    communication protocols with real-time data processing.
    """
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        
        # Sensor configurations
        self.sensor_configs: Dict[str, SensorConfig] = {}
        self.active_sensors: Dict[str, Any] = {}
        
        # Data buffers
        self.sensor_buffers: Dict[str, queue.Queue] = {}
        self.buffer_size = 100
        
        # Thread safety
        self.lock = threading.RLock()
        self.running = False
        
        # Performance monitoring
        self.performance_metrics = {
            'total_readings': 0,
            'failed_readings': 0,
            'average_latency': 0.0,
            'last_update': time.time()
        }
        
        # Load configuration
        if config_file:
            self._load_config(config_file)
        
        # Initialize default sensors
        self._initialize_default_sensors()
    
    def _load_config(self, config_file: str):
        """Load sensor configuration from file"""
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            for sensor_config in config_data.get('sensors', []):
                config = SensorConfig(
                    sensor_id=sensor_config['id'],
                    sensor_type=SensorType(sensor_config['type']),
                    protocol=CommunicationProtocol(sensor_config['protocol']),
                    address=sensor_config['address'],
                    baud_rate=sensor_config.get('baud_rate', 115200),
                    timeout=sensor_config.get('timeout', 1.0),
                    retry_count=sensor_config.get('retry_count', 3),
                    calibration_data=sensor_config.get('calibration', {})
                )
                self.sensor_configs[config.sensor_id] = config
                
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
    
    def _initialize_default_sensors(self):
        """Initialize default sensor configurations"""
        
        # Vision sensor (camera)
        vision_config = SensorConfig(
            sensor_id="vision_01",
            sensor_type=SensorType.VISION,
            protocol=CommunicationProtocol.ETHERNET,
            address="/dev/video0",
            calibration_data={
                'focal_length': 3.6,
                'sensor_width': 6.17,
                'sensor_height': 4.55
            }
        )
        self.sensor_configs[vision_config.sensor_id] = vision_config
        
        # Audio sensor (microphone array)
        audio_config = SensorConfig(
            sensor_id="audio_01",
            sensor_type=SensorType.AUDIO,
            protocol=CommunicationProtocol.I2C,
            address="0x48",
            calibration_data={
                'sensitivity': -38.0,
                'noise_floor': -60.0
            }
        )
        self.sensor_configs[audio_config.sensor_id] = audio_config
        
        # Tactile sensor (force sensors)
        tactile_config = SensorConfig(
            sensor_id="tactile_01",
            sensor_type=SensorType.TACTILE,
            protocol=CommunicationProtocol.I2C,
            address="0x50",
            calibration_data={
                'force_scale': 100.0,  # N
                'pressure_scale': 1000.0  # Pa
            }
        )
        self.sensor_configs[tactile_config.sensor_id] = tactile_config
        
        # Proprioceptive sensor (IMU)
        proprioceptive_config = SensorConfig(
            sensor_id="proprioceptive_01",
            sensor_type=SensorType.PROPRIOCEPTIVE,
            protocol=CommunicationProtocol.I2C,
            address="0x68",
            calibration_data={
                'accel_scale': 2.0,  # g
                'gyro_scale': 250.0,  # deg/s
                'mag_scale': 4900.0  # uT
            }
        )
        self.sensor_configs[proprioceptive_config.sensor_id] = proprioceptive_config
    
    def start(self):
        """Start hardware interface"""
        
        with self.lock:
            if self.running:
                return
            
            self.running = True
            
            # Initialize sensor connections
            for sensor_id, config in self.sensor_configs.items():
                self._initialize_sensor(sensor_id, config)
            
            # Start sensor reading threads
            for sensor_id in self.sensor_configs:
                self._start_sensor_thread(sensor_id)
            
            logging.info("Hardware interface started")
    
    def stop(self):
        """Stop hardware interface"""
        
        with self.lock:
            if not self.running:
                return
            
            self.running = False
            
            # Close sensor connections
            for sensor_id, sensor in self.active_sensors.items():
                self._close_sensor(sensor_id, sensor)
            
            logging.info("Hardware interface stopped")
    
    def _initialize_sensor(self, sensor_id: str, config: SensorConfig):
        """Initialize a sensor connection"""
        
        try:
            if config.protocol == CommunicationProtocol.I2C:
                sensor = self._init_i2c_sensor(config)
            elif config.protocol == CommunicationProtocol.UART:
                sensor = self._init_uart_sensor(config)
            elif config.protocol == CommunicationProtocol.ETHERNET:
                sensor = self._init_ethernet_sensor(config)
            else:
                logging.warning(f"Unsupported protocol: {config.protocol}")
                return
            
            self.active_sensors[sensor_id] = sensor
            
            # Initialize buffer
            self.sensor_buffers[sensor_id] = queue.Queue(maxsize=self.buffer_size)
            
            logging.info(f"Initialized sensor {sensor_id}")
            
        except Exception as e:
            logging.error(f"Error initializing sensor {sensor_id}: {e}")
    
    def _init_i2c_sensor(self, config: SensorConfig):
        """Initialize I2C sensor"""
        
        # Simulated I2C sensor for demo
        class I2CSensor:
            def __init__(self, config):
                self.config = config
                self.address = int(config.address, 16)
                self.calibration = config.calibration_data or {}
            
            def read_data(self):
                # Simulate sensor reading
                if config.sensor_type == SensorType.AUDIO:
                    # Simulate audio data
                    data = np.random.normal(0, 0.1, 1024)
                    quality = 0.9
                elif config.sensor_type == SensorType.TACTILE:
                    # Simulate tactile data
                    data = np.random.uniform(0, 1, 16)
                    quality = 0.8
                elif config.sensor_type == SensorType.PROPRIOCEPTIVE:
                    # Simulate IMU data
                    data = np.array([
                        np.random.normal(0, 0.1, 3),  # Accelerometer
                        np.random.normal(0, 0.1, 3),  # Gyroscope
                        np.random.normal(0, 0.1, 3)   # Magnetometer
                    ]).flatten()
                    quality = 0.85
                else:
                    data = np.zeros(10)
                    quality = 0.0
                
                return data, quality
        
        return I2CSensor(config)
    
    def _init_uart_sensor(self, config: SensorConfig):
        """Initialize UART sensor"""
        
        # Simulated UART sensor for demo
        class UARTSensor:
            def __init__(self, config):
                self.config = config
                self.baud_rate = config.baud_rate
                self.timeout = config.timeout
            
            def read_data(self):
                # Simulate UART reading
                data = np.random.uniform(0, 1, 8)
                quality = 0.75
                return data, quality
        
        return UARTSensor(config)
    
    def _init_ethernet_sensor(self, config: SensorConfig):
        """Initialize Ethernet sensor (camera)"""
        
        # Simulated camera sensor for demo
        class CameraSensor:
            def __init__(self, config):
                self.config = config
                self.calibration = config.calibration_data or {}
                self.frame_count = 0
            
            def read_data(self):
                # Simulate camera frame
                self.frame_count += 1
                
                # Simulate image data (simplified)
                height, width = 480, 640
                data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                
                # Add some structure to simulate real image
                if self.frame_count % 30 == 0:  # Every 30 frames
                    # Simulate object detection
                    data[200:300, 250:350] = [255, 0, 0]  # Red rectangle
                
                quality = 0.95
                return data, quality
        
        return CameraSensor(config)
    
    def _start_sensor_thread(self, sensor_id: str):
        """Start sensor reading thread"""
        
        def sensor_reading_loop():
            while self.running:
                try:
                    # Read sensor data
                    reading = self._read_sensor(sensor_id)
                    
                    if reading:
                        # Add to buffer
                        buffer = self.sensor_buffers.get(sensor_id)
                        if buffer and not buffer.full():
                            buffer.put(reading)
                        
                        # Update performance metrics
                        self._update_performance_metrics(sensor_id, reading)
                    
                    # Sleep based on sensor type
                    config = self.sensor_configs[sensor_id]
                    if config.sensor_type == SensorType.VISION:
                        time.sleep(1.0 / 30.0)  # 30 FPS
                    elif config.sensor_type == SensorType.AUDIO:
                        time.sleep(1.0 / 16000.0)  # 16 kHz
                    else:
                        time.sleep(1.0 / 100.0)  # 100 Hz
                
                except Exception as e:
                    logging.error(f"Error in sensor reading loop for {sensor_id}: {e}")
                    time.sleep(1.0)
        
        thread = threading.Thread(target=sensor_reading_loop, daemon=True)
        thread.start()
    
    def _read_sensor(self, sensor_id: str) -> Optional[SensorReading]:
        """Read data from a sensor"""
        
        try:
            sensor = self.active_sensors.get(sensor_id)
            if not sensor:
                return None
            
            # Read raw data
            data, quality = sensor.read_data()
            
            # Create sensor reading
            reading = SensorReading(
                sensor_id=sensor_id,
                timestamp=time.time(),
                data=data,
                quality=quality,
                status='ok' if quality > 0.5 else 'warning',
                metadata={
                    'sensor_type': self.sensor_configs[sensor_id].sensor_type.value,
                    'protocol': self.sensor_configs[sensor_id].protocol.value
                }
            )
            
            return reading
            
        except Exception as e:
            logging.error(f"Error reading sensor {sensor_id}: {e}")
            return None
    
    def _update_performance_metrics(self, sensor_id: str, reading: SensorReading):
        """Update performance metrics"""
        
        self.performance_metrics['total_readings'] += 1
        
        if reading.status == 'error':
            self.performance_metrics['failed_readings'] += 1
        
        # Update average latency
        current_time = time.time()
        latency = current_time - reading.timestamp
        self.performance_metrics['average_latency'] = (
            self.performance_metrics['average_latency'] * 0.9 + latency * 0.1
        )
        
        self.performance_metrics['last_update'] = current_time
    
    def _close_sensor(self, sensor_id: str, sensor: Any):
        """Close sensor connection"""
        
        try:
            # Close sensor-specific resources
            if hasattr(sensor, 'close'):
                sensor.close()
            
            logging.info(f"Closed sensor {sensor_id}")
            
        except Exception as e:
            logging.error(f"Error closing sensor {sensor_id}: {e}")
    
    def get_sensor_reading(self, sensor_id: str) -> Optional[SensorReading]:
        """Get latest reading from a sensor"""
        
        buffer = self.sensor_buffers.get(sensor_id)
        if not buffer:
            return None
        
        try:
            return buffer.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_readings(self) -> Dict[str, SensorReading]:
        """Get latest readings from all sensors"""
        
        readings = {}
        
        for sensor_id in self.sensor_configs:
            reading = self.get_sensor_reading(sensor_id)
            if reading:
                readings[sensor_id] = reading
        
        return readings
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """Get status of all sensors"""
        
        status = {}
        
        for sensor_id, config in self.sensor_configs.items():
            buffer = self.sensor_buffers.get(sensor_id)
            
            sensor_status = {
                'type': config.sensor_type.value,
                'protocol': config.protocol.value,
                'address': config.address,
                'active': sensor_id in self.active_sensors,
                'buffer_size': buffer.qsize() if buffer else 0,
                'buffer_full': buffer.full() if buffer else False
            }
            
            # Add latest reading info
            reading = self.get_sensor_reading(sensor_id)
            if reading:
                sensor_status.update({
                    'last_reading_time': reading.timestamp,
                    'last_reading_quality': reading.quality,
                    'last_reading_status': reading.status
                })
            
            status[sensor_id] = sensor_status
        
        return status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        return self.performance_metrics.copy()
    
    def calibrate_sensor(self, sensor_id: str, calibration_data: Dict[str, float]):
        """Calibrate a sensor"""
        
        try:
            config = self.sensor_configs.get(sensor_id)
            if not config:
                logging.error(f"Sensor {sensor_id} not found")
                return False
            
            # Update calibration data
            if config.calibration_data is None:
                config.calibration_data = {}
            
            config.calibration_data.update(calibration_data)
            
            # Apply calibration to sensor
            sensor = self.active_sensors.get(sensor_id)
            if sensor and hasattr(sensor, 'apply_calibration'):
                sensor.apply_calibration(calibration_data)
            
            logging.info(f"Calibrated sensor {sensor_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error calibrating sensor {sensor_id}: {e}")
            return False
    
    def emergency_stop(self):
        """Emergency stop - immediately stop all sensors"""
        
        logging.warning("Emergency stop triggered")
        
        with self.lock:
            self.running = False
            
            # Immediately close all sensors
            for sensor_id, sensor in self.active_sensors.items():
                self._close_sensor(sensor_id, sensor)
            
            # Clear all buffers
            for buffer in self.sensor_buffers.values():
                while not buffer.empty():
                    try:
                        buffer.get_nowait()
                    except queue.Empty:
                        break
        
        logging.info("Emergency stop completed")


class RealTimeController:
    """
    Real-time controller for safety-critical operations
    
    Provides deterministic timing and safety monitoring
    for hardware control operations.
    """
    
    def __init__(self, control_rate: float = 1000.0):
        self.control_rate = control_rate
        self.control_period = 1.0 / control_rate
        
        # Control state
        self.running = False
        self.control_thread = None
        
        # Safety monitoring
        self.safety_violations = []
        self.last_safety_check = time.time()
        
        # Performance monitoring
        self.control_metrics = {
            'control_cycles': 0,
            'missed_deadlines': 0,
            'average_cycle_time': 0.0,
            'max_cycle_time': 0.0
        }
    
    def start(self):
        """Start real-time control loop"""
        
        if self.running:
            return
        
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        logging.info(f"Real-time controller started at {self.control_rate} Hz")
    
    def stop(self):
        """Stop real-time control loop"""
        
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        
        logging.info("Real-time controller stopped")
    
    def _control_loop(self):
        """Main control loop with deterministic timing"""
        
        next_cycle = time.time()
        
        while self.running:
            cycle_start = time.time()
            
            try:
                # Perform control operations
                self._control_cycle()
                
                # Safety check
                self._safety_check()
                
                # Update metrics
                cycle_time = time.time() - cycle_start
                self._update_control_metrics(cycle_time)
                
                # Check for missed deadline
                if cycle_time > self.control_period:
                    self.control_metrics['missed_deadlines'] += 1
                    logging.warning(f"Missed control deadline: {cycle_time:.3f}s")
                
            except Exception as e:
                logging.error(f"Error in control cycle: {e}")
            
            # Wait for next cycle
            next_cycle += self.control_period
            sleep_time = next_cycle - time.time()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # We're behind schedule
                next_cycle = time.time() + self.control_period
    
    def _control_cycle(self):
        """Perform one control cycle"""
        
        # This is where actual control operations would go
        # For now, just increment cycle counter
        self.control_metrics['control_cycles'] += 1
    
    def _safety_check(self):
        """Perform safety checks"""
        
        current_time = time.time()
        
        # Check for safety violations
        if self.control_metrics['missed_deadlines'] > 10:
            violation = {
                'timestamp': current_time,
                'type': 'missed_deadlines',
                'count': self.control_metrics['missed_deadlines']
            }
            self.safety_violations.append(violation)
            
            # Reset counter
            self.control_metrics['missed_deadlines'] = 0
        
        # Check cycle time
        if self.control_metrics['max_cycle_time'] > self.control_period * 2:
            violation = {
                'timestamp': current_time,
                'type': 'excessive_cycle_time',
                'max_time': self.control_metrics['max_cycle_time']
            }
            self.safety_violations.append(violation)
        
        self.last_safety_check = current_time
    
    def _update_control_metrics(self, cycle_time: float):
        """Update control performance metrics"""
        
        # Update average cycle time
        self.control_metrics['average_cycle_time'] = (
            self.control_metrics['average_cycle_time'] * 0.99 + cycle_time * 0.01
        )
        
        # Update max cycle time
        if cycle_time > self.control_metrics['max_cycle_time']:
            self.control_metrics['max_cycle_time'] = cycle_time
    
    def get_control_status(self) -> Dict[str, Any]:
        """Get control system status"""
        
        return {
            'running': self.running,
            'control_rate': self.control_rate,
            'control_period': self.control_period,
            'metrics': self.control_metrics.copy(),
            'safety_violations': len(self.safety_violations),
            'last_safety_check': self.last_safety_check
        }
    
    def get_safety_violations(self) -> List[Dict[str, Any]]:
        """Get recent safety violations"""
        
        return self.safety_violations.copy() 