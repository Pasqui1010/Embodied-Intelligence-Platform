#!/usr/bin/env python3
"""
Multi-Modal Safety Fusion Tests

This module provides comprehensive tests for the multi-modal safety fusion system,
including sensor fusion algorithms, safety event detection, and performance validation.
"""

import unittest
import numpy as np
import time
import sys
import os
from typing import Dict, List, Any

# Add the intelligence package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../intelligence/eip_multimodal_safety'))

from eip_multimodal_safety.sensor_fusion import (
    SensorFusionEngine, SensorData, SensorType, FusionMethod,
    SafetyEvent, FusionResult
)


class TestSensorFusionEngine(unittest.TestCase):
    """Test cases for the Sensor Fusion Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fusion_engine = SensorFusionEngine(FusionMethod.WEIGHTED_AVERAGE)
        self.test_data = self._create_test_sensor_data()
    
    def _create_test_sensor_data(self) -> Dict[SensorType, List[SensorData]]:
        """Create test sensor data for all modalities"""
        test_data = {}
        current_time = time.time()
        
        # Vision data
        test_data[SensorType.VISION] = [
            SensorData(
                sensor_type=SensorType.VISION,
                timestamp=current_time,
                data="test_image_data",
                confidence=0.8,
                source_id="test_camera"
            ),
            SensorData(
                sensor_type=SensorType.VISION,
                timestamp=current_time + 0.1,
                data="test_depth_data",
                confidence=0.9,
                source_id="test_depth_camera"
            )
        ]
        
        # Audio data
        test_data[SensorType.AUDIO] = [
            SensorData(
                sensor_type=SensorType.AUDIO,
                timestamp=current_time,
                data="test_audio_data",
                confidence=0.7,
                source_id="test_microphone"
            )
        ]
        
        # Tactile data
        test_data[SensorType.TACTILE] = [
            SensorData(
                sensor_type=SensorType.TACTILE,
                timestamp=current_time,
                data="test_tactile_data",
                confidence=0.85,
                source_id="test_tactile_sensor"
            )
        ]
        
        # Proprioceptive data
        test_data[SensorType.PROPRIOCEPTIVE] = [
            SensorData(
                sensor_type=SensorType.PROPRIOCEPTIVE,
                timestamp=current_time,
                data="test_imu_data",
                confidence=0.9,
                source_id="test_imu"
            ),
            SensorData(
                sensor_type=SensorType.PROPRIOCEPTIVE,
                timestamp=current_time + 0.05,
                data="test_joint_data",
                confidence=0.8,
                source_id="test_joint_controller"
            )
        ]
        
        # LIDAR data
        test_data[SensorType.LIDAR] = [
            SensorData(
                sensor_type=SensorType.LIDAR,
                timestamp=current_time,
                data="test_lidar_data",
                confidence=0.95,
                source_id="test_lidar"
            )
        ]
        
        return test_data
    
    def test_initialization(self):
        """Test sensor fusion engine initialization"""
        self.assertIsNotNone(self.fusion_engine)
        self.assertEqual(self.fusion_engine.fusion_method, FusionMethod.WEIGHTED_AVERAGE)
        self.assertIsNotNone(self.fusion_engine.sensor_weights)
        self.assertIsNotNone(self.fusion_engine.safety_thresholds)
        self.assertIsNotNone(self.fusion_engine.correlation_matrix)
    
    def test_sensor_data_validation(self):
        """Test sensor data validation"""
        # Valid data
        valid_data = SensorData(
            sensor_type=SensorType.VISION,
            timestamp=time.time(),
            data="test",
            confidence=0.8,
            source_id="test"
        )
        self.assertTrue(self.fusion_engine._validate_sensor_data(valid_data))
        
        # Invalid timestamp
        invalid_timestamp = SensorData(
            sensor_type=SensorType.VISION,
            timestamp=0,
            data="test",
            confidence=0.8,
            source_id="test"
        )
        self.assertFalse(self.fusion_engine._validate_sensor_data(invalid_timestamp))
        
        # Invalid confidence
        invalid_confidence = SensorData(
            sensor_type=SensorType.VISION,
            timestamp=time.time(),
            data="test",
            confidence=1.5,  # > 1.0
            source_id="test"
        )
        self.assertFalse(self.fusion_engine._validate_sensor_data(invalid_confidence))
        
        # Invalid source_id
        invalid_source = SensorData(
            sensor_type=SensorType.VISION,
            timestamp=time.time(),
            data="test",
            confidence=0.8,
            source_id=""
        )
        self.assertFalse(self.fusion_engine._validate_sensor_data(invalid_source))
    
    def test_add_sensor_data(self):
        """Test adding sensor data to the fusion engine"""
        test_data = self.test_data[SensorType.VISION][0]
        
        # Add valid data
        initial_buffer_size = len(self.fusion_engine.sensor_buffers[SensorType.VISION])
        self.fusion_engine.add_sensor_data(test_data)
        final_buffer_size = len(self.fusion_engine.sensor_buffers[SensorType.VISION])
        
        self.assertEqual(final_buffer_size, initial_buffer_size + 1)
    
    def test_weighted_average_fusion(self):
        """Test weighted average sensor fusion"""
        # Add test data to all sensors
        for sensor_type, data_list in self.test_data.items():
            for data in data_list:
                self.fusion_engine.add_sensor_data(data)
        
        # Perform fusion
        fusion_result = self.fusion_engine.fuse_sensors()
        
        # Validate result
        self.assertIsInstance(fusion_result, FusionResult)
        self.assertGreaterEqual(fusion_result.safety_score, 0.0)
        self.assertLessEqual(fusion_result.safety_score, 1.0)
        self.assertIsInstance(fusion_result.events, list)
        self.assertIsInstance(fusion_result.sensor_health, dict)
        self.assertGreaterEqual(fusion_result.fusion_confidence, 0.0)
        self.assertLessEqual(fusion_result.fusion_confidence, 1.0)
        self.assertGreaterEqual(fusion_result.processing_time, 0.0)
    
    def test_sensor_health_calculation(self):
        """Test sensor health calculation"""
        # Add test data
        for sensor_type, data_list in self.test_data.items():
            for data in data_list:
                self.fusion_engine.add_sensor_data(data)
        
        # Get health metrics
        health = self.fusion_engine._calculate_sensor_health()
        
        # Validate health metrics
        self.assertIsInstance(health, dict)
        for sensor_type in SensorType:
            self.assertIn(sensor_type.value, health)
            self.assertGreaterEqual(health[sensor_type.value], 0.0)
            self.assertLessEqual(health[sensor_type.value], 1.0)
    
    def test_fusion_confidence_calculation(self):
        """Test fusion confidence calculation"""
        # Add test data
        for sensor_type, data_list in self.test_data.items():
            for data in data_list:
                self.fusion_engine.add_sensor_data(data)
        
        # Get recent data
        recent_data = self.fusion_engine._collect_recent_data()
        
        # Calculate confidence
        confidence = self.fusion_engine._calculate_fusion_confidence(recent_data)
        
        # Validate confidence
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_safety_event_generation(self):
        """Test safety event generation"""
        # Add test data
        for sensor_type, data_list in self.test_data.items():
            for data in data_list:
                self.fusion_engine.add_sensor_data(data)
        
        # Generate events for each sensor type
        for sensor_type, data_list in self.test_data.items():
            events = self.fusion_engine._generate_safety_events(sensor_type, data_list)
            self.assertIsInstance(events, list)
            
            for event in events:
                self.assertIsInstance(event, SafetyEvent)
                self.assertGreaterEqual(event.severity, 0.0)
                self.assertLessEqual(event.severity, 1.0)
                self.assertGreaterEqual(event.confidence, 0.0)
                self.assertLessEqual(event.confidence, 1.0)
                self.assertIsInstance(event.sensor_sources, list)
                self.assertGreater(event.timestamp, 0)
    
    def test_sensor_weights_update(self):
        """Test sensor weights update"""
        initial_weights = self.fusion_engine.sensor_weights.copy()
        
        # Update weights
        new_weights = {
            SensorType.VISION: 0.4,
            SensorType.AUDIO: 0.3,
            SensorType.TACTILE: 0.2,
            SensorType.PROPRIOCEPTIVE: 0.08,
            SensorType.LIDAR: 0.02
        }
        
        self.fusion_engine.update_sensor_weights(new_weights)
        
        # Check that weights were updated
        for sensor_type, weight in new_weights.items():
            self.assertAlmostEqual(
                self.fusion_engine.sensor_weights[sensor_type],
                weight,
                places=2
            )
        
        # Check that weights sum to 1.0
        total_weight = sum(self.fusion_engine.sensor_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        # Add some test data and perform fusion
        for sensor_type, data_list in self.test_data.items():
            for data in data_list:
                self.fusion_engine.add_sensor_data(data)
        
        self.fusion_engine.fuse_sensors()
        
        # Get performance metrics
        metrics = self.fusion_engine.get_performance_metrics()
        
        # Validate metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn('avg_processing_time', metrics)
        self.assertIn('fusion_accuracy', metrics)
        self.assertIn('sensor_weights', metrics)
        self.assertIn('buffer_sizes', metrics)
        self.assertIn('queue_size', metrics)
        
        self.assertGreaterEqual(metrics['avg_processing_time'], 0.0)
        self.assertGreaterEqual(metrics['fusion_accuracy'], 0.0)
        self.assertLessEqual(metrics['fusion_accuracy'], 1.0)
    
    def test_correlation_matrix_initialization(self):
        """Test correlation matrix initialization"""
        matrix = self.fusion_engine.correlation_matrix
        
        # Check matrix structure
        sensor_names = [s.value for s in SensorType]
        for sensor1 in sensor_names:
            self.assertIn(sensor1, matrix)
            for sensor2 in sensor_names:
                self.assertIn(sensor2, matrix[sensor1])
                self.assertGreaterEqual(matrix[sensor1][sensor2], 0.0)
                self.assertLessEqual(matrix[sensor1][sensor2], 1.0)
        
        # Check diagonal elements (self-correlation)
        for sensor in sensor_names:
            self.assertEqual(matrix[sensor][sensor], 1.0)
    
    def test_processing_thread_lifecycle(self):
        """Test processing thread lifecycle"""
        # Start processing
        self.fusion_engine.start_processing()
        self.assertTrue(self.fusion_engine.running)
        self.assertIsNotNone(self.fusion_engine.processing_thread)
        self.assertTrue(self.fusion_engine.processing_thread.is_alive())
        
        # Stop processing
        self.fusion_engine.stop_processing()
        self.assertFalse(self.fusion_engine.running)
    
    def test_data_buffer_limits(self):
        """Test data buffer size limits"""
        max_buffer_size = 100
        
        # Add more data than buffer size
        for i in range(max_buffer_size + 10):
            data = SensorData(
                sensor_type=SensorType.VISION,
                timestamp=time.time() + i,
                data=f"test_data_{i}",
                confidence=0.8,
                source_id="test"
            )
            self.fusion_engine.add_sensor_data(data)
        
        # Check buffer size is limited
        buffer_size = len(self.fusion_engine.sensor_buffers[SensorType.VISION])
        self.assertLessEqual(buffer_size, max_buffer_size)
    
    def test_empty_fusion_result(self):
        """Test fusion with no sensor data"""
        # Perform fusion without adding any data
        fusion_result = self.fusion_engine.fuse_sensors()
        
        # Should return valid result with low scores
        self.assertIsInstance(fusion_result, FusionResult)
        self.assertEqual(fusion_result.safety_score, 0.0)
        self.assertEqual(len(fusion_result.events), 0)
        self.assertEqual(fusion_result.fusion_confidence, 0.0)


class TestMultiModalSafetyIntegration(unittest.TestCase):
    """Integration tests for multi-modal safety system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.fusion_engine = SensorFusionEngine(FusionMethod.WEIGHTED_AVERAGE)
    
    def test_cross_modal_correlation(self):
        """Test cross-modal correlation in safety events"""
        # Create correlated sensor data
        timestamp = time.time()
        
        # Vision detects human
        vision_data = SensorData(
            sensor_type=SensorType.VISION,
            timestamp=timestamp,
            data="human_detected",
            confidence=0.9,
            source_id="vision"
        )
        
        # Audio detects human voice
        audio_data = SensorData(
            sensor_type=SensorType.AUDIO,
            timestamp=timestamp,
            data="human_voice",
            confidence=0.8,
            source_id="audio"
        )
        
        # Add both data points
        self.fusion_engine.add_sensor_data(vision_data)
        self.fusion_engine.add_sensor_data(audio_data)
        
        # Perform fusion
        fusion_result = self.fusion_engine.fuse_sensors()
        
        # Should have higher confidence due to correlation
        self.assertGreater(fusion_result.fusion_confidence, 0.7)
    
    def test_safety_threshold_violations(self):
        """Test safety threshold violation detection"""
        # Create high-risk sensor data
        timestamp = time.time()
        
        # High collision risk from LIDAR
        lidar_data = SensorData(
            sensor_type=SensorType.LIDAR,
            timestamp=timestamp,
            data="obstacle_close",
            confidence=0.95,
            source_id="lidar"
        )
        
        self.fusion_engine.add_sensor_data(lidar_data)
        
        # Perform fusion
        fusion_result = self.fusion_engine.fuse_sensors()
        
        # Should generate safety events
        self.assertGreater(len(fusion_result.events), 0)
        
        # Check for high-severity events
        high_severity_events = [e for e in fusion_result.events if e.severity > 0.7]
        self.assertGreater(len(high_severity_events), 0)
    
    def test_sensor_failure_handling(self):
        """Test handling of sensor failures"""
        # Add some valid data
        valid_data = SensorData(
            sensor_type=SensorType.VISION,
            timestamp=time.time(),
            data="valid_data",
            confidence=0.8,
            source_id="vision"
        )
        self.fusion_engine.add_sensor_data(valid_data)
        
        # Add invalid data (simulating sensor failure)
        invalid_data = SensorData(
            sensor_type=SensorType.AUDIO,
            timestamp=time.time(),
            data="invalid_data",
            confidence=-0.1,  # Invalid confidence
            source_id="audio"
        )
        self.fusion_engine.add_sensor_data(invalid_data)
        
        # Perform fusion
        fusion_result = self.fusion_engine.fuse_sensors()
        
        # Should still work with partial data
        self.assertIsInstance(fusion_result, FusionResult)
        self.assertGreaterEqual(fusion_result.safety_score, 0.0)
    
    def test_real_time_performance(self):
        """Test real-time performance requirements"""
        # Add test data
        for i in range(10):
            data = SensorData(
                sensor_type=SensorType.VISION,
                timestamp=time.time() + i * 0.1,
                data=f"test_data_{i}",
                confidence=0.8,
                source_id="test"
            )
            self.fusion_engine.add_sensor_data(data)
        
        # Measure fusion time
        start_time = time.time()
        fusion_result = self.fusion_engine.fuse_sensors()
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (100ms)
        self.assertLess(processing_time, 0.1)
        self.assertGreaterEqual(fusion_result.processing_time, 0.0)


def run_performance_benchmarks():
    """Run performance benchmarks for the multi-modal safety system"""
    print("Running Multi-Modal Safety Performance Benchmarks...")
    
    # Initialize fusion engine
    fusion_engine = SensorFusionEngine(FusionMethod.WEIGHTED_AVERAGE)
    
    # Performance test parameters
    num_iterations = 1000
    num_sensors = 5
    data_per_sensor = 10
    
    print(f"Testing with {num_iterations} iterations, {num_sensors} sensors, {data_per_sensor} data points per sensor")
    
    # Generate test data
    test_data = []
    for i in range(num_iterations):
        for sensor_type in SensorType:
            for j in range(data_per_sensor):
                data = SensorData(
                    sensor_type=sensor_type,
                    timestamp=time.time() + i * 0.01 + j * 0.001,
                    data=f"test_data_{i}_{j}",
                    confidence=np.random.uniform(0.5, 1.0),
                    source_id=f"test_sensor_{sensor_type.value}"
                )
                test_data.append(data)
    
    # Performance test
    start_time = time.time()
    
    for i in range(0, len(test_data), num_sensors * data_per_sensor):
        batch = test_data[i:i + num_sensors * data_per_sensor]
        
        # Add data to fusion engine
        for data in batch:
            fusion_engine.add_sensor_data(data)
        
        # Perform fusion
        fusion_result = fusion_engine.fuse_sensors()
        
        # Validate result
        assert isinstance(fusion_result, FusionResult)
        assert 0.0 <= fusion_result.safety_score <= 1.0
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    avg_time_per_iteration = total_time / num_iterations
    throughput = num_iterations / total_time
    
    print(f"Performance Results:")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Average time per iteration: {avg_time_per_iteration:.6f} seconds")
    print(f"  Throughput: {throughput:.2f} iterations/second")
    print(f"  Average processing time per fusion: {fusion_result.processing_time:.6f} seconds")
    
    # Performance requirements
    max_time_per_iteration = 0.01  # 10ms per iteration
    min_throughput = 100  # 100 iterations/second
    
    print(f"\nPerformance Requirements:")
    print(f"  Max time per iteration: {max_time_per_iteration:.3f} seconds")
    print(f"  Min throughput: {min_throughput:.0f} iterations/second")
    
    # Check requirements
    assert avg_time_per_iteration < max_time_per_iteration, f"Average time per iteration ({avg_time_per_iteration:.6f}s) exceeds limit ({max_time_per_iteration:.3f}s)"
    assert throughput > min_throughput, f"Throughput ({throughput:.2f} it/s) below minimum ({min_throughput:.0f} it/s)"
    
    print("✅ All performance benchmarks passed!")


if __name__ == '__main__':
    # Run unit tests
    print("Running Multi-Modal Safety Fusion Tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmarks
    print("\n" + "="*60)
    run_performance_benchmarks() 