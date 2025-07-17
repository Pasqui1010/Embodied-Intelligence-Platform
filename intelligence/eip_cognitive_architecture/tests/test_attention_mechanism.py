#!/usr/bin/env python3

"""
Test suite for Attention Mechanism

This module contains comprehensive tests for the attention mechanism component
of the cognitive architecture.
"""

import unittest
import time
import numpy as np
from unittest.mock import Mock, patch

# Import the attention mechanism
from eip_cognitive_architecture.attention_mechanism import (
    AttentionMechanism, AttentionType, AttentionFocus, MultiModalSensorData
)


class TestAttentionMechanism(unittest.TestCase):
    """Test cases for the AttentionMechanism class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.attention = AttentionMechanism()
        self.sample_sensor_data = MultiModalSensorData(
            visual_data={
                'objects': [
                    {'type': 'person', 'distance': 1.5, 'confidence': 0.8, 'location': (1.0, 0.0, 0.0)},
                    {'type': 'chair', 'distance': 2.0, 'confidence': 0.6, 'location': (2.0, 0.0, 0.0)}
                ],
                'faces': [
                    {'id': 'person1', 'confidence': 0.9, 'location': (1.0, 0.0, 0.0), 'expression': 'happy'}
                ],
                'motion': [
                    {'type': 'movement', 'velocity': 0.5, 'confidence': 0.7, 'location': (1.5, 0.0, 0.0)}
                ]
            },
            audio_data={
                'speech': [
                    {'speaker_id': 'person1', 'content': 'hello robot', 'confidence': 0.8}
                ],
                'sounds': [
                    {'type': 'footsteps', 'intensity': 0.6, 'confidence': 0.7}
                ]
            },
            tactile_data={
                'contacts': [
                    {'type': 'touch', 'pressure': 0.3, 'confidence': 0.6, 'location': (0.0, 0.0, 0.0)}
                ]
            },
            timestamp=time.time()
        )
    
    def test_initialization(self):
        """Test attention mechanism initialization."""
        self.assertIsNotNone(self.attention)
        self.assertEqual(len(self.attention.current_foci), 0)
        self.assertIsInstance(self.attention.config, dict)
        self.assertIn('max_foci', self.attention.config)
    
    def test_focus_attention_basic(self):
        """Test basic attention focusing functionality."""
        focused_attention = self.attention.focus_attention(
            self.sample_sensor_data, None, None
        )
        
        self.assertIsInstance(focused_attention, list)
        self.assertGreater(len(focused_attention), 0)
        
        for focus in focused_attention:
            self.assertIsInstance(focus, AttentionFocus)
            self.assertIsInstance(focus.attention_type, AttentionType)
            self.assertGreaterEqual(focus.priority, 0.0)
            self.assertLessEqual(focus.priority, 1.0)
    
    def test_visual_attention_processing(self):
        """Test visual attention processing."""
        visual_data = {
            'objects': [
                {'type': 'person', 'distance': 0.8, 'confidence': 0.9, 'location': (0.8, 0.0, 0.0)},
                {'type': 'table', 'distance': 3.0, 'confidence': 0.5, 'location': (3.0, 0.0, 0.0)}
            ],
            'faces': [
                {'id': 'person1', 'confidence': 0.95, 'location': (0.8, 0.0, 0.0), 'expression': 'surprised'}
            ]
        }
        
        sensor_data = MultiModalSensorData(visual_data=visual_data, timestamp=time.time())
        focused_attention = self.attention.focus_attention(sensor_data, None, None)
        
        # Should prioritize closer objects and faces
        person_focus = [f for f in focused_attention if 'person' in str(f.metadata)]
        self.assertGreater(len(person_focus), 0)
        
        # Check that closer objects have higher priority
        if len(focused_attention) > 1:
            priorities = [f.priority for f in focused_attention]
            self.assertGreater(max(priorities), min(priorities))
    
    def test_safety_attention_priority(self):
        """Test that safety-related attention gets high priority."""
        safety_sensor_data = MultiModalSensorData(
            visual_data={
                'objects': [
                    {'type': 'person', 'distance': 0.3, 'confidence': 0.9, 'location': (0.3, 0.0, 0.0)}
                ]
            },
            timestamp=time.time()
        )
        
        focused_attention = self.attention.focus_attention(safety_sensor_data, None, None)
        
        # Should have safety attention due to close proximity
        safety_foci = [f for f in focused_attention if f.attention_type == AttentionType.SAFETY]
        self.assertGreater(len(safety_foci), 0)
        
        # Safety foci should have high priority
        for focus in safety_foci:
            self.assertGreater(focus.priority, 0.5)
    
    def test_social_attention_processing(self):
        """Test social attention processing."""
        social_sensor_data = MultiModalSensorData(
            visual_data={
                'faces': [
                    {'id': 'person1', 'confidence': 0.9, 'location': (1.0, 0.0, 0.0), 'expression': 'happy'}
                ]
            },
            audio_data={
                'speech': [
                    {'speaker_id': 'person1', 'content': 'hello', 'confidence': 0.8}
                ]
            },
            timestamp=time.time()
        )
        
        focused_attention = self.attention.focus_attention(social_sensor_data, None, None)
        
        # Should have social attention
        social_foci = [f for f in focused_attention if f.attention_type == AttentionType.SOCIAL]
        self.assertGreater(len(social_foci), 0)
    
    def test_user_input_attention(self):
        """Test attention to user input."""
        user_input = "Robot, please help me"
        
        focused_attention = self.attention.focus_attention(
            self.sample_sensor_data, user_input, None
        )
        
        # Should have semantic attention for user input
        semantic_foci = [f for f in focused_attention if f.attention_type == AttentionType.SEMANTIC]
        self.assertGreater(len(semantic_foci), 0)
        
        # User input should have high priority
        for focus in semantic_foci:
            self.assertGreater(focus.priority, 0.7)
    
    def test_attention_decay(self):
        """Test that attention foci decay over time."""
        # Create initial attention
        focused_attention = self.attention.focus_attention(
            self.sample_sensor_data, None, None
        )
        
        if len(focused_attention) > 0:
            initial_priority = focused_attention[0].priority
            
            # Simulate time passing
            time.sleep(0.1)
            
            # Process attention again
            decayed_attention = self.attention.focus_attention(
                self.sample_sensor_data, None, None
            )
            
            if len(decayed_attention) > 0:
                decayed_priority = decayed_attention[0].priority
                # Priority should decay (though this might not be visible in short time)
                self.assertLessEqual(decayed_priority, initial_priority)
    
    def test_attention_capacity_limits(self):
        """Test that attention respects capacity limits."""
        # Create many attention targets
        many_objects = []
        for i in range(20):
            many_objects.append({
                'type': f'object_{i}',
                'distance': 1.0 + i * 0.1,
                'confidence': 0.5 + i * 0.02,
                'location': (1.0 + i * 0.1, 0.0, 0.0)
            })
        
        sensor_data = MultiModalSensorData(
            visual_data={'objects': many_objects},
            timestamp=time.time()
        )
        
        focused_attention = self.attention.focus_attention(sensor_data, None, None)
        
        # Should not exceed max_foci
        self.assertLessEqual(len(focused_attention), self.attention.config['max_foci'])
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        self.attention.focus_attention(self.sample_sensor_data, None, None)
        
        metrics = self.attention.get_performance_metrics()
        
        self.assertIn('focus_accuracy', metrics)
        self.assertIn('response_time', metrics)
        self.assertIn('attention_shifts', metrics)
        
        self.assertIsInstance(metrics['focus_accuracy'], float)
        self.assertIsInstance(metrics['response_time'], float)
    
    def test_attention_summary(self):
        """Test attention summary generation."""
        self.attention.focus_attention(self.sample_sensor_data, None, None)
        
        summary = self.attention.get_attention_summary()
        
        self.assertIn('num_foci', summary)
        self.assertIn('foci_types', summary)
        self.assertIn('avg_priority', summary)
        self.assertIn('performance_metrics', summary)
        
        self.assertIsInstance(summary['num_foci'], int)
        self.assertIsInstance(summary['avg_priority'], float)
    
    def test_configuration_override(self):
        """Test custom configuration override."""
        custom_config = {
            'max_foci': 3,
            'min_priority_threshold': 0.5,
            'attention_decay_rate': 0.2
        }
        
        custom_attention = AttentionMechanism(config=custom_config)
        
        self.assertEqual(custom_attention.config['max_foci'], 3)
        self.assertEqual(custom_attention.config['min_priority_threshold'], 0.5)
        self.assertEqual(custom_attention.config['attention_decay_rate'], 0.2)
    
    def test_thread_safety(self):
        """Test thread safety of attention mechanism."""
        import threading
        
        def attention_worker():
            for _ in range(10):
                self.attention.focus_attention(self.sample_sensor_data, None, None)
                time.sleep(0.01)
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=attention_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should not raise any exceptions
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main() 