#!/usr/bin/env python3

"""
Test file for Emotion Recognition Module

This file contains unit tests for the emotion recognition functionality
to ensure proper operation and accuracy.
"""

import unittest
import numpy as np
import sys
import os

# Add the package to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eip_social_intelligence'))

from emotion_recognizer import EmotionRecognizer, HumanInput, EmotionType, EmotionAnalysis


class TestEmotionRecognition(unittest.TestCase):
    """Test cases for emotion recognition functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.emotion_recognizer = EmotionRecognizer()
        
        # Create test human input
        self.test_human_input = HumanInput(
            facial_image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            voice_audio=np.random.randn(16000).astype(np.float32),
            body_pose=np.random.randn(17, 3),  # 17 keypoints
            speech_text="Hello, how are you?",
            gesture_data={'gesture_type': 'wave', 'confidence': 0.8},
            timestamp=1234567890.0
        )
        
        # Create test social context
        self.test_social_context = {
            'environment': 'indoor',
            'relationship': 'friendly',
            'cultural_context': 'western',
            'time_of_day': 'day',
            'privacy_level': 'public'
        }
    
    def test_initialization(self):
        """Test emotion recognizer initialization"""
        self.assertIsNotNone(self.emotion_recognizer)
        self.assertIsNotNone(self.emotion_recognizer.config)
        self.assertIsNotNone(self.emotion_recognizer.facial_recognizer)
        self.assertIsNotNone(self.emotion_recognizer.voice_recognizer)
        self.assertIsNotNone(self.emotion_recognizer.body_recognizer)
    
    def test_analyze_emotions(self):
        """Test emotion analysis functionality"""
        # Test with valid input
        result = self.emotion_recognizer.analyze_emotions(
            self.test_human_input, self.test_social_context
        )
        
        # Check that result is an EmotionAnalysis object
        self.assertIsInstance(result, EmotionAnalysis)
        
        # Check that primary emotion is valid
        self.assertIn(result.primary_emotion, list(EmotionType))
        
        # Check confidence is within valid range
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        
        # Check intensity is within valid range
        self.assertGreaterEqual(result.intensity, 0.0)
        self.assertLessEqual(result.intensity, 1.0)
        
        # Check emotional stability is within valid range
        self.assertGreaterEqual(result.emotional_stability, 0.0)
        self.assertLessEqual(result.emotional_stability, 1.0)
    
    def test_analyze_emotions_with_none_input(self):
        """Test emotion analysis with None input"""
        # Test with None facial image
        input_none_facial = HumanInput(
            facial_image=None,
            voice_audio=self.test_human_input.voice_audio,
            body_pose=self.test_human_input.body_pose,
            speech_text=self.test_human_input.speech_text
        )
        
        result = self.emotion_recognizer.analyze_emotions(
            input_none_facial, self.test_social_context
        )
        
        self.assertIsInstance(result, EmotionAnalysis)
        self.assertIn(result.primary_emotion, list(EmotionType))
    
    def test_analyze_emotions_with_empty_input(self):
        """Test emotion analysis with empty input"""
        # Test with empty input
        empty_input = HumanInput()
        
        result = self.emotion_recognizer.analyze_emotions(
            empty_input, self.test_social_context
        )
        
        self.assertIsInstance(result, EmotionAnalysis)
        self.assertIn(result.primary_emotion, list(EmotionType))
    
    def test_emotion_history(self):
        """Test emotion history tracking"""
        # Perform multiple analyses
        for i in range(5):
            result = self.emotion_recognizer.analyze_emotions(
                self.test_human_input, self.test_social_context
            )
        
        # Check that history is being tracked
        self.assertGreater(len(self.emotion_recognizer.emotion_history), 0)
        self.assertLessEqual(len(self.emotion_recognizer.emotion_history), 
                           self.emotion_recognizer.config.get('max_history_length', 10))
    
    def test_facial_emotion_analysis(self):
        """Test facial emotion analysis specifically"""
        # Test facial emotion analysis
        facial_emotion = self.emotion_recognizer._analyze_facial_emotion(
            self.test_human_input.facial_image
        )
        
        self.assertIsInstance(facial_emotion, dict)
        self.assertIn('emotion', facial_emotion)
        self.assertIn('confidence', facial_emotion)
        self.assertIn(facial_emotion['emotion'], list(EmotionType))
        self.assertGreaterEqual(facial_emotion['confidence'], 0.0)
        self.assertLessEqual(facial_emotion['confidence'], 1.0)
    
    def test_voice_emotion_analysis(self):
        """Test voice emotion analysis specifically"""
        # Test voice emotion analysis
        voice_emotion = self.emotion_recognizer._analyze_voice_emotion(
            self.test_human_input.voice_audio,
            self.test_human_input.speech_text
        )
        
        self.assertIsInstance(voice_emotion, dict)
        self.assertIn('emotion', voice_emotion)
        self.assertIn('confidence', voice_emotion)
        self.assertIn(voice_emotion['emotion'], list(EmotionType))
        self.assertGreaterEqual(voice_emotion['confidence'], 0.0)
        self.assertLessEqual(voice_emotion['confidence'], 1.0)
    
    def test_body_emotion_analysis(self):
        """Test body emotion analysis specifically"""
        # Test body emotion analysis
        body_emotion = self.emotion_recognizer._analyze_body_emotion(
            self.test_human_input.body_pose,
            self.test_human_input.gesture_data
        )
        
        self.assertIsInstance(body_emotion, dict)
        self.assertIn('emotion', body_emotion)
        self.assertIn('confidence', body_emotion)
        self.assertIn(body_emotion['emotion'], list(EmotionType))
        self.assertGreaterEqual(body_emotion['confidence'], 0.0)
        self.assertLessEqual(body_emotion['confidence'], 1.0)
    
    def test_emotion_fusion(self):
        """Test emotion fusion from multiple modalities"""
        # Test emotion fusion
        facial_emotion = {'emotion': EmotionType.HAPPY, 'confidence': 0.8}
        voice_emotion = {'emotion': EmotionType.NEUTRAL, 'confidence': 0.6}
        body_emotion = {'emotion': EmotionType.HAPPY, 'confidence': 0.7}
        
        fused_emotion = self.emotion_recognizer._fuse_emotions(
            facial_emotion, voice_emotion, body_emotion
        )
        
        self.assertIsInstance(fused_emotion, dict)
        self.assertIn('emotion', fused_emotion)
        self.assertIn('confidence', fused_emotion)
        self.assertIn('modalities', fused_emotion)
        self.assertIn(fused_emotion['emotion'], list(EmotionType))
        self.assertGreaterEqual(fused_emotion['confidence'], 0.0)
        self.assertLessEqual(fused_emotion['confidence'], 1.0)
    
    def test_emotional_stability_calculation(self):
        """Test emotional stability calculation"""
        # Add some emotions to history
        emotions = [EmotionType.HAPPY, EmotionType.HAPPY, EmotionType.NEUTRAL, 
                   EmotionType.HAPPY, EmotionType.HAPPY]
        
        for emotion in emotions:
            self.emotion_recognizer.emotion_history.append({
                'emotion': emotion,
                'confidence': 0.8
            })
        
        stability = self.emotion_recognizer._calculate_emotional_stability()
        
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
    
    def test_secondary_emotions_generation(self):
        """Test secondary emotions generation"""
        # Test with different primary emotions
        test_cases = [
            EmotionType.HAPPY,
            EmotionType.SAD,
            EmotionType.NEUTRAL
        ]
        
        for primary_emotion in test_cases:
            emotion_dict = {'emotion': primary_emotion, 'confidence': 0.8}
            secondary_emotions = self.emotion_recognizer._generate_secondary_emotions(emotion_dict)
            
            self.assertIsInstance(secondary_emotions, list)
            for emotion_tuple in secondary_emotions:
                self.assertIsInstance(emotion_tuple, tuple)
                self.assertEqual(len(emotion_tuple), 2)
                self.assertIn(emotion_tuple[0], list(EmotionType))
                self.assertGreaterEqual(emotion_tuple[1], 0.0)
                self.assertLessEqual(emotion_tuple[1], 1.0)
    
    def test_error_handling(self):
        """Test error handling in emotion analysis"""
        # Test with invalid input that should cause errors
        invalid_input = HumanInput(
            facial_image="invalid_image",  # Should be numpy array
            voice_audio="invalid_audio",   # Should be numpy array
            body_pose="invalid_pose"       # Should be numpy array
        )
        
        # Should not raise an exception, should return default analysis
        result = self.emotion_recognizer.analyze_emotions(
            invalid_input, self.test_social_context
        )
        
        self.assertIsInstance(result, EmotionAnalysis)
        self.assertIn(result.primary_emotion, list(EmotionType))


if __name__ == '__main__':
    unittest.main() 