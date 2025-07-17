#!/usr/bin/env python3

"""
Test file for Social Behavior Engine Module

This file contains unit tests for the social behavior generation functionality
to ensure proper operation and appropriateness.
"""

import unittest
import sys
import os

# Add the package to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eip_social_intelligence'))

from social_behavior_engine import (
    SocialBehaviorEngine, SocialBehavior, SocialContext, RobotState,
    BehaviorType, ResponseModality
)
from emotion_recognizer import EmotionAnalysis, EmotionType


class TestSocialBehaviorEngine(unittest.TestCase):
    """Test cases for social behavior engine functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.behavior_engine = SocialBehaviorEngine()
        
        # Create test emotion analysis
        self.test_emotion_analysis = EmotionAnalysis(
            primary_emotion=EmotionType.HAPPY,
            confidence=0.8,
            intensity=0.7,
            secondary_emotions=[(EmotionType.EXCITED, 0.6)],
            facial_features={'smile': 0.8},
            voice_features={'pitch': 0.6},
            body_language={'posture': 'open'},
            overall_emotional_state="moderately happy",
            emotional_stability=0.8
        )
        
        # Create test social context
        self.test_social_context = SocialContext(
            environment='indoor',
            relationship='friendly',
            cultural_context='western',
            social_norms={'personal_space': 1.2},
            interaction_history=[],
            current_task='assistance',
            time_of_day='day',
            privacy_level='public'
        )
        
        # Create test robot state
        self.test_robot_state = RobotState(
            capabilities=['verbal', 'gestural', 'facial', 'proxemic'],
            current_emotion='neutral',
            energy_level=0.8,
            task_engagement=0.7,
            social_comfort=0.8,
            safety_status='safe'
        )
    
    def test_initialization(self):
        """Test behavior engine initialization"""
        self.assertIsNotNone(self.behavior_engine)
        self.assertIsNotNone(self.behavior_engine.config)
        self.assertIsNotNone(self.behavior_engine.behavior_templates)
        self.assertIsNotNone(self.behavior_engine.response_generator)
        self.assertIsNotNone(self.behavior_engine.safety_validator)
        self.assertIsNotNone(self.behavior_engine.context_analyzer)
    
    def test_generate_behavior(self):
        """Test behavior generation functionality"""
        # Test with valid input
        result = self.behavior_engine.generate_behavior(
            self.test_emotion_analysis,
            self.test_social_context,
            self.test_robot_state
        )
        
        # Check that result is a SocialBehavior object
        self.assertIsInstance(result, SocialBehavior)
        
        # Check that behavior type is valid
        self.assertIn(result.behavior_type, list(BehaviorType))
        
        # Check that modality is valid
        self.assertIn(result.modality, list(ResponseModality))
        
        # Check confidence is within valid range
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        
        # Check appropriateness score is within valid range
        self.assertGreaterEqual(result.appropriateness_score, 0.0)
        self.assertLessEqual(result.appropriateness_score, 1.0)
        
        # Check safety score is within valid range
        self.assertGreaterEqual(result.safety_score, 0.0)
        self.assertLessEqual(result.safety_score, 1.0)
        
        # Check duration is positive
        self.assertGreater(result.duration, 0.0)
        
        # Check priority is valid
        self.assertGreaterEqual(result.priority, 1)
        self.assertLessEqual(result.priority, 10)
    
    def test_behavior_type_determination(self):
        """Test behavior type determination"""
        # Test different emotion scenarios
        test_cases = [
            (EmotionType.HAPPY, 0.8, 'assistance', BehaviorType.ASSISTANCE),
            (EmotionType.SAD, 0.8, None, BehaviorType.COMFORT),
            (EmotionType.CONFUSED, 0.7, None, BehaviorType.EXPLANATION),
            (EmotionType.NEUTRAL, 0.3, None, BehaviorType.CONVERSATION)
        ]
        
        for emotion, intensity, task, expected_behavior in test_cases:
            emotion_analysis = EmotionAnalysis(
                primary_emotion=emotion,
                confidence=0.8,
                intensity=intensity,
                secondary_emotions=[],
                facial_features={},
                voice_features={},
                body_language={},
                overall_emotional_state="test",
                emotional_stability=0.8
            )
            
            social_context = SocialContext(
                environment='indoor',
                relationship='friendly',
                cultural_context='western',
                social_norms={},
                interaction_history=[],
                current_task=task,
                time_of_day='day',
                privacy_level='public'
            )
            
            behavior_type = self.behavior_engine._determine_behavior_type(
                emotion_analysis, social_context, self.test_robot_state
            )
            
            self.assertEqual(behavior_type, expected_behavior)
    
    def test_response_modality_selection(self):
        """Test response modality selection"""
        # Test with different robot capabilities
        test_cases = [
            (['verbal'], BehaviorType.GREETING, ResponseModality.VERBAL),
            (['gestural'], BehaviorType.CELEBRATION, ResponseModality.GESTURAL),
            (['verbal', 'gestural'], BehaviorType.GREETING, ResponseModality.VERBAL),
            (['facial'], BehaviorType.COMFORT, ResponseModality.FACIAL)
        ]
        
        for capabilities, behavior_type, expected_modality in test_cases:
            robot_state = RobotState(
                capabilities=capabilities,
                current_emotion='neutral',
                energy_level=0.8,
                task_engagement=0.7,
                social_comfort=0.8,
                safety_status='safe'
            )
            
            modality = self.behavior_engine._select_response_modality(
                behavior_type, self.test_emotion_analysis, 
                self.test_social_context, robot_state
            )
            
            self.assertEqual(modality, expected_modality)
    
    def test_behavior_content_generation(self):
        """Test behavior content generation"""
        # Test verbal content generation
        content = self.behavior_engine._generate_behavior_content(
            BehaviorType.GREETING,
            ResponseModality.VERBAL,
            self.test_emotion_analysis,
            self.test_social_context,
            self.test_robot_state
        )
        
        self.assertIsInstance(content, dict)
        self.assertIn('verbal_response', content)
        self.assertIsInstance(content['verbal_response'], str)
        self.assertGreater(len(content['verbal_response']), 0)
        
        # Test gestural content generation
        content = self.behavior_engine._generate_behavior_content(
            BehaviorType.CELEBRATION,
            ResponseModality.GESTURAL,
            self.test_emotion_analysis,
            self.test_social_context,
            self.test_robot_state
        )
        
        self.assertIsInstance(content, dict)
        self.assertIn('gesture_type', content)
        self.assertIsInstance(content['gesture_type'], str)
    
    def test_verbal_content_personalization(self):
        """Test verbal content personalization"""
        # Test with different emotions
        test_cases = [
            (EmotionType.HAPPY, "Hello!", "Hello!"),
            (EmotionType.SAD, "Hello!", "Hello."),
            (EmotionType.NEUTRAL, "Hi there!", "Hi there!")
        ]
        
        for emotion, template, expected in test_cases:
            emotion_analysis = EmotionAnalysis(
                primary_emotion=emotion,
                confidence=0.8,
                intensity=0.7,
                secondary_emotions=[],
                facial_features={},
                voice_features={},
                body_language={},
                overall_emotional_state="test",
                emotional_stability=0.8
            )
            
            personalized = self.behavior_engine._personalize_verbal_content(
                template, emotion_analysis, self.test_social_context
            )
            
            self.assertEqual(personalized, expected)
    
    def test_behavior_validation(self):
        """Test behavior validation"""
        # Test safety validation
        content = {'verbal_response': 'Hello, how are you?'}
        safety_score = self.behavior_engine._validate_behavior_safety(
            content, self.test_social_context
        )
        
        self.assertGreaterEqual(safety_score, 0.0)
        self.assertLessEqual(safety_score, 1.0)
        
        # Test appropriateness validation
        appropriateness_score = self.behavior_engine._validate_behavior_appropriateness(
            content, self.test_emotion_analysis, self.test_social_context
        )
        
        self.assertGreaterEqual(appropriateness_score, 0.0)
        self.assertLessEqual(appropriateness_score, 1.0)
    
    def test_confidence_calculation(self):
        """Test confidence calculation"""
        content = {'verbal_response': 'Hello!'}
        confidence = self.behavior_engine._calculate_behavior_confidence(
            BehaviorType.GREETING,
            ResponseModality.VERBAL,
            content,
            self.test_emotion_analysis,
            self.test_social_context
        )
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_priority_calculation(self):
        """Test priority calculation"""
        # Test different behavior types
        test_cases = [
            (BehaviorType.APOLOGY, 1),
            (BehaviorType.COMFORT, 2),
            (BehaviorType.GREETING, 4),
            (BehaviorType.CONVERSATION, 5)
        ]
        
        for behavior_type, expected_priority in test_cases:
            priority = self.behavior_engine._calculate_behavior_priority(
                behavior_type, self.test_emotion_analysis, self.test_social_context
            )
            
            self.assertEqual(priority, expected_priority)
    
    def test_duration_estimation(self):
        """Test duration estimation"""
        # Test verbal duration estimation
        verbal_content = {'verbal_response': 'Hello, how are you today?'}
        duration = self.behavior_engine._estimate_behavior_duration(
            verbal_content, ResponseModality.VERBAL
        )
        
        self.assertGreater(duration, 0.0)
        
        # Test gestural duration estimation
        gestural_content = {'gesture_duration': 1.5}
        duration = self.behavior_engine._estimate_behavior_duration(
            gestural_content, ResponseModality.GESTURAL
        )
        
        self.assertEqual(duration, 1.5)
    
    def test_error_handling(self):
        """Test error handling in behavior generation"""
        # Test with invalid input that should cause errors
        invalid_emotion = "invalid_emotion"  # Should be EmotionAnalysis object
        
        # Should not raise an exception, should return default behavior
        result = self.behavior_engine.generate_behavior(
            invalid_emotion, self.test_social_context, self.test_robot_state
        )
        
        self.assertIsInstance(result, SocialBehavior)
        self.assertIn(result.behavior_type, list(BehaviorType))
    
    def test_behavior_history_tracking(self):
        """Test behavior history tracking"""
        # Generate multiple behaviors
        for i in range(5):
            result = self.behavior_engine.generate_behavior(
                self.test_emotion_analysis,
                self.test_social_context,
                self.test_robot_state
            )
        
        # Check that history is being tracked
        self.assertGreater(len(self.behavior_engine.behavior_history), 0)
        self.assertLessEqual(len(self.behavior_engine.behavior_history), 
                           self.behavior_engine.config.get('max_behavior_history', 20))


if __name__ == '__main__':
    unittest.main() 