#!/usr/bin/env python3

"""
Test file for Cultural Adaptation Module

This file contains unit tests for the cultural adaptation functionality
to ensure proper cultural sensitivity and appropriateness.
"""

import unittest
import sys
import os

# Add the package to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eip_social_intelligence'))

from cultural_adaptation import (
    CulturalAdaptationEngine, CulturalAdaptation, CulturalProfile,
    CulturalDimension, CommunicationStyle
)


class TestCulturalAdaptation(unittest.TestCase):
    """Test cases for cultural adaptation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cultural_adaptation = CulturalAdaptationEngine()
        
        # Create test behavior
        self.test_behavior = {
            'verbal_response': 'Hello! How are you today?',
            'gesture_type': 'wave',
            'facial_expression': 'smile',
            'proxemic_behavior': 'approach_politely',
            'eye_contact': 0.8,
            'formality_level': 0.4
        }
        
        # Create test social context
        self.test_social_context = {
            'environment': 'indoor',
            'relationship': 'friendly',
            'cultural_context': 'western',
            'time_of_day': 'day',
            'privacy_level': 'public'
        }
    
    def test_initialization(self):
        """Test cultural adaptation engine initialization"""
        self.assertIsNotNone(self.cultural_adaptation)
        self.assertIsNotNone(self.cultural_adaptation.config)
        self.assertIsNotNone(self.cultural_adaptation.cultural_database)
        self.assertIsNotNone(self.cultural_adaptation.adaptation_rules)
        self.assertIsNotNone(self.cultural_adaptation.sensitivity_checker)
        self.assertIsNotNone(self.cultural_adaptation.cultural_analyzer)
    
    def test_adapt_behavior(self):
        """Test behavior adaptation functionality"""
        # Test with valid input
        result = self.cultural_adaptation.adapt_behavior(
            self.test_behavior,
            'western',
            self.test_social_context
        )
        
        # Check that result is a CulturalAdaptation object
        self.assertIsInstance(result, CulturalAdaptation)
        
        # Check that adapted behavior is a dictionary
        self.assertIsInstance(result.adapted_behavior, dict)
        
        # Check sensitivity score is within valid range
        self.assertGreaterEqual(result.cultural_sensitivity_score, 0.0)
        self.assertLessEqual(result.cultural_sensitivity_score, 1.0)
        
        # Check appropriateness score is within valid range
        self.assertGreaterEqual(result.appropriateness_score, 0.0)
        self.assertLessEqual(result.appropriateness_score, 1.0)
        
        # Check adaptation confidence is within valid range
        self.assertGreaterEqual(result.adaptation_confidence, 0.0)
        self.assertLessEqual(result.adaptation_confidence, 1.0)
        
        # Check cultural notes is a list
        self.assertIsInstance(result.cultural_notes, list)
        
        # Check risk factors is a list
        self.assertIsInstance(result.risk_factors, list)
    
    def test_cultural_profile_retrieval(self):
        """Test cultural profile retrieval"""
        # Test different cultural contexts
        test_cases = [
            ('western', 'western'),
            ('eastern', 'eastern'),
            ('asian', 'eastern'),
            ('middle_eastern', 'middle_eastern'),
            ('arabic', 'middle_eastern'),
            ('latin_american', 'latin_american'),
            ('hispanic', 'latin_american'),
            ('unknown', 'western')  # Default fallback
        ]
        
        for input_context, expected_profile in test_cases:
            profile = self.cultural_adaptation._get_cultural_profile(input_context)
            
            self.assertIsInstance(profile, CulturalProfile)
            self.assertEqual(profile.culture_name.lower(), expected_profile.title())
    
    def test_cultural_requirements_analysis(self):
        """Test cultural requirements analysis"""
        # Test with western culture
        western_profile = self.cultural_adaptation._get_cultural_profile('western')
        requirements = self.cultural_adaptation._analyze_cultural_requirements(
            self.test_behavior, western_profile, self.test_social_context
        )
        
        self.assertIsInstance(requirements, dict)
        self.assertIn('communication_style', requirements)
        self.assertIn('interaction_distance', requirements)
        self.assertIn('eye_contact_level', requirements)
        self.assertIn('gesture_sensitivity', requirements)
        self.assertIn('formality_level', requirements)
        self.assertIn('taboo_avoidance', requirements)
        self.assertIn('social_norms', requirements)
        
        # Check that values are within expected ranges
        self.assertGreaterEqual(requirements['interaction_distance'], 0.0)
        self.assertLessEqual(requirements['eye_contact_level'], 1.0)
        self.assertGreaterEqual(requirements['gesture_sensitivity'], 0.0)
        self.assertLessEqual(requirements['gesture_sensitivity'], 1.0)
        self.assertGreaterEqual(requirements['formality_level'], 0.0)
        self.assertLessEqual(requirements['formality_level'], 1.0)
    
    def test_communication_style_determination(self):
        """Test communication style determination"""
        # Test western culture (individualistic)
        western_profile = CulturalProfile(
            culture_name='Western',
            region='North America',
            language='English',
            cultural_dimensions={
                CulturalDimension.INDIVIDUALISM_COLLECTIVISM: 0.8
            },
            communication_style=CommunicationStyle.DIRECT,
            social_norms={},
            taboos=[],
            preferred_interaction_distance=1.2,
            eye_contact_preference=0.8,
            gesture_sensitivity=0.3,
            formality_level=0.4
        )
        
        style = self.cultural_adaptation._determine_communication_style(western_profile)
        self.assertEqual(style, CommunicationStyle.DIRECT)
        
        # Test eastern culture (collectivistic)
        eastern_profile = CulturalProfile(
            culture_name='Eastern',
            region='Asia',
            language='Various',
            cultural_dimensions={
                CulturalDimension.INDIVIDUALISM_COLLECTIVISM: 0.3
            },
            communication_style=CommunicationStyle.INDIRECT,
            social_norms={},
            taboos=[],
            preferred_interaction_distance=0.8,
            eye_contact_preference=0.4,
            gesture_sensitivity=0.7,
            formality_level=0.8
        )
        
        style = self.cultural_adaptation._determine_communication_style(eastern_profile)
        self.assertEqual(style, CommunicationStyle.INDIRECT)
    
    def test_verbal_communication_adaptation(self):
        """Test verbal communication adaptation"""
        # Test formality adaptation
        requirements = {'formality_level': 0.8, 'communication_style': CommunicationStyle.INDIRECT}
        western_profile = self.cultural_adaptation._get_cultural_profile('western')
        
        adapted_verbal = self.cultural_adaptation._adapt_verbal_communication(
            'Hi! How are you?', requirements, western_profile
        )
        
        self.assertIsInstance(adapted_verbal, str)
        self.assertGreater(len(adapted_verbal), 0)
        
        # Test taboo replacement
        requirements = {'formality_level': 0.5, 'communication_style': CommunicationStyle.DIRECT}
        taboo_verbal = self.cultural_adaptation._adapt_verbal_communication(
            'How old are you?', requirements, western_profile
        )
        
        self.assertIsInstance(taboo_verbal, str)
        self.assertNotIn('age', taboo_verbal.lower())
    
    def test_proxemic_behavior_adaptation(self):
        """Test proxemic behavior adaptation"""
        # Test western culture (larger distance)
        requirements = {'interaction_distance': 1.2}
        western_profile = self.cultural_adaptation._get_cultural_profile('western')
        
        adapted_proxemic = self.cultural_adaptation._adapt_proxemic_behavior(
            'approach_closer', requirements, western_profile
        )
        
        self.assertIsInstance(adapted_proxemic, str)
        self.assertIn('maintain_distance', adapted_proxemic)
        
        # Test eastern culture (smaller distance)
        requirements = {'interaction_distance': 0.8}
        eastern_profile = self.cultural_adaptation._get_cultural_profile('eastern')
        
        adapted_proxemic = self.cultural_adaptation._adapt_proxemic_behavior(
            'maintain_distance', requirements, eastern_profile
        )
        
        self.assertIsInstance(adapted_proxemic, str)
        self.assertIn('approach_closer', adapted_proxemic)
    
    def test_eye_contact_adaptation(self):
        """Test eye contact adaptation"""
        # Test western culture (higher eye contact)
        requirements = {'eye_contact_level': 0.8}
        western_profile = self.cultural_adaptation._get_cultural_profile('western')
        
        adapted_eye_contact = self.cultural_adaptation._adapt_eye_contact(
            0.5, requirements, western_profile
        )
        
        self.assertIsInstance(adapted_eye_contact, float)
        self.assertGreaterEqual(adapted_eye_contact, 0.0)
        self.assertLessEqual(adapted_eye_contact, 1.0)
        
        # Test eastern culture (lower eye contact)
        requirements = {'eye_contact_level': 0.4}
        eastern_profile = self.cultural_adaptation._get_cultural_profile('eastern')
        
        adapted_eye_contact = self.cultural_adaptation._adapt_eye_contact(
            0.8, requirements, eastern_profile
        )
        
        self.assertIsInstance(adapted_eye_contact, float)
        self.assertGreaterEqual(adapted_eye_contact, 0.0)
        self.assertLessEqual(adapted_eye_contact, 1.0)
    
    def test_gesture_adaptation(self):
        """Test gesture adaptation"""
        # Test western culture (lower sensitivity)
        requirements = {'gesture_sensitivity': 0.3}
        western_profile = self.cultural_adaptation._get_cultural_profile('western')
        
        adapted_gesture = self.cultural_adaptation._adapt_gestures(
            'pointing', requirements, western_profile
        )
        
        self.assertIsInstance(adapted_gesture, str)
        self.assertEqual(adapted_gesture, 'pointing')  # Should remain unchanged
        
        # Test eastern culture (higher sensitivity)
        requirements = {'gesture_sensitivity': 0.7}
        eastern_profile = self.cultural_adaptation._get_cultural_profile('eastern')
        
        adapted_gesture = self.cultural_adaptation._adapt_gestures(
            'pointing', requirements, eastern_profile
        )
        
        self.assertIsInstance(adapted_gesture, str)
        self.assertEqual(adapted_gesture, 'neutral_gesture')  # Should be changed
    
    def test_taboo_replacement(self):
        """Test taboo content replacement"""
        # Test various taboo replacements
        test_cases = [
            ('personal_questions', 'general_questions'),
            ('age_questions', 'experience_questions'),
            ('direct_refusal', 'polite_decline'),
            ('pointing_feet', 'pointing_hand'),
            ('touching_head', 'respectful_gesture'),
            ('left_hand_use', 'right_hand_use'),
            ('showing_soles', 'respectful_posture'),
            ('direct_criticism', 'constructive_feedback')
        ]
        
        for taboo, expected_replacement in test_cases:
            replaced_text = self.cultural_adaptation._replace_taboo_content(
                f"This contains {taboo} content", taboo
            )
            
            self.assertIsInstance(replaced_text, str)
            self.assertIn(expected_replacement, replaced_text)
            self.assertNotIn(taboo, replaced_text)
    
    def test_cultural_sensitivity_validation(self):
        """Test cultural sensitivity validation"""
        # Test with appropriate behavior
        appropriate_behavior = {
            'verbal_response': 'Hello, how are you?',
            'gesture_type': 'wave',
            'facial_expression': 'smile'
        }
        
        western_profile = self.cultural_adaptation._get_cultural_profile('western')
        sensitivity_score = self.cultural_adaptation._validate_cultural_sensitivity(
            appropriate_behavior, western_profile
        )
        
        self.assertGreaterEqual(sensitivity_score, 0.0)
        self.assertLessEqual(sensitivity_score, 1.0)
        
        # Test with taboo behavior
        taboo_behavior = {
            'verbal_response': 'How old are you?',
            'gesture_type': 'pointing_feet',
            'facial_expression': 'smile'
        }
        
        sensitivity_score = self.cultural_adaptation._validate_cultural_sensitivity(
            taboo_behavior, western_profile
        )
        
        self.assertGreaterEqual(sensitivity_score, 0.0)
        self.assertLessEqual(sensitivity_score, 1.0)
    
    def test_cultural_appropriateness_validation(self):
        """Test cultural appropriateness validation"""
        # Test with appropriate behavior
        appropriate_behavior = {
            'verbal_response': 'Hello, how are you?',
            'formality_level': 0.4
        }
        
        western_profile = self.cultural_adaptation._get_cultural_profile('western')
        appropriateness_score = self.cultural_adaptation._validate_cultural_appropriateness(
            appropriate_behavior, western_profile, self.test_social_context
        )
        
        self.assertGreaterEqual(appropriateness_score, 0.0)
        self.assertLessEqual(appropriateness_score, 1.0)
    
    def test_adaptation_confidence_calculation(self):
        """Test adaptation confidence calculation"""
        # Test confidence calculation
        adapted_behavior = {'verbal_response': 'Hello!'}
        western_profile = self.cultural_adaptation._get_cultural_profile('western')
        requirements = {'formality_level': 0.4, 'communication_style': CommunicationStyle.DIRECT}
        
        confidence = self.cultural_adaptation._calculate_adaptation_confidence(
            adapted_behavior, western_profile, requirements
        )
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_cultural_notes_generation(self):
        """Test cultural notes generation"""
        # Test notes generation
        adapted_behavior = {'verbal_response': 'Hello!'}
        western_profile = self.cultural_adaptation._get_cultural_profile('western')
        
        notes = self.cultural_adaptation._generate_cultural_notes(
            adapted_behavior, western_profile
        )
        
        self.assertIsInstance(notes, list)
        for note in notes:
            self.assertIsInstance(note, str)
            self.assertGreater(len(note), 0)
    
    def test_risk_factors_identification(self):
        """Test risk factors identification"""
        # Test risk identification
        adapted_behavior = {'gesture_type': 'pointing'}
        western_profile = self.cultural_adaptation._get_cultural_profile('western')
        
        risk_factors = self.cultural_adaptation._identify_risk_factors(
            adapted_behavior, western_profile
        )
        
        self.assertIsInstance(risk_factors, list)
        for risk in risk_factors:
            self.assertIsInstance(risk, str)
            self.assertGreater(len(risk), 0)
    
    def test_error_handling(self):
        """Test error handling in cultural adaptation"""
        # Test with invalid input that should cause errors
        invalid_behavior = "invalid_behavior"  # Should be dict
        
        # Should not raise an exception, should return default adaptation
        result = self.cultural_adaptation.adapt_behavior(
            invalid_behavior, 'western', self.test_social_context
        )
        
        self.assertIsInstance(result, CulturalAdaptation)
        self.assertIsInstance(result.adapted_behavior, dict)
    
    def test_adaptation_history_tracking(self):
        """Test adaptation history tracking"""
        # Perform multiple adaptations
        for i in range(5):
            result = self.cultural_adaptation.adapt_behavior(
                self.test_behavior, 'western', self.test_social_context
            )
        
        # Check that history is being tracked
        self.assertGreater(len(self.cultural_adaptation.adaptation_history), 0)
        self.assertLessEqual(len(self.cultural_adaptation.adaptation_history), 
                           self.cultural_adaptation.config.get('max_adaptation_history', 50))


if __name__ == '__main__':
    unittest.main() 