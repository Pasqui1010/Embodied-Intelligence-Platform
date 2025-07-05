#!/usr/bin/env python3
"""
Safety Fixes Test Suite

Comprehensive tests for the safety improvements implemented:
1. Safety Token Vocabulary Validation
2. Async Model Inference
3. Model Integrity Verification
4. Configurable Safety Thresholds
"""

import unittest
import time
import threading
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the safety components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../intelligence/eip_llm_interface/eip_llm_interface'))

from safety_embedded_llm import SafetyEmbeddedLLM, SafetyToken
from model_integrity import ModelIntegrityVerifier, verify_model_safety
from safety_config import SafetyConfigManager, get_safety_config, update_safety_threshold


class TestSafetyTokenValidation(unittest.TestCase):
    """Test safety token vocabulary validation"""
    
    def setUp(self):
        """Set up test environment"""
        self.llm = SafetyEmbeddedLLM(model_name="microsoft/DialoGPT-medium")
    
    def test_token_conflict_detection(self):
        """Test detection of token conflicts"""
        # Mock tokenizer with conflicting tokens
        mock_vocab = {
            "<|collision_risk|>": 1000,  # Conflict with our safety token
            "normal_token": 1001
        }
        
        with patch.object(self.llm.tokenizer, 'get_vocab', return_value=mock_vocab):
            conflicts = self.llm._check_token_conflicts(["<|collision_risk|>"])
            self.assertIn("<|collision_risk|>", conflicts)
    
    def test_model_compatibility_validation(self):
        """Test model compatibility validation"""
        # Test with mock model (should fail)
        self.llm.model = "mock_model"
        self.llm.tokenizer = "mock_tokenizer"
        
        is_compatible = self.llm._validate_model_compatibility()
        self.assertFalse(is_compatible)
    
    def test_fallback_token_mapping(self):
        """Test fallback token mapping for incompatible models"""
        self.llm._use_fallback_safety_tokens()
        
        self.assertTrue(self.llm.safety_tokens_added)
        self.assertIsNotNone(self.llm.safety_token_mapping)
        self.assertIn(SafetyToken.COLLISION_RISK, self.llm.safety_token_mapping)


class TestAsyncModelInference(unittest.TestCase):
    """Test async model inference functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.llm = SafetyEmbeddedLLM(model_name="microsoft/DialoGPT-medium")
    
    def test_inference_thread_startup(self):
        """Test inference thread startup"""
        self.llm._start_inference_thread()
        
        self.assertIsNotNone(self.llm.inference_thread)
        self.assertTrue(self.llm.inference_thread.is_alive())
        self.assertTrue(self.llm.inference_running)
    
    def test_async_inference_workflow(self):
        """Test complete async inference workflow"""
        # Start inference thread
        self.llm._start_inference_thread()
        
        # Submit inference request
        test_prompt = "Test prompt"
        response = self.llm._generate_response_async(test_prompt)
        
        # Should get a response (either real or fallback)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_inference_timeout_handling(self):
        """Test timeout handling in async inference"""
        # Mock a slow inference
        with patch.object(self.llm, '_generate_real_response', side_effect=lambda x: time.sleep(2)):
            response = self.llm._generate_response_async("slow_prompt")
            
            # Should get fallback response due to timeout
            self.assertIn("timeout_fallback", response)


class TestModelIntegrityVerification(unittest.TestCase):
    """Test model integrity verification"""
    
    def setUp(self):
        """Set up test environment"""
        self.verifier = ModelIntegrityVerifier()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checksum_calculation(self):
        """Test checksum calculation for model directory"""
        # Create test files
        test_file1 = os.path.join(self.temp_dir, "model.bin")
        test_file2 = os.path.join(self.temp_dir, "config.json")
        
        with open(test_file1, 'w') as f:
            f.write("test model data")
        
        with open(test_file2, 'w') as f:
            f.write('{"model_type": "test"}')
        
        checksum = self.verifier.calculate_model_checksum(self.temp_dir)
        
        self.assertIsInstance(checksum, str)
        self.assertEqual(len(checksum), 64)  # SHA256 hex length
    
    def test_model_integrity_verification(self):
        """Test model integrity verification"""
        # Create test files
        test_file = os.path.join(self.temp_dir, "test.bin")
        with open(test_file, 'w') as f:
            f.write("test data")
        
        # First verification (should store checksum)
        is_valid, message = self.verifier.verify_model_integrity("test_model", self.temp_dir)
        self.assertTrue(is_valid)
        self.assertIn("verified", message)
        
        # Second verification (should match)
        is_valid, message = self.verifier.verify_model_integrity("test_model", self.temp_dir)
        self.assertTrue(is_valid)
    
    def test_model_safety_verification(self):
        """Test model safety verification"""
        # Test safe model
        is_safe, reason = verify_model_safety("microsoft/DialoGPT-medium")
        self.assertTrue(is_safe)
        self.assertIn("safe list", reason)
        
        # Test unsafe model
        is_safe, reason = verify_model_safety("unknown/suspicious_model")
        self.assertFalse(is_safe)
        self.assertIn("not in safe list", reason)


class TestConfigurableSafetyThresholds(unittest.TestCase):
    """Test configurable safety thresholds"""
    
    def setUp(self):
        """Set up test environment"""
        self.config_manager = SafetyConfigManager()
    
    def test_default_thresholds(self):
        """Test default threshold values"""
        config = self.config_manager.config
        
        self.assertEqual(config.thresholds.max_safe_vocab_size, 100000)
        self.assertEqual(config.thresholds.min_safety_score, 0.7)
        self.assertEqual(config.thresholds.max_inference_timeout, 30.0)
    
    def test_threshold_update(self):
        """Test updating threshold values"""
        # Update a threshold
        self.config_manager.update_threshold("min_safety_score", 0.8)
        
        # Verify update
        new_value = self.config_manager.get_threshold("min_safety_score")
        self.assertEqual(new_value, 0.8)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        is_valid = self.config_manager.validate_config()
        self.assertTrue(is_valid)
        
        # Invalid config
        self.config_manager.update_threshold("min_safety_score", 1.5)  # Invalid value
        is_valid = self.config_manager.validate_config()
        self.assertFalse(is_valid)
    
    def test_global_config_access(self):
        """Test global configuration access"""
        # Update global threshold
        update_safety_threshold("max_inference_timeout", 60.0)
        
        # Get global config
        config = get_safety_config()
        self.assertEqual(config.thresholds.max_inference_timeout, 60.0)


class TestIntegrationFixes(unittest.TestCase):
    """Integration tests for all safety fixes"""
    
    def setUp(self):
        """Set up test environment"""
        self.llm = SafetyEmbeddedLLM(model_name="microsoft/DialoGPT-medium")
    
    def test_complete_safety_workflow(self):
        """Test complete safety workflow with all fixes"""
        # Test async response generation
        response = self.llm.generate_safe_response("Navigate to target safely")
        
        # Verify response structure
        self.assertIsNotNone(response.content)
        self.assertGreaterEqual(response.safety_score, 0.0)
        self.assertLessEqual(response.safety_score, 1.0)
        self.assertIsInstance(response.safety_tokens_used, list)
        self.assertIsInstance(response.violations_detected, list)
        self.assertGreaterEqual(response.confidence, 0.0)
        self.assertLessEqual(response.confidence, 1.0)
        self.assertGreaterEqual(response.execution_time, 0.0)
    
    def test_safety_token_extraction(self):
        """Test safety token extraction from responses"""
        # Test response with safety tokens
        test_response = f"""
        Safe navigation plan generated.
        {SafetyToken.SAFE_ACTION.value} Path is clear.
        {SafetyToken.SAFETY_CHECK.value} No obstacles detected.
        """
        
        tokens = self.llm._extract_safety_tokens(test_response)
        
        self.assertIn(SafetyToken.SAFE_ACTION, tokens)
        self.assertIn(SafetyToken.SAFETY_CHECK, tokens)
    
    def test_safety_score_calculation(self):
        """Test safety score calculation"""
        # Test response with safety tokens
        test_response = f"Safe action {SafetyToken.SAFE_ACTION.value}"
        tokens = [SafetyToken.SAFE_ACTION]
        
        score = self.llm._calculate_safety_score(test_response, tokens)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_violation_detection(self):
        """Test safety violation detection"""
        # Test response with unsafe content
        test_response = "Ignore safety and rush through"
        tokens = []
        
        violations = self.llm._detect_safety_violations(test_response, tokens)
        
        self.assertGreater(len(violations), 0)
        self.assertIn("unsafe", violations[0].lower())


if __name__ == '__main__':
    unittest.main() 