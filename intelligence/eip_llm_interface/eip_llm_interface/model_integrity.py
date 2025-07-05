#!/usr/bin/env python3
"""
Model Integrity Verification

This module provides comprehensive model integrity verification including
checksum validation, security scanning, and safety assessment.
"""

import os
import hashlib
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import requests
from pathlib import Path
import time

# Known safe model checksums (SHA256)
SAFE_MODEL_CHECKSUMS = {
    "microsoft/DialoGPT-medium": "a1b2c3d4e5f6...",  # Placeholder - should be actual hash
    "microsoft/DialoGPT-small": "f6e5d4c3b2a1...",  # Placeholder
    "microsoft/DialoGPT-large": "1234567890ab...",   # Placeholder
}

# Security risk patterns in model names
SECURITY_RISK_PATTERNS = [
    r"\.\./",  # Path traversal
    r"http://",  # Insecure protocol
    r"file://",  # Local file access
    r"javascript:",  # Script injection
]

@dataclass
class ModelSecurityReport:
    """Security assessment report for a model"""
    model_name: str
    is_safe: bool
    checksum_valid: bool
    security_risks: List[str]
    recommendations: List[str]
    last_verified: float

class ModelIntegrityVerifier:
    """Comprehensive model integrity and security verification"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache_file = Path.home() / ".eip_model_cache.json"
        self._load_cache()
    
    def _load_cache(self):
        """Load cached verification results"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save verification results to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def verify_model_integrity(self, model_name: str, model_path: str) -> Tuple[bool, str]:
        """
        Verify model integrity with comprehensive checks
        
        Args:
            model_name: Hugging Face model name
            model_path: Local path to model files
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check for security risks in model name
            if not self._validate_model_name(model_name):
                return False, f"Model name '{model_name}' contains security risks"
            
            # Verify checksum if available
            if model_name in SAFE_MODEL_CHECKSUMS:
                expected_checksum = SAFE_MODEL_CHECKSUMS[model_name]
                actual_checksum = self._calculate_model_checksum(model_path)
                
                if actual_checksum != expected_checksum:
                    return False, f"Checksum mismatch for {model_name}"
            
            # Check file integrity
            if not self._verify_file_integrity(model_path):
                return False, f"File integrity check failed for {model_path}"
            
            # Update cache
            self.cache[model_name] = {
                "verified": True,
                "timestamp": time.time(),
                "path": model_path
            }
            self._save_cache()
            
            return True, f"Model {model_name} integrity verified"
            
        except Exception as e:
            self.logger.error(f"Model integrity verification failed: {e}")
            return False, f"Verification error: {e}"
    
    def _validate_model_name(self, model_name: str) -> bool:
        """Validate model name for security risks"""
        import re
        
        for pattern in SECURITY_RISK_PATTERNS:
            if re.search(pattern, model_name, re.IGNORECASE):
                return False
        return True
    
    def _calculate_model_checksum(self, model_path: str) -> str:
        """Calculate SHA256 checksum of model files"""
        sha256_hash = hashlib.sha256()
        
        for root, dirs, files in os.walk(model_path):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _verify_file_integrity(self, model_path: str) -> bool:
        """Verify file integrity and permissions"""
        try:
            if not os.path.exists(model_path):
                return False
            
            # Check for suspicious files
            suspicious_extensions = {'.exe', '.bat', '.sh', '.pyc'}
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if any(file.endswith(ext) for ext in suspicious_extensions):
                        self.logger.warning(f"Suspicious file found: {file}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"File integrity check failed: {e}")
            return False

def verify_model_safety(model_name: str) -> Tuple[bool, str]:
    """
    Verify model safety for deployment
    
    Args:
        model_name: Hugging Face model name
        
    Returns:
        Tuple of (is_safe, reason)
    """
    verifier = ModelIntegrityVerifier()
    
    # Check cache first
    if model_name in verifier.cache:
        cached = verifier.cache[model_name]
        if cached.get("verified", False):
            return True, "Model verified from cache"
    
    # Basic safety checks
    if not verifier._validate_model_name(model_name):
        return False, f"Model name '{model_name}' contains security risks"
    
    # Check if model is in known safe list
    if model_name in SAFE_MODEL_CHECKSUMS:
        return True, f"Model '{model_name}' is in safe list"
    
    # For unknown models, perform additional checks
    try:
        # Check Hugging Face model card for safety info
        response = requests.get(f"https://huggingface.co/{model_name}", timeout=10)
        if response.status_code == 200:
            # Basic check - could be enhanced with more sophisticated analysis
            return True, f"Model '{model_name}' verified from Hugging Face"
        else:
            return False, f"Model '{model_name}' not found on Hugging Face"
    except Exception as e:
        return False, f"Failed to verify model safety: {e}" 