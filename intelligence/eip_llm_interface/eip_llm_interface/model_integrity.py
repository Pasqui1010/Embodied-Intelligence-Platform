#!/usr/bin/env python3
"""
Model Integrity Verification

This module provides integrity verification for downloaded models to ensure
they haven't been tampered with and are safe to use.
"""

import hashlib
import json
import os
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelIntegrityVerifier:
    """Verifies model integrity and provides checksum validation"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the model integrity verifier
        
        Args:
            cache_dir: Directory to store model checksums
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/eip_models")
        self.checksum_file = os.path.join(self.cache_dir, "model_checksums.json")
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing checksums
        self.checksums = self._load_checksums()
    
    def _load_checksums(self) -> Dict[str, str]:
        """Load existing model checksums"""
        try:
            if os.path.exists(self.checksum_file):
                with open(self.checksum_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load checksums: {e}")
        
        return {}
    
    def _save_checksums(self):
        """Save model checksums to file"""
        try:
            with open(self.checksum_file, 'w') as f:
                json.dump(self.checksums, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save checksums: {e}")
    
    def calculate_model_checksum(self, model_path: str) -> str:
        """
        Calculate SHA256 checksum for a model directory
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            SHA256 checksum
        """
        sha256_hash = hashlib.sha256()
        
        try:
            path_obj = Path(model_path)
            if not path_obj.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
            # Walk through all files in the model directory
            for file_path in sorted(path_obj.rglob('*')):
                if file_path.is_file():
                    # Add file path to hash
                    sha256_hash.update(str(file_path.relative_to(path_obj)).encode())
                    
                    # Add file content to hash
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Failed to calculate checksum: {e}")
            return ""
    
    def verify_model_integrity(self, model_name: str, model_path: str) -> Tuple[bool, str]:
        """
        Verify model integrity by comparing checksums
        
        Args:
            model_name: Name of the model
            model_path: Path to the model directory
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Calculate current checksum
            current_checksum = self.calculate_model_checksum(model_path)
            if not current_checksum:
                return False, "Failed to calculate model checksum"
            
            # Check if we have a stored checksum
            if model_name in self.checksums:
                stored_checksum = self.checksums[model_name]
                if current_checksum != stored_checksum:
                    return False, f"Model checksum mismatch. Expected: {stored_checksum}, Got: {current_checksum}"
            else:
                # First time seeing this model, store the checksum
                self.checksums[model_name] = current_checksum
                self._save_checksums()
                self.logger.info(f"Stored checksum for new model: {model_name}")
            
            return True, "Model integrity verified"
            
        except Exception as e:
            return False, f"Integrity verification failed: {e}"
    
    def download_model_with_verification(self, model_name: str, target_dir: str) -> Tuple[bool, str]:
        """
        Download model with integrity verification
        
        Args:
            model_name: Hugging Face model name
            target_dir: Directory to download the model to
            
        Returns:
            Tuple of (success, message)
        """
        try:
            self.logger.info(f"Downloading model: {model_name}")
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=target_dir)
            
            # Download model
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=target_dir)
            
            # Verify integrity
            is_valid, message = self.verify_model_integrity(model_name, target_dir)
            
            if is_valid:
                self.logger.info(f"Model downloaded and verified: {model_name}")
                return True, "Model downloaded and verified successfully"
            else:
                self.logger.error(f"Model integrity check failed: {message}")
                return False, message
                
        except Exception as e:
            error_msg = f"Failed to download model {model_name}: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """
        Get information about a model including its checksum
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        info = {
            'name': model_name,
            'checksum': self.checksums.get(model_name, 'Not stored'),
            'verified': model_name in self.checksums
        }
        
        return info


def verify_model_safety(model_name: str) -> Tuple[bool, str]:
    """
    Verify that a model is safe to use
    
    Args:
        model_name: Name of the model to verify
        
    Returns:
        Tuple of (is_safe, reason)
    """
    # List of known safe models
    safe_models = [
        'microsoft/DialoGPT-medium',
        'microsoft/DialoGPT-small',
        'microsoft/DialoGPT-large',
        'gpt2',
        'gpt2-medium',
        'gpt2-large',
        'gpt2-xl'
    ]
    
    # Check if model is in safe list
    if model_name in safe_models:
        return True, "Model is in safe list"
    
    # Check if model is from a trusted organization
    trusted_orgs = ['microsoft', 'openai', 'huggingface', 'meta']
    org = model_name.split('/')[0] if '/' in model_name else ''
    
    if org in trusted_orgs:
        return True, f"Model is from trusted organization: {org}"
    
    return False, f"Model {model_name} is not in safe list and not from trusted organization" 