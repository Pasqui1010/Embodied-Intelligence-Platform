#!/usr/bin/env python3
"""
Safety Configuration System

This module provides configurable safety thresholds and parameters for the
Safety-Embedded LLM system.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SafetyThresholds:
    """Configurable safety thresholds"""
    # Model compatibility thresholds
    max_safe_vocab_size: int = 100000
    min_safety_score: float = 0.7
    max_inference_timeout: float = 30.0
    
    # Safety validation thresholds
    collision_risk_threshold: float = 0.8
    human_proximity_threshold: float = 1.0
    velocity_limit_threshold: float = 0.8
    workspace_boundary_threshold: float = 0.7
    emergency_stop_threshold: float = 1.0
    
    # Performance thresholds
    max_response_time_ms: float = 1000.0
    min_confidence_score: float = 0.5
    max_violations_allowed: int = 0
    
    # Async processing thresholds
    inference_queue_timeout: float = 1.0
    result_wait_timeout: float = 30.0
    busy_wait_delay: float = 0.01


@dataclass
class SafetyConfig:
    """Complete safety configuration"""
    thresholds: SafetyThresholds
    enable_integrity_verification: bool = True
    enable_async_processing: bool = True
    enable_fallback_tokens: bool = True
    log_level: str = "INFO"
    cache_directory: str = "~/.cache/eip_models"


class SafetyConfigManager:
    """Manages safety configuration loading and validation"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or self._get_default_config_path()
        self.config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        config_dir = os.path.expanduser("~/.config/eip")
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, "safety_config.json")
    
    def _load_config(self) -> SafetyConfig:
        """Load configuration from file or create default"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Load thresholds
                thresholds_data = config_data.get('thresholds', {})
                thresholds = SafetyThresholds(**thresholds_data)
                
                # Load other config
                config = SafetyConfig(
                    thresholds=thresholds,
                    enable_integrity_verification=config_data.get('enable_integrity_verification', True),
                    enable_async_processing=config_data.get('enable_async_processing', True),
                    enable_fallback_tokens=config_data.get('enable_fallback_tokens', True),
                    log_level=config_data.get('log_level', 'INFO'),
                    cache_directory=config_data.get('cache_directory', '~/.cache/eip_models')
                )
                
                print(f"Loaded safety configuration from {self.config_file}")
                return config
                
        except Exception as e:
            print(f"Failed to load configuration: {e}, using defaults")
        
        # Return default configuration
        return SafetyConfig(thresholds=SafetyThresholds())
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_dict = asdict(self.config)
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"Saved safety configuration to {self.config_file}")
        except Exception as e:
            print(f"Failed to save configuration: {e}")
    
    def update_threshold(self, threshold_name: str, value: Any):
        """
        Update a specific threshold value
        
        Args:
            threshold_name: Name of the threshold to update
            value: New value for the threshold
        """
        if hasattr(self.config.thresholds, threshold_name):
            setattr(self.config.thresholds, threshold_name, value)
            print(f"Updated {threshold_name} to {value}")
        else:
            print(f"Unknown threshold: {threshold_name}")
    
    def get_threshold(self, threshold_name: str) -> Any:
        """
        Get a specific threshold value
        
        Args:
            threshold_name: Name of the threshold to get
            
        Returns:
            Threshold value
        """
        if hasattr(self.config.thresholds, threshold_name):
            return getattr(self.config.thresholds, threshold_name)
        else:
            print(f"Unknown threshold: {threshold_name}")
            return None
    
    def validate_config(self) -> bool:
        """
        Validate configuration values
        
        Returns:
            True if configuration is valid
        """
        thresholds = self.config.thresholds
        
        # Validate threshold ranges
        if not (0.0 <= thresholds.min_safety_score <= 1.0):
            print(f"Invalid min_safety_score: {thresholds.min_safety_score}")
            return False
        
        if not (0.0 <= thresholds.collision_risk_threshold <= 1.0):
            print(f"Invalid collision_risk_threshold: {thresholds.collision_risk_threshold}")
            return False
        
        if not (0.0 <= thresholds.human_proximity_threshold <= 1.0):
            print(f"Invalid human_proximity_threshold: {thresholds.human_proximity_threshold}")
            return False
        
        if thresholds.max_safe_vocab_size <= 0:
            print(f"Invalid max_safe_vocab_size: {thresholds.max_safe_vocab_size}")
            return False
        
        if thresholds.max_inference_timeout <= 0:
            print(f"Invalid max_inference_timeout: {thresholds.max_inference_timeout}")
            return False
        
        print("Configuration validation passed")
        return True
    
    def create_default_config(self):
        """Create a default configuration file"""
        default_config = SafetyConfig(thresholds=SafetyThresholds())
        self.config = default_config
        self.save_config()
        print(f"Created default configuration at {self.config_file}")


# Global configuration instance
_safety_config_manager: Optional[SafetyConfigManager] = None


def get_safety_config() -> SafetyConfig:
    """Get the global safety configuration"""
    global _safety_config_manager
    if _safety_config_manager is None:
        _safety_config_manager = SafetyConfigManager()
    return _safety_config_manager.config


def update_safety_threshold(threshold_name: str, value: Any):
    """Update a safety threshold globally"""
    global _safety_config_manager
    if _safety_config_manager is None:
        _safety_config_manager = SafetyConfigManager()
    _safety_config_manager.update_threshold(threshold_name, value)


def get_safety_threshold(threshold_name: str) -> Any:
    """Get a safety threshold globally"""
    global _safety_config_manager
    if _safety_config_manager is None:
        _safety_config_manager = SafetyConfigManager()
    return _safety_config_manager.get_threshold(threshold_name) 