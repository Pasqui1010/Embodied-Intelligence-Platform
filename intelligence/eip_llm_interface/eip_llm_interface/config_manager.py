#!/usr/bin/env python3
"""
Configuration Management for Safety-Embedded LLM

This module provides centralized configuration management with validation,
environment-specific settings, and runtime configuration updates.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import threading


class ConfigSource(Enum):
    """Configuration source types"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DEFAULT = "default"
    RUNTIME = "runtime"


@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_name: str = "microsoft/DialoGPT-medium"
    device: str = "auto"
    max_length: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    cache_size: int = 128
    enable_gpu_optimization: bool = True
    batch_size: int = 4
    timeout_seconds: float = 30.0


@dataclass
class SafetyConfig:
    """Safety configuration settings"""
    safety_level: str = "high"
    enable_attention_masking: bool = True
    safety_score_threshold: float = 0.7
    max_violations_allowed: int = 0
    enable_constitutional_ai: bool = True
    safety_token_weight: float = 2.0
    emergency_stop_threshold: float = 0.9


@dataclass
class PerformanceConfig:
    """Performance configuration settings"""
    enable_caching: bool = True
    enable_batch_processing: bool = True
    max_concurrent_requests: int = 4
    memory_limit_mb: float = 2048.0
    gpu_memory_fraction: float = 0.8
    enable_profiling: bool = False
    log_performance_metrics: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "safety_llm.log"
    max_log_size_mb: float = 100.0
    backup_count: int = 5
    enable_structured_logging: bool = True


@dataclass
class EIPConfig:
    """Main configuration container"""
    model: ModelConfig
    safety: SafetyConfig
    performance: PerformanceConfig
    logging: LoggingConfig
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values"""
        # Model validation
        if self.model.temperature < 0.0 or self.model.temperature > 2.0:
            raise ValueError(f"Invalid temperature: {self.model.temperature}")
        
        if self.model.max_length < 1 or self.model.max_length > 4096:
            raise ValueError(f"Invalid max_length: {self.model.max_length}")
        
        # Safety validation
        if self.safety.safety_score_threshold < 0.0 or self.safety.safety_score_threshold > 1.0:
            raise ValueError(f"Invalid safety_score_threshold: {self.safety.safety_score_threshold}")
        
        # Performance validation
        if self.performance.memory_limit_mb < 512:
            raise ValueError(f"Memory limit too low: {self.performance.memory_limit_mb}")
        
        if self.performance.gpu_memory_fraction < 0.1 or self.performance.gpu_memory_fraction > 1.0:
            raise ValueError(f"Invalid GPU memory fraction: {self.performance.gpu_memory_fraction}")


class ConfigManager:
    """Centralized configuration management system"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self.config_file = self.config_dir / "eip_config.yaml"
        self.config_sources = {}
        self.config_lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Set up configuration validation
        self._setup_validation()
    
    def _load_configuration(self) -> EIPConfig:
        """Load configuration from multiple sources"""
        with self.config_lock:
            # Start with defaults
            config_dict = self._get_default_config()
            self.config_sources['default'] = config_dict.copy()
            
            # Override with file configuration
            file_config = self._load_from_file()
            if file_config:
                config_dict = self._merge_configs(config_dict, file_config)
                self.config_sources['file'] = file_config
            
            # Override with environment variables
            env_config = self._load_from_environment()
            if env_config:
                config_dict = self._merge_configs(config_dict, env_config)
                self.config_sources['environment'] = env_config
            
            # Create configuration object
            try:
                return EIPConfig(
                    model=ModelConfig(**config_dict.get('model', {})),
                    safety=SafetyConfig(**config_dict.get('safety', {})),
                    performance=PerformanceConfig(**config_dict.get('performance', {})),
                    logging=LoggingConfig(**config_dict.get('logging', {}))
                )
            except Exception as e:
                self.logger.error(f"Configuration validation failed: {e}")
                # Return default configuration
                return self._get_default_eip_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration dictionary"""
        return {
            'model': asdict(ModelConfig()),
            'safety': asdict(SafetyConfig()),
            'performance': asdict(PerformanceConfig()),
            'logging': asdict(LoggingConfig())
        }
    
    def _get_default_eip_config(self) -> EIPConfig:
        """Get default EIP configuration object"""
        return EIPConfig(
            model=ModelConfig(),
            safety=SafetyConfig(),
            performance=PerformanceConfig(),
            logging=LoggingConfig()
        )
    
    def _load_from_file(self) -> Optional[Dict[str, Any]]:
        """Load configuration from file"""
        if not self.config_file.exists():
            self.logger.info(f"Config file not found: {self.config_file}")
            return None
        
        try:
            with open(self.config_file, 'r') as f:
                if self.config_file.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            
            self.logger.info(f"Loaded configuration from {self.config_file}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load config file {self.config_file}: {e}")
            return None
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Define environment variable mappings
        env_mappings = {
            'EIP_MODEL_NAME': ('model', 'model_name'),
            'EIP_DEVICE': ('model', 'device'),
            'EIP_TEMPERATURE': ('model', 'temperature'),
            'EIP_MAX_LENGTH': ('model', 'max_length'),
            'EIP_SAFETY_LEVEL': ('safety', 'safety_level'),
            'EIP_SAFETY_THRESHOLD': ('safety', 'safety_score_threshold'),
            'EIP_CACHE_SIZE': ('model', 'cache_size'),
            'EIP_BATCH_SIZE': ('model', 'batch_size'),
            'EIP_MEMORY_LIMIT': ('performance', 'memory_limit_mb'),
            'EIP_GPU_MEMORY_FRACTION': ('performance', 'gpu_memory_fraction'),
            'EIP_LOG_LEVEL': ('logging', 'log_level'),
            'EIP_LOG_FILE': ('logging', 'log_file_path'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Initialize section if not exists
                if section not in env_config:
                    env_config[section] = {}
                
                # Convert value to appropriate type
                env_config[section][key] = self._convert_env_value(value, section, key)
        
        if env_config:
            self.logger.info(f"Loaded {len(env_config)} environment configurations")
        
        return env_config
    
    def _convert_env_value(self, value: str, section: str, key: str) -> Any:
        """Convert environment variable value to appropriate type"""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # String values
        return value
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _setup_validation(self):
        """Set up configuration validation rules"""
        # This could be extended with more sophisticated validation
        pass
    
    def get_config(self) -> EIPConfig:
        """Get current configuration"""
        with self.config_lock:
            return self.config
    
    def update_config(self, updates: Dict[str, Any], source: str = "runtime"):
        """Update configuration at runtime"""
        with self.config_lock:
            try:
                # Create updated configuration dictionary
                current_dict = {
                    'model': asdict(self.config.model),
                    'safety': asdict(self.config.safety),
                    'performance': asdict(self.config.performance),
                    'logging': asdict(self.config.logging)
                }
                
                updated_dict = self._merge_configs(current_dict, updates)
                
                # Validate new configuration
                new_config = EIPConfig(
                    model=ModelConfig(**updated_dict.get('model', {})),
                    safety=SafetyConfig(**updated_dict.get('safety', {})),
                    performance=PerformanceConfig(**updated_dict.get('performance', {})),
                    logging=LoggingConfig(**updated_dict.get('logging', {}))
                )
                
                # Update configuration
                self.config = new_config
                self.config_sources[source] = updates
                
                self.logger.info(f"Configuration updated from {source}")
                
            except Exception as e:
                self.logger.error(f"Failed to update configuration: {e}")
                raise
    
    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file"""
        with self.config_lock:
            save_path = Path(file_path) if file_path else self.config_file
            
            try:
                config_dict = {
                    'model': asdict(self.config.model),
                    'safety': asdict(self.config.safety),
                    'performance': asdict(self.config.performance),
                    'logging': asdict(self.config.logging)
                }
                
                with open(save_path, 'w') as f:
                    if save_path.suffix.lower() == '.json':
                        json.dump(config_dict, f, indent=2)
                    else:
                        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                
                self.logger.info(f"Configuration saved to {save_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to save configuration: {e}")
                raise
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        with self.config_lock:
            return {
                'model_name': self.config.model.model_name,
                'device': self.config.model.device,
                'safety_level': self.config.safety.safety_level,
                'safety_threshold': self.config.safety.safety_score_threshold,
                'cache_enabled': self.config.performance.enable_caching,
                'batch_processing': self.config.performance.enable_batch_processing,
                'log_level': self.config.logging.log_level,
                'config_sources': list(self.config_sources.keys())
            }
    
    def validate_config(self) -> List[str]:
        """Validate current configuration and return any issues"""
        issues = []
        
        try:
            # This will raise an exception if validation fails
            self.config._validate_config()
        except ValueError as e:
            issues.append(str(e))
        
        # Additional custom validations
        if self.config.model.model_name == "":
            issues.append("Model name cannot be empty")
        
        if self.config.performance.max_concurrent_requests < 1:
            issues.append("Max concurrent requests must be at least 1")
        
        return issues
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        with self.config_lock:
            self.config = self._get_default_eip_config()
            self.config_sources = {'default': self._get_default_config()}
            self.logger.info("Configuration reset to defaults")


# Global configuration manager instance
_config_manager = None
_config_lock = threading.Lock()


def get_config_manager(config_dir: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    
    with _config_lock:
        if _config_manager is None:
            _config_manager = ConfigManager(config_dir)
        return _config_manager


def get_config() -> EIPConfig:
    """Get current configuration"""
    return get_config_manager().get_config()


def update_config(updates: Dict[str, Any], source: str = "runtime"):
    """Update global configuration"""
    get_config_manager().update_config(updates, source)


def save_config(file_path: Optional[str] = None):
    """Save global configuration"""
    get_config_manager().save_config(file_path)