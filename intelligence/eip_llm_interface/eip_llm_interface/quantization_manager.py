from typing import Optional, Dict, Any
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """Supported quantization types"""
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bf16"

class QuantizationConfig:
    """Configuration for model quantization"""
    def __init__(
        self,
        quantization_type: QuantizationType = QuantizationType.INT8,
        bits: int = 8,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_double_quant: bool = True,
        quantization_config: Optional[Dict[str, Any]] = None
    ):
        self.quantization_type = quantization_type
        self.bits = bits
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_double_quant = use_double_quant
        self.quantization_config = quantization_config or {}

class QuantizationManager:
    """
    Manages model quantization for GPU optimization
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        
    def quantize_model(self, model_name: str, device: str) -> nn.Module:
        """
        Quantize the model based on the configuration
        
        Args:
            model_name: Name of the model to quantize
            device: Target device (cuda/cpu)
            
        Returns:
            Quantized model instance
        """
        try:
            if self.config.load_in_8bit:
                return self._quantize_8bit(model_name, device)
            elif self.config.load_in_4bit:
                return self._quantize_4bit(model_name, device)
            else:
                return self._quantize_fp(model_name, device)
        except Exception as e:
            logger.error(f"Error during model quantization: {e}")
            raise
    
    def _quantize_8bit(self, model_name: str, device: str) -> nn.Module:
        """Quantize model to 8-bit precision"""
        import bitsandbytes as bnb
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            quantization_config=bnb.config.QuantizationConfig(
                bits=8,
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        )
        
        return model
    
    def _quantize_4bit(self, model_name: str, device: str) -> nn.Module:
        """Quantize model to 4-bit precision"""
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=self.config.use_double_quant,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        return model
    
    def _quantize_fp(self, model_name: str, device: str) -> nn.Module:
        """Quantize model to FP16/BF16 precision"""
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config.quantization_type == QuantizationType.FP16 else torch.bfloat16,
            device_map="auto"
        )
        
        return model
    
    def get_model_size(self, model: nn.Module) -> float:
        """
        Get the size of the quantized model in MB
        
        Args:
            model: Quantized model
            
        Returns:
            Model size in MB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def validate_quantization(self, model: nn.Module) -> bool:
        """
        Validate the quantization of the model
        
        Args:
            model: Quantized model
            
        Returns:
            True if quantization is valid, False otherwise
        """
        try:
            # Check if model is quantized
            if self.config.load_in_8bit or self.config.load_in_4bit:
                for param in model.parameters():
                    if not hasattr(param, "quant_state"):
                        logger.warning("Model parameters not properly quantized")
                        return False
                        
            # Check model size
            model_size = self.get_model_size(model)
            logger.info(f"Quantized model size: {model_size:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Quantization validation failed: {e}")
            return False
