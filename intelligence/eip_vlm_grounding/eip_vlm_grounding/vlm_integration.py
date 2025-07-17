#!/usr/bin/env python3
"""
VLM Integration

This module integrates vision-language models with the Safety-Embedded LLM
to provide enhanced reasoning capabilities for robotics applications.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import logging
import json
from enum import Enum

# ROS 2 imports
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import String, Float32, Bool

# Vision-Language Model imports
try:
    import clip
    from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    logging.warning("VLM libraries not available, using fallback integration")

# Import from other modules
from .spatial_reference_resolver import SceneData, ObjectDetection
from .scene_understanding import SceneDescription
from .object_affordance_estimator import AffordanceSet


class VLMType(Enum):
    """Types of vision-language models"""
    CLIP = "clip"
    FLAMINGO = "flamingo"
    BLIP = "blip"
    LLAVA = "llava"
    CUSTOM = "custom"


@dataclass
class VLMResponse:
    """Response from vision-language model"""
    text: str
    confidence: float
    visual_features: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class VisualPrompt:
    """Visual prompt for VLM"""
    image: np.ndarray
    text_prompt: str
    visual_context: Optional[Dict[str, Any]] = None
    safety_constraints: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class VLMReasoningResult:
    """Result of VLM reasoning"""
    query: str
    response: VLMResponse
    scene_context: SceneDescription
    spatial_references: List[str]
    safety_validation: bool
    confidence: float
    reasoning_steps: List[str]
    metadata: Optional[Dict[str, Any]] = None


class VLMIntegration:
    """
    Integration of vision-language models with Safety-Embedded LLM
    
    Provides enhanced reasoning capabilities by combining visual
    understanding with language processing and safety validation.
    """
    
    def __init__(self, 
                 vlm_type: VLMType = VLMType.CLIP,
                 model_name: str = "openai/clip-vit-base-patch32",
                 safety_llm_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize VLM
        self.vlm_type = vlm_type
        self.vlm_model, self.vlm_processor = self._load_vlm_model(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize Safety-Embedded LLM integration
        self.safety_llm = self._load_safety_llm(safety_llm_path)
        
        # Integration parameters
        self.min_confidence = 0.6
        self.max_response_length = 200
        self.safety_validation_enabled = True
        
        # Prompt templates
        self.prompt_templates = {
            "spatial_reasoning": "Given the image, {query}",
            "object_identification": "What objects can you see in this image? {query}",
            "affordance_analysis": "What can I do with the objects in this image? {query}",
            "safety_assessment": "Is it safe to {query} in this scene?",
            "scene_description": "Describe what you see in this image: {query}"
        }
        
    def _load_vlm_model(self, model_name: str):
        """Load vision-language model"""
        if not VLM_AVAILABLE:
            self.logger.warning("VLM libraries not available")
            return None, None
        
        try:
            if self.vlm_type == VLMType.CLIP:
                model = CLIPModel.from_pretrained(model_name)
                processor = CLIPProcessor.from_pretrained(model_name)
                self.logger.info(f"Loaded CLIP model: {model_name}")
                return model, processor
            else:
                self.logger.warning(f"VLM type {self.vlm_type} not implemented")
                return None, None
        except Exception as e:
            self.logger.error(f"Failed to load VLM model: {e}")
            return None, None
    
    def _load_safety_llm(self, safety_llm_path: Optional[str]):
        """Load Safety-Embedded LLM for validation"""
        try:
            if safety_llm_path:
                # Load custom safety LLM
                self.logger.info(f"Loading Safety-Embedded LLM from {safety_llm_path}")
                return None  # Placeholder for actual implementation
            else:
                # Use default safety validation
                self.logger.info("Using default safety validation")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load Safety-Embedded LLM: {e}")
            return None
    
    def process_visual_query(self, 
                           query: str, 
                           scene_data: SceneData,
                           scene_description: SceneDescription) -> VLMReasoningResult:
        """
        Process a visual query using VLM integration
        
        Args:
            query: Natural language query about the scene
            scene_data: Current scene data with visual information
            scene_description: Scene understanding results
            
        Returns:
            VLMReasoningResult with reasoning and safety validation
        """
        self.logger.info(f"Processing visual query: {query}")
        
        # Create visual prompt
        visual_prompt = self._create_visual_prompt(query, scene_data, scene_description)
        
        # Process with VLM
        vlm_response = self._process_with_vlm(visual_prompt)
        
        # Extract spatial references
        spatial_references = self._extract_spatial_references(query, scene_description)
        
        # Validate safety
        safety_validation = self._validate_safety(query, vlm_response, scene_description)
        
        # Generate reasoning steps
        reasoning_steps = self._generate_reasoning_steps(query, vlm_response, scene_description)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(vlm_response, safety_validation, scene_description)
        
        return VLMReasoningResult(
            query=query,
            response=vlm_response,
            scene_context=scene_description,
            spatial_references=spatial_references,
            safety_validation=safety_validation,
            confidence=confidence,
            reasoning_steps=reasoning_steps
        )
    
    def _create_visual_prompt(self, 
                            query: str, 
                            scene_data: SceneData,
                            scene_description: SceneDescription) -> VisualPrompt:
        """Create visual prompt for VLM processing"""
        # Prepare image
        if scene_data.image is not None:
            image = scene_data.image
        else:
            # Create placeholder image
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Determine prompt template
        prompt_template = self._select_prompt_template(query)
        text_prompt = prompt_template.format(query=query)
        
        # Prepare visual context
        visual_context = {
            "scene_type": scene_description.scene_type,
            "num_objects": len(scene_description.elements),
            "complexity_score": scene_description.complexity_score,
            "safety_score": scene_description.safety_score
        }
        
        # Prepare safety constraints
        safety_constraints = self._extract_safety_constraints(query, scene_description)
        
        return VisualPrompt(
            image=image,
            text_prompt=text_prompt,
            visual_context=visual_context,
            safety_constraints=safety_constraints
        )
    
    def _select_prompt_template(self, query: str) -> str:
        """Select appropriate prompt template based on query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["left", "right", "behind", "front", "near", "far"]):
            return self.prompt_templates["spatial_reasoning"]
        elif any(word in query_lower for word in ["what", "see", "object", "thing"]):
            return self.prompt_templates["object_identification"]
        elif any(word in query_lower for word in ["do", "can", "affordance", "grasp", "pick"]):
            return self.prompt_templates["affordance_analysis"]
        elif any(word in query_lower for word in ["safe", "dangerous", "risk"]):
            return self.prompt_templates["safety_assessment"]
        else:
            return self.prompt_templates["scene_description"]
    
    def _process_with_vlm(self, visual_prompt: VisualPrompt) -> VLMResponse:
        """Process visual prompt with VLM"""
        if self.vlm_model is None or self.vlm_processor is None:
            # Fallback response
            return VLMResponse(
                text="VLM not available, using fallback response",
                confidence=0.5
            )
        
        try:
            if self.vlm_type == VLMType.CLIP:
                return self._process_with_clip(visual_prompt)
            else:
                return VLMResponse(
                    text=f"VLM type {self.vlm_type} not implemented",
                    confidence=0.0
                )
        except Exception as e:
            self.logger.error(f"Error in VLM processing: {e}")
            return VLMResponse(
                text=f"Error processing with VLM: {str(e)}",
                confidence=0.0
            )
    
    def _process_with_clip(self, visual_prompt: VisualPrompt) -> VLMResponse:
        """Process with CLIP model"""
        try:
            # Prepare inputs
            inputs = self.vlm_processor(
                text=[visual_prompt.text_prompt],
                images=visual_prompt.image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get CLIP embeddings
            with torch.no_grad():
                outputs = self.vlm_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)
            
            # Get confidence
            confidence = probs[0][0].item()
            
            # Generate response based on confidence
            if confidence > self.min_confidence:
                response_text = f"Based on the image, {visual_prompt.text_prompt.lower()}"
            else:
                response_text = "I'm not confident about what I see in the image."
            
            return VLMResponse(
                text=response_text,
                confidence=confidence,
                visual_features=outputs.image_embeds.cpu().numpy(),
                attention_weights=probs.cpu().numpy()
            )
            
        except Exception as e:
            self.logger.error(f"Error in CLIP processing: {e}")
            return VLMResponse(
                text="Error processing with CLIP",
                confidence=0.0
            )
    
    def _extract_spatial_references(self, query: str, scene_description: SceneDescription) -> List[str]:
        """Extract spatial references from query and scene"""
        spatial_references = []
        
        # Extract spatial relations from scene
        for relation in scene_description.spatial_relations:
            spatial_ref = f"{relation.element1_id} {relation.relation_type.value} {relation.element2_id}"
            spatial_references.append(spatial_ref)
        
        # Extract spatial terms from query
        spatial_terms = ["left", "right", "behind", "front", "near", "far", "above", "below"]
        query_lower = query.lower()
        for term in spatial_terms:
            if term in query_lower:
                spatial_references.append(f"query mentions {term}")
        
        return spatial_references
    
    def _extract_safety_constraints(self, query: str, scene_description: SceneDescription) -> List[str]:
        """Extract safety constraints from query and scene"""
        constraints = []
        
        # Scene-based constraints
        if scene_description.safety_score < 0.7:
            constraints.append("low_safety_scene")
        
        if scene_description.complexity_score > 0.8:
            constraints.append("high_complexity_scene")
        
        # Query-based constraints
        query_lower = query.lower()
        dangerous_actions = ["hit", "break", "drop", "throw", "push_hard"]
        for action in dangerous_actions:
            if action in query_lower:
                constraints.append(f"dangerous_action_{action}")
        
        return constraints
    
    def _validate_safety(self, 
                        query: str, 
                        vlm_response: VLMResponse, 
                        scene_description: SceneDescription) -> bool:
        """Validate safety of the proposed action"""
        if not self.safety_validation_enabled:
            return True
        
        # Check scene safety score
        if scene_description.safety_score < 0.5:
            self.logger.warning("Scene has low safety score")
            return False
        
        # Check for dangerous actions in query
        query_lower = query.lower()
        dangerous_keywords = ["dangerous", "unsafe", "risk", "harm", "break", "destroy"]
        if any(keyword in query_lower for keyword in dangerous_keywords):
            self.logger.warning("Query contains dangerous keywords")
            return False
        
        # Check VLM response confidence
        if vlm_response.confidence < self.min_confidence:
            self.logger.warning("VLM response has low confidence")
            return False
        
        # Additional safety checks could be implemented here
        # using the Safety-Embedded LLM
        
        return True
    
    def _generate_reasoning_steps(self, 
                                query: str, 
                                vlm_response: VLMResponse, 
                                scene_description: SceneDescription) -> List[str]:
        """Generate reasoning steps for the response"""
        steps = []
        
        # Step 1: Scene analysis
        steps.append(f"Analyzed scene type: {scene_description.scene_type}")
        steps.append(f"Detected {len(scene_description.elements)} scene elements")
        
        # Step 2: Query understanding
        steps.append(f"Processed query: {query}")
        
        # Step 3: VLM processing
        steps.append(f"Applied VLM with confidence: {vlm_response.confidence:.2f}")
        
        # Step 4: Safety validation
        safety_status = "passed" if vlm_response.confidence > self.min_confidence else "failed"
        steps.append(f"Safety validation: {safety_status}")
        
        # Step 5: Response generation
        steps.append("Generated response based on visual understanding")
        
        return steps
    
    def _calculate_confidence(self, 
                            vlm_response: VLMResponse, 
                            safety_validation: bool, 
                            scene_description: SceneDescription) -> float:
        """Calculate overall confidence in the reasoning result"""
        # Base confidence from VLM
        base_confidence = vlm_response.confidence
        
        # Adjust for safety validation
        if not safety_validation:
            base_confidence *= 0.5
        
        # Adjust for scene complexity
        complexity_factor = 1.0 - scene_description.complexity_score * 0.3
        base_confidence *= complexity_factor
        
        # Adjust for scene safety
        safety_factor = scene_description.safety_score
        base_confidence *= safety_factor
        
        return max(0.0, min(1.0, base_confidence))
    
    def integrate_with_safety_llm(self, 
                                vlm_result: VLMReasoningResult,
                                safety_constraints: List[str]) -> VLMReasoningResult:
        """Integrate VLM result with Safety-Embedded LLM"""
        if self.safety_llm is None:
            # No safety LLM available, return original result
            return vlm_result
        
        try:
            # Create safety validation prompt
            safety_prompt = self._create_safety_prompt(vlm_result, safety_constraints)
            
            # Process with safety LLM (placeholder)
            safety_validation = self._process_safety_validation(safety_prompt)
            
            # Update result
            vlm_result.safety_validation = safety_validation
            vlm_result.reasoning_steps.append("Applied Safety-Embedded LLM validation")
            
            return vlm_result
            
        except Exception as e:
            self.logger.error(f"Error in safety LLM integration: {e}")
            return vlm_result
    
    def _create_safety_prompt(self, 
                            vlm_result: VLMReasoningResult, 
                            safety_constraints: List[str]) -> str:
        """Create safety validation prompt"""
        prompt = f"""
        Safety Validation Request:
        
        Query: {vlm_result.query}
        VLM Response: {vlm_result.response.text}
        Scene Type: {vlm_result.scene_context.scene_type}
        Scene Safety Score: {vlm_result.scene_context.safety_score:.2f}
        
        Safety Constraints: {', '.join(safety_constraints)}
        
        Is this action safe to perform? Consider:
        1. Object safety and fragility
        2. Human safety and proximity
        3. Environmental safety
        4. Robot capability limitations
        
        Respond with SAFE or UNSAFE and reasoning.
        """
        return prompt
    
    def _process_safety_validation(self, safety_prompt: str) -> bool:
        """Process safety validation with Safety-Embedded LLM"""
        # Placeholder implementation
        # In practice, this would use the actual Safety-Embedded LLM
        
        # Simple heuristic-based validation
        unsafe_keywords = ["unsafe", "dangerous", "risk", "harm", "break", "destroy"]
        if any(keyword in safety_prompt.lower() for keyword in unsafe_keywords):
            return False
        
        return True
    
    def get_visual_features(self, image: np.ndarray) -> np.ndarray:
        """Extract visual features from image"""
        if self.vlm_model is None:
            return np.zeros(512)  # Default feature size
        
        try:
            # Process image with VLM
            inputs = self.vlm_processor(
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.vlm_model(**inputs)
                features = outputs.image_embeds.cpu().numpy()
            
            return features.flatten()
            
        except Exception as e:
            self.logger.error(f"Error extracting visual features: {e}")
            return np.zeros(512)
    
    def compare_visual_similarity(self, 
                                image1: np.ndarray, 
                                image2: np.ndarray) -> float:
        """Compare visual similarity between two images"""
        features1 = self.get_visual_features(image1)
        features2 = self.get_visual_features(image2)
        
        # Calculate cosine similarity
        similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        
        return max(0.0, similarity)  # Ensure non-negative 