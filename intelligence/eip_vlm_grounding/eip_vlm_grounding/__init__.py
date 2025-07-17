#!/usr/bin/env python3
"""
Vision-Language Grounding Package

This package provides vision-language grounding capabilities for spatial reference
resolution and object affordance estimation in robotics applications.
"""

__version__ = "0.1.0"
__author__ = "Embodied Intelligence Platform Team"
__email__ = "maintainer@embodied-intelligence.org"

from .vlm_grounding_node import VLMGroundingNode
from .spatial_reference_resolver import SpatialReferenceResolver
from .object_affordance_estimator import ObjectAffordanceEstimator
from .scene_understanding import SceneUnderstanding
from .vlm_integration import VLMIntegration

__all__ = [
    'VLMGroundingNode',
    'SpatialReferenceResolver', 
    'ObjectAffordanceEstimator',
    'SceneUnderstanding',
    'VLMIntegration'
] 