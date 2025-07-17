"""
Cognitive Architecture for Embodied Intelligence Platform

This package implements a comprehensive cognitive architecture that orchestrates
all AI components (perception, reasoning, planning, execution) to create a unified
intelligent system capable of complex autonomous behavior while maintaining safety
and social awareness.
"""

from .cognitive_architecture_node import CognitiveArchitectureNode
from .attention_mechanism import AttentionMechanism
from .working_memory import WorkingMemory
from .long_term_memory import LongTermMemory
from .executive_control import ExecutiveControl
from .learning_engine import LearningEngine
from .social_intelligence import SocialIntelligence

__version__ = '0.1.0'
__author__ = 'AI Team'
__email__ = 'ai-team@embodied-intelligence.com'

__all__ = [
    'CognitiveArchitectureNode',
    'AttentionMechanism',
    'WorkingMemory',
    'LongTermMemory',
    'ExecutiveControl',
    'LearningEngine',
    'SocialIntelligence',
] 