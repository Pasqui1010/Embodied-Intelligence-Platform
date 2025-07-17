"""
Attention Mechanism for Cognitive Architecture

This module implements an attention mechanism that focuses on relevant stimuli
and filters out distractions, enabling the cognitive system to prioritize
important information for processing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
import threading
from collections import deque


class AttentionType(Enum):
    """Types of attention mechanisms"""
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    SOCIAL = "social"
    SAFETY = "safety"


@dataclass
class AttentionFocus:
    """Represents a focused attention target"""
    attention_type: AttentionType
    priority: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    location: Optional[Tuple[float, float, float]] = None
    timestamp: float = None
    metadata: Dict[str, Any] = None


@dataclass
class MultiModalSensorData:
    """Multi-modal sensor data structure"""
    visual_data: Optional[Dict[str, Any]] = None
    audio_data: Optional[Dict[str, Any]] = None
    tactile_data: Optional[Dict[str, Any]] = None
    proprioceptive_data: Optional[Dict[str, Any]] = None
    timestamp: float = None


class AttentionMechanism:
    """
    Attention mechanism that focuses on relevant stimuli and filters distractions.
    
    This component implements multiple attention types:
    - Spatial attention: Focus on specific locations in space
    - Temporal attention: Focus on time-sensitive events
    - Semantic attention: Focus on meaningful content
    - Social attention: Focus on human interactions
    - Safety attention: Focus on safety-critical information
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the attention mechanism.
        
        Args:
            config: Configuration dictionary for attention parameters
        """
        self.config = config or self._get_default_config()
        
        # Attention focus history
        self.attention_history = deque(maxlen=100)
        
        # Current attention foci
        self.current_foci: List[AttentionFocus] = []
        
        # Attention weights for different modalities
        self.modality_weights = {
            'visual': 0.4,
            'audio': 0.3,
            'tactile': 0.2,
            'proprioceptive': 0.1
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self.performance_metrics = {
            'focus_accuracy': 0.0,
            'response_time': 0.0,
            'attention_shifts': 0
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for attention mechanism."""
        return {
            'max_foci': 5,
            'min_priority_threshold': 0.3,
            'attention_decay_rate': 0.1,
            'safety_priority_boost': 2.0,
            'social_priority_boost': 1.5,
            'temporal_window_ms': 1000
        }
    
    def focus_attention(self, 
                       sensor_data: MultiModalSensorData,
                       user_input: Optional[str] = None,
                       current_context: Optional[Dict[str, Any]] = None) -> List[AttentionFocus]:
        """
        Focus attention on relevant stimuli from multi-modal sensor data.
        
        Args:
            sensor_data: Current multi-modal sensor readings
            user_input: Optional user command or interaction
            current_context: Current task and environmental context
            
        Returns:
            List of attention foci with priorities and metadata
        """
        start_time = time.time()
        
        with self._lock:
            # Clear old attention foci
            self._decay_attention_foci()
            
            # Process each modality for attention targets
            attention_targets = []
            
            # Visual attention processing
            if sensor_data.visual_data:
                visual_targets = self._process_visual_attention(
                    sensor_data.visual_data, current_context
                )
                attention_targets.extend(visual_targets)
            
            # Audio attention processing
            if sensor_data.audio_data:
                audio_targets = self._process_audio_attention(
                    sensor_data.audio_data, current_context
                )
                attention_targets.extend(audio_targets)
            
            # Tactile attention processing
            if sensor_data.tactile_data:
                tactile_targets = self._process_tactile_attention(
                    sensor_data.tactile_data, current_context
                )
                attention_targets.extend(tactile_targets)
            
            # User input attention processing
            if user_input:
                user_targets = self._process_user_input_attention(
                    user_input, current_context
                )
                attention_targets.extend(user_targets)
            
            # Safety attention processing (always active)
            safety_targets = self._process_safety_attention(
                sensor_data, current_context
            )
            attention_targets.extend(safety_targets)
            
            # Apply attention selection and prioritization
            self.current_foci = self._select_attention_foci(attention_targets)
            
            # Update attention history
            self._update_attention_history()
            
            # Update performance metrics
            self._update_performance_metrics(start_time)
            
            return self.current_foci.copy()
    
    def _process_visual_attention(self, 
                                 visual_data: Dict[str, Any],
                                 context: Optional[Dict[str, Any]]) -> List[AttentionFocus]:
        """Process visual data for attention targets."""
        targets = []
        
        # Extract visual features
        objects = visual_data.get('objects', [])
        faces = visual_data.get('faces', [])
        motion = visual_data.get('motion', [])
        gestures = visual_data.get('gestures', [])
        
        # Object attention
        for obj in objects:
            priority = self._calculate_object_priority(obj, context)
            if priority > self.config['min_priority_threshold']:
                targets.append(AttentionFocus(
                    attention_type=AttentionType.SPATIAL,
                    priority=priority,
                    confidence=obj.get('confidence', 0.5),
                    location=obj.get('location'),
                    timestamp=time.time(),
                    metadata={'object_type': obj.get('type'), 'object_id': obj.get('id')}
                ))
        
        # Face attention (social)
        for face in faces:
            priority = self._calculate_face_priority(face, context) * self.config['social_priority_boost']
            targets.append(AttentionFocus(
                attention_type=AttentionType.SOCIAL,
                priority=priority,
                confidence=face.get('confidence', 0.5),
                location=face.get('location'),
                timestamp=time.time(),
                metadata={'face_id': face.get('id'), 'expression': face.get('expression')}
            ))
        
        # Motion attention
        for motion_event in motion:
            priority = self._calculate_motion_priority(motion_event, context)
            targets.append(AttentionFocus(
                attention_type=AttentionType.TEMPORAL,
                priority=priority,
                confidence=motion_event.get('confidence', 0.5),
                location=motion_event.get('location'),
                timestamp=time.time(),
                metadata={'motion_type': motion_event.get('type'), 'velocity': motion_event.get('velocity')}
            ))
        
        return targets
    
    def _process_audio_attention(self, 
                                audio_data: Dict[str, Any],
                                context: Optional[Dict[str, Any]]) -> List[AttentionFocus]:
        """Process audio data for attention targets."""
        targets = []
        
        # Extract audio features
        speech = audio_data.get('speech', [])
        sounds = audio_data.get('sounds', [])
        voice_characteristics = audio_data.get('voice_characteristics', {})
        
        # Speech attention (social)
        for speech_event in speech:
            priority = self._calculate_speech_priority(speech_event, context) * self.config['social_priority_boost']
            targets.append(AttentionFocus(
                attention_type=AttentionType.SOCIAL,
                priority=priority,
                confidence=speech_event.get('confidence', 0.5),
                timestamp=time.time(),
                metadata={'speaker_id': speech_event.get('speaker_id'), 'content': speech_event.get('content')}
            ))
        
        # Sound attention
        for sound in sounds:
            priority = self._calculate_sound_priority(sound, context)
            targets.append(AttentionFocus(
                attention_type=AttentionType.TEMPORAL,
                priority=priority,
                confidence=sound.get('confidence', 0.5),
                timestamp=time.time(),
                metadata={'sound_type': sound.get('type'), 'intensity': sound.get('intensity')}
            ))
        
        return targets
    
    def _process_tactile_attention(self, 
                                  tactile_data: Dict[str, Any],
                                  context: Optional[Dict[str, Any]]) -> List[AttentionFocus]:
        """Process tactile data for attention targets."""
        targets = []
        
        # Extract tactile features
        contacts = tactile_data.get('contacts', [])
        forces = tactile_data.get('forces', [])
        
        # Contact attention
        for contact in contacts:
            priority = self._calculate_contact_priority(contact, context)
            targets.append(AttentionFocus(
                attention_type=AttentionType.SPATIAL,
                priority=priority,
                confidence=contact.get('confidence', 0.5),
                location=contact.get('location'),
                timestamp=time.time(),
                metadata={'contact_type': contact.get('type'), 'pressure': contact.get('pressure')}
            ))
        
        return targets
    
    def _process_user_input_attention(self, 
                                     user_input: str,
                                     context: Optional[Dict[str, Any]]) -> List[AttentionFocus]:
        """Process user input for attention targets."""
        targets = []
        
        # High priority for direct user input
        priority = 0.9  # High priority for user commands
        
        targets.append(AttentionFocus(
            attention_type=AttentionType.SEMANTIC,
            priority=priority,
            confidence=0.8,
            timestamp=time.time(),
            metadata={'input_type': 'user_command', 'content': user_input}
        ))
        
        return targets
    
    def _process_safety_attention(self, 
                                 sensor_data: MultiModalSensorData,
                                 context: Optional[Dict[str, Any]]) -> List[AttentionFocus]:
        """Process safety-critical information for attention."""
        targets = []
        
        # Check for safety violations across all modalities
        safety_events = self._detect_safety_events(sensor_data, context)
        
        for event in safety_events:
            priority = event.get('priority', 0.5) * self.config['safety_priority_boost']
            targets.append(AttentionFocus(
                attention_type=AttentionType.SAFETY,
                priority=priority,
                confidence=event.get('confidence', 0.8),
                location=event.get('location'),
                timestamp=time.time(),
                metadata={'safety_type': event.get('type'), 'severity': event.get('severity')}
            ))
        
        return targets
    
    def _calculate_object_priority(self, obj: Dict[str, Any], context: Optional[Dict[str, Any]]) -> float:
        """Calculate priority for visual objects."""
        base_priority = 0.5
        
        # Task relevance
        if context and 'current_task' in context:
            task_objects = context['current_task'].get('relevant_objects', [])
            if obj.get('type') in task_objects:
                base_priority += 0.3
        
        # Proximity
        if obj.get('distance'):
            distance = obj['distance']
            if distance < 1.0:  # Close objects
                base_priority += 0.2
            elif distance > 5.0:  # Far objects
                base_priority -= 0.1
        
        # Movement
        if obj.get('velocity') and np.linalg.norm(obj['velocity']) > 0.1:
            base_priority += 0.1
        
        return min(1.0, max(0.0, base_priority))
    
    def _calculate_face_priority(self, face: Dict[str, Any], context: Optional[Dict[str, Any]]) -> float:
        """Calculate priority for detected faces."""
        base_priority = 0.6  # High base priority for faces
        
        # Expression-based priority
        expression = face.get('expression', 'neutral')
        if expression in ['happy', 'sad', 'angry', 'surprised']:
            base_priority += 0.2
        
        # Familiarity
        if face.get('familiar', False):
            base_priority += 0.1
        
        return min(1.0, base_priority)
    
    def _calculate_motion_priority(self, motion: Dict[str, Any], context: Optional[Dict[str, Any]]) -> float:
        """Calculate priority for motion events."""
        base_priority = 0.4
        
        # Velocity-based priority
        velocity = motion.get('velocity', 0)
        if velocity > 1.0:  # Fast motion
            base_priority += 0.3
        elif velocity > 0.5:  # Moderate motion
            base_priority += 0.1
        
        # Direction (towards robot)
        if motion.get('direction_towards_robot', False):
            base_priority += 0.2
        
        return min(1.0, base_priority)
    
    def _calculate_speech_priority(self, speech: Dict[str, Any], context: Optional[Dict[str, Any]]) -> float:
        """Calculate priority for speech events."""
        base_priority = 0.7  # High base priority for speech
        
        # Content relevance
        content = speech.get('content', '').lower()
        if any(keyword in content for keyword in ['robot', 'help', 'stop', 'emergency']):
            base_priority += 0.2
        
        # Speaker familiarity
        if speech.get('speaker_familiar', False):
            base_priority += 0.1
        
        return min(1.0, base_priority)
    
    def _calculate_sound_priority(self, sound: Dict[str, Any], context: Optional[Dict[str, Any]]) -> float:
        """Calculate priority for sound events."""
        base_priority = 0.3
        
        # Intensity-based priority
        intensity = sound.get('intensity', 0)
        if intensity > 0.8:  # Loud sounds
            base_priority += 0.4
        elif intensity > 0.5:  # Moderate sounds
            base_priority += 0.2
        
        # Sound type
        sound_type = sound.get('type', '')
        if sound_type in ['alarm', 'scream', 'crash']:
            base_priority += 0.3
        
        return min(1.0, base_priority)
    
    def _calculate_contact_priority(self, contact: Dict[str, Any], context: Optional[Dict[str, Any]]) -> float:
        """Calculate priority for tactile contacts."""
        base_priority = 0.6  # High base priority for contacts
        
        # Pressure-based priority
        pressure = contact.get('pressure', 0)
        if pressure > 0.8:  # High pressure
            base_priority += 0.3
        elif pressure > 0.5:  # Moderate pressure
            base_priority += 0.1
        
        return min(1.0, base_priority)
    
    def _detect_safety_events(self, 
                             sensor_data: MultiModalSensorData,
                             context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect safety-critical events across all sensor modalities."""
        safety_events = []
        
        # Check for proximity violations
        if sensor_data.visual_data:
            objects = sensor_data.visual_data.get('objects', [])
            for obj in objects:
                if obj.get('distance', float('inf')) < 0.5:  # Very close objects
                    safety_events.append({
                        'type': 'proximity_violation',
                        'priority': 0.8,
                        'confidence': 0.9,
                        'location': obj.get('location'),
                        'severity': 'high'
                    })
        
        # Check for sudden movements
        if sensor_data.visual_data:
            motion = sensor_data.visual_data.get('motion', [])
            for motion_event in motion:
                velocity = motion_event.get('velocity', 0)
                if velocity > 2.0:  # Very fast motion
                    safety_events.append({
                        'type': 'sudden_movement',
                        'priority': 0.7,
                        'confidence': 0.8,
                        'location': motion_event.get('location'),
                        'severity': 'medium'
                    })
        
        # Check for loud sounds
        if sensor_data.audio_data:
            sounds = sensor_data.audio_data.get('sounds', [])
            for sound in sounds:
                if sound.get('intensity', 0) > 0.9:  # Very loud sounds
                    safety_events.append({
                        'type': 'loud_sound',
                        'priority': 0.6,
                        'confidence': 0.7,
                        'severity': 'medium'
                    })
        
        return safety_events
    
    def _select_attention_foci(self, targets: List[AttentionFocus]) -> List[AttentionFocus]:
        """Select and prioritize attention foci from all targets."""
        if not targets:
            return []
        
        # Sort by priority (highest first)
        sorted_targets = sorted(targets, key=lambda x: x.priority, reverse=True)
        
        # Select top targets up to max_foci
        selected = sorted_targets[:self.config['max_foci']]
        
        # Apply attention decay to existing foci
        current_time = time.time()
        for focus in selected:
            if focus.timestamp:
                age = current_time - focus.timestamp
                decay = self.config['attention_decay_rate'] * age
                focus.priority = max(0.0, focus.priority - decay)
        
        # Filter out low-priority foci
        selected = [focus for focus in selected if focus.priority > self.config['min_priority_threshold']]
        
        return selected
    
    def _decay_attention_foci(self):
        """Apply temporal decay to current attention foci."""
        current_time = time.time()
        decayed_foci = []
        
        for focus in self.current_foci:
            if focus.timestamp:
                age = current_time - focus.timestamp
                decay = self.config['attention_decay_rate'] * age
                focus.priority = max(0.0, focus.priority - decay)
                
                if focus.priority > self.config['min_priority_threshold']:
                    decayed_foci.append(focus)
        
        self.current_foci = decayed_foci
    
    def _update_attention_history(self):
        """Update attention history for learning and analysis."""
        if self.current_foci:
            self.attention_history.append({
                'timestamp': time.time(),
                'foci': self.current_foci.copy(),
                'num_foci': len(self.current_foci)
            })
    
    def _update_performance_metrics(self, start_time: float):
        """Update performance metrics."""
        response_time = time.time() - start_time
        self.performance_metrics['response_time'] = response_time
        
        # Calculate focus accuracy (simplified)
        if self.current_foci:
            avg_confidence = sum(focus.confidence for focus in self.current_foci) / len(self.current_foci)
            self.performance_metrics['focus_accuracy'] = avg_confidence
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def get_attention_summary(self) -> Dict[str, Any]:
        """Get a summary of current attention state."""
        return {
            'num_foci': len(self.current_foci),
            'foci_types': [focus.attention_type.value for focus in self.current_foci],
            'avg_priority': sum(focus.priority for focus in self.current_foci) / max(1, len(self.current_foci)),
            'performance_metrics': self.performance_metrics
        } 