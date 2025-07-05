#!/usr/bin/env python3
"""
Swarm Coordination Module

Manages communication, consensus, and coordination among swarm safety nodes.
Implements distributed decision-making and conflict resolution.
"""

import numpy as np
import threading
import time
import json
import uuid
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import random


class CoordinationEventType(Enum):
    """Types of coordination events"""
    CONSENSUS_REACHED = "consensus_reached"
    CONFLICT_DETECTED = "conflict_detected"
    EVOLUTION_TRIGGERED = "evolution_triggered"
    NODE_JOINED = "node_joined"
    NODE_LEFT = "node_left"
    DECISION_MADE = "decision_made"


@dataclass
class SwarmNode:
    """Represents a node in the swarm"""
    node_id: str
    cell_type: str
    join_time: float
    last_seen: float
    decision_count: int = 0
    consensus_count: int = 0
    conflict_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusEvent:
    """Represents a consensus event"""
    event_id: str
    timestamp: float
    participating_nodes: List[str]
    consensus_level: float
    decision_type: str
    decision_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictEvent:
    """Represents a conflict event"""
    event_id: str
    timestamp: float
    conflicting_nodes: List[str]
    conflict_type: str
    conflict_data: Dict[str, Any]
    resolution_strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SwarmCoordinator:
    """
    Swarm coordination manager
    
    Handles:
    - Node discovery and tracking
    - Consensus building
    - Conflict resolution
    - Decision propagation
    - Swarm evolution
    """
    
    def __init__(self, node_id: str, swarm_size: int = 5):
        self.node_id = node_id
        self.swarm_size = swarm_size
        
        # Node tracking
        self.nodes: Dict[str, SwarmNode] = {}
        self.active_nodes: Set[str] = set()
        
        # Decision tracking
        self.recent_decisions: deque = deque(maxlen=100)
        self.consensus_history: deque = deque(maxlen=50)
        self.conflict_history: deque = deque(maxlen=50)
        
        # Coordination parameters
        self.consensus_threshold = 0.7
        self.conflict_threshold = 0.3
        self.decision_timeout = 5.0  # seconds
        self.node_timeout = 30.0  # seconds
        
        # Evolution tracking
        self.evolution_stage = 0
        self.coordination_round = 0
        
        # Event queue
        self.event_queue: deque = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Register self
        self._register_node(node_id, "fusion")
        
        # Start background coordination
        self._start_coordination_thread()
    
    def _register_node(self, node_id: str, cell_type: str):
        """Register a node in the swarm"""
        
        with self.lock:
            if node_id not in self.nodes:
                node = SwarmNode(
                    node_id=node_id,
                    cell_type=cell_type,
                    join_time=time.time(),
                    last_seen=time.time()
                )
                self.nodes[node_id] = node
                self.active_nodes.add(node_id)
                
                # Create join event
                self._create_event(CoordinationEventType.NODE_JOINED, {
                    'node_id': node_id,
                    'cell_type': cell_type,
                    'total_nodes': len(self.nodes)
                })
    
    def _start_coordination_thread(self):
        """Start background coordination thread"""
        
        def coordination_loop():
            while True:
                try:
                    self._coordination_round()
                    time.sleep(1.0)  # Coordinate every second
                except Exception as e:
                    print(f"Error in coordination loop: {e}")
                    time.sleep(5.0)
        
        thread = threading.Thread(target=coordination_loop, daemon=True)
        thread.start()
    
    def _coordination_round(self):
        """Perform one round of coordination"""
        
        with self.lock:
            self.coordination_round += 1
            
            # Update node status
            self._update_node_status()
            
            # Check for consensus opportunities
            self._check_consensus_opportunities()
            
            # Check for conflicts
            self._check_conflicts()
            
            # Clean up old data
            self._cleanup_old_data()
    
    def _update_node_status(self):
        """Update status of all nodes"""
        
        current_time = time.time()
        inactive_nodes = []
        
        for node_id, node in self.nodes.items():
            if current_time - node.last_seen > self.node_timeout:
                inactive_nodes.append(node_id)
            else:
                # Update last seen if node is still active
                if node_id in self.active_nodes:
                    node.last_seen = current_time
        
        # Remove inactive nodes
        for node_id in inactive_nodes:
            self._remove_node(node_id)
    
    def _remove_node(self, node_id: str):
        """Remove a node from the swarm"""
        
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # Create leave event
            self._create_event(CoordinationEventType.NODE_LEFT, {
                'node_id': node_id,
                'cell_type': node.cell_type,
                'total_nodes': len(self.nodes) - 1
            })
            
            # Remove from tracking
            del self.nodes[node_id]
            self.active_nodes.discard(node_id)
    
    def _check_consensus_opportunities(self):
        """Check for consensus opportunities among recent decisions"""
        
        if len(self.recent_decisions) < 3:
            return
        
        # Group decisions by similarity
        decision_groups = self._group_similar_decisions()
        
        for group in decision_groups:
            if len(group) >= 3:  # Need at least 3 similar decisions
                consensus_level = self._calculate_consensus_level(group)
                
                if consensus_level >= self.consensus_threshold:
                    self._trigger_consensus(group, consensus_level)
    
    def _group_similar_decisions(self) -> List[List[Dict]]:
        """Group similar decisions together"""
        
        groups = []
        used_decisions = set()
        
        for i, decision1 in enumerate(self.recent_decisions):
            if i in used_decisions:
                continue
            
            group = [decision1]
            used_decisions.add(i)
            
            for j, decision2 in enumerate(self.recent_decisions):
                if j in used_decisions:
                    continue
                
                if self._decisions_similar(decision1, decision2):
                    group.append(decision2)
                    used_decisions.add(j)
            
            if len(group) >= 2:
                groups.append(group)
        
        return groups
    
    def _decisions_similar(self, decision1: Dict, decision2: Dict) -> bool:
        """Check if two decisions are similar"""
        
        # Compare safety levels
        safety_diff = abs(decision1.get('safety_level', 0) - decision2.get('safety_level', 0))
        if safety_diff > 0.1:
            return False
        
        # Compare confidence levels
        confidence_diff = abs(decision1.get('confidence', 0) - decision2.get('confidence', 0))
        if confidence_diff > 0.2:
            return False
        
        # Compare timestamps (should be recent)
        time_diff = abs(decision1.get('timestamp', 0) - decision2.get('timestamp', 0))
        if time_diff > 10.0:  # 10 seconds
            return False
        
        return True
    
    def _calculate_consensus_level(self, decisions: List[Dict]) -> float:
        """Calculate consensus level for a group of decisions"""
        
        if not decisions:
            return 0.0
        
        # Calculate average safety level
        safety_levels = [d.get('safety_level', 0) for d in decisions]
        avg_safety = np.mean(safety_levels)
        
        # Calculate variance (lower variance = higher consensus)
        variance = np.var(safety_levels)
        consensus_from_variance = max(0, 1 - variance * 10)
        
        # Calculate confidence consensus
        confidences = [d.get('confidence', 0) for d in decisions]
        avg_confidence = np.mean(confidences)
        
        # Combine metrics
        consensus_level = (consensus_from_variance + avg_confidence) / 2
        
        return min(1.0, consensus_level)
    
    def _trigger_consensus(self, decisions: List[Dict], consensus_level: float):
        """Trigger a consensus event"""
        
        # Create consensus event
        event_id = f"consensus_{int(time.time() * 1000)}"
        
        # Calculate average decision
        avg_safety = np.mean([d.get('safety_level', 0) for d in decisions])
        avg_confidence = np.mean([d.get('confidence', 0) for d in decisions])
        
        participating_nodes = list(set([d.get('node_id', 'unknown') for d in decisions]))
        
        consensus_event = ConsensusEvent(
            event_id=event_id,
            timestamp=time.time(),
            participating_nodes=participating_nodes,
            consensus_level=consensus_level,
            decision_type="safety_assessment",
            decision_data={
                'safety_level': avg_safety,
                'confidence': avg_confidence,
                'is_safe': avg_safety > 0.7,
                'decision_count': len(decisions)
            }
        )
        
        # Store consensus event
        self.consensus_history.append(consensus_event)
        
        # Create coordination event
        self._create_event(CoordinationEventType.CONSENSUS_REACHED, {
            'event_id': event_id,
            'consensus_level': consensus_level,
            'participating_nodes': participating_nodes,
            'decision_data': consensus_event.decision_data
        })
        
        # Update node statistics
        for node_id in participating_nodes:
            if node_id in self.nodes:
                self.nodes[node_id].consensus_count += 1
    
    def _check_conflicts(self):
        """Check for conflicts among recent decisions"""
        
        if len(self.recent_decisions) < 2:
            return
        
        # Find conflicting decisions
        conflicts = self._find_conflicts()
        
        for conflict in conflicts:
            if len(conflict) >= 2:
                self._resolve_conflict(conflict)
    
    def _find_conflicts(self) -> List[List[Dict]]:
        """Find groups of conflicting decisions"""
        
        conflicts = []
        
        for i, decision1 in enumerate(self.recent_decisions):
            for j, decision2 in enumerate(self.recent_decisions[i+1:], i+1):
                if self._decisions_conflict(decision1, decision2):
                    # Check if either decision is already in a conflict group
                    conflict_group = [decision1, decision2]
                    
                    # Add other conflicting decisions
                    for k, decision3 in enumerate(self.recent_decisions):
                        if k != i and k != j:
                            if (self._decisions_conflict(decision1, decision3) or 
                                self._decisions_conflict(decision2, decision3)):
                                conflict_group.append(decision3)
                    
                    conflicts.append(conflict_group)
        
        return conflicts
    
    def _decisions_conflict(self, decision1: Dict, decision2: Dict) -> bool:
        """Check if two decisions conflict"""
        
        # Check if decisions are recent
        time_diff = abs(decision1.get('timestamp', 0) - decision2.get('timestamp', 0))
        if time_diff > 5.0:  # 5 seconds
            return False
        
        # Check for safety level conflict
        safety1 = decision1.get('safety_level', 0)
        safety2 = decision2.get('safety_level', 0)
        
        # Conflict if one is safe and other is unsafe
        if (safety1 > 0.7 and safety2 < 0.3) or (safety2 > 0.7 and safety1 < 0.3):
            return True
        
        # Check for high confidence conflicts
        conf1 = decision1.get('confidence', 0)
        conf2 = decision2.get('confidence', 0)
        
        if conf1 > 0.8 and conf2 > 0.8:
            safety_diff = abs(safety1 - safety2)
            if safety_diff > 0.3:
                return True
        
        return False
    
    def _resolve_conflict(self, conflicting_decisions: List[Dict]):
        """Resolve a conflict among decisions"""
        
        # Create conflict event
        event_id = f"conflict_{int(time.time() * 1000)}"
        
        conflicting_nodes = list(set([d.get('node_id', 'unknown') for d in conflicting_decisions]))
        
        # Determine resolution strategy
        resolution_strategy = self._determine_resolution_strategy(conflicting_decisions)
        
        conflict_event = ConflictEvent(
            event_id=event_id,
            timestamp=time.time(),
            conflicting_nodes=conflicting_nodes,
            conflict_type="safety_assessment",
            conflict_data={
                'decisions': conflicting_decisions,
                'safety_levels': [d.get('safety_level', 0) for d in conflicting_decisions],
                'confidences': [d.get('confidence', 0) for d in conflicting_decisions]
            },
            resolution_strategy=resolution_strategy
        )
        
        # Store conflict event
        self.conflict_history.append(conflict_event)
        
        # Create coordination event
        self._create_event(CoordinationEventType.CONFLICT_DETECTED, {
            'event_id': event_id,
            'conflicting_nodes': conflicting_nodes,
            'resolution_strategy': resolution_strategy,
            'conflict_data': conflict_event.conflict_data
        })
        
        # Update node statistics
        for node_id in conflicting_nodes:
            if node_id in self.nodes:
                self.nodes[node_id].conflict_count += 1
        
        # Apply resolution
        self._apply_resolution(conflicting_decisions, resolution_strategy)
    
    def _determine_resolution_strategy(self, conflicting_decisions: List[Dict]) -> str:
        """Determine resolution strategy for conflict"""
        
        # Count safe vs unsafe decisions
        safe_count = sum(1 for d in conflicting_decisions if d.get('safety_level', 0) > 0.7)
        unsafe_count = sum(1 for d in conflicting_decisions if d.get('safety_level', 0) < 0.3)
        
        # Count high confidence decisions
        high_conf_count = sum(1 for d in conflicting_decisions if d.get('confidence', 0) > 0.8)
        
        if safe_count > unsafe_count:
            return "majority_safe"
        elif unsafe_count > safe_count:
            return "majority_unsafe"
        elif high_conf_count > len(conflicting_decisions) / 2:
            return "high_confidence_weighted"
        else:
            return "conservative_unsafe"
    
    def _apply_resolution(self, conflicting_decisions: List[Dict], strategy: str):
        """Apply resolution strategy to conflict"""
        
        if strategy == "majority_safe":
            # Default to safe
            resolved_safety = 0.8
        elif strategy == "majority_unsafe":
            # Default to unsafe
            resolved_safety = 0.2
        elif strategy == "high_confidence_weighted":
            # Weight by confidence
            total_weight = 0
            weighted_safety = 0
            
            for decision in conflicting_decisions:
                weight = decision.get('confidence', 0)
                weighted_safety += weight * decision.get('safety_level', 0)
                total_weight += weight
            
            if total_weight > 0:
                resolved_safety = weighted_safety / total_weight
            else:
                resolved_safety = 0.5
        else:  # conservative_unsafe
            # Default to unsafe for safety
            resolved_safety = 0.2
        
        # Create resolved decision
        resolved_decision = {
            'node_id': self.node_id,
            'safety_level': resolved_safety,
            'confidence': 0.6,  # Lower confidence due to conflict
            'timestamp': time.time(),
            'resolution_strategy': strategy,
            'conflict_resolved': True
        }
        
        # Add to recent decisions
        self.recent_decisions.append(resolved_decision)
    
    def _cleanup_old_data(self):
        """Clean up old data"""
        
        current_time = time.time()
        
        # Clean up old decisions
        while self.recent_decisions and current_time - self.recent_decisions[0].get('timestamp', 0) > 60:
            self.recent_decisions.popleft()
        
        # Clean up old events
        while self.event_queue and len(self.event_queue) > 50:
            self.event_queue.popleft()
    
    def _create_event(self, event_type: CoordinationEventType, data: Dict[str, Any]):
        """Create a coordination event"""
        
        event = {
            'event_id': f"{event_type.value}_{int(time.time() * 1000)}",
            'event_type': event_type.value,
            'timestamp': time.time(),
            'node_id': self.node_id,
            'data': data
        }
        
        self.event_queue.append(event)
    
    def update(self):
        """Update coordination state"""
        
        with self.lock:
            # Update node last seen
            if self.node_id in self.nodes:
                self.nodes[self.node_id].last_seen = time.time()
    
    def add_decision(self, decision: Dict[str, Any]):
        """Add a decision to the coordination system"""
        
        with self.lock:
            # Add node ID if not present
            if 'node_id' not in decision:
                decision['node_id'] = self.node_id
            
            # Add timestamp if not present
            if 'timestamp' not in decision:
                decision['timestamp'] = time.time()
            
            # Add to recent decisions
            self.recent_decisions.append(decision)
            
            # Update node statistics
            if self.node_id in self.nodes:
                self.nodes[self.node_id].decision_count += 1
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get pending coordination events"""
        
        with self.lock:
            events = list(self.event_queue)
            self.event_queue.clear()
            return events
    
    def evolve(self, evolution_stage: int):
        """Evolve the coordination system"""
        
        with self.lock:
            self.evolution_stage = evolution_stage
            
            # Adjust coordination parameters based on evolution
            if evolution_stage > 0:
                # Increase consensus threshold for higher evolution stages
                self.consensus_threshold = min(0.9, self.consensus_threshold + 0.05)
                
                # Decrease conflict threshold
                self.conflict_threshold = max(0.1, self.conflict_threshold - 0.05)
            
            # Create evolution event
            self._create_event(CoordinationEventType.EVOLUTION_TRIGGERED, {
                'evolution_stage': evolution_stage,
                'new_consensus_threshold': self.consensus_threshold,
                'new_conflict_threshold': self.conflict_threshold
            })
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of the coordination system"""
        
        with self.lock:
            return {
                'node_id': self.node_id,
                'swarm_size': self.swarm_size,
                'active_nodes': len(self.active_nodes),
                'total_nodes': len(self.nodes),
                'evolution_stage': self.evolution_stage,
                'coordination_round': self.coordination_round,
                'recent_decisions': len(self.recent_decisions),
                'consensus_history': len(self.consensus_history),
                'conflict_history': len(self.conflict_history),
                'pending_events': len(self.event_queue),
                'consensus_threshold': self.consensus_threshold,
                'conflict_threshold': self.conflict_threshold,
                'node_statistics': {
                    node_id: {
                        'cell_type': node.cell_type,
                        'decision_count': node.decision_count,
                        'consensus_count': node.consensus_count,
                        'conflict_count': node.conflict_count,
                        'last_seen': node.last_seen
                    }
                    for node_id, node in self.nodes.items()
                }
            } 