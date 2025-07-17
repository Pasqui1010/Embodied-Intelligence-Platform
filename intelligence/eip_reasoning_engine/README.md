# Advanced Multi-Modal Reasoning Engine

The Advanced Multi-Modal Reasoning Engine is a sophisticated reasoning system for autonomous robotic platforms that combines visual perception, natural language understanding, spatial awareness, and safety constraints to enable intelligent decision-making.

## Overview

This reasoning engine provides advanced cognitive capabilities for robots by integrating multiple reasoning modalities:

- **Multi-Modal Reasoning**: Combines visual, language, spatial, and safety contexts
- **Spatial Reasoning**: Understanding object relationships and navigation
- **Temporal Reasoning**: Planning sequences and understanding time constraints
- **Causal Reasoning**: Understanding cause-effect relationships and risks
- **Safety Reasoning**: Ensuring safe operation with constraint validation

## Architecture

The reasoning engine consists of several key components:

```
eip_reasoning_engine/
├── eip_reasoning_engine/
│   ├── reasoning_engine_node.py      # Main ROS node
│   ├── multi_modal_reasoner.py       # Multi-modal reasoning core
│   ├── spatial_reasoner.py           # Spatial reasoning engine
│   ├── temporal_reasoner.py          # Temporal reasoning engine
│   ├── causal_reasoner.py            # Causal reasoning engine
│   └── reasoning_orchestrator.py     # Component coordination
├── launch/
│   └── reasoning_engine_demo.launch.py
├── config/
│   └── reasoning_engine.yaml
└── tests/
    ├── test_multi_modal_reasoning.py
    ├── test_spatial_reasoning.py
    └── test_causal_reasoning.py
```

## Features

### Multi-Modal Reasoning
- **Visual Context Integration**: Processes visual scene understanding
- **Language Grounding**: Grounds natural language commands in visual context
- **Spatial Awareness**: Integrates spatial relationships and navigation
- **Safety Validation**: Ensures all reasoning respects safety constraints

### Reasoning Modes
- **Fast Mode**: Quick reasoning for real-time decisions (< 200ms)
- **Balanced Mode**: Balanced speed and accuracy (< 500ms)
- **Thorough Mode**: Comprehensive reasoning for complex tasks (< 2s)
- **Safety Critical Mode**: Maximum safety focus with enhanced validation

### Performance Features
- **Async Processing**: Non-blocking reasoning with priority queues
- **Performance Monitoring**: Real-time performance tracking and optimization
- **Fallback Mechanisms**: Graceful degradation when reasoning fails
- **Extensible Architecture**: Easy integration of new reasoning capabilities

## Installation

### Prerequisites
- ROS 2 Humble or later
- Python 3.8+
- Required dependencies (see `requirements.txt`)

### Building
```bash
# Navigate to your workspace
cd /path/to/your/workspace

# Build the package
colcon build --packages-select eip_reasoning_engine

# Source the workspace
source install/setup.bash
```

## Usage

### Basic Usage

1. **Launch the reasoning engine**:
```bash
ros2 launch eip_reasoning_engine reasoning_engine_demo.launch.py
```

2. **Publish visual context**:
```bash
ros2 topic pub /eip/vision/context std_msgs/msg/String "data: '{\"objects\": [{\"name\": \"red_cube\", \"position\": [1.0, 0.0, 0.5]}], \"scene_description\": \"A table with a red cube\", \"confidence\": 0.8}'"
```

3. **Send a language command**:
```bash
ros2 topic pub /eip/language/commands std_msgs/msg/String "data: '{\"command\": \"Move to the red cube\"}'"
```

4. **Monitor reasoning results**:
```bash
ros2 topic echo /eip/reasoning/results
```

### Configuration

The reasoning engine can be configured through the `config/reasoning_engine.yaml` file:

```yaml
reasoning_engine_node:
  ros__parameters:
    reasoning_mode: "balanced"
    max_reasoning_time: 0.5
    enable_visual_reasoning: true
    enable_spatial_reasoning: true
    enable_temporal_reasoning: true
    enable_causal_reasoning: true
    enable_safety_reasoning: true
```

### API Reference

#### Multi-Modal Reasoner

```python
from eip_reasoning_engine import MultiModalReasoner, VisualContext, SpatialContext, SafetyConstraints

# Initialize reasoner
reasoner = MultiModalReasoner()

# Perform reasoning
result = reasoner.reason_about_scene(
    visual_context=visual_context,
    language_command="Move to the red cube",
    spatial_context=spatial_context,
    safety_constraints=safety_constraints
)

# Access results
print(f"Confidence: {result.confidence}")
print(f"Safety Score: {result.safety_score}")
print(f"Plan: {result.plan}")
```

#### Spatial Reasoner

```python
from eip_reasoning_engine import SpatialReasoner

# Initialize spatial reasoner
spatial_reasoner = SpatialReasoner()

# Analyze spatial relationships
spatial_understanding = spatial_reasoner.analyze_scene(
    visual_context, spatial_context
)

# Access spatial information
print(f"Object relationships: {spatial_understanding.object_relationships}")
print(f"Navigation paths: {spatial_understanding.navigation_paths}")
```

#### Causal Reasoner

```python
from eip_reasoning_engine import CausalReasoner

# Initialize causal reasoner
causal_reasoner = CausalReasoner()

# Analyze causal effects
causal_analysis = causal_reasoner.analyze_effects(
    task_plan, spatial_understanding, safety_constraints
)

# Access risk assessment
print(f"Risk level: {causal_analysis.risk_assessment.risk_level}")
print(f"Risk score: {causal_analysis.risk_assessment.risk_score}")
```

## Testing

Run the test suite:

```bash
# Run all tests
colcon test --packages-select eip_reasoning_engine

# Run specific test
python3 -m pytest intelligence/eip_reasoning_engine/tests/test_multi_modal_reasoning.py -v

# Run with coverage
python3 -m pytest intelligence/eip_reasoning_engine/tests/ --cov=eip_reasoning_engine --cov-report=html
```

## Performance

### Benchmarks
- **Fast Mode**: < 200ms response time
- **Balanced Mode**: < 500ms response time
- **Thorough Mode**: < 2s response time
- **Safety Critical Mode**: < 1s response time

### Memory Usage
- **Base Memory**: ~50MB
- **Peak Memory**: ~200MB (complex scenes)
- **Memory per Request**: ~10MB

## Integration

### With Vision System
The reasoning engine integrates with vision systems through the `/eip/vision/context` topic:

```json
{
  "objects": [
    {
      "name": "red_cube",
      "position": [1.0, 0.0, 0.5],
      "dimensions": [0.1, 0.1, 0.1],
      "confidence": 0.9
    }
  ],
  "scene_description": "A table with a red cube",
  "spatial_relationships": {
    "red_cube": ["near_table", "above_surface"]
  },
  "affordances": {
    "red_cube": ["graspable", "movable"]
  },
  "confidence": 0.8
}
```

### With Language System
Integration with language systems through `/eip/language/commands`:

```json
{
  "command": "Pick up the red cube and place it on the table",
  "confidence": 0.9,
  "timestamp": 1234567890.123
}
```

### With Safety System
Safety constraints through `/eip/safety/constraints`:

```json
{
  "collision_threshold": 0.7,
  "human_proximity_threshold": 0.8,
  "velocity_limits": {
    "linear": 1.0,
    "angular": 1.0
  },
  "workspace_boundaries": {
    "min_x": -5.0,
    "max_x": 5.0,
    "min_y": -5.0,
    "max_y": 5.0
  }
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **ROS Topic Issues**: Check topic connections
   ```bash
   ros2 topic list
   ros2 topic info /eip/reasoning/results
   ```

3. **Performance Issues**: Monitor system resources
   ```bash
   ros2 topic echo /eip/reasoning/performance_stats
   ```

### Debug Mode

Enable debug logging:

```yaml
reasoning_engine_node:
  ros__parameters:
    debug:
      enable_debug_mode: true
      log_detailed_reasoning: true
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the test examples

## Roadmap

- [ ] Integration with advanced vision models
- [ ] Support for more reasoning modalities
- [ ] Real-time learning capabilities
- [ ] Enhanced safety validation
- [ ] Performance optimization
- [ ] Extended language understanding 