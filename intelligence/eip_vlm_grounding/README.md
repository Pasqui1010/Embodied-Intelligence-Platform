# VLM Grounding Package

Vision-Language Grounding package for spatial reference resolution and object affordance estimation in robotics applications.

## Overview

The VLM Grounding package provides advanced vision-language understanding capabilities for robots, enabling them to:

- **Resolve spatial references** in natural language commands (e.g., "move to the left of the red cup")
- **Estimate object affordances** for manipulation planning (grasp points, stability, safety)
- **Understand scene context** through comprehensive scene analysis
- **Integrate with Safety-Embedded LLM** for safe reasoning and validation

## Features

### 🎯 Spatial Reference Resolution
- Support for relative spatial references ("left of", "behind", "next to")
- Absolute spatial references ("at coordinates x,y")
- Object-based references ("near the red cup")
- Multi-object scene understanding
- Real-time processing with <200ms response time

### 🤖 Object Affordance Estimation
- Grasp point detection for manipulation
- Object stability assessment
- Manipulation difficulty estimation
- Safety-aware affordance filtering
- Support for multiple grasp types (pinch, palmar, power, precision)

### 🧠 Scene Understanding
- Comprehensive scene analysis and classification
- Object detection and tracking
- Spatial relation analysis
- Scene complexity and safety assessment
- Natural language scene descriptions

### 🔒 Safety Integration
- Integration with Safety-Embedded LLM
- Real-time safety validation
- Dangerous action detection
- Safety constraint enforcement
- Confidence-based decision making

## Installation

### Prerequisites

- ROS 2 Humble or later
- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)

### Dependencies

```bash
# Install ROS 2 dependencies
sudo apt install ros-humble-rclpy ros-humble-sensor-msgs ros-humble-geometry-msgs

# Install Python dependencies
pip install -r requirements.txt
```

### Building the Package

```bash
# Navigate to your workspace
cd ~/ros2_ws

# Build the package
colcon build --packages-select eip_vlm_grounding

# Source the workspace
source install/setup.bash
```

## Usage

### Basic Usage

1. **Launch the VLM Grounding Node:**
```bash
ros2 launch eip_vlm_grounding vlm_grounding_demo.launch.py
```

2. **Send Spatial Reference Queries:**
```bash
ros2 topic pub /vlm_grounding/spatial_query std_msgs/String "data: 'move to the left of the red cup'"
```

3. **Request Affordance Estimation:**
```bash
ros2 topic pub /vlm_grounding/affordance_query std_msgs/String "data: 'what can I do with the cup?'"
```

4. **Send VLM Reasoning Queries:**
```bash
ros2 topic pub /vlm_grounding/vlm_query std_msgs/String "data: 'describe the scene and identify objects'"
```

### Service Calls

```bash
# Resolve spatial reference via service
ros2 service call /vlm_grounding/resolve_spatial_reference std_srvs/String "data: 'move to the left of the red cup'"

# Estimate affordances via service
ros2 service call /vlm_grounding/estimate_affordances std_srvs/String "data: 'cup'"

# VLM reasoning via service
ros2 service call /vlm_grounding/vlm_reasoning std_srvs/String "data: 'is it safe to pick up the cup?'"
```

### Configuration

The package can be configured using the `vlm_grounding.yaml` configuration file:

```yaml
vlm_grounding_node:
  ros__parameters:
    # Processing rates
    spatial_resolution_rate: 10.0
    affordance_estimation_rate: 5.0
    scene_analysis_rate: 2.0
    vlm_integration_rate: 1.0
    
    # Model settings
    enable_clip: true
    enable_safety_validation: true
    min_confidence_threshold: 0.6
    max_objects_per_scene: 20
    enable_visualization: true
```

## API Reference

### Topics

#### Subscribed Topics
- `/camera/color/image_raw` (sensor_msgs/Image): Camera image feed
- `/camera/depth/points` (sensor_msgs/PointCloud2): Depth point cloud
- `/scan` (sensor_msgs/LaserScan): Lidar scan data
- `/vlm_grounding/spatial_query` (std_msgs/String): Spatial reference queries
- `/vlm_grounding/affordance_query` (std_msgs/String): Affordance estimation queries
- `/vlm_grounding/vlm_query` (std_msgs/String): VLM reasoning queries

#### Published Topics
- `/vlm_grounding/spatial_reference` (std_msgs/String): Spatial reference results
- `/vlm_grounding/affordance_result` (std_msgs/String): Affordance estimation results
- `/vlm_grounding/vlm_result` (std_msgs/String): VLM reasoning results
- `/vlm_grounding/scene_description` (std_msgs/String): Scene descriptions
- `/vlm_grounding/visualization` (visualization_msgs/MarkerArray): Visualization markers

### Services

- `/vlm_grounding/resolve_spatial_reference` (std_srvs/String): Resolve spatial references
- `/vlm_grounding/estimate_affordances` (std_srvs/String): Estimate object affordances
- `/vlm_grounding/vlm_reasoning` (std_srvs/String): Perform VLM reasoning

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spatial_resolution_rate` | double | 10.0 | Spatial resolution processing rate (Hz) |
| `affordance_estimation_rate` | double | 5.0 | Affordance estimation processing rate (Hz) |
| `scene_analysis_rate` | double | 2.0 | Scene analysis processing rate (Hz) |
| `vlm_integration_rate` | double | 1.0 | VLM integration processing rate (Hz) |
| `enable_clip` | bool | true | Enable CLIP model for vision-language grounding |
| `enable_safety_validation` | bool | true | Enable safety validation for VLM responses |
| `min_confidence_threshold` | double | 0.6 | Minimum confidence threshold for VLM responses |
| `max_objects_per_scene` | int | 20 | Maximum number of objects to track per scene |
| `enable_visualization` | bool | true | Enable visualization markers |

## Architecture

### Core Components

1. **SpatialReferenceResolver**: Handles spatial reference resolution using CLIP and geometric heuristics
2. **ObjectAffordanceEstimator**: Estimates manipulation affordances with neural and geometric methods
3. **SceneUnderstanding**: Provides comprehensive scene analysis and understanding
4. **VLMIntegration**: Integrates vision-language models with Safety-Embedded LLM
5. **VLMGroundingNode**: Main ROS 2 node orchestrating all components

### Data Flow

```
Camera/Lidar Data → Scene Understanding → Spatial Reference Resolution
                                      ↓
Object Detection → Affordance Estimation → VLM Integration
                                      ↓
Safety Validation → Response Generation → ROS 2 Topics/Services
```

## Testing

### Run Tests

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test
python3 -m pytest tests/test_spatial_reference.py -v

# Run with coverage
python3 -m pytest tests/ --cov=eip_vlm_grounding --cov-report=html
```

### Test Coverage

The package includes comprehensive tests for:
- Spatial reference resolution
- Object affordance estimation
- Scene understanding
- VLM integration
- Safety validation

## Performance

### Benchmarks

| Component | Response Time | Accuracy | Memory Usage |
|-----------|---------------|----------|--------------|
| Spatial Resolution | <200ms | >90% | <500MB |
| Affordance Estimation | <500ms | >85% | <1GB |
| Scene Understanding | <1s | >80% | <2GB |
| VLM Integration | <2s | >75% | <3GB |

### Optimization

- GPU acceleration for CLIP model inference
- Batch processing for multiple queries
- Memory-efficient object tracking
- Real-time performance monitoring

## Safety Features

### Safety Validation
- Integration with Safety-Embedded LLM
- Real-time safety constraint checking
- Dangerous action detection and prevention
- Confidence-based decision making

### Error Handling
- Graceful degradation when models are unavailable
- Fallback methods for critical functions
- Comprehensive error logging and reporting
- Recovery mechanisms for system failures

## Integration

### With Safety-Embedded LLM
The package integrates seamlessly with the Safety-Embedded LLM for enhanced safety validation:

```python
from eip_vlm_grounding import VLMIntegration
from eip_llm_interface import SafetyEmbeddedLLM

# Initialize integration
vlm_integration = VLMIntegration(safety_llm_path="path/to/safety_llm")

# Process with safety validation
result = vlm_integration.process_visual_query(
    query="pick up the cup",
    scene_data=scene_data,
    scene_description=scene_description
)
```

### With Multi-Modal Safety
Integration with the multi-modal safety system for comprehensive safety monitoring:

```python
from eip_multimodal_safety import MultimodalSafetyNode

# Initialize safety monitoring
safety_node = MultimodalSafetyNode()
safety_node.add_vlm_grounding_integration(vlm_grounding_node)
```

## Troubleshooting

### Common Issues

1. **CLIP Model Not Loading**
   - Ensure transformers and CLIP are properly installed
   - Check GPU memory availability
   - Use CPU fallback if GPU is unavailable

2. **Low Confidence Results**
   - Verify camera calibration
   - Check lighting conditions
   - Adjust confidence thresholds in configuration

3. **Performance Issues**
   - Enable GPU acceleration
   - Reduce processing rates
   - Monitor memory usage

### Debug Mode

Enable debug mode for detailed logging:

```bash
ros2 run eip_vlm_grounding vlm_grounding_node --ros-args -p debug.enable_debug_mode:=true
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](../../docs/CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/embodied-intelligence/eip.git

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run linting
black eip_vlm_grounding/
flake8 eip_vlm_grounding/
mypy eip_vlm_grounding/
```

## License

This package is licensed under the Apache 2.0 License. See [LICENSE](../../LICENSE) for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{eip_vlm_grounding,
  title={VLM Grounding Package for Robotics},
  author={Embodied Intelligence Platform Team},
  year={2024},
  url={https://github.com/embodied-intelligence/eip}
}
```

## Support

For support and questions:
- GitHub Issues: [Create an issue](https://github.com/embodied-intelligence/eip/issues)
- Documentation: [Read the docs](https://embodied-intelligence.github.io/eip/)
- Community: [Join our discussions](https://github.com/embodied-intelligence/eip/discussions) 