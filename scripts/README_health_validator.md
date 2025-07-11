# Adaptive Safety Health Validator

## Overview

The Adaptive Safety Health Validator implements the **System Health Validation Prompt** from the prompt engineering framework. This tool provides automated health assessment of the adaptive safety learning system in the embodied intelligence platform.

## Features

✅ **Automated Health Assessment** - Real-time system health monitoring  
✅ **Comprehensive Metrics Analysis** - Learning effectiveness, safety reliability, performance  
✅ **Intelligent Recommendations** - Actionable improvement suggestions  
✅ **Alert System** - Critical and warning alerts for system issues  
✅ **Continuous Monitoring** - Long-running validation with configurable intervals  
✅ **Configuration Management** - Flexible configuration via JSON files  
✅ **Historical Tracking** - Validation history with automatic cleanup  
✅ **Integration Ready** - Can integrate with existing adaptive safety systems  

## Architecture

### Core Components

1. **SystemHealthValidator** - Main validation engine
2. **SystemMetrics** - Data structure for system metrics
3. **HealthAssessment** - Validation results container
4. **Configuration Management** - Flexible configuration system

### Validation Process

```
Input Metrics → Analysis → Scoring → Assessment → Recommendations → Alerts
     ↓              ↓         ↓          ↓             ↓            ↓
Learning    → Effectiveness → Health → Status → Actionable → Critical
Safety      → Reliability   → Score   → Risk   → Suggestions → Warnings
Performance → Bottlenecks   → Report  → Level  → Improvements → Alerts
```

## Installation

### Prerequisites

```bash
pip install numpy
pip install torch  # If using adaptive learning engine
```

### Setup

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd Embodied-Intelligence-Platform
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
cd scripts
python test_health_validator.py
```

## Usage

### Command Line Interface

#### Single Validation
```bash
# Basic validation with mock data
python adaptive_safety_health_validator.py

# Validation with custom configuration
python adaptive_safety_health_validator.py --config health_validator_config.json

# Validation with mock metrics for testing
python adaptive_safety_health_validator.py --mock
```

#### Continuous Monitoring
```bash
# Continuous validation with 60-second intervals
python adaptive_safety_health_validator.py --continuous

# Continuous validation with custom interval
python adaptive_safety_health_validator.py --continuous --interval 30

# Continuous validation with iteration limit
python adaptive_safety_health_validator.py --continuous --iterations 100
```

### Programmatic Usage

```python
from adaptive_safety_health_validator import SystemHealthValidator, SystemMetrics

# Initialize validator
validator = SystemHealthValidator("config.json")

# Create system metrics
metrics = SystemMetrics(
    learning_rounds=500,
    pattern_discoveries=45,
    rule_evolutions=12,
    adaptation_count=85,
    safety_level=0.92,
    confidence=0.88,
    memory_usage=0.65,
    processing_latency=35.0
)

# Perform validation
assessment = validator.validate_health(metrics)

# Get validation report
report = validator.get_validation_report(assessment)
print(json.dumps(report, indent=2))

# Check for alerts
alerts = validator.check_alerts(assessment)
for alert in alerts:
    print(f"[{alert['level'].upper()}] {alert['message']}")
```

### Integration with Adaptive Safety System

```python
from eip_adaptive_safety.adaptive_learning_engine import AdaptiveLearningEngine
from adaptive_safety_health_validator import SystemHealthValidator

# Initialize adaptive safety system
learning_engine = AdaptiveLearningEngine()

# Initialize health validator
validator = SystemHealthValidator()

# Collect metrics from actual system
metrics = validator.collect_system_metrics(learning_engine)

# Perform validation
assessment = validator.validate_health(metrics)

# Handle results
if assessment.status == "critical":
    # Take immediate action
    print("CRITICAL: System health requires immediate attention")
elif assessment.status == "degraded":
    # Monitor closely
    print("WARNING: System health is degraded")
else:
    # Normal operation
    print("INFO: System health is good")
```

## Configuration

### Configuration File Format

```json
{
  "validation_interval": 60.0,
  "history_retention_hours": 24,
  "alert_thresholds": {
    "health_score_critical": 50,
    "health_score_warning": 70,
    "safety_reliability_min": 0.8,
    "learning_effectiveness_min": 0.6,
    "memory_usage_max": 0.9,
    "latency_max": 100.0
  },
  "scoring_weights": {
    "learning_effectiveness": 0.25,
    "safety_reliability": 0.35,
    "performance": 0.25,
    "stability": 0.15
  }
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `validation_interval` | float | 60.0 | Validation interval in seconds |
| `history_retention_hours` | int | 24 | Hours to retain validation history |
| `alert_thresholds.health_score_critical` | int | 50 | Critical health score threshold |
| `alert_thresholds.health_score_warning` | int | 70 | Warning health score threshold |
| `alert_thresholds.safety_reliability_min` | float | 0.8 | Minimum safety reliability |
| `alert_thresholds.learning_effectiveness_min` | float | 0.6 | Minimum learning effectiveness |
| `scoring_weights.learning_effectiveness` | float | 0.25 | Weight for learning effectiveness |
| `scoring_weights.safety_reliability` | float | 0.35 | Weight for safety reliability |
| `scoring_weights.performance` | float | 0.25 | Weight for performance |
| `scoring_weights.stability` | float | 0.15 | Weight for stability |

## Validation Metrics

### Input Metrics

The validator analyzes the following system metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `learning_rounds` | int | Total learning rounds completed |
| `pattern_discoveries` | int | Number of safety patterns discovered |
| `rule_evolutions` | int | Number of safety rule evolutions |
| `adaptation_count` | int | Total adaptations performed |
| `safety_level` | float | Current safety level (0.0-1.0) |
| `confidence` | float | System confidence score (0.0-1.0) |
| `memory_usage` | float | Memory usage ratio (0.0-1.0) |
| `processing_latency` | float | Processing latency in milliseconds |

### Output Metrics

The validator produces comprehensive health assessments:

| Metric | Type | Description |
|--------|------|-------------|
| `health_score` | float | Overall health score (0-100) |
| `status` | string | System status (healthy/degraded/critical) |
| `risk_level` | string | Risk level (low/medium/high) |
| `learning_effectiveness` | object | Learning effectiveness analysis |
| `safety_reliability` | object | Safety reliability analysis |
| `performance_analysis` | object | Performance and bottleneck analysis |
| `recommendations` | array | Actionable improvement suggestions |

## Health Scoring Algorithm

### Score Calculation

The health score is calculated using weighted components:

```
Health Score = (
    Learning Effectiveness Score × 0.25 +
    Safety Reliability Score × 0.35 +
    Performance Score × 0.25 +
    Stability Score × 0.15
) × 100
```

### Component Scoring

#### Learning Effectiveness
- **Learning Rate**: Pattern discoveries per learning round
- **Adaptation Efficiency**: Adaptations per rule evolution
- **Score Range**: 0.0 - 1.0

#### Safety Reliability
- **Safety Level**: Current safety assurance level
- **Confidence**: System confidence in decisions
- **Score Range**: 0.0 - 1.0

#### Performance
- **Memory Usage**: System memory utilization
- **Processing Latency**: Response time performance
- **Bottlenecks**: Number of performance bottlenecks
- **Score Range**: 0.0 - 1.0

#### Stability
- **Consistency**: System behavior consistency
- **Predictability**: System response predictability
- **Score Range**: 0.0 - 1.0

## Status Classification

| Health Score | Status | Description | Action Required |
|--------------|--------|-------------|-----------------|
| 80-100 | Healthy | System operating optimally | Monitor and maintain |
| 60-79 | Degraded | System performance below optimal | Investigate and improve |
| 0-59 | Critical | System health requires attention | Immediate action required |

## Alert System

### Alert Types

#### Critical Alerts
- Health score below 50
- Safety reliability below 0.8
- System failures or errors

#### Warning Alerts
- Health score below 70
- Learning effectiveness below 0.6
- Performance degradation

### Alert Handling

```python
# Check for alerts
alerts = validator.check_alerts(assessment)

# Handle alerts
for alert in alerts:
    if alert['level'] == 'critical':
        # Send immediate notification
        send_critical_alert(alert['message'])
    elif alert['level'] == 'warning':
        # Log warning for monitoring
        log_warning(alert['message'])
```

## Testing

### Run Test Suite

```bash
cd scripts
python test_health_validator.py
```

### Test Scenarios

The test suite covers:

1. **Healthy System** - Optimal performance metrics
2. **Degraded System** - Below-optimal performance
3. **Critical System** - Poor performance requiring attention
4. **Edge Cases** - Boundary conditions and extreme values
5. **Continuous Validation** - Long-running monitoring
6. **Configuration Testing** - Configuration loading and validation
7. **Recommendation Testing** - Recommendation generation for different scenarios

### Test Output Example

```
============================================================
TEST 1: Healthy System
============================================================
Input Metrics:
  Learning Rounds: 500
  Pattern Discoveries: 45
  Rule Evolutions: 12
  Adaptation Count: 85
  Safety Level: 0.92
  Confidence: 0.88
  Memory Usage: 0.65
  Processing Latency: 35.0ms

Validation Report:
{
  "health_score": 87.2,
  "status": "healthy",
  "learning_effectiveness": {
    "score": 0.85,
    "assessment": "System is learning effectively with good pattern recognition",
    "concerns": []
  },
  "safety_reliability": {
    "score": 0.90,
    "assessment": "High safety assurance with robust rule evolution",
    "concerns": []
  },
  "performance_analysis": {
    "bottlenecks": [],
    "optimization_opportunities": []
  },
  "recommendations": [
    "Continue monitoring system performance",
    "Schedule periodic safety rule review"
  ],
  "risk_level": "low"
}

No alerts generated - system is healthy!
```

## Integration Examples

### ROS 2 Integration

```python
import rclpy
from rclpy.node import Node
from adaptive_safety_health_validator import SystemHealthValidator

class HealthMonitoringNode(Node):
    def __init__(self):
        super().__init__('health_monitoring_node')
        self.validator = SystemHealthValidator()
        self.timer = self.create_timer(60.0, self.health_check)
    
    def health_check(self):
        # Collect metrics from ROS 2 topics
        metrics = self.collect_ros_metrics()
        
        # Perform validation
        assessment = self.validator.validate_health(metrics)
        
        # Publish health status
        self.publish_health_status(assessment)
```

### Docker Integration

```dockerfile
# Add health validator to Docker image
COPY scripts/adaptive_safety_health_validator.py /opt/health_validator/
COPY scripts/health_validator_config.json /opt/health_validator/

# Run health monitoring
CMD ["python", "/opt/health_validator/adaptive_safety_health_validator.py", "--continuous"]
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Health Validation
  run: |
    cd scripts
    python adaptive_safety_health_validator.py --config health_validator_config.json
    python test_health_validator.py
```

## Troubleshooting

### Common Issues

#### Import Errors
```
Warning: Could not import adaptive safety modules. Running in standalone mode.
```
**Solution**: This is normal when running in standalone mode. The validator will use mock data.

#### Configuration Errors
```
Failed to load config file: [Errno 2] No such file or directory
```
**Solution**: Ensure the configuration file exists or use default configuration.

#### Performance Issues
```
High processing latency detected
```
**Solution**: Check system resources and optimize processing pipeline.

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### Log Files

Logs are written to `adaptive_safety_health.log` with rotation support.

## Contributing

### Adding New Metrics

1. Update `SystemMetrics` dataclass
2. Modify `collect_system_metrics()` method
3. Update scoring algorithms
4. Add tests for new metrics

### Adding New Validation Rules

1. Create new validation method
2. Update `validate_health()` method
3. Add configuration parameters
4. Update test suite

### Code Style

- Follow PEP 8 guidelines
- Add type hints
- Include docstrings
- Write comprehensive tests

## License

This implementation is part of the Embodied Intelligence Platform and follows the same licensing terms.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review test outputs
3. Examine log files
4. Create an issue in the repository

## Version History

- **v1.0.0** - Initial implementation of System Health Validation Prompt
- **v1.1.0** - Added continuous monitoring and alert system
- **v1.2.0** - Enhanced configuration management and testing 