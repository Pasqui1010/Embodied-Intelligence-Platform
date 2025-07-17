# Safety Monitor Test Suite

This directory contains test scripts for the EIP Safety Monitor, specifically focusing on the LLM-based safety evaluation functionality.

## Test Script: `test_llm_safety_evaluation.py`

This script demonstrates the LLM-based safety evaluation by testing different task plan scenarios:

1. **Safe Plan**: A simple navigation task to a charging station
2. **Risky Plan**: A task involving moving near a human
3. **Dangerous Plan**: A task that enters a restricted area

### Prerequisites

- ROS 2 Humble (or newer)
- Python 3.8+
- Required Python packages (install via `pip` or `apt`):
  - rclpy
  - std_msgs
  - geometry_msgs
  - sensor_msgs
  - nav_msgs
  - eip_interfaces

### Running the Tests

#### Method 1: Using the Launch File (Recommended)

```bash
# In your ROS 2 workspace
source install/setup.bash
ros2 launch eip_safety_arbiter test_llm_safety.launch.py
```

#### Method 2: Manual Execution

1. Start the safety monitor in one terminal:
   ```bash
   ros2 run eip_safety_arbiter safety_monitor.py
   ```

2. In another terminal, run the test script:
   ```bash
   ros2 run eip_safety_arbiter test_llm_safety_evaluation.py
   ```

### Expected Output

The test script will output the safety evaluation results for each test scenario, including:
- Whether the task plan is considered safe
- The safety level (SAFE, LOW_RISK, MEDIUM_RISK, HIGH_RISK, CRITICAL)
- Any detected safety violations
- Suggested modifications to make the plan safer
- A detailed explanation of the safety evaluation

### Adding New Test Cases

To add a new test case, edit the `create_test_plan` method in `test_llm_safety_evaluation.py` to include your new scenario. Follow the existing examples for the format.

### Troubleshooting

- **Safety monitor not found**: Ensure you've built and sourced your workspace
- **Import errors**: Check that all dependencies are installed
- **LLM not loading**: Verify that the Mistral 7B model is properly installed and accessible

## License

This test suite is part of the Embodied Intelligence Platform and is licensed under the Apache 2.0 License.
