#!/usr/bin/env python3
"""
Emergency Stop Safety Tests

Critical tests to ensure emergency stop functionality works reliably.
These tests MUST pass before any robot deployment.
"""

import pytest
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from eip_interfaces.msg import EmergencyStop
import threading
import time


class EmergencyStopTester(Node):
    """Test node for emergency stop functionality"""
    
    def __init__(self):
        super().__init__('emergency_stop_tester')
        
        self.emergency_received = False
        self.cmd_vel_stopped = False
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)
        
        # Subscribers
        self.emergency_sub = self.create_subscription(
            EmergencyStop,
            '/safety/emergency_stop',
            self.emergency_callback,
            10
        )
        
        self.cmd_vel_safe_sub = self.create_subscription(
            Twist,
            '/cmd_vel_safe',
            self.cmd_vel_safe_callback,
            10
        )

    def emergency_callback(self, msg):
        """Record emergency stop reception"""
        self.emergency_received = True
        self.get_logger().info(f"Emergency stop received: {msg.reason}")

    def cmd_vel_safe_callback(self, msg):
        """Check if velocity commands are zeroed"""
        if (abs(msg.linear.x) < 0.001 and 
            abs(msg.linear.y) < 0.001 and 
            abs(msg.angular.z) < 0.001):
            self.cmd_vel_stopped = True


@pytest.fixture
def ros_context():
    """Setup ROS 2 context for testing"""
    rclpy.init()
    yield
    rclpy.shutdown()


def test_emergency_stop_triggers_on_critical_violation(ros_context):
    """Test that emergency stop triggers on critical safety violations"""
    
    tester = EmergencyStopTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    # Start ROS node in background
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Simulate dangerous velocity command
    dangerous_cmd = Twist()
    dangerous_cmd.linear.x = 10.0  # Way above safety limits
    
    # Publish dangerous command
    tester.cmd_vel_pub.publish(dangerous_cmd)
    
    # Wait for safety system to respond
    timeout = 5.0  # seconds
    start_time = time.time()
    
    while (time.time() - start_time) < timeout:
        if tester.emergency_received and tester.cmd_vel_stopped:
            break
        time.sleep(0.1)
    
    # Verify emergency stop was triggered
    assert tester.emergency_received, "Emergency stop was not triggered"
    assert tester.cmd_vel_stopped, "Robot velocity was not stopped"
    
    tester.destroy_node()


def test_emergency_stop_response_time(ros_context):
    """Test that emergency stop responds within 100ms"""
    
    tester = EmergencyStopTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Record start time
    start_time = time.time()
    
    # Trigger emergency condition
    dangerous_cmd = Twist()
    dangerous_cmd.linear.x = 5.0
    dangerous_cmd.angular.z = 5.0
    
    tester.cmd_vel_pub.publish(dangerous_cmd)
    
    # Wait for response
    while not tester.emergency_received:
        if (time.time() - start_time) > 1.0:  # 1 second timeout
            break
        time.sleep(0.001)
    
    response_time = time.time() - start_time
    
    # Critical requirement: <100ms response time
    assert response_time < 0.1, f"Emergency stop took {response_time:.3f}s (max: 0.1s)"
    assert tester.emergency_received, "Emergency stop was not triggered"
    
    tester.destroy_node()


def test_emergency_stop_persistence(ros_context):
    """Test that emergency stop persists until explicitly reset"""
    
    tester = EmergencyStopTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Trigger emergency stop
    dangerous_cmd = Twist()
    dangerous_cmd.linear.x = 10.0
    tester.cmd_vel_pub.publish(dangerous_cmd)
    
    # Wait for emergency stop
    time.sleep(0.5)
    assert tester.emergency_received, "Emergency stop not triggered"
    
    # Try to send normal command
    normal_cmd = Twist()
    normal_cmd.linear.x = 0.5
    tester.cmd_vel_pub.publish(normal_cmd)
    
    # Verify robot remains stopped
    time.sleep(0.5)
    assert tester.cmd_vel_stopped, "Robot should remain stopped after emergency"
    
    tester.destroy_node()


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 