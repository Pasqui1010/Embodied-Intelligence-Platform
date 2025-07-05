#!/usr/bin/env python3
"""
Velocity Limits Safety Tests

Tests to ensure velocity limits are properly enforced.
"""

import pytest
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from eip_interfaces.msg import SafetyVerificationResponse
import threading
import time


class VelocityLimitsTester(Node):
    """Test node for velocity limits functionality"""
    
    def __init__(self):
        super().__init__('velocity_limits_tester')
        
        self.safety_violation_received = False
        self.cmd_vel_filtered = False
        self.latest_safety_response = None
        self.received_velocities = []
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/robot_pose', 10)
        
        # Subscribers
        self.safety_status_sub = self.create_subscription(
            SafetyVerificationResponse,
            '/safety/status',
            self.safety_status_callback,
            10
        )
        
        self.cmd_vel_safe_sub = self.create_subscription(
            Twist,
            '/cmd_vel_safe',
            self.cmd_vel_safe_callback,
            10
        )

    def safety_status_callback(self, msg):
        """Record safety status updates"""
        self.latest_safety_response = msg
        if not msg.is_safe and 'invalid_manipulation' in msg.violations:
            self.safety_violation_received = True
        self.get_logger().info(f"Safety status: {msg.explanation}")

    def cmd_vel_safe_callback(self, msg):
        """Record filtered velocity commands"""
        self.received_velocities.append({
            'linear_x': msg.linear.x,
            'angular_z': msg.angular.z,
            'timestamp': time.time()
        })
        
        # Check if velocity was filtered (reduced from original)
        if len(self.received_velocities) > 1:
            prev_vel = self.received_velocities[-2]
            if (abs(msg.linear.x) < abs(prev_vel['linear_x']) or 
                abs(msg.angular.z) < abs(prev_vel['angular_z'])):
                self.cmd_vel_filtered = True

    def publish_robot_pose(self, x=0.0, y=0.0):
        """Publish robot pose"""
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        
        self.pose_pub.publish(pose)


@pytest.fixture
def ros_context():
    """Setup ROS 2 context for testing"""
    rclpy.init()
    yield
    rclpy.shutdown()


def test_linear_velocity_limit_enforcement(ros_context):
    """Test that linear velocity limits are enforced"""
    
    tester = VelocityLimitsTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose
    tester.publish_robot_pose()
    
    # Publish velocity command exceeding linear limit (1.5 m/s > 1.0 m/s default)
    high_vel_cmd = Twist()
    high_vel_cmd.linear.x = 1.5
    tester.cmd_vel_pub.publish(high_vel_cmd)
    
    # Wait for safety system to respond
    timeout = 3.0
    start_time = time.time()
    
    while (time.time() - start_time) < timeout:
        if tester.safety_violation_received:
            break
        time.sleep(0.1)
    
    # Verify velocity limit violation was detected
    assert tester.safety_violation_received, "Linear velocity limit violation was not detected"
    assert tester.latest_safety_response is not None, "No safety response received"
    
    tester.destroy_node()


def test_angular_velocity_limit_enforcement(ros_context):
    """Test that angular velocity limits are enforced"""
    
    tester = VelocityLimitsTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose
    tester.publish_robot_pose()
    
    # Publish velocity command exceeding angular limit (1.5 rad/s > 1.0 rad/s default)
    high_vel_cmd = Twist()
    high_vel_cmd.angular.z = 1.5
    tester.cmd_vel_pub.publish(high_vel_cmd)
    
    # Wait for safety system to respond
    timeout = 3.0
    start_time = time.time()
    
    while (time.time() - start_time) < timeout:
        if tester.safety_violation_received:
            break
        time.sleep(0.1)
    
    # Verify velocity limit violation was detected
    assert tester.safety_violation_received, "Angular velocity limit violation was not detected"
    
    tester.destroy_node()


def test_velocity_filtering_effectiveness(ros_context):
    """Test that velocity commands are properly filtered to stay within limits"""
    
    tester = VelocityLimitsTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose
    tester.publish_robot_pose()
    
    # Publish multiple velocity commands
    velocities = [
        (2.0, 0.0),   # High linear velocity
        (0.0, 2.0),   # High angular velocity
        (1.5, 1.5),   # Both high
    ]
    
    for linear_vel, angular_vel in velocities:
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        tester.cmd_vel_pub.publish(cmd)
        time.sleep(0.5)
    
    # Wait for filtering to occur
    time.sleep(2.0)
    
    # Verify that velocities were filtered
    assert len(tester.received_velocities) > 0, "No velocity commands received"
    
    # Check that filtered velocities are within reasonable limits
    for vel in tester.received_velocities:
        assert abs(vel['linear_x']) <= 1.1, f"Linear velocity {vel['linear_x']} exceeds limit"
        assert abs(vel['angular_z']) <= 1.1, f"Angular velocity {vel['angular_z']} exceeds limit"
    
    tester.destroy_node()


def test_safe_velocity_commands_pass_through(ros_context):
    """Test that safe velocity commands pass through without filtering"""
    
    tester = VelocityLimitsTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose
    tester.publish_robot_pose()
    
    # Publish safe velocity command (within limits)
    safe_cmd = Twist()
    safe_cmd.linear.x = 0.5  # Well within 1.0 m/s limit
    safe_cmd.angular.z = 0.5  # Well within 1.0 rad/s limit
    tester.cmd_vel_pub.publish(safe_cmd)
    
    # Wait for processing
    time.sleep(1.0)
    
    # Verify no safety violation was detected
    assert not tester.safety_violation_received, "Safety violation detected for safe velocity"
    if tester.latest_safety_response:
        assert tester.latest_safety_response.is_safe, "Safety system reported unsafe for safe velocity"
    
    # Verify velocity command was passed through
    assert len(tester.received_velocities) > 0, "No velocity command received"
    last_vel = tester.received_velocities[-1]
    assert abs(last_vel['linear_x'] - 0.5) < 0.1, "Safe linear velocity was modified"
    assert abs(last_vel['angular_z'] - 0.5) < 0.1, "Safe angular velocity was modified"
    
    tester.destroy_node()


def test_velocity_limit_response_time(ros_context):
    """Test that velocity limit enforcement responds quickly"""
    
    tester = VelocityLimitsTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose
    tester.publish_robot_pose()
    
    # Record start time
    start_time = time.time()
    
    # Publish high velocity command
    high_vel_cmd = Twist()
    high_vel_cmd.linear.x = 2.0
    tester.cmd_vel_pub.publish(high_vel_cmd)
    
    # Wait for response
    while not tester.safety_violation_received:
        if (time.time() - start_time) > 1.0:  # 1 second timeout
            break
        time.sleep(0.001)
    
    response_time = time.time() - start_time
    
    # Verify response time is acceptable (<200ms for velocity limits)
    assert response_time < 0.2, f"Velocity limit enforcement took {response_time:.3f}s (max: 0.2s)"
    assert tester.safety_violation_received, "Velocity limit violation was not detected"
    
    tester.destroy_node()


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 