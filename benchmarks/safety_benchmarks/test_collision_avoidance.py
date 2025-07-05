#!/usr/bin/env python3
"""
Collision Avoidance Safety Tests

Tests to ensure collision detection and avoidance work reliably.
"""

import pytest
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from eip_interfaces.msg import SafetyVerificationResponse
import threading
import time
import numpy as np


class CollisionAvoidanceTester(Node):
    """Test node for collision avoidance functionality"""
    
    def __init__(self):
        super().__init__('collision_avoidance_tester')
        
        self.safety_violation_received = False
        self.cmd_vel_filtered = False
        self.latest_safety_response = None
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)
        self.laser_pub = self.create_publisher(LaserScan, '/scan', 10)
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
        if not msg.is_safe and 'collision_risk' in msg.violations:
            self.safety_violation_received = True
        self.get_logger().info(f"Safety status: {msg.explanation}")

    def cmd_vel_safe_callback(self, msg):
        """Check if velocity commands are filtered"""
        # Check if velocity was reduced due to collision risk
        if abs(msg.linear.x) < 0.1:  # Significantly reduced
            self.cmd_vel_filtered = True

    def publish_laser_scan(self, distances):
        """Publish a laser scan with specified distances"""
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = 'base_link'
        scan.angle_min = -np.pi
        scan.angle_max = np.pi
        scan.angle_increment = 2 * np.pi / len(distances)
        scan.range_min = 0.1
        scan.range_max = 10.0
        scan.ranges = distances
        
        self.laser_pub.publish(scan)

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


def test_collision_detection_with_close_obstacle(ros_context):
    """Test that collision risk is detected when obstacle is too close"""
    
    tester = CollisionAvoidanceTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose
    tester.publish_robot_pose()
    
    # Publish laser scan with close obstacle (0.3m - below threshold)
    close_distances = [0.3] * 360  # 360 laser readings
    tester.publish_laser_scan(close_distances)
    
    # Publish normal velocity command
    normal_cmd = Twist()
    normal_cmd.linear.x = 0.5
    tester.cmd_vel_pub.publish(normal_cmd)
    
    # Wait for safety system to respond
    timeout = 3.0
    start_time = time.time()
    
    while (time.time() - start_time) < timeout:
        if tester.safety_violation_received:
            break
        time.sleep(0.1)
    
    # Verify collision risk was detected
    assert tester.safety_violation_received, "Collision risk was not detected"
    assert tester.latest_safety_response is not None, "No safety response received"
    
    tester.destroy_node()


def test_velocity_filtering_on_collision_risk(ros_context):
    """Test that velocity is filtered when collision risk is detected"""
    
    tester = CollisionAvoidanceTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose
    tester.publish_robot_pose()
    
    # Publish laser scan with close obstacle
    close_distances = [0.4] * 360
    tester.publish_laser_scan(close_distances)
    
    # Publish high velocity command
    high_vel_cmd = Twist()
    high_vel_cmd.linear.x = 1.0
    tester.cmd_vel_pub.publish(high_vel_cmd)
    
    # Wait for velocity filtering
    timeout = 3.0
    start_time = time.time()
    
    while (time.time() - start_time) < timeout:
        if tester.cmd_vel_filtered:
            break
        time.sleep(0.1)
    
    # Verify velocity was filtered
    assert tester.cmd_vel_filtered, "Velocity was not filtered for collision risk"
    
    tester.destroy_node()


def test_safe_operation_with_distant_obstacles(ros_context):
    """Test that normal operation continues when obstacles are far away"""
    
    tester = CollisionAvoidanceTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose
    tester.publish_robot_pose()
    
    # Publish laser scan with distant obstacles (2m - safe distance)
    safe_distances = [2.0] * 360
    tester.publish_laser_scan(safe_distances)
    
    # Publish normal velocity command
    normal_cmd = Twist()
    normal_cmd.linear.x = 0.5
    tester.cmd_vel_pub.publish(normal_cmd)
    
    # Wait for safety evaluation
    time.sleep(1.0)
    
    # Verify no collision risk was detected
    assert not tester.safety_violation_received, "Collision risk detected when obstacles are safe distance"
    if tester.latest_safety_response:
        assert tester.latest_safety_response.is_safe, "Safety system reported unsafe when should be safe"
    
    tester.destroy_node()


def test_collision_detection_response_time(ros_context):
    """Test that collision detection responds within acceptable time"""
    
    tester = CollisionAvoidanceTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose
    tester.publish_robot_pose()
    
    # Record start time
    start_time = time.time()
    
    # Publish laser scan with close obstacle
    close_distances = [0.3] * 360
    tester.publish_laser_scan(close_distances)
    
    # Publish velocity command
    cmd = Twist()
    cmd.linear.x = 0.5
    tester.cmd_vel_pub.publish(cmd)
    
    # Wait for response
    while not tester.safety_violation_received:
        if (time.time() - start_time) > 1.0:  # 1 second timeout
            break
        time.sleep(0.001)
    
    response_time = time.time() - start_time
    
    # Verify response time is acceptable (<500ms for collision detection)
    assert response_time < 0.5, f"Collision detection took {response_time:.3f}s (max: 0.5s)"
    assert tester.safety_violation_received, "Collision risk was not detected"
    
    tester.destroy_node()


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 