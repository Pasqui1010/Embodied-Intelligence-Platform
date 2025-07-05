#!/usr/bin/env python3
"""
Human Proximity Safety Tests

Tests to ensure human proximity detection and safety enforcement work reliably.
"""

import pytest
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from eip_interfaces.msg import SafetyVerificationResponse
import threading
import time


class HumanProximityTester(Node):
    """Test node for human proximity functionality"""
    
    def __init__(self):
        super().__init__('human_proximity_tester')
        
        self.safety_violation_received = False
        self.cmd_vel_filtered = False
        self.latest_safety_response = None
        self.human_proximity_violations = []
        
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
        if not msg.is_safe and 'human_proximity' in msg.violations:
            self.safety_violation_received = True
            self.human_proximity_violations.append({
                'explanation': msg.explanation,
                'timestamp': time.time()
            })
        self.get_logger().info(f"Safety status: {msg.explanation}")

    def cmd_vel_safe_callback(self, msg):
        """Check if velocity commands are filtered due to human proximity"""
        # Check if velocity was reduced due to human proximity
        if abs(msg.linear.x) < 0.1:  # Significantly reduced
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


def test_human_proximity_detection_within_threshold(ros_context):
    """Test that human proximity violation is detected when robot is close to simulated human"""
    
    tester = HumanProximityTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose close to simulated human (0.8m to the left - within 1.0m threshold)
    tester.publish_robot_pose(x=0.0, y=-0.8)
    
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
    
    # Verify human proximity violation was detected
    assert tester.safety_violation_received, "Human proximity violation was not detected"
    assert tester.latest_safety_response is not None, "No safety response received"
    assert len(tester.human_proximity_violations) > 0, "No human proximity violations recorded"
    
    tester.destroy_node()


def test_human_proximity_detection_outside_threshold(ros_context):
    """Test that no violation is detected when robot is far from simulated humans"""
    
    tester = HumanProximityTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose far from simulated humans (3m away - outside 1.0m threshold)
    tester.publish_robot_pose(x=3.0, y=3.0)
    
    # Publish normal velocity command
    normal_cmd = Twist()
    normal_cmd.linear.x = 0.5
    tester.cmd_vel_pub.publish(normal_cmd)
    
    # Wait for safety evaluation
    time.sleep(2.0)
    
    # Verify no human proximity violation was detected
    assert not tester.safety_violation_received, "Human proximity violation detected when robot is far away"
    if tester.latest_safety_response:
        assert tester.latest_safety_response.is_safe, "Safety system reported unsafe when should be safe"
    
    tester.destroy_node()


def test_multiple_human_detection(ros_context):
    """Test that multiple simulated humans are detected correctly"""
    
    tester = HumanProximityTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Test different positions near simulated humans
    test_positions = [
        (2.0, 0.0),   # Close to human 1 (2m in front)
        (-1.5, 1.0),  # Close to human 2 (1.5m behind and 1m right)
        (0.0, -0.8),  # Close to human 3 (0.8m to left)
    ]
    
    violations_detected = 0
    
    for x, y in test_positions:
        # Reset violation state
        tester.safety_violation_received = False
        tester.human_proximity_violations.clear()
        
        # Publish robot pose
        tester.publish_robot_pose(x, y)
        
        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = 0.5
        tester.cmd_vel_pub.publish(cmd)
        
        # Wait for safety evaluation
        time.sleep(1.0)
        
        if tester.safety_violation_received:
            violations_detected += 1
    
    # Verify that violations were detected for close positions
    assert violations_detected >= 2, f"Expected at least 2 violations, got {violations_detected}"
    
    tester.destroy_node()


def test_human_proximity_response_time(ros_context):
    """Test that human proximity detection responds within acceptable time"""
    
    tester = HumanProximityTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose close to simulated human
    tester.publish_robot_pose(x=0.0, y=-0.8)
    
    # Record start time
    start_time = time.time()
    
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
    
    # Verify response time is acceptable (<500ms for human proximity)
    assert response_time < 0.5, f"Human proximity detection took {response_time:.3f}s (max: 0.5s)"
    assert tester.safety_violation_received, "Human proximity violation was not detected"
    
    tester.destroy_node()


def test_human_proximity_violation_details(ros_context):
    """Test that human proximity violations include detailed information"""
    
    tester = HumanProximityTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose close to simulated human
    tester.publish_robot_pose(x=0.0, y=-0.8)
    
    # Publish velocity command
    cmd = Twist()
    cmd.linear.x = 0.5
    tester.cmd_vel_pub.publish(cmd)
    
    # Wait for safety evaluation
    time.sleep(2.0)
    
    # Verify violation details are provided
    assert len(tester.human_proximity_violations) > 0, "No human proximity violations recorded"
    
    violation = tester.human_proximity_violations[0]
    assert 'explanation' in violation, "Violation missing explanation"
    assert 'Human' in violation['explanation'], "Explanation should mention human detection"
    assert 'distance' in violation['explanation'].lower(), "Explanation should include distance information"
    
    tester.destroy_node()


def test_safe_operation_with_distant_humans(ros_context):
    """Test that normal operation continues when humans are at safe distance"""
    
    tester = HumanProximityTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose at safe distance from all simulated humans
    tester.publish_robot_pose(x=5.0, y=5.0)
    
    # Publish normal velocity command
    normal_cmd = Twist()
    normal_cmd.linear.x = 0.5
    tester.cmd_vel_pub.publish(normal_cmd)
    
    # Wait for safety evaluation
    time.sleep(2.0)
    
    # Verify no safety violation was detected
    assert not tester.safety_violation_received, "Safety violation detected when humans are at safe distance"
    if tester.latest_safety_response:
        assert tester.latest_safety_response.is_safe, "Safety system reported unsafe when should be safe"
    
    tester.destroy_node()


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 