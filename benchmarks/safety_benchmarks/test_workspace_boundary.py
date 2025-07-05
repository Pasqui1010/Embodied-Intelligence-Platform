#!/usr/bin/env python3
"""
Workspace Boundary Safety Tests

Tests to ensure workspace boundary detection and enforcement work reliably.
"""

import pytest
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from eip_interfaces.msg import SafetyVerificationResponse
import threading
import time


class WorkspaceBoundaryTester(Node):
    """Test node for workspace boundary functionality"""
    
    def __init__(self):
        super().__init__('workspace_boundary_tester')
        
        self.safety_violation_received = False
        self.cmd_vel_filtered = False
        self.latest_safety_response = None
        self.workspace_violations = []
        
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
        if not msg.is_safe and 'workspace_boundary' in msg.violations:
            self.safety_violation_received = True
            self.workspace_violations.append({
                'explanation': msg.explanation,
                'timestamp': time.time()
            })
        self.get_logger().info(f"Safety status: {msg.explanation}")

    def cmd_vel_safe_callback(self, msg):
        """Check if velocity commands are filtered due to workspace boundary"""
        # Check if velocity was reduced due to workspace boundary violation
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


def test_circular_workspace_boundary_violation(ros_context):
    """Test that circular workspace boundary violation is detected"""
    
    tester = WorkspaceBoundaryTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose outside circular workspace (6m from center > 5m radius)
    tester.publish_robot_pose(x=6.0, y=0.0)
    
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
    
    # Verify workspace boundary violation was detected
    assert tester.safety_violation_received, "Workspace boundary violation was not detected"
    assert tester.latest_safety_response is not None, "No safety response received"
    assert len(tester.workspace_violations) > 0, "No workspace boundary violations recorded"
    
    tester.destroy_node()


def test_circular_workspace_safe_operation(ros_context):
    """Test that normal operation continues when robot is within circular workspace"""
    
    tester = WorkspaceBoundaryTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose within circular workspace (3m from center < 5m radius)
    tester.publish_robot_pose(x=3.0, y=0.0)
    
    # Publish normal velocity command
    normal_cmd = Twist()
    normal_cmd.linear.x = 0.5
    tester.cmd_vel_pub.publish(normal_cmd)
    
    # Wait for safety evaluation
    time.sleep(2.0)
    
    # Verify no workspace boundary violation was detected
    assert not tester.safety_violation_received, "Workspace boundary violation detected when robot is within bounds"
    if tester.latest_safety_response:
        assert tester.latest_safety_response.is_safe, "Safety system reported unsafe when should be safe"
    
    tester.destroy_node()


def test_rectangular_workspace_boundary_violation(ros_context):
    """Test that rectangular workspace boundary violation is detected"""
    
    tester = WorkspaceBoundaryTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose outside rectangular workspace (3m > 2.5m max_x)
    tester.publish_robot_pose(x=3.0, y=0.0)
    
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
    
    # Verify workspace boundary violation was detected
    assert tester.safety_violation_received, "Rectangular workspace boundary violation was not detected"
    
    tester.destroy_node()


def test_rectangular_workspace_safe_operation(ros_context):
    """Test that normal operation continues when robot is within rectangular workspace"""
    
    tester = WorkspaceBoundaryTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose within rectangular workspace (1m, well within -2.5 to 2.5 bounds)
    tester.publish_robot_pose(x=1.0, y=1.0)
    
    # Publish normal velocity command
    normal_cmd = Twist()
    normal_cmd.linear.x = 0.5
    tester.cmd_vel_pub.publish(normal_cmd)
    
    # Wait for safety evaluation
    time.sleep(2.0)
    
    # Verify no workspace boundary violation was detected
    assert not tester.safety_violation_received, "Rectangular workspace boundary violation detected when robot is within bounds"
    if tester.latest_safety_response:
        assert tester.latest_safety_response.is_safe, "Safety system reported unsafe when should be safe"
    
    tester.destroy_node()


def test_workspace_boundary_response_time(ros_context):
    """Test that workspace boundary detection responds within acceptable time"""
    
    tester = WorkspaceBoundaryTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose outside workspace
    tester.publish_robot_pose(x=6.0, y=0.0)
    
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
    
    # Verify response time is acceptable (<500ms for workspace boundary)
    assert response_time < 0.5, f"Workspace boundary detection took {response_time:.3f}s (max: 0.5s)"
    assert tester.safety_violation_received, "Workspace boundary violation was not detected"
    
    tester.destroy_node()


def test_workspace_boundary_violation_details(ros_context):
    """Test that workspace boundary violations include detailed information"""
    
    tester = WorkspaceBoundaryTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Publish robot pose outside workspace
    tester.publish_robot_pose(x=6.0, y=0.0)
    
    # Publish velocity command
    cmd = Twist()
    cmd.linear.x = 0.5
    tester.cmd_vel_pub.publish(cmd)
    
    # Wait for safety evaluation
    time.sleep(2.0)
    
    # Verify violation details are provided
    assert len(tester.workspace_violations) > 0, "No workspace boundary violations recorded"
    
    violation = tester.workspace_violations[0]
    assert 'explanation' in violation, "Violation missing explanation"
    assert 'workspace' in violation['explanation'].lower(), "Explanation should mention workspace"
    assert 'outside' in violation['explanation'].lower(), "Explanation should indicate robot is outside"
    
    tester.destroy_node()


def test_multiple_workspace_boundary_scenarios(ros_context):
    """Test multiple workspace boundary scenarios"""
    
    tester = WorkspaceBoundaryTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Test different positions relative to workspace boundaries
    test_scenarios = [
        (6.0, 0.0, True),   # Outside circular workspace (should violate)
        (3.0, 0.0, False),  # Inside circular workspace (should be safe)
        (3.0, 0.0, True),   # Outside rectangular workspace (should violate)
        (1.0, 1.0, False),  # Inside rectangular workspace (should be safe)
    ]
    
    for x, y, should_violate in test_scenarios:
        # Reset violation state
        tester.safety_violation_received = False
        tester.workspace_violations.clear()
        
        # Publish robot pose
        tester.publish_robot_pose(x, y)
        
        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = 0.5
        tester.cmd_vel_pub.publish(cmd)
        
        # Wait for safety evaluation
        time.sleep(1.0)
        
        if should_violate:
            assert tester.safety_violation_received, f"Expected violation at position ({x}, {y})"
        else:
            assert not tester.safety_violation_received, f"Unexpected violation at position ({x}, {y})"
    
    tester.destroy_node()


def test_workspace_center_offset(ros_context):
    """Test that workspace center offset is properly handled"""
    
    tester = WorkspaceBoundaryTester()
    
    def spin_node():
        rclpy.spin(tester)
    
    thread = threading.Thread(target=spin_node, daemon=True)
    thread.start()
    
    # Test position relative to workspace center (0,0) with 5m radius
    # Position at (5.1, 0) should be just outside the boundary
    tester.publish_robot_pose(x=5.1, y=0.0)
    
    # Publish velocity command
    cmd = Twist()
    cmd.linear.x = 0.5
    tester.cmd_vel_pub.publish(cmd)
    
    # Wait for safety evaluation
    time.sleep(2.0)
    
    # Verify boundary violation was detected
    assert tester.safety_violation_received, "Workspace boundary violation not detected at edge case"
    
    tester.destroy_node()


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 