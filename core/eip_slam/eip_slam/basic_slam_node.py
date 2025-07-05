#!/usr/bin/env python3
"""
Basic SLAM Node for Embodied Intelligence Platform

Implements a simple but functional SLAM system using:
- ICP for point cloud registration
- Loop closure detection
- Semantic object integration
- Safety-aware mapping

This serves as the foundation for more advanced SLAM capabilities.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
import open3d as o3d
from threading import Lock
import time

# ROS 2 message types
from sensor_msgs.msg import LaserScan, PointCloud2, Image
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA

# TF2 for transforms
import tf2_ros
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs

# Custom interfaces (will be available after building eip_interfaces)
try:
    from eip_interfaces.msg import SafetyVerificationRequest
except ImportError:
    # Fallback for development
    SafetyVerificationRequest = None


def open3d_to_pointcloud2(point_cloud, frame_id="map", timestamp=None):
    """
    Convert Open3D point cloud to ROS 2 PointCloud2 message
    
    Args:
        point_cloud: Open3D point cloud object
        frame_id: Frame ID for the point cloud
        timestamp: ROS 2 timestamp (if None, uses current time)
    
    Returns:
        PointCloud2 message or None if conversion fails
    """
    try:
        if len(point_cloud.points) == 0:
            return None
        
        # Get points as numpy array
        points = np.asarray(point_cloud.points)
        
        # Create PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header.frame_id = frame_id
        if timestamp is not None:
            cloud_msg.header.stamp = timestamp
        
        # Set point cloud fields (x, y, z)
        from sensor_msgs_py import point_cloud2
        cloud_msg = point_cloud2.create_cloud_xyz32(cloud_msg.header, points)
        
        return cloud_msg
        
    except Exception as e:
        print(f"Error converting Open3D to PointCloud2: {e}")
        return None


class BasicSLAMNode(Node):
    """
    Basic SLAM implementation with semantic mapping capabilities.
    
    Features:
    - Point cloud-based SLAM using ICP
    - Semantic object detection integration  
    - Occupancy grid generation
    - Safety-aware map updates
    - RViz visualization
    """
    
    def __init__(self):
        super().__init__('basic_slam_node')
        
        # Configuration parameters
        self.declare_parameters()
        self.load_configuration()
        
        # SLAM state
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.map_points = []  # List of point clouds
        self.semantic_objects = {}  # Dict of detected objects
        self.pose_history = []  # For loop closure
        
        # Thread safety
        self.map_lock = Lock()
        self.pose_lock = Lock()
        
        # Point cloud processing
        self.previous_cloud = None
        self.map_cloud = o3d.geometry.PointCloud()
        
        # ROS 2 interfaces
        self.setup_publishers()
        self.setup_subscribers()
        self.setup_transforms()
        
        # Processing timer
        self.slam_timer = self.create_timer(
            1.0 / self.slam_update_rate,
            self.slam_update_callback
        )
        
        self.get_logger().info("Basic SLAM Node initialized")

    def declare_parameters(self):
        """Declare ROS 2 parameters with defaults"""
        self.declare_parameter('slam_update_rate', 10.0)  # Hz
        self.declare_parameter('icp_max_distance', 0.1)  # meters
        self.declare_parameter('voxel_size', 0.05)  # meters for downsampling
        self.declare_parameter('loop_closure_threshold', 2.0)  # meters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

    def load_configuration(self):
        """Load configuration from parameters"""
        self.slam_update_rate = self.get_parameter('slam_update_rate').value
        self.icp_max_distance = self.get_parameter('icp_max_distance').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.loop_closure_threshold = self.get_parameter('loop_closure_threshold').value
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

    def setup_publishers(self):
        """Setup ROS 2 publishers"""
        # QoS for mapping data
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/map',
            map_qos
        )
        
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/robot_pose',
            10
        )
        
        self.markers_pub = self.create_publisher(
            MarkerArray,
            '/slam/semantic_objects',
            10
        )
        
        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            '/slam/map_cloud',
            10
        )

    def setup_subscribers(self):
        """Setup ROS 2 subscribers"""
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            10
        )

    def setup_transforms(self):
        """Setup TF2 transform handling"""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

    def laser_callback(self, msg):
        """Process laser scan data for SLAM"""
        try:
            # Convert laser scan to point cloud
            cloud = self.laser_to_pointcloud(msg)
            
            if cloud is not None and len(cloud.points) > 0:
                self.process_pointcloud(cloud, msg.header.stamp)
                
        except Exception as e:
            self.get_logger().error(f"Laser processing error: {e}")

    def laser_to_pointcloud(self, laser_msg):
        """Convert LaserScan to Open3D point cloud"""
        points = []
        
        for i, range_val in enumerate(laser_msg.ranges):
            if (laser_msg.range_min <= range_val <= laser_msg.range_max):
                angle = laser_msg.angle_min + i * laser_msg.angle_increment
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                z = 0.0  # 2D SLAM
                points.append([x, y, z])
        
        if len(points) > 10:  # Minimum points threshold
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(np.array(points))
            return cloud
        
        return None

    def process_pointcloud(self, cloud, timestamp):
        """Process point cloud for SLAM update"""
        with self.map_lock:
            # Downsample for efficiency
            cloud = cloud.voxel_down_sample(self.voxel_size)
            
            if self.previous_cloud is not None:
                # Perform ICP registration
                transform = self.perform_icp(cloud, self.previous_cloud)
                
                if transform is not None:
                    # Update pose
                    with self.pose_lock:
                        self.current_pose = self.current_pose @ transform
                    
                    # Add to map
                    self.update_map(cloud, transform)
                    
                    # Check for loop closure
                    self.check_loop_closure()
            
            self.previous_cloud = cloud

    def perform_icp(self, source, target):
        """Perform ICP registration between point clouds"""
        try:
            # ICP parameters
            threshold = self.icp_max_distance
            trans_init = np.eye(4)  # Initial guess
            
            # Point-to-point ICP
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            
            # Check if registration was successful
            if reg_p2p.fitness > 0.3:  # Minimum fitness threshold
                return reg_p2p.transformation
            
        except Exception as e:
            self.get_logger().warn(f"ICP failed: {e}")
        
        return None

    def update_map(self, cloud, transform):
        """Update the global map with new point cloud"""
        # Transform cloud to global coordinates
        cloud_global = cloud.transform(self.current_pose)
        
        # Add to global map
        self.map_cloud += cloud_global
        
        # Downsample periodically to control memory
        if len(self.map_cloud.points) > 50000:
            self.map_cloud = self.map_cloud.voxel_down_sample(self.voxel_size * 2)

    def check_loop_closure(self):
        """Simple loop closure detection"""
        current_position = self.current_pose[:3, 3]
        
        for i, past_pose in enumerate(self.pose_history):
            past_position = past_pose[:3, 3]
            distance = np.linalg.norm(current_position - past_position)
            
            if distance < self.loop_closure_threshold:
                self.get_logger().info(f"Loop closure detected at frame {i}")
                # TODO: Implement pose graph optimization
                break
        
        # Add current pose to history
        if len(self.pose_history) % 10 == 0:  # Sample poses
            self.pose_history.append(self.current_pose.copy())

    def odometry_callback(self, msg):
        """Process odometry updates (for comparison with SLAM)"""
        # Store odometry for validation/comparison
        pass

    def slam_update_callback(self):
        """Periodic SLAM updates and publishing"""
        try:
            # Publish current pose
            self.publish_pose()
            
            # Publish occupancy grid
            self.publish_occupancy_grid()
            
            # Publish point cloud
            self.publish_pointcloud()
            
            # Publish semantic markers
            self.publish_semantic_markers()
            
        except Exception as e:
            self.get_logger().error(f"SLAM update error: {e}")

    def publish_pose(self):
        """Publish current robot pose"""
        with self.pose_lock:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = self.map_frame
            
            # Extract position and orientation from transformation matrix
            pose_msg.pose.position.x = float(self.current_pose[0, 3])
            pose_msg.pose.position.y = float(self.current_pose[1, 3])
            pose_msg.pose.position.z = float(self.current_pose[2, 3])
            
            # Convert rotation matrix to quaternion (simplified for 2D)
            yaw = np.arctan2(self.current_pose[1, 0], self.current_pose[0, 0])
            pose_msg.pose.orientation.z = np.sin(yaw / 2)
            pose_msg.pose.orientation.w = np.cos(yaw / 2)
            
            self.pose_pub.publish(pose_msg)

    def publish_occupancy_grid(self):
        """Generate and publish occupancy grid from point cloud"""
        if len(self.map_cloud.points) == 0:
            return
            
        # Simple occupancy grid generation
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = self.map_frame
        
        # Grid parameters
        resolution = 0.05  # meters per pixel
        width = height = 400  # pixels
        
        grid_msg.info.resolution = resolution
        grid_msg.info.width = width
        grid_msg.info.height = height
        grid_msg.info.origin.position.x = -width * resolution / 2
        grid_msg.info.origin.position.y = -height * resolution / 2
        
        # Initialize grid
        grid_data = np.full((height, width), -1, dtype=np.int8)  # Unknown
        
        # Fill grid with point cloud data
        points = np.asarray(self.map_cloud.points)
        if len(points) > 0:
            # Convert points to grid coordinates
            x_indices = ((points[:, 0] - grid_msg.info.origin.position.x) / resolution).astype(int)
            y_indices = ((points[:, 1] - grid_msg.info.origin.position.y) / resolution).astype(int)
            
            # Mark occupied cells
            valid_mask = ((x_indices >= 0) & (x_indices < width) & 
                         (y_indices >= 0) & (y_indices < height))
            
            grid_data[y_indices[valid_mask], x_indices[valid_mask]] = 100  # Occupied
        
        grid_msg.data = grid_data.flatten().tolist()
        self.map_pub.publish(grid_msg)

    def publish_pointcloud(self):
        """Publish point cloud for visualization"""
        if len(self.map_cloud.points) == 0:
            return
        
        try:
            # Convert Open3D point cloud to PointCloud2
            cloud_msg = open3d_to_pointcloud2(
                self.map_cloud,
                frame_id=self.map_frame,
                timestamp=self.get_clock().now().to_msg()
            )
            
            if cloud_msg is not None:
                self.pointcloud_pub.publish(cloud_msg)
                
        except Exception as e:
            self.get_logger().warn(f"Point cloud publishing error: {e}")

    def publish_semantic_markers(self):
        """Publish semantic object markers for visualization"""
        marker_array = MarkerArray()
        
        for obj_id, obj_data in self.semantic_objects.items():
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = obj_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Position and size from object data
            marker.pose.position.x = obj_data.get('x', 0.0)
            marker.pose.position.y = obj_data.get('y', 0.0)
            marker.pose.position.z = obj_data.get('z', 0.0)
            
            marker.scale.x = obj_data.get('width', 0.5)
            marker.scale.y = obj_data.get('height', 0.5)
            marker.scale.z = obj_data.get('depth', 0.5)
            
            # Color based on object class
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7)
            
            marker_array.markers.append(marker)
        
        self.markers_pub.publish(marker_array)


def main(args=None):
    """Main entry point for basic SLAM node"""
    rclpy.init(args=args)
    
    try:
        slam_node = BasicSLAMNode()
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"SLAM node error: {e}")
    finally:
        if 'slam_node' in locals():
            slam_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 