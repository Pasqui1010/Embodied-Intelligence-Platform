#!/usr/bin/env python3
"""
VLM Grounding Node

Main node that orchestrates vision-language grounding components and provides
ROS 2 interface for spatial reference resolution and object affordance estimation.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import cv2
import time
import logging
import threading
from typing import Dict, List, Optional, Any
import json

# ROS 2 imports
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Point, Pose, PoseStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray
from eip_interfaces.msg import SafetyViolation, SafetyVerificationRequest, SafetyVerificationResponse
from eip_interfaces.srv import ValidateTaskPlan

# Import VLM grounding components
from .spatial_reference_resolver import SpatialReferenceResolver, SceneData, ObjectDetection, SpatialReference
from .object_affordance_estimator import ObjectAffordanceEstimator, AffordanceSet
from .scene_understanding import SceneUnderstanding, SceneDescription
from .vlm_integration import VLMIntegration, VLMReasoningResult


class VLMGroundingNode(Node):
    """
    VLM Grounding Node for spatial reference resolution and object affordance estimation
    
    This node integrates vision-language models with robotics applications to enable
    natural language spatial reasoning and object manipulation planning.
    """
    
    def __init__(self):
        super().__init__('vlm_grounding_node')
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('spatial_resolution_rate', 10.0),
                ('affordance_estimation_rate', 5.0),
                ('scene_analysis_rate', 2.0),
                ('vlm_integration_rate', 1.0),
                ('enable_clip', True),
                ('enable_safety_validation', True),
                ('min_confidence_threshold', 0.6),
                ('max_objects_per_scene', 20),
                ('enable_visualization', True),
                ('model_path', ''),
                ('safety_llm_path', '')
            ]
        )
        
        # Initialize VLM grounding components
        self._initialize_components()
        
        # Set up ROS 2 publishers and subscribers
        self._setup_ros_interface()
        
        # Initialize state variables
        self.current_scene_data = None
        self.current_scene_description = None
        self.spatial_references = []
        self.object_affordances = {}
        self.vlm_results = {}
        
        # Performance tracking
        self.processing_times = {
            'spatial_resolution': [],
            'affordance_estimation': [],
            'scene_analysis': [],
            'vlm_integration': []
        }
        
        # Start processing threads
        self._start_processing_threads()
        
        self.logger.info("VLM Grounding Node initialized successfully")
    
    def _initialize_components(self):
        """Initialize VLM grounding components"""
        model_path = self.get_parameter('model_path').value
        safety_llm_path = self.get_parameter('safety_llm_path').value
        
        # Initialize spatial reference resolver
        self.spatial_resolver = SpatialReferenceResolver()
        self.logger.info("Spatial reference resolver initialized")
        
        # Initialize object affordance estimator
        self.affordance_estimator = ObjectAffordanceEstimator(model_path)
        self.logger.info("Object affordance estimator initialized")
        
        # Initialize scene understanding
        self.scene_understanding = SceneUnderstanding(model_path)
        self.logger.info("Scene understanding initialized")
        
        # Initialize VLM integration
        self.vlm_integration = VLMIntegration(
            safety_llm_path=safety_llm_path
        )
        self.logger.info("VLM integration initialized")
    
    def _setup_ros_interface(self):
        """Set up ROS 2 publishers and subscribers"""
        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self._image_callback,
            sensor_qos
        )
        
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/camera/pointcloud',
            self._pointcloud_callback,
            sensor_qos
        )
        
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/lidar/scan',
            self._lidar_callback,
            sensor_qos
        )
        
        self.spatial_query_sub = self.create_subscription(
            String,
            '/vlm_grounding/spatial_query',
            self._spatial_query_callback,
            command_qos
        )
        
        self.affordance_query_sub = self.create_subscription(
            String,
            '/vlm_grounding/affordance_query',
            self._affordance_query_callback,
            command_qos
        )
        
        self.vlm_query_sub = self.create_subscription(
            String,
            '/vlm_grounding/vlm_query',
            self._vlm_query_callback,
            command_qos
        )
        
        # Publishers
        self.spatial_reference_pub = self.create_publisher(
            String,
            '/vlm_grounding/spatial_reference',
            command_qos
        )
        
        self.affordance_result_pub = self.create_publisher(
            String,
            '/vlm_grounding/affordance_result',
            command_qos
        )
        
        self.vlm_result_pub = self.create_publisher(
            String,
            '/vlm_grounding/vlm_result',
            command_qos
        )
        
        self.scene_description_pub = self.create_publisher(
            String,
            '/vlm_grounding/scene_description',
            command_qos
        )
        
        # Visualization publishers
        if self.get_parameter('enable_visualization').value:
            self.marker_pub = self.create_publisher(
                MarkerArray,
                '/vlm_grounding/visualization',
                command_qos
            )
        
        # Services
        self.spatial_resolution_srv = self.create_service(
            String,
            '/vlm_grounding/resolve_spatial_reference',
            self._spatial_resolution_service
        )
        
        self.affordance_estimation_srv = self.create_service(
            String,
            '/vlm_grounding/estimate_affordances',
            self._affordance_estimation_service
        )
        
        self.vlm_reasoning_srv = self.create_service(
            String,
            '/vlm_grounding/vlm_reasoning',
            self._vlm_reasoning_service
        )
        
        self.logger.info("ROS 2 interface setup completed")
    
    def _start_processing_threads(self):
        """Start background processing threads"""
        # Spatial resolution thread
        spatial_rate = self.get_parameter('spatial_resolution_rate').value
        self.spatial_timer = self.create_timer(
            1.0 / spatial_rate,
            self._spatial_resolution_loop
        )
        
        # Affordance estimation thread
        affordance_rate = self.get_parameter('affordance_estimation_rate').value
        self.affordance_timer = self.create_timer(
            1.0 / affordance_rate,
            self._affordance_estimation_loop
        )
        
        # Scene analysis thread
        scene_rate = self.get_parameter('scene_analysis_rate').value
        self.scene_timer = self.create_timer(
            1.0 / scene_rate,
            self._scene_analysis_loop
        )
        
        # VLM integration thread
        vlm_rate = self.get_parameter('vlm_integration_rate').value
        self.vlm_timer = self.create_timer(
            1.0 / vlm_rate,
            self._vlm_integration_loop
        )
        
        self.logger.info("Processing threads started")
    
    def _image_callback(self, msg: Image):
        """Handle incoming image messages"""
        try:
            # Convert ROS Image to OpenCV format
            image = self._ros_image_to_cv2(msg)
            
            # Update scene data
            if self.current_scene_data is None:
                self.current_scene_data = SceneData(
                    timestamp=time.time(),
                    objects=[],
                    image=image
                )
            else:
                self.current_scene_data.image = image
                self.current_scene_data.timestamp = time.time()
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
    
    def _pointcloud_callback(self, msg: PointCloud2):
        """Handle incoming point cloud messages"""
        try:
            # Convert ROS PointCloud2 to numpy array
            pointcloud = self._ros_pointcloud_to_numpy(msg)
            
            # Update scene data
            if self.current_scene_data is not None:
                self.current_scene_data.point_cloud = pointcloud
            
        except Exception as e:
            self.logger.error(f"Error processing point cloud: {e}")
    
    def _lidar_callback(self, msg: LaserScan):
        """Handle incoming lidar scan messages"""
        try:
            # Process lidar data for obstacle detection
            # This is a simplified implementation
            pass
            
        except Exception as e:
            self.logger.error(f"Error processing lidar scan: {e}")
    
    def _spatial_query_callback(self, msg: String):
        """Handle spatial reference queries"""
        try:
            query = msg.data
            self.logger.info(f"Received spatial query: {query}")
            
            if self.current_scene_data is None:
                self.logger.warning("No scene data available for spatial query")
                return
            
            # Resolve spatial reference
            start_time = time.time()
            spatial_reference = self.spatial_resolver.resolve_reference(
                query, self.current_scene_data
            )
            processing_time = time.time() - start_time
            
            # Store result
            self.spatial_references.append(spatial_reference)
            self.processing_times['spatial_resolution'].append(processing_time)
            
            # Publish result
            result_msg = String()
            result_msg.data = json.dumps({
                'query': query,
                'position': spatial_reference.position,
                'confidence': spatial_reference.confidence,
                'reference_type': spatial_reference.reference_type.value,
                'reference_object': spatial_reference.reference_object,
                'description': spatial_reference.description,
                'processing_time': processing_time
            })
            self.spatial_reference_pub.publish(result_msg)
            
        except Exception as e:
            self.logger.error(f"Error processing spatial query: {e}")
    
    def _affordance_query_callback(self, msg: String):
        """Handle affordance estimation queries"""
        try:
            query = msg.data
            self.logger.info(f"Received affordance query: {query}")
            
            if self.current_scene_data is None or not self.current_scene_data.objects:
                self.logger.warning("No objects available for affordance estimation")
                return
            
            # Estimate affordances for all objects
            start_time = time.time()
            affordances = {}
            
            for obj in self.current_scene_data.objects:
                affordance_set = self.affordance_estimator.estimate_affordances(obj)
                affordances[obj.class_name] = affordance_set
            
            processing_time = time.time() - start_time
            
            # Store results
            self.object_affordances.update(affordances)
            self.processing_times['affordance_estimation'].append(processing_time)
            
            # Publish results
            result_msg = String()
            result_msg.data = json.dumps({
                'query': query,
                'affordances': {
                    name: {
                        'grasp_points': len(aff.grasp_points),
                        'safety_score': aff.safety_score,
                        'confidence': aff.confidence,
                        'available_affordances': [a.value for a in aff.available_affordances]
                    }
                    for name, aff in affordances.items()
                },
                'processing_time': processing_time
            })
            self.affordance_result_pub.publish(result_msg)
            
        except Exception as e:
            self.logger.error(f"Error processing affordance query: {e}")
    
    def _vlm_query_callback(self, msg: String):
        """Handle VLM reasoning queries"""
        try:
            query = msg.data
            self.logger.info(f"Received VLM query: {query}")
            
            if self.current_scene_data is None or self.current_scene_description is None:
                self.logger.warning("No scene data available for VLM query")
                return
            
            # Process VLM query
            start_time = time.time()
            vlm_result = self.vlm_integration.process_visual_query(
                query, self.current_scene_data, self.current_scene_description
            )
            processing_time = time.time() - start_time
            
            # Store result
            self.vlm_results[query] = vlm_result
            self.processing_times['vlm_integration'].append(processing_time)
            
            # Publish result
            result_msg = String()
            result_msg.data = json.dumps({
                'query': query,
                'response': vlm_result.response.text,
                'confidence': vlm_result.confidence,
                'safety_validation': vlm_result.safety_validation,
                'spatial_references': vlm_result.spatial_references,
                'reasoning_steps': vlm_result.reasoning_steps,
                'processing_time': processing_time
            })
            self.vlm_result_pub.publish(result_msg)
            
        except Exception as e:
            self.logger.error(f"Error processing VLM query: {e}")
    
    def _spatial_resolution_loop(self):
        """Background spatial resolution processing"""
        if self.current_scene_data is None:
            return
        
        # Process pending spatial queries
        # This could be extended to handle queued queries
        
        # Update visualization
        if self.get_parameter('enable_visualization').value:
            self._publish_visualization()
    
    def _affordance_estimation_loop(self):
        """Background affordance estimation processing"""
        if self.current_scene_data is None or not self.current_scene_data.objects:
            return
        
        # Update affordances for current objects
        # This could be extended to handle dynamic object tracking
    
    def _scene_analysis_loop(self):
        """Background scene analysis processing"""
        if self.current_scene_data is None:
            return
        
        try:
            # Perform scene analysis
            start_time = time.time()
            scene_description = self.scene_understanding.analyze_scene(self.current_scene_data)
            processing_time = time.time() - start_time
            
            # Update current scene description
            self.current_scene_description = scene_description
            self.processing_times['scene_analysis'].append(processing_time)
            
            # Publish scene description
            scene_msg = String()
            scene_msg.data = json.dumps({
                'scene_type': scene_description.scene_type,
                'num_elements': len(scene_description.elements),
                'complexity_score': scene_description.complexity_score,
                'safety_score': scene_description.safety_score,
                'description_text': scene_description.description,
                'processing_time': processing_time
            })
            self.scene_description_pub.publish(scene_msg)
            
        except Exception as e:
            self.logger.error(f"Error in scene analysis: {e}")
    
    def _vlm_integration_loop(self):
        """Background VLM integration processing"""
        # Process any pending VLM tasks
        # This could be extended to handle queued VLM operations
    
    def _spatial_resolution_service(self, request: String, response: String) -> String:
        """Service for spatial reference resolution"""
        try:
            query = request.data
            self.logger.info(f"Service request: spatial resolution for '{query}'")
            
            if self.current_scene_data is None:
                response.data = json.dumps({'error': 'No scene data available'})
                return response
            
            # Resolve spatial reference
            spatial_reference = self.spatial_resolver.resolve_reference(
                query, self.current_scene_data
            )
            
            # Prepare response
            response.data = json.dumps({
                'position': spatial_reference.position,
                'confidence': spatial_reference.confidence,
                'reference_type': spatial_reference.reference_type.value,
                'reference_object': spatial_reference.reference_object,
                'description': spatial_reference.description
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in spatial resolution service: {e}")
            response.data = json.dumps({'error': str(e)})
            return response
    
    def _affordance_estimation_service(self, request: String, response: String) -> String:
        """Service for affordance estimation"""
        try:
            query = request.data
            self.logger.info(f"Service request: affordance estimation for '{query}'")
            
            if self.current_scene_data is None or not self.current_scene_data.objects:
                response.data = json.dumps({'error': 'No objects available'})
                return response
            
            # Estimate affordances
            affordances = {}
            for obj in self.current_scene_data.objects:
                if query.lower() in obj.class_name.lower():
                    affordance_set = self.affordance_estimator.estimate_affordances(obj)
                    affordances[obj.class_name] = {
                        'grasp_points': len(affordance_set.grasp_points),
                        'safety_score': affordance_set.safety_score,
                        'confidence': affordance_set.confidence,
                        'available_affordances': [a.value for a in affordance_set.available_affordances]
                    }
            
            # Prepare response
            response.data = json.dumps({'affordances': affordances})
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in affordance estimation service: {e}")
            response.data = json.dumps({'error': str(e)})
            return response
    
    def _vlm_reasoning_service(self, request: String, response: String) -> String:
        """Service for VLM reasoning"""
        try:
            query = request.data
            self.logger.info(f"Service request: VLM reasoning for '{query}'")
            
            if self.current_scene_data is None or self.current_scene_description is None:
                response.data = json.dumps({'error': 'No scene data available'})
                return response
            
            # Process VLM query
            vlm_result = self.vlm_integration.process_visual_query(
                query, self.current_scene_data, self.current_scene_description
            )
            
            # Prepare response
            response.data = json.dumps({
                'response': vlm_result.response.text,
                'confidence': vlm_result.confidence,
                'safety_validation': vlm_result.safety_validation,
                'spatial_references': vlm_result.spatial_references,
                'reasoning_steps': vlm_result.reasoning_steps
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in VLM reasoning service: {e}")
            response.data = json.dumps({'error': str(e)})
            return response
    
    def _ros_image_to_cv2(self, ros_image: Image) -> np.ndarray:
        """Convert ROS Image message to OpenCV format"""
        # Simplified conversion - in practice, this would handle different encodings
        height = ros_image.height
        width = ros_image.width
        
        # Convert to numpy array
        image_array = np.frombuffer(ros_image.data, dtype=np.uint8)
        image_array = image_array.reshape((height, width, 3))
        
        # Convert BGR to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        return image_array
    
    def _ros_pointcloud_to_numpy(self, ros_pointcloud: PointCloud2) -> np.ndarray:
        """Convert ROS PointCloud2 message to numpy array"""
        # Simplified conversion - in practice, this would handle different point formats
        # For now, return empty array
        return np.array([])
    
    def _publish_visualization(self):
        """Publish visualization markers"""
        if not self.get_parameter('enable_visualization').value:
            return
        
        try:
            marker_array = MarkerArray()
            
            # Add markers for spatial references
            for i, ref in enumerate(self.spatial_references[-10:]):  # Last 10 references
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "spatial_references"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                marker.pose.position.x = ref.position[0]
                marker.pose.position.y = ref.position[1]
                marker.pose.position.z = ref.position[2]
                
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.8
                
                marker_array.markers.append(marker)
            
            # Add markers for objects
            if self.current_scene_data and self.current_scene_data.objects:
                for i, obj in enumerate(self.current_scene_data.objects):
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = "objects"
                    marker.id = i
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    
                    marker.pose.position.x = obj.center[0]
                    marker.pose.position.y = obj.center[1]
                    marker.pose.position.z = 0.0
                    
                    marker.scale.x = obj.bbox[2] / 100.0
                    marker.scale.y = obj.bbox[3] / 100.0
                    marker.scale.z = 0.1
                    
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 0.6
                    
                    marker_array.markers.append(marker)
            
            self.marker_pub.publish(marker_array)
            
        except Exception as e:
            self.logger.error(f"Error publishing visualization: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        for metric, times in self.processing_times.items():
            if times:
                stats[metric] = {
                    'mean_time': np.mean(times),
                    'max_time': np.max(times),
                    'min_time': np.min(times),
                    'count': len(times)
                }
            else:
                stats[metric] = {
                    'mean_time': 0.0,
                    'max_time': 0.0,
                    'min_time': 0.0,
                    'count': 0
                }
        
        return stats


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    node = VLMGroundingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Print performance stats
        stats = node.get_performance_stats()
        node.logger.info(f"Performance stats: {json.dumps(stats, indent=2)}")
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 