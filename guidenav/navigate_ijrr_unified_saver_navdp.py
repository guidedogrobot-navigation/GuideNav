import os
import sys
import time
import yaml
import copy
import argparse
import parser
import glob
import pdb

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from tqdm.auto import tqdm
from pathlib import Path

import torch

# Place recognition
from place_recognition.bayesian_querier import PlaceRecognitionTopologicalFilter
from place_recognition.sliding_window_querier import PlaceRecognitionSlidingWindowFilter
from place_recognition.feature_extractor import FeatureExtractor
from place_recognition import extract_database


# feature matching for rel pose est.
from match_to_control import feature_match, se2_estimate, control # estimate_pose_test

# Go2 control
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# realsense stream input
import threading
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# debug
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import CompressedImage


from utils import to_numpy, read_image, read_depth_image, get_image_transform
matplotlib.use('Agg')  # Use non-interactive backend

# tactile stop
from std_msgs.msg import Bool

# smooth behavior
from collections import deque

# navdp
import requests
import json

class FakeRGBDSubscriber(Node):
    def __init__(self, guidenav_node, image_directory, use_odometry=False, fps=30):
        super().__init__('fake_rgbd_subscriber')
        self.guidenav_node = guidenav_node
        self.bridge = CvBridge()
        self.use_odometry = use_odometry
        self.image_directory = image_directory
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        # Load image paths
        self.rgb_paths = []
        self.depth_paths = []
        self._load_image_paths()
        
        self.current_index = 0
        self.is_running = False
        self.thread = None
        
        self.get_logger().info(f'Loaded {len(self.rgb_paths)} RGB images and {len(self.depth_paths)} depth images')
    
    def _load_image_paths(self):
        """Load RGB and depth image paths from directory"""
        # Option 1: Separate rgb/ and depth/ subdirectories
        rgb_dir = os.path.join(self.image_directory, 'color')
        depth_dir = os.path.join(self.image_directory, 'depth')
        
        if os.path.exists(rgb_dir):
            self.rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, '*')))
            self.rgb_paths = [p for p in self.rgb_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if os.path.exists(depth_dir):
            self.depth_paths = sorted(glob.glob(os.path.join(depth_dir, '*')))
            self.depth_paths = [p for p in self.depth_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        

    def start_streaming(self, loop=True):
        """Start streaming images from directory"""
        if self.is_running:
            return
            
        self.is_running = True
        self.loop = loop
        self.thread = threading.Thread(target=self._stream_images)
        self.thread.daemon = True
        self.thread.start()
        self.get_logger().info('Started fake RGBD streaming')
    
    def stop_streaming(self):
        """Stop streaming images"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        self.get_logger().info('Stopped fake RGBD streaming')
    
    def _stream_images(self):
        """Main streaming loop"""
        while self.is_running:
            start_time = time.time()
            
            # Check if we have images to process
            if not self.rgb_paths and not self.depth_paths:
                self.get_logger().error('No images found in directory')
                break
            
            # Get current images
            rgb_image = None
            depth_image = None
            
            if self.rgb_paths and self.current_index < len(self.rgb_paths):
                rgb_path = self.rgb_paths[self.current_index]
                rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                if rgb_image is None:
                    self.get_logger().warning(f'Failed to load RGB image: {rgb_path}')
            
            if self.depth_paths and self.current_index < len(self.depth_paths):
                depth_path = self.depth_paths[self.current_index]
                # Load depth as 16-bit if possible, otherwise as grayscale
                depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth_image is None:
                    depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                if depth_image is None:
                    self.get_logger().warning(f'Failed to load depth image: {depth_path}')
            
            # Process images if available
            if rgb_image is not None or depth_image is not None:
                if self.use_odometry:
                    # For odometry mode, create a fake odometry message or pass None
                    fake_odom = None  # You can create a proper Odometry message here if needed
                    self.guidenav_node.process_stream_image(rgb_image, depth_image, fake_odom)
                else:
                    self.guidenav_node.process_stream_image(rgb_image, depth_image, None)
            
            # Advance to next image
            self.current_index += 1
            
            # Handle looping or stopping
            max_images = max(len(self.rgb_paths), len(self.depth_paths)) if (self.rgb_paths or self.depth_paths) else 0
            if self.current_index >= max_images:
                if self.loop:
                    self.current_index = 0
                    self.get_logger().info('Looping back to start of image sequence')
                else:
                    self.get_logger().info('Reached end of image sequence')
                    break
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)
        
        self.is_running = False

class RGBDSubscriber(Node):
    def __init__(self, guidenav_node, use_odometry=False):
        super().__init__('rgbd_subscriber')
        self.guidenav_node = guidenav_node
        self.bridge = CvBridge()
        self.use_odometry = use_odometry

        if self.use_odometry:
            # Use message_filters for synced subscription
            from message_filters import Subscriber, ApproximateTimeSynchronizer
            from nav_msgs.msg import Odometry

            # self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
            # self.depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
            self.rgb_sub = Subscriber(self, Image, '/d435i/color/image_raw')
            self.depth_sub = Subscriber(self, Image, '/d435i/aligned_depth_to_color/image_raw')
            self.odom_sub = Subscriber(self, Odometry, '/visual_slam/tracking/odometry')

            self.ts = ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub, self.odom_sub],
                queue_size=10,
                slop=0.1
            )
            self.ts.registerCallback(self.synced_callback)
        else:
            # Use regular rclpy subscription
            self.rgb_image = None
            self.depth_image = None

            # self.rgb_sub = self.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, 10)
            # self.depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
            self.rgb_sub = self.create_subscription(Image, '/d435i/color/image_raw', self.rgb_callback, 10)
            self.depth_sub = self.create_subscription(Image, '/d435i/aligned_depth_to_color/image_raw', self.depth_callback, 10)

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_process()

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.try_process()

    def try_process(self):
        if self.rgb_image is not None and self.depth_image is not None:
            self.guidenav_node.process_stream_image(self.rgb_image, self.depth_image, None)
            self.rgb_image = None
            self.depth_image = None

    def synced_callback(self, rgb_msg, depth_msg, odom_msg):
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        self.guidenav_node.process_stream_image(rgb, depth, odom_msg)

class GuideNavNode:

    def __init__(self, args: argparse.Namespace):
        if args.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.dev_name = torch.cuda.get_device_name(0)
            print(f"Using CUDA Device: {self.dev_name}")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

        # Tactile segmentation related
        self.stop_tactile = False
        self.tactile_signal_count = 0
        self.last_tactile_time = None
        self.tactile_stop_duration = 0.0
        self.tactile_lock = threading.Lock()


        # NavDP
        self.navdp_enabled = args.navdp_enabled
        print(f"navdp enabled: {self.navdp_enabled}")

        # Get the image preprocessing transform for the neural network models
        self._image_transform = get_image_transform(args.img_size)

        self.img_suffix = ['png', 'jpg', 'jpeg']

        self.pose_errors = []

        self.bridge = CvBridge()

        # Load the topological map
        self.topomap_filenames = []
        self._load_topomap(args.topomap_base_dir, args.topomap_dir) # started from 4

        print(f"Topomap loaded with {self.map_size} images")

        self.context_size = args.context_size if hasattr(args, 'context_size') else 4
        self.context_queue = []

        self.vis_dir = args.vis_dir if hasattr(args, 'vis_dir') else 'visualization_results'

        self.unvalidPnpCount = 0

        # Add error tracking
        self.prediction_errors = []
        self.frame_indices = []
        self.ground_truth_indices = []
        self.prediction_indices = []

        # feature matching method
        self.fm_method = args.feature_matching

        # Load robot config
        with args.robot_config_path.open(mode="r", encoding="utf-8") as f:
            robot_configs = yaml.safe_load(f)
        self.robot_config = robot_configs[args.robot]

        self.K = np.array([self.robot_config['fx'], 0, self.robot_config['cx'],
                            0, self.robot_config['fy'], self.robot_config['cy'],
                            0, 0, 1]).reshape(3, 3).astype(np.float32)

        print(f"Camera intrinsic matrix K:\n{self.K}")

        # interpolation for smooth movements
        self.use_smoothing = getattr(args, 'use_smoothing', False)

        if self.use_smoothing:
            print("[INFO] Using smoothing for pose and command")
        else:
            print("[INFO] Smoothing is disabled")

        self.smooth_filter = None
        self.last_smooth_pose = None
        self.last_smooth_cmd = None
        self.last_cmd_time = time.time()

        if self.use_smoothing:
            self.pose_hist = deque(maxlen=5)
            self.cmd_hist = deque(maxlen=3)
            self.sm_pose_alpha_x = self.robot_config['smooth_pose_alpha_x']
            self.sm_pose_alpha_y = self.robot_config['smooth_pose_alpha_y']
            self.sm_pose_alpha_yaw = self.robot_config['smooth_pose_alpha_yaw']
            self.sm_cmd_alpha = self.robot_config['smooth_cmd_alpha']
            self.sm_max_dv = self.robot_config['max_v'] * 0.5
            self.sm_max_dw = self.robot_config['max_w'] * 0.6

        # Init fm_model
        if self.fm_method == 'loftr':
            self.fm_model = feature_match.init_loftr()

        elif self.fm_method == 'roma':
           self.fm_model = feature_match.init_roma() 

        elif self.fm_method == 'mast3r':
           self.fm_model = feature_match.init_mast3r() 

        elif self.fm_method == 'liftfeat':
           self.fm_model = feature_match.init_liftFeat() 

        elif self.fm_method == 'reloc3r':
            self.fm_model, self.img_reso = feature_match.init_reloc3r()

        elif self.fm_method == 'navdp':
            self.init_navdp(args)
            

        # Initialize place recognition
        if args.subgoal_mode == 'place_recognition':
            self._setup_place_recognition(
                args.model_config_path,
                args.model_weight_dir,
                args.pr_model,
                filter_mode=args.filter_mode,
                transition_model_window_lower=args.transition_model_window_lower,
                transition_model_window_upper=args.transition_model_window_upper,
                bayesian_filter_delta=args.filter_delta,
                recompute_db=args.recompute_place_recognition_db,
                )
                
        print(f"[INFO] Place recognition model {args.pr_model} loaded")
        rclpy.init()  # call this once globally

        # Create ROS2 node and publisher
        self.ros_node = rclpy.create_node('guidenav_vel_pub')
        self.cmd_vel_pub = self.ros_node.create_publisher(Twist, '/cmd_vel', 10)
        self.tactile_sub = self.ros_node.create_subscription(
            Bool, 
            '/tactile_stop_signal', 
            self.stop_tactile_callback, 
            10  # QoS depth - can handle high frequency
        )
        self.tactile_stats_timer = self.ros_node.create_timer(5.0, self.print_tactile_stats)


        
        self.enable_debug = getattr(args, 'enable_debug', True)

        if self.enable_debug:
            from std_msgs.msg import Float64MultiArray
            from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
            print(f"DEBUG ENABLED")
            
            # Fast QoS for minimal latency
            fast_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,  # Drop frames vs lag
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1  # Only keep latest
            )
            
            # Publishers
            self.debug_image_pub = self.ros_node.create_publisher(
                CompressedImage, '/debug/image/compressed', fast_qos)
            self.debug_nav_pub = self.ros_node.create_publisher(
                Float64MultiArray, '/debug/nav_data', fast_qos)

        
        # Navigation parameters
        self.closest_node_idx = args.start_node_idx if hasattr(args, 'start_node_idx') else 0
        self.goal_node_idx = self.map_size - 1
        self.lookahead = args.lookahead if hasattr(args, 'lookahead') else 1
        self.filter_mode = args.filter_mode
        self.window_radius = args.window_radius if hasattr(args, 'window_radius') else 2
        
        # Navigation state
        self.first_pass = True
        self.reached_goal = False
        self.navigation_active = False

        # Add thread lock for thread safety
        self.processing_lock = threading.Lock()

    # def publish_cmd_vel(self, v: float, w: float):
    #     twist = Twist()
    #     twist.linear.x = v
    #     twist.angular.z = w
    #     self.cmd_vel_pub.publish(twist)
    #     self.ros_node.get_logger().info(f"Sent cmd_vel: v={v:.2f}, w={w:.2f}")

    def init_navdp(self, args):
        """Initialize NavDP connection"""
        
        self.navdp_port = getattr(args, 'navdp_port', 8000)
        self.navdp_url = f"http://localhost:{self.navdp_port}"
        
        print(f"[NavDP] Initializing connection to {self.navdp_url}")
        
        # Test connection and reset
        if self.navdp_reset(self.K):
            print("[NavDP] Successfully initialized")
        else:
            print("[NavDP] Failed to initialize - will use fallback navigation")

    def navdp_reset(self, intrinsic):
        """Reset NavDP system"""
        try:
            url = f"{self.navdp_url}/navdp_reset"
            response = requests.post(url, json={
                'intrinsic': intrinsic.tolist(),
                'stop_threshold': -4.0,
                'batch_size': 1
            })
            print("[NavDP] Reset successful")
            return response.status_code == 200
        except Exception as e:
            print(f"[NavDP] Reset failed: {e}")
            return False
    
    def navdp_get_trajectory_point_goal(self, rgb_img, depth_img, goal_x, goal_y):
        """Get NavDP trajectory with point goal"""
        try:
            url = f"{self.navdp_url}/navdp_step_xy"
            
            # Prepare images
            _, rgb_encoded = cv2.imencode('.jpg', rgb_img)
            depth_scaled = np.clip(depth_img * 10000.0, 0, 65535.0).astype(np.uint16)
            _, depth_encoded = cv2.imencode('.png', depth_scaled)
            
            files = {
                'image': ('image.jpg', rgb_encoded.tobytes(), 'image/jpeg'),
                'depth': ('depth.png', depth_encoded.tobytes(), 'image/png'),
            }
            data = {
                'goal_data': json.dumps({
                    'goal_x': [goal_x],
                    'goal_y': [goal_y]
                })
            }
            
            response = requests.post(url, files=files, data=data)
            if response.status_code == 200:
                result = json.loads(response.text)
                return np.array(result['trajectory']), np.array(result['all_values'])
            else:
                print(f"[NavDP] API call failed with status {response.status_code}")
                return None, None
                
        except Exception as e:
            print(f"[NavDP] Trajectory request failed: {e}")
            return None, None

    def navdp_no_goal_step(self, rgb_img, depth_img):
        """Get NavDP trajectory in exploration mode (no specific goal)"""
        try:
            url = f"{self.navdp_url}/navdp_step_nogoal"
            
            # Prepare images
            _, rgb_encoded = cv2.imencode('.jpg', rgb_img)
            depth_scaled = np.clip(depth_img * 10000.0, 0, 65535.0).astype(np.uint16)
            _, depth_encoded = cv2.imencode('.png', depth_scaled)
            
            files = {
                'image': ('image.jpg', rgb_encoded.tobytes(), 'image/jpeg'),
                'depth': ('depth.png', depth_encoded.tobytes(), 'image/png'),
            }
            data = {
                'goal_data': json.dumps({
                    'goal_x': [0.0],  # Dummy values for no-goal mode
                    'goal_y': [0.0]
                })
            }
            
            response = requests.post(url, files=files, data=data)
            if response.status_code == 200:
                result = json.loads(response.text)
                return np.array(result['trajectory']), np.array(result['all_values'])
            else:
                print(f"[NavDP] No-goal API call failed with status {response.status_code}")
                return None, None
                
        except Exception as e:
            print(f"[NavDP] No-goal request failed: {e}")
            return None, None

    def trajectory_to_velocity(self, trajectory, dt=0.1):
        """Convert NavDP trajectory to velocity commands"""
        if trajectory is None or len(trajectory) == 0:
            return 0.0, 0.0
        
        # Use the first action in the trajectory
        if len(trajectory[0]) > 0:
            next_action = trajectory[0][1] if len(trajectory[0]) > 1 else trajectory[0][0]
            
            # NavDP outputs relative pose changes (Δx, Δy, Δω)
            dx, dy = next_action[0], next_action[1]
            
            # Convert to velocity commands
            v = np.sqrt(dx**2 + dy**2) / dt  # Linear velocity
            w = np.arctan2(dy, dx) / dt       # Angular velocity approximation
            
            # Clamp to robot limits
            v = np.clip(v, 0, self.robot_config['max_v'])
            w = np.clip(w, -self.robot_config['max_w'], self.robot_config['max_w'])
            
            return v, w
        
        return 0.0, 0.0

    def subgoal_to_approximate_goal(self, subgoal_idx):
        """Convert subgoal index to approximate spatial goal"""
        steps_ahead = max(subgoal_idx - self.closest_node_idx, 1)
        
        # Estimate distance based on steps ahead
        if steps_ahead <= 1:
            return 2.0, 0.0  # Close target
        elif steps_ahead <= 3:
            return 5.0, 0.0  # Medium distance
        else:
            return 8.0, 0.0  # Far target

    def publish_cmd_vel(self, v: float, w: float):
        """Publish velocity commands with tactile safety check"""
        with self.tactile_lock:
            tactile_blocked = self.stop_tactile
        
        if tactile_blocked:
            # Override commands with stop if tactile sensor triggered
            v, w = 0.0, 0.0
            self.ros_node.get_logger().info("TACTILE OVERRIDE: Stopping robot!")
        
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_vel_pub.publish(twist)
        
        # Log with tactile status
        status = "[TACTILE BLOCKED]" if tactile_blocked else "[NORMAL]"
        self.ros_node.get_logger().info(f"{status} cmd_vel: v={v:.2f}, w={w:.2f}")

    def stop_tactile_callback(self, msg):
        """Enhanced tactile stop callback with statistics tracking"""
        current_time = time.time()
        
        with self.tactile_lock:
            # Track signal changes
            state_changed = self.stop_tactile != msg.data
            self.stop_tactile = msg.data
            
            # Update statistics
            self.tactile_signal_count += 1
            self.last_tactile_time = current_time
            
            # Log state changes with more detail
            if state_changed:
                if self.stop_tactile:
                    # self.ros_node.get_logger().warn("🛑 TACTILE STOP ACTIVATED - Navigation will be blocked!")
                    self.ros_node.get_logger().warn("🛑 TACTILE STOP ACTIVATED - Robot will stop but navigation continues processing")

                    # Immediately stop the robot
                    self.publish_cmd_vel(0.0, 0.0)
                else:
                    # self.ros_node.get_logger().info("✅ Tactile clear - Navigation can resume")
                    self.ros_node.get_logger().info("✅ TACTILE CLEAR - Navigation will resume automatically!")

        
        # Optional: Log high-frequency signals occasionally
        if self.tactile_signal_count % 100 == 0:  # Every 100 signals (2 seconds at 50Hz)
            with self.tactile_lock:
                state_str = "STOP" if self.stop_tactile else "GO" 
            self.ros_node.get_logger().info(f"Tactile signals received: {self.tactile_signal_count}, Current: {state_str}")

    def print_tactile_stats(self):
        """Print tactile sensor statistics"""
        current_time = time.time()
        
        with self.tactile_lock:
            if self.last_tactile_time is None:
                self.ros_node.get_logger().warn("⚠️  No tactile signals received!")
                return
            
            time_since_last = current_time - self.last_tactile_time
            state_str = "🛑 STOP" if self.stop_tactile else "✅ GO"
            
            # Calculate approximate rate
            if hasattr(self, 'tactile_stats_start_time'):
                elapsed = current_time - self.tactile_stats_start_time
                rate = self.tactile_signal_count / elapsed if elapsed > 0 else 0
            else:
                self.tactile_stats_start_time = current_time
                rate = 0
            
            self.ros_node.get_logger().info(
                f"Tactile Stats - Signals: {self.tactile_signal_count}, "
                f"Rate: {rate:.1f} Hz, State: {state_str}, "
                f"Last signal: {time_since_last:.1f}s ago"
            )
            
            # Warn if no recent signals (potential connection issue)
            if time_since_last > 1.0:  # No signal for 1 second
                self.ros_node.get_logger().warn(f"⚠️  No tactile signals for {time_since_last:.1f}s - Check connection!")



    def process_stream_image(self, rgb_img, depth_img, odom_msg):
        """Process incoming RealSense images in real-time"""
        with self.processing_lock:
            with self.tactile_lock:
                tactile_stopped = self.stop_tactile
            # Check if navigation is complete
            if self.reached_goal or not self.navigation_active:
                return

            # if tactile_stopped:
            #     # Still process for debugging/monitoring, but don't send commands
            #     self.ros_node.get_logger().info("Tactile active - skipping navigation commands")
            #     return

            # RGB image transform
            current_obs = self._image_transform(rgb_img).unsqueeze(0).to(self.device)

            if depth_img.max() > 1000:  # likely in mm
                depth_img = depth_img / 1000.0

            # Maintain context queue
            self.context_queue.append({'img': current_obs})
            if len(self.context_queue) > self.context_size + 1:
                self.context_queue.pop(0)

            if len(self.context_queue) < self.context_size + 1:
                return  # Wait until enough context

            # Process navigation step
            # time_nav_start = time.time()
            self.navigate_one_step(rgb_img, depth_img, current_obs, odom_msg)
            # time_nav_end = time.time()
            # print(f"[INFO] Navigation step took {time_nav_end - time_nav_start:.3f} seconds")
    def smooth_pose(self, x, y, yaw):
        """Apply exponential smoothing to pose estimates"""
        if not self.use_smoothing:
            return x, y, yaw
            
        yaw_radians = np.radians(yaw)
        # current_pose = np.array([x, y, yaw])
        current_pose = np.array([x, y, yaw_radians])
        self.pose_hist.append(current_pose)
        
        if self.last_smooth_pose is None:
            self.last_smooth_pose = current_pose.copy()
            return x, y, yaw
        
        # Exponential smoothing
        x_smooth = self.sm_pose_alpha_x * x + (1 - self.sm_pose_alpha_x) * self.last_smooth_pose[0]
        y_smooth = self.sm_pose_alpha_y * y + (1 - self.sm_pose_alpha_y) * self.last_smooth_pose[1]
        
        # Handle angle smoothing properly
        angle_diff = yaw_radians - self.last_smooth_pose[2]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # Wrap to [-pi, pi]
        yaw_smooth_rad = self.last_smooth_pose[2] + self.sm_pose_alpha_yaw * angle_diff
        
        self.last_smooth_pose = np.array([x_smooth, y_smooth, yaw_smooth_rad])

        yaw_smooth = np.degrees(yaw_smooth_rad)
        return x_smooth, y_smooth, yaw_smooth


    def smooth_commands(self, v, w):
        """Apply smoothing and rate limiting to velocity commands"""
        if not self.use_smoothing:
            return v, w
            
        current_time = time.time()
        dt = current_time - self.last_cmd_time
        self.last_cmd_time = current_time
        
        current_cmd = np.array([v, w])
        self.cmd_history.append(current_cmd)
        
        if self.last_smooth_cmd is None:
            self.last_smooth_cmd = current_cmd.copy()
            return v, w
        
        # Exponential smoothing
        smoothed_cmd = (self.cmd_alpha * current_cmd + 
                       (1 - self.cmd_alpha) * self.last_smooth_cmd)
        
        # Rate limiting to prevent jerky motion
        if dt > 0 and dt < 1.0:  # Reasonable dt range
            max_change = np.array([self.max_linear_change * dt, 
                                 self.max_angular_change * dt])
            
            cmd_diff = smoothed_cmd - self.last_smooth_cmd
            cmd_diff = np.clip(cmd_diff, -max_change, max_change)
            final_cmd = self.last_smooth_cmd + cmd_diff
        else:
            final_cmd = smoothed_cmd
        
        # Optional: Moving average for additional smoothing
        if len(self.cmd_history) >= 3:
            recent_cmds = list(self.cmd_history)[-3:]
            avg_cmd = np.mean(recent_cmds, axis=0)
            # Blend with rate-limited command
            final_cmd = 0.8 * final_cmd + 0.2 * avg_cmd
        
        self.last_smooth_cmd = final_cmd.copy()
        return final_cmd[0], final_cmd[1]

    def navigate_one_step(self, rgb_img, depth_img, current_obs, odom_msg):
        """Execute one navigation step using NavDP + Place Recognition"""
        try:
            start_time = time.time()
            
            # Keep your existing place recognition (this works well!)
            context = copy.deepcopy(self.context_queue)
            current_obs = context[-1]['img'].to(self.device)
            context = torch.cat([obs['img'] for obs in context], dim=1).to(self.device)

            # Initialize Bayesian filter on first pass
            if (self.first_pass and self.filter_mode == 'bayesian'):
                self.place_recognition_querier.initialize_model(current_obs)
                self.first_pass = False
                
            # Place recognition (keep this - it's your strong point!)
            if self.filter_mode == 'bayesian':
                self.closest_node_idx, _score = self.place_recognition_querier.match(current_obs)
            elif self.filter_mode == 'sliding_window':
                start = max(self.closest_node_idx - self.window_radius, 0)
                end = min(self.closest_node_idx + self.window_radius + 1, self.goal_node_idx + 1)
                self.closest_node_idx = self.place_recognition_querier.match(current_obs, start, end)
            else:
                raise ValueError(f"Filter mode {self.filter_mode} not recognized")
            
            # Determine subgoal
            subgoal_idx = min(self.closest_node_idx + self.lookahead, self.goal_node_idx)
            
            print(f"Frame {self.frame_counter}: Closest node: {self.closest_node_idx}, "
                f"Subgoal: {subgoal_idx}, Goal: {self.goal_node_idx}")

            # ===== NEW: Use NavDP for navigation =====
            if self.fm_method == 'navdp' and self.navdp_enabled:
                v, w = self.navigate_with_navdp(rgb_img, depth_img, subgoal_idx)
            else:
                # Fallback to your existing method
                x,y,yaw, v, w = self.navigate_with_traditional_method(rgb_img, depth_img, subgoal_idx)
            
            
            # Publish commands
            self.publish_cmd_vel(v, w)
            
            # Check if goal is reached
            self.reached_goal = (self.closest_node_idx >= self.goal_node_idx)
            if self.reached_goal:
                print("Goal reached! Stopping navigation.")
                self.publish_cmd_vel(0.0, 0.0)
                self.navigation_active = False

            # Update frame counter
            self.frame_counter += 1
            
            # Print timing info
            loop_duration = time.time() - start_time
            print(f"Processing time: {loop_duration:.3f}s")

        except Exception as e:
            print(f"[ERROR] Navigation step failed: {e}")
            self.publish_cmd_vel(0.0, 0.0)

        # Debug publishing (keep your existing code)
        if self.enable_debug:
            try:
                _, img_encoded = cv2.imencode('.jpg', rgb_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                compressed_msg = CompressedImage()
                compressed_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
                compressed_msg.header.frame_id = "camera_frame"
                compressed_msg.format = "jpeg"
                compressed_msg.data = img_encoded.tobytes()
                
                nav_msg = Float64MultiArray()
                
                if self.navdp_enabled:
                    nav_msg.data = [float(self.frame_counter), float(subgoal_idx), 
                                0.0, 0.0, 0.0, float(v), float(w)]  # No pose data for NavDP
                else:
                    nav_msg.data = [float(self.frame_counter), float(subgoal_idx), 
                                float(x), float(y), float(yaw), float(v), float(w)]
                
                self.debug_image_pub.publish(compressed_msg)
                self.debug_nav_pub.publish(nav_msg)
            except Exception as e:
                print(f"[DEBUG] Publish failed: {e}")

    def navigate_with_navdp(self, rgb_img, depth_img, subgoal_idx):
        """Use NavDP for robust navigation"""
        
        # Strategy 1: Try with approximate goal based on subgoal distance
        steps_ahead = subgoal_idx - self.closest_node_idx
        
        if steps_ahead <= 1:
            # Very close to subgoal - use exploration mode (most robust)
            print("[NavDP] Close to subgoal - using exploration mode")
            trajectory, values = self.navdp_no_goal_step(rgb_img, depth_img)
            
            if trajectory is not None:
                v, w = self.trajectory_to_velocity(trajectory)
                print(f"[NavDP] Exploration: v={v:.3f}, w={w:.3f}")
                return v, w
        
        else:
            # Use point goal based on subgoal distance
            goal_x, goal_y = self.subgoal_to_approximate_goal(subgoal_idx)
            print(f"[NavDP] Using point goal: ({goal_x:.1f}, {goal_y:.1f})")
            
            trajectory, values = self.navdp_get_trajectory_point_goal(rgb_img, depth_img, goal_x, goal_y)
            
            if trajectory is not None:
                v, w = self.trajectory_to_velocity(trajectory)
                print(f"[NavDP] Point goal: v={v:.3f}, w={w:.3f}")
                return v, w
        

    def navigate_with_traditional_method(self, rgb_img, depth_img, subgoal_idx):
        """Fallback to your existing navigation method"""
        
        sg_img = self.topomap_images[subgoal_idx]
        
        try:
            if self.fm_method == 'reloc3r':
                x, y, yaw = feature_match.matching_features_reloc3r_inv(
                    rgb_img, sg_img, self.fm_model, self.img_reso)
                
                x, y, yaw = self.smooth_pose(x, y, yaw)
                
                # Handle negative x (behind robot)
                while x < 0:
                    subgoal_idx = subgoal_idx + 1
                    print(f"[Traditional] subgoal from {subgoal_idx-1} to {subgoal_idx} due to negative x")
                    if subgoal_idx >= len(self.topomap_images):
                        print(f"[WARNING] Subgoal index {subgoal_idx} exceeds topomap length. Stopping.")
                        return 0.0, 0.0
                    
                    sg_img = self.topomap_images[subgoal_idx]
                    x, y, yaw = feature_match.matching_features_reloc3r_inv(
                        rgb_img, sg_img, self.fm_model, self.img_reso)

                if x is None or y is None or yaw is None:
                    print(f"[WARNING] Traditional method failed for frame {self.frame_counter}")
                    self.unvalidPnpCount += 1
                    return 0.0, 0.0

                # Generate control commands
                v, w = control.vtr_controller(x, y, yaw, 
                                            self.robot_config['max_v'], 
                                            self.robot_config['max_w'])
                
                print(f"[Traditional] v={v:.3f}, w={w:.3f}, pose=({x:.2f},{y:.2f},{yaw:.1f})")
                return x,y,yaw, v, w
                
            else:
                # Handle other feature matching methods
                kp1, kp2, matches = self.do_feature_matching(rgb_img, sg_img)
                x, y, yaw = se2_estimate.pnpRansac(kp1, kp2, matches, depth_img, self.K)
                
                if x is None or y is None or yaw is None:
                    print(f"[WARNING] Traditional PnP failed for frame {self.frame_counter}")
                    self.unvalidPnpCount += 1
                    return 0.0, 0.0

                v, w = control.vtr_controller(x, y, yaw, 
                                            self.robot_config['max_v'], 
                                            self.robot_config['max_w'])
                return x,y,yaw,v, w
                
        except Exception as e:
            print(f"[ERROR] Traditional method failed: {e}")
            return 0.0, 0.0

    def do_feature_matching(self, rgb_img, sg_img):
        """Helper function for feature matching"""
        if self.fm_method == 'loftr':
            return feature_match.matching_features_loftr(rgb_img, sg_img, self.fm_model)
        elif self.fm_method == 'roma':
            return feature_match.matching_features_roma(rgb_img, sg_img, self.fm_model)
        elif self.fm_method == 'mast3r':
            return feature_match.matching_features_mast3r(rgb_img, sg_img, self.fm_model)
        elif self.fm_method == 'liftfeat':
            return feature_match.matching_features_liftFeat(rgb_img, sg_img, self.fm_model)
        else:
            raise ValueError(f"Feature matching method {self.fm_method} not supported")

    # def navigate_one_step(self, rgb_img, depth_img, current_obs, odom_msg):
    #     """Execute one navigation step with current sensor data"""
    #     try:
    #         start_time = time.time()
            

    #         # TODO: current_obs for VPR vs. rgb_img for matching
    #         context = copy.deepcopy(self.context_queue)
    #         current_obs = context[-1]['img'].to(self.device)
            
    #         # Cat the context images along the channel dimension
    #         context = torch.cat([obs['img'] for obs in context], dim=1).to(self.device)
    #         # pdb.set_trace()

    #         # Initialize Bayesian filter on first pass
    #         if (self.first_pass and self.filter_mode == 'bayesian'):
    #             self.place_recognition_querier.initialize_model(current_obs)
    #             self.first_pass = False
                
    #         # Place recognition
    #         if self.filter_mode == 'bayesian':
    #             self.closest_node_idx, _score = self.place_recognition_querier.match(current_obs)
    #         elif self.filter_mode == 'sliding_window':
    #             start = max(self.closest_node_idx - self.window_radius, 0)
    #             end = min(self.closest_node_idx + self.window_radius + 1, self.goal_node_idx + 1)
    #             self.closest_node_idx = self.place_recognition_querier.match(current_obs, start, end)
    #         else:
    #             raise ValueError(f"Filter mode {self.filter_mode} not recognized")
            
    #         # Determine subgoal
    #         subgoal_idx = min(self.closest_node_idx + self.lookahead, self.goal_node_idx)
    #         sg_img = self.topomap_images[subgoal_idx]

    #         # Save sg_image for visualization
    #         # imwrite_path = os.path.join("/home/orin2/Repository/GuideNav/src/guidenav/tool", f'subgoal_{subgoal_idx}.jpg')
    #         # cv2.imwrite(imwrite_path, sg_img)
    #         # pdb.set_trace()

    #         print(f"Frame {self.frame_counter}: Closest node: {self.closest_node_idx}, "
    #               f"Subgoal: {subgoal_idx}, Goal: {self.goal_node_idx}")

    #         # Feature matching
    #         if self.fm_method == 'loftr':
    #             kp1, kp2, matches = feature_match.matching_features_loftr(
    #                 rgb_img, sg_img, self.fm_model)
    #         elif self.fm_method == 'roma':
    #             # Note: Roma might need image paths - you may need to save temp images
    #             kp1, kp2, matches = feature_match.matching_features_roma(
    #                 rgb_img, sg_img, self.fm_model)
    #         elif self.fm_method == 'mast3r':
    #             kp1, kp2, matches = feature_match.matching_features_mast3r(
    #                 rgb_img, sg_img, self.fm_model)
    #         elif self.fm_method == 'liftfeat':
    #             kp1, kp2, matches = feature_match.matching_features_liftFeat(
    #                 rgb_img, sg_img, self.fm_model)
    #         elif self.fm_method == 'reloc3r':
    #             x,y,yaw= feature_match.matching_features_reloc3r_inv(
    #                 rgb_img, sg_img, self.fm_model, self.img_reso)

    #             x, y, yaw = self.smooth_pose(x,y,yaw)
                
    #             while x < 0:
    #                 subgoal_idx = subgoal_idx +1
    #                 print(f"[IN LOOP] subgoal from {subgoal_idx-1} to {subgoal_idx} due to negative x")
    #                 if subgoal_idx >= len(self.topomap_images):
    #                     print(f"[WARNING] Subgoal index {subgoal_idx} exceeds topomap image length. Stopping navigation.")
    #                     return
    #                 # Get the new subgoal image
    #                 sg_img = self.topomap_images[subgoal_idx]
    #                 x,y,yaw= feature_match.matching_features_reloc3r_inv(
    #                     rgb_img, sg_img, self.fm_model, self.img_reso)

    #             # REMOVE
    #             # if x < 0:
    #             #     subgoal_idx += 1
    #             #     sg_img = self.topomap_images[subgoal_idx]
    #             #     x,y,yaw= feature_match.matching_features_reloc3r_inv(
    #             #         rgb_img, sg_img, self.fm_model, self.img_reso)
    #         else:
    #             raise ValueError(f"Feature matching method {self.fm_method} not recognized")

    #         # pdb.set_trace()
            
    #         # Pose estimation
    #         try:
    #             if self.fm_method == 'reloc3r':
    #                 # x,y ,yaw = pose2to1[0, 3], pose2to1[1, 3], np.arctan2(pose2to1[1, 0], pose2to1[0, 0])
    #                 if x is None or y is None or yaw is None:
    #                     print(f"[WARNING] Failed to estimate pose for frame {self.frame_counter}")
    #                     self.unvalidPnpCount += 1
    #                     return

    #                 # Generate control commands
    #                 v, w = control.vtr_controller(x, y, yaw, 
    #                                             self.robot_config['max_v'], 
    #                                             self.robot_config['max_w'])
                    
    #                 # Publish velocity command
    #                 self.publish_cmd_vel(v, w)
                    
    #                 # print(f"Control Command: v={v:.3f} m/s, w={w:.3f} rad/s")
    #                 # print(f"Relative pose: x={x:.2f}m, y={y:.2f}m, yaw={yaw:.2f}deg")

    #                 # Save visualization (optional - can be disabled for performance)
    #                 # TODO: Need new plot saving 
    #                 # if hasattr(self, 'save_visualizations') and self.save_visualizations:
    #                 #     self.visualize_and_save_results_no_matching(
    #                 #         rgb_img, self.frame_counter, None, None,
    #                 #         sg_img, subgoal_idx, x, y, yaw, v, w,
    #                 #         self.vis_dir
    #                 #     )
    #             else:
    #                 x, y, yaw = se2_estimate.pnpRansac(kp1, kp2, matches, depth_img, self.K)
                
    #                 if x is None or y is None or yaw is None:
    #                     print(f"[WARNING] Failed to estimate pose for frame {self.frame_counter}")
    #                     self.unvalidPnpCount += 1
    #                     return

    #                 # Generate control commands
    #                 v, w = control.vtr_controller(x, y, yaw, 
    #                                             self.robot_config['max_v'], 
    #                                             self.robot_config['max_w'])
                    
    #                 # Publish velocity command
    #                 self.publish_cmd_vel(v, w)
                    
    #                 print(f"Control Command: v={v:.3f} m/s, w={w:.3f} rad/s")
    #                 print(f"Relative pose: x={x:.2f}m, y={y:.2f}m, yaw={yaw:.2f}deg")

    #                 # Save visualization (optional - can be disabled for performance)
    #                 # if hasattr(self, 'save_visualizations') and self.save_visualizations:
    #                 #     self.visualize_and_save_results(
    #                 #         rgb_img, self.frame_counter, None, depth_img,
    #                 #         sg_img, subgoal_idx, x, y, yaw, v, w,
    #                 #         kp1, kp2, matches, self.unvalidPnpCount, self.vis_dir
    #                 #     )

    #             # Check if goal is reached
    #             self.reached_goal = (self.closest_node_idx >= self.goal_node_idx)
                
    #             if self.reached_goal:
    #                 print("Goal reached! Stopping navigation.")
    #                 self.publish_cmd_vel(0.0, 0.0)  # Stop the robot
    #                 self.navigation_active = False

    #         except Exception as e:
    #             print(f"[ERROR] Pose estimation failed: {e}")
    #             self.unvalidPnpCount += 1
    #             return

    #         # Update frame counter
    #         self.frame_counter += 1
            
    #         # Print timing info
    #         loop_duration = time.time() - start_time
    #         print(f"Processing time: {loop_duration:.3f}s")

    #     except Exception as e:
    #         print(f"[ERROR] Navigation step failed: {e}")
    #         # Send stop command on error
    #         self.publish_cmd_vel(0.0, 0.0)

    #     if self.enable_debug:
    #         try:
    #             # Method 1: Manual compression (more control)
    #             _, img_encoded = cv2.imencode('.jpg', rgb_img, 
    #                                         [cv2.IMWRITE_JPEG_QUALITY, 70])
                
    #             compressed_msg = CompressedImage()
    #             compressed_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
    #             compressed_msg.header.frame_id = "camera_frame"
    #             compressed_msg.format = "jpeg"
    #             compressed_msg.data = img_encoded.tobytes()
                
    #             # Navigation data
    #             nav_msg = Float64MultiArray()
    #             nav_msg.data = [float(self.frame_counter), float(subgoal_idx), 
    #                         float(x), float(y), float(yaw), float(v), float(w)]
                
    #             # Publish
    #             self.debug_image_pub.publish(compressed_msg)
    #             self.debug_nav_pub.publish(nav_msg)
                
    #             print(f"[DEBUG] Published frame {self.frame_counter}")
                
    #         except Exception as e:
    #             print(f"[DEBUG] Publish failed: {e}")
    def start_navigation(self):
        """Start the navigation process"""
        print("[INFO] Starting real-time navigation...")
        self.navigation_active = True
        self.reached_goal = False
        self.frame_counter = 0
        with self.tactile_lock:
            self.tactile_signal_count = 0
            self.tactile_stats_start_time = time.time()

    def stop_navigation(self):
        """Stop the navigation process"""
        print("[INFO] Stopping navigation...")
        self.navigation_active = False
        self.publish_cmd_vel(0.0, 0.0)

    def _load_topomap(self, topomap_images_base_dir, topomap_dir: Path):
        # List the topomap images with suffix img_suffix in the directory,
        # extract the filenames and sort them        
        self.topomap_img_dir = topomap_images_base_dir / topomap_dir

        topomap_images = []
        for img_suffix in self.img_suffix:
            topomap_images.extend(list(self.topomap_img_dir.glob(f"*.{img_suffix}")))
        self.topomap_filenames = [img.name for img in topomap_images]
        self.topomap_filenames = sorted(self.topomap_filenames, key=lambda filename: int(filename.split(".")[0]))

        # Load the topomap images
        map_size = len(self.topomap_filenames)
        topomap_images = []
        for i in range(map_size):
            image_path = self.topomap_img_dir / self.topomap_filenames[i]
            img = read_image(image_path)
            topomap_images.append(img)
            img = self._image_transform(img.astype(np.uint8)).unsqueeze(0)
        self.map_size = map_size
        self.topomap_images = topomap_images

    import pandas as pd

    def load_trajectory_csv(self, csv_path):
        """
        Load ground truth trajectory CSV.

        Args:
            csv_path (str or Path): Path to the trajectory_2d.csv file

        Returns:
            obs_gt_dict: mapping from frame_id (int) to (x, y, yaw)
            topo_gt_dict: mapping from topomap_node_idx (int) to (x, y, yaw)
        """
        df = pd.read_csv(csv_path)

        # Observation ground truth indexed by frame_id
        obs_gt_dict = {
            int(row['frame_id']): (float(row['x']), float(row['y']), float(row['yaw']))
            for _, row in df.iterrows()
        }

        # Topomap ground truth indexed by topomap_node_idx (assume first unique ones)
        topo_gt_dict = {}
        for _, row in df.iterrows():
            node_idx = int(row['topomap_node_idx'])
            if node_idx not in topo_gt_dict:
                topo_gt_dict[node_idx] = (
                    float(row['topomap_node_x']),
                    float(row['topomap_node_y']),
                    float(row['topomap_node_yaw']),
                )

        return obs_gt_dict, topo_gt_dict


    def _setup_place_recognition(
            self,
            model_config_path: Path,
            model_weight_dir: Path,
            place_recognition_model: str,
            filter_mode: str,
            transition_model_window_lower: int,
            transition_model_window_upper: int,
            bayesian_filter_delta: int,
            recompute_db: bool = False,
        ):

        # Load the place recognition model config
        with model_config_path.open(mode="r", encoding="utf-8") as f:
            confs = yaml.safe_load(f)
            conf = confs[place_recognition_model]
        conf['model']["checkpoint_path"] = model_weight_dir / conf['model']["checkpoint_path"]

        # Extract the global descriptors from the topomap images
        place_recognition_db_path = self.topomap_img_dir / f"global-feats-{place_recognition_model}.h5"

        if not place_recognition_db_path.exists():
            print(f"Extracting features from topomaps in {self.topomap_img_dir}")
            extract_database.main(
                conf,
                self.topomap_img_dir,
                self._image_transform,
                self.topomap_img_dir,
                as_half=False,
                )
            
        elif recompute_db:
            print(f"Recomputing features from topomaps in {self.topomap_img_dir}")
            place_recognition_db_path.unlink()
            extract_database.main(
                conf,
                self.topomap_img_dir,
                self._image_transform,
                self.topomap_img_dir,
                as_half=False,
                )

        extractor = FeatureExtractor(conf, self.device) # ready to extract global descriptor (prediction)

        if filter_mode == 'bayesian':
            self.place_recognition_querier = PlaceRecognitionTopologicalFilter(
                extractor,
                place_recognition_db_path,
                self.topomap_img_dir,
                delta=bayesian_filter_delta,
                window_lower=transition_model_window_lower,
                window_upper=transition_model_window_upper,
                )
            
        elif filter_mode == 'sliding_window':
            self.place_recognition_querier = PlaceRecognitionSlidingWindowFilter(
                extractor,
                place_recognition_db_path,
                str(self.topomap_img_dir))
        else:
            raise ValueError(f"Filter mode {filter_mode} not recognized")

    def extract_gt_idx_from_path(self, file_path):
        # Get just the filename without directory
        filename = os.path.basename(file_path)
        
        # Split by underscore and take the first part
        parts = filename.split('_')
        if parts and parts[0].isdigit():
            return int(parts[0])
        else:
            return None

    def nice_text_overlay(self, image, text, position, font_scale=0.7, thickness=2):
        # Make a copy to avoid modifying the original
        img = image.copy()
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Coordinates of the background rectangle
        rect_x, rect_y = position
        rect_w = text_size[0] + 20  # Add padding
        rect_h = text_size[1] + 10  # Add padding
        
        # Create a semi-transparent overlay for better text visibility
        overlay = img.copy()
        cv2.rectangle(overlay, (rect_x, rect_y), 
                     (rect_x + rect_w, rect_y + rect_h), 
                     (0, 0, 0), -1)
        
        # Apply the overlay with transparency
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Add text
        text_x = rect_x + 10  # Add padding
        text_y = rect_y + text_size[1] + 5  # Align with padding
        cv2.putText(img, text, (text_x, text_y), 
                    font, font_scale, (255, 255, 255), thickness)
        
        return img

    def create_error_distribution_plot(self, current_frame_idx=None):
        """Create a plot showing the distribution of prediction errors over time"""
        plt.figure(figsize=(10, 4))
        
        # Calculate cumulative error
        frame_count = len(self.frame_indices)
        if frame_count == 0:
            return None
            
        # Calculate cumulative errors (sum of binary errors)
        cumulative_errors = np.zeros(frame_count)
        for i in range(frame_count):
            if i > 0:
                cumulative_errors[i] = cumulative_errors[i-1] + self.prediction_errors[i]
            else:
                cumulative_errors[i] = self.prediction_errors[i]

        # Plot cumulative error
        plt.plot(self.frame_indices, cumulative_errors, 'b-', linewidth=2)
        
        # Highlight current frame if provided
        if current_frame_idx is not None and current_frame_idx < len(self.frame_indices):
            current_idx = self.frame_indices.index(current_frame_idx)
            plt.plot(current_frame_idx, cumulative_errors[current_idx], 'ro', markersize=8)
        
        plt.title('Cumulative Prediction Errors (closest_node+1 != subgoal_idx)')
        plt.xlabel('Frame Index')
        plt.ylabel('Cumulative Error Count')
        plt.grid(True)
        
        # Calculate error rate percentage
        if self.prediction_errors:
            error_rate = (sum(self.prediction_errors) / len(self.prediction_errors)) * 100
            # plt.figtext(0.5, 0.01, f'Error Rate: {error_rate:.2f}%', ha='center', fontsize=12)
            plt.figtext(0.5, 0.93, f'Error Rate: {error_rate:.2f}%', ha='center', fontsize=12)
            # --- New: Show mean pose error (x, y, yaw) ---
            if hasattr(self, 'pose_errors') and self.pose_errors:
                x_errs = [abs(e['x_err']) for e in self.pose_errors]
                y_errs = [abs(e['y_err']) for e in self.pose_errors]
                yaw_errs = [abs(e['yaw_err']) for e in self.pose_errors]
                mean_x = np.mean(x_errs)
                mean_y = np.mean(y_errs)
                mean_yaw = np.mean(yaw_errs)
                pose_error_text = f'Avg Pose Error: x={mean_x:.2f}m, y={mean_y:.2f}m, yaw={mean_yaw:.2f}deg'
                plt.figtext(0.5, 0.85, pose_error_text, ha='center', fontsize=11)
            plt.figtext(0.5, 0.78, f'unvalid PnP counts: {self.unvalidPnpCount:.2f}', ha='center', fontsize=11)

        plt.tight_layout()
        
        # Save plot to file
        plot_path = os.path.join(self.vis_dir, 'error_distribution.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        return plot_path




    def draw_feature_matches(self, img1, img2, kp1, kp2, matches, max_matches=100):
        """
        Draw feature matches between two images.
        
        Args:
            img1, img2: Input images (numpy arrays)
            kp1, kp2: Keypoints from the feature matching
            matches: List of matches
            max_matches: Maximum number of matches to display
        
        Returns:
            matched_vis: Image with drawn matches
        """
        # Convert keypoints to cv2.KeyPoint format if they aren't already
        if len(kp1) > 0 and not isinstance(kp1[0], cv2.KeyPoint):
            # Convert from numpy array format to cv2.KeyPoint
            cv_kp1 = [cv2.KeyPoint(float(kp[0]), float(kp[1]), 1) for kp in kp1]
            cv_kp2 = [cv2.KeyPoint(float(kp[0]), float(kp[1]), 1) for kp in kp2]
        else:
            cv_kp1 = kp1
            cv_kp2 = kp2
        
        # Convert matches to cv2.DMatch format if needed
        if len(matches) > 0 and not isinstance(matches[0], cv2.DMatch):
            # Assuming matches is a list of indices or coordinate pairs
            cv_matches = []
            for i, match in enumerate(matches[:max_matches]):
                if isinstance(match, (list, tuple)) and len(match) >= 2:
                    # If matches are index pairs
                    cv_matches.append(cv2.DMatch(match[0], match[1], 0, 0))
                else:
                    # If matches are just indices, create sequential matching
                    cv_matches.append(cv2.DMatch(i, i, 0, 0))
        else:
            cv_matches = matches[:max_matches]
        
        # Ensure images are in the right format (BGR for OpenCV)
        if len(img1.shape) == 3 and img1.shape[2] == 3:
            # Already BGR
            draw_img1 = img1.copy()
            draw_img2 = img2.copy()
        # else:
        #     # Convert if needed
        #     draw_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR) if img1.shape[2] == 3 else img1
        #     draw_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR) if img2.shape[2] == 3 else img2
        
        # Draw matches
        matched_vis = cv2.drawMatches(
            draw_img1, cv_kp1, 
            draw_img2, cv_kp2,
            cv_matches, None,
            matchColor=(0, 255, 0),  # Green lines for matches
            singlePointColor=(255, 0, 0),  # Red points for unmatched keypoints
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return matched_vis






if __name__ == '__main__':
    args = parser.parse_args()
    
    
    guidenav_node = GuideNavNode(args)
    guidenav_node.save_visualizations = True  # Optional: control whether to save visualizations

    # Start navigation logic
    guidenav_node.start_navigation()
     
    try:
        if args.offline_images:
            # Use fake RGBD subscriber for offline testing
            print(f"[INFO] Using fake camera with images from: {args.img_dir}")
            rgbd_sub_node = FakeRGBDSubscriber(
                guidenav_node, 
                image_directory=args.img_dir,
                use_odometry=False,  # Match your original setting
                fps=args.offline_fps
            )
            
            # Start streaming after a short delay
            def start_streaming_delayed():
                time.sleep(1.0)  # Give ROS time to initialize
                rgbd_sub_node.start_streaming(loop=True)  # loop=True to repeat images
                
            streaming_thread = threading.Thread(target=start_streaming_delayed)
            streaming_thread.daemon = True
            streaming_thread.start()
            
        else:
            # Use real RGBD subscriber for online testing
            print("[INFO] Using real RealSense camera")
            rgbd_sub_node = RGBDSubscriber(guidenav_node, use_odometry=False)

        # Use MultiThreadedExecutor
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(rgbd_sub_node)
        executor.add_node(guidenav_node.ros_node)

        # Spin the nodes to keep receiving images
        print("[INFO] Starting ROS2 spin. Press Ctrl+C to stop.")
        executor.spin()

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user. Shutting down...")

    finally:
        # Stop fake camera streaming if using it
        if args.offline_images and hasattr(rgbd_sub_node, 'stop_streaming'):
            rgbd_sub_node.stop_streaming()
            
        # Stop robot safely
        guidenav_node.stop_navigation()

        # Save video summary
        output_video_path = os.path.join(guidenav_node.vis_dir, 'guidenav_summary.mp4')
        # guidenav_node.create_video_from_images(guidenav_node.vis_dir, output_video_path)

        # Shutdown ROS
        guidenav_node.ros_node.destroy_node()
        rgbd_sub_node.destroy_node()
        rclpy.shutdown()