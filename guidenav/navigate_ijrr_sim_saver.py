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

# debug
import threading
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import CompressedImage

from utils import to_numpy, read_image, read_depth_image, get_image_transform
matplotlib.use('Agg')  # Use non-interactive backend


# smooth behavior
from collections import deque

class GuideNavNode:

    def __init__(self, args: argparse.Namespace):
        if args.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.dev_name = torch.cuda.get_device_name(0)
            print(f"Using CUDA Device: {self.dev_name}")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

        # Get the image preprocessing transform for the neural network models
        self._image_transform = get_image_transform(args.img_size)

        self.img_suffix = ['png', 'jpg', 'jpeg']

        self.pose_errors = []

        # Load the topological map
        self.topomap_filenames = []
        self._load_topomap(args.topomap_base_dir, args.topomap_dir) # started from 4

        print(f"Topomap loaded with {self.map_size} images")


        # self.obs_gt, self.topo_gt = self.load_trajectory_csv(args.gt_csv_path)


        self.image_dir = args.img_dir
        self.context_size = args.context_size if hasattr(args, 'context_size') else 4
        self.context_queue = []
        self.all_image_paths = []

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
        self.use_smoothing = getattr(args, 'use_smoothing', True)
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
        self.tactile_stats_timer = self.ros_node.create_timer(5.0, self.print_tactile_stats)

        self.enable_debug = getattr(args, 'enable_debug', False)
        if self.enable_debug:
            from std_msgs.msg import Float64MultiArray
            from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
            
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

    # def publish_cmd_vel(self, v: float, w: float):
    #     twist = Twist()
    #     twist.linear.x = v
    #     twist.angular.z = w
    #     self.cmd_vel_pub.publish(twist)
    #     self.ros_node.get_logger().info(f"Sent cmd_vel: v={v:.2f}, w={w:.2f}")
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



    def smooth_pose(self, x, y, yaw):
        """Apply exponential smoothing to pose estimates"""
        if not self.use_smoothing:
            return x, y, yaw
            
        current_pose = np.array([x, y, yaw])
        self.pose_hist.append(current_pose)
        
        if self.last_smooth_pose is None:
            self.last_smooth_pose = current_pose.copy()
            return x, y, yaw
        
        # Exponential smoothing
        x_smooth = self.sm_pose_alpha_x * x + (1 - self.sm_pose_alpha_x) * self.last_smooth_pose[0]
        y_smooth = self.sm_pose_alpha_y * y + (1 - self.sm_pose_alpha_y) * self.last_smooth_pose[1]
        
        # Handle angle smoothing properly
        angle_diff = yaw - self.last_smooth_pose[2]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # Wrap to [-pi, pi]
        yaw_smooth = self.last_smooth_pose[2] + self.sm_pose_alpha_yaw * angle_diff
        
        self.last_smooth_pose = np.array([x_smooth, y_smooth, yaw_smooth])
        return x_smooth, y_smooth, yaw_smooth

    def get_sorted_image_files(self):
        """Get sorted image files from directory without processing them."""
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory {self.image_dir} not found")
            
        image_files = [f for f in os.listdir(self.image_dir) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sort numerically
        image_files = sorted(image_files, key=self.extract_number)
        
        if not image_files:
            raise ValueError(f"No image files found in {self.image_dir}")
            
        return image_files

    def extract_number(self, filename):
        import re
        nums = re.findall(r'\d+', filename)
        return tuple(map(int, nums))

    def load_images_for_offline_test(self):
        """Load images from directory and process them."""
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory {self.image_dir} not found")

        image_files = self.get_sorted_image_files() # original sorting
        # print(f"Image ori: {image_files_ori}")
        # image_files = self.get_sorted_image_files_with_topo_idx() # original sorting
        
        # Store full paths for later processing
        self.all_image_paths = [os.path.join(self.image_dir, img_file) for img_file in image_files]
        
        # Initialize context queue with first set of images
        for i in range(min(self.context_size + 1, len(self.all_image_paths))):
            self.process_image(self.all_image_paths[i])

            
        
    def process_image(self, img_path):
        """Process a single image and add it to the context queue."""
        try:
            img = cv2.imread(img_path)
            
            # Transform the image
            img = self._image_transform(img).unsqueeze(0)
            
            # Manage the context queue
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append({'img': img})
            else:            
                self.context_queue.pop(0)
                self.context_queue.append({'img': img})
                
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

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



    def save_pose_plot(self, x_rel, y_rel, yaw_deg, save_path, v,w, margin=1.0, gt_x_rel=None, gt_y_rel=None, gt_yaw_deg=None):
        """
        Plot the estimated target pose relative to robot origin at (0, 0).
        - Forward (0°) is +X.
        - Lateral (left) is +Y.
        - Always show robot at the bottom and adjust axis limits to include target.
        """
        # Convert yaw to radians
        yaw_rad = np.radians(yaw_deg)

        # Compute arrow direction at target location
        arrow_length = 0.5
        dx = arrow_length * np.cos(yaw_rad)
        dy = arrow_length * np.sin(yaw_rad)

        # Determine axis limits dynamically
        x_vals = [0, x_rel]
        y_vals = [0, y_rel]

        if gt_x_rel is not None and gt_y_rel is not None:
            x_vals.append(gt_x_rel)
            y_vals.append(gt_y_rel)

        x_min = min(x_vals) - margin
        x_max = max(x_vals) + margin
        y_min = min(y_vals) - margin
        y_max = max(y_vals) + margin
        # x_min = min(0, x_rel) - margin
        # x_max = max(0, x_rel) + margin
        # y_min = min(0, y_rel) - margin
        # y_max = max(0, y_rel) + margin

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect('equal')

        # Plot the robot at origin
        ax.plot(0, 0, 'bo', label='Robot')

        # Draw arrow from (0,0) to target (optional gray line)
        ax.quiver(0, 0, x_rel, y_rel, angles='xy', scale_units='xy', scale=1, color='gray', width=0.005, alpha=0.3)
        ax.quiver(0, 0, 0.5, 0, angles='xy', scale_units='xy', scale=1, color='blue', width=0.01)

        # Draw the target position and its heading
        ax.plot(x_rel, y_rel, 'ro', label='Target')
        ax.quiver(x_rel, y_rel, dx, dy, angles='xy', scale_units='xy', scale=1, color='red', width=0.01)

        # Annotate the target pose
        ax.text(x_rel, y_rel + 0.1, f"({x_rel:.2f}, {y_rel:.2f})", ha='center')
        ax.text(0, 0 - 0.1, f"v: {v:.2f} m/s \n w: {w:.2f}rad/s", ha='center')

        if gt_x_rel is not None and gt_y_rel is not None and gt_yaw_deg is not None:
            gt_yaw_rad = np.radians(gt_yaw_deg)
            dx_gt = 0.5 * np.cos(gt_yaw_rad)
            dy_gt = 0.5 * np.sin(gt_yaw_rad)
            ax.plot(gt_x_rel, gt_y_rel, 'go', label='GT Target')
            ax.quiver(gt_x_rel, gt_y_rel, dx_gt, dy_gt, angles='xy', scale_units='xy', scale=1,
                    color='green', width=0.01)
            ax.text(gt_x_rel, gt_y_rel + 0.1, f"({gt_x_rel:.2f}, {gt_y_rel:.2f})", ha='center', color='green')


        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X (Forward, meters)")
        ax.set_ylabel("Y (Lateral, meters)")
        ax.set_title("Estimated Relative Target Pose")
        ax.grid(True)
        ax.legend(loc='upper left')

        # Always have robot at bottom (visually)
        ax.set_ylim(bottom=min(0, y_min), top=max(y_max, 0))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def draw_images(self, img1, img2):
        """
        Draw two images side by side, matching cv2.drawMatches() layout exactly.
        
        Args:
            img1, img2: Input images (numpy arrays in BGR format)
        
        Returns:
            combined_img: Image with both images placed side by side (same layout as cv2.drawMatches)
        """
        # Get image dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Calculate dimensions for combined image (same as cv2.drawMatches)
        max_height = max(h1, h2)
        total_width = w1 + w2
        
        # Create combined image with black background
        combined_img = np.zeros((max_height, total_width, 3), dtype=np.uint8)
        
        # Place first image (top-left aligned)
        combined_img[0:h1, 0:w1] = img1
        
        # Place second image (top-left aligned, offset by first image width)
        combined_img[0:h2, w1:w1+w2] = img2
        
        # Optional: Add a vertical line separator between images
        separator_x = w1
        cv2.line(combined_img, (separator_x, 0), (separator_x, max_height), (255, 255, 255), 1)
        
        return combined_img


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
        else:
            # Convert if needed
            draw_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR) if img1.shape[2] == 3 else img1
            draw_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR) if img2.shape[2] == 3 else img2
        
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

    def visualize_and_save_results_no_matching(self, current_obs, current_idx, gt_idx, current_obs_depth, topomap_img, topomap_idx, x, y, yaw, v,w, output_dir):
        """Save visualization of current observation and matched topomap image side by side."""
        os.makedirs(output_dir, exist_ok=True)

        # Convert to BGR for OpenCV
        current_obs_np = current_obs
        subgoal_img = topomap_img
        # current_obs_np = cv2.cvtColor(current_obs_np, cv2.COLOR_RGB2BGR)
        # subgoal_img = cv2.cvtColor(subgoal_img, cv2.COLOR_RGB2BGR)
        


        pose_plot_path = os.path.join(output_dir, f"relpose_{current_idx:04d}_{topomap_idx:04d}.jpg")
        self.save_pose_plot(x, y, yaw,pose_plot_path ,v,w, 1.0)


        feature_matches_img = self.draw_images(
            current_obs_np, subgoal_img
        )
        
        gt_text = f" (GT: {gt_idx})" if gt_idx is not None else ""
        current_text = f"Observation ({current_idx}): {gt_text}"
        topomap_text = f"Topomap Node: {topomap_idx}"

        
        feature_matches_img = self.nice_text_overlay(feature_matches_img, current_text, (10, 10))
        feature_matches_img = self.nice_text_overlay(feature_matches_img, topomap_text, (10, 50))


        if gt_idx is not None and current_idx in self.obs_gt and topomap_idx in self.topo_gt:
            # Get current robot pose and matched GT node pose (both world)
            x_cur, y_cur, yaw_cur = self.obs_gt[current_idx]
            x_gt, y_gt, yaw_gt = self.topo_gt[topomap_idx]

            # ---- [1] For GT visualization (relative to robot) ----
            dx = x_gt - x_cur
            dy = y_gt - y_cur
            cos_yaw = np.cos(yaw_cur)
            sin_yaw = np.sin(yaw_cur)
            gt_x_rel =  cos_yaw * dx + sin_yaw * dy
            gt_y_rel = -sin_yaw * dx + cos_yaw * dy
            gt_yaw_rel = (yaw_gt - yaw_cur + np.pi) % (2 * np.pi) - np.pi
            gt_yaw_deg = np.degrees(gt_yaw_rel)

            self.save_pose_plot(x, y, yaw, pose_plot_path, v, w,
                                gt_x_rel=gt_x_rel, gt_y_rel=gt_y_rel, gt_yaw_deg=gt_yaw_deg)

            # ---- [2] For pose estimation error (in world frame) ----
            # dx_err = x_cur - x_gt
            # dy_err = y_cur - y_gt
            # dyaw_err = np.degrees((yaw_cur - yaw_gt + np.pi) % (2 * np.pi) - np.pi)
            dx_err = x - gt_x_rel
            dy_err = y - gt_y_rel
            dyaw_err = (yaw - gt_yaw_deg + 180) % 360 - 180  # Normalize to [-180, 180]

            self.pose_errors.append({
                'idx': current_idx,
                'x_err': dx_err,
                'y_err': dy_err,
                'yaw_err': dyaw_err,
            })

            # ---- [3] Node-level (topological) error ----
            error = abs(topomap_idx - gt_idx)
            error_text = f"Error: {error} nodes"
            position_y = feature_matches_img.shape[0] - 50
            feature_matches_img = self.nice_text_overlay(feature_matches_img, error_text, (10, position_y))

        else:
            self.save_pose_plot(x, y, yaw, pose_plot_path, v, w)

        pose_img = cv2.imread(pose_plot_path)
        matches_h, matches_w = feature_matches_img.shape[:2]
        pose_h, pose_w = pose_img.shape[:2]
        
        new_pose_w = matches_w
        new_pose_h = int(pose_h * (new_pose_w / pose_w))
        pose_img_rszed = cv2.resize(pose_img, (new_pose_w, new_pose_h))

        # stacked_img = np.vstack([combined_img, pose_img])
        stacked_img = np.vstack([feature_matches_img, pose_img_rszed])

        output_path = os.path.join(output_dir, f"match_{current_idx:04d}_{topomap_idx:04d}.jpg")
        cv2.imwrite(output_path, stacked_img)

        return output_path


    def navigate(self, args: argparse.Namespace):
        """
            Load data
            Init loc based on bayesian filter
            Get closest node, subgoal node 
            Feature matching
            Estimate relative pose
            Send velocity command to robot
        """
        print("[INFO] Start navigation")
        self.load_images_for_offline_test()
        print("[INFO] Offline observationn data loaded")

        next_image_idx = self.context_size + 1
        total_images = len(self.all_image_paths)
        
        closest_node_idx = args.start_node_idx
        assert -1 <= args.goal_node_idx < self.map_size, "Invalid goal index"
        if args.goal_node_idx == -1:
            args.goal_node_idx = self.map_size - 1

        # Initialize iterative average calculation for runtime
        avg_runtime = 0
        avg_runtime_count = 0
        obs_idx = self.context_size

        first_pass = True
        reached_goal = False

        while len(self.context_queue) >= self.context_size + 1 and (not reached_goal) or (obs_idx < args.goal_node_idx):   
            start_time = time.time()

            context = copy.deepcopy(self.context_queue)
            
            current_obs = context[-1]['img'].to(self.device)
            # Cat the context images along the channel dimension
            context = torch.cat([obs['img'] for obs in context], dim=1).to(self.device)

            if (first_pass and args.subgoal_mode == 'place_recognition' and args.filter_mode == 'bayesian'):
                # Initialize the belief distribution of the Bayesian filter prior to first query
                self.place_recognition_querier.initialize_model(current_obs)
                first_pass = False
                
            ############### If subgoal mode is place recognition #################
            # Place recognition with Bayesian filter
            if args.filter_mode == 'bayesian':
                # start_pr_match_time = time.time()
                closest_node_idx, _score = self.place_recognition_querier.match(current_obs) # output index of subgoal node and probability of subgoal
                # end_pr_match_time = time.time()
                # print(f"[INFO] Place recognition match time: {end_pr_match_time - start_pr_match_time:.2f} seconds")

            # Place recognition with
            elif args.filter_mode == 'sliding_window':
                start = max(closest_node_idx - args.window_radius, 0)
                end = min(closest_node_idx + args.window_radius +1, args.goal_node_idx +1)
                closest_node_idx = self.place_recognition_querier.match(current_obs, start, end)

            else:
                raise ValueError(f"Filter mode {args.filter_mode} not recognized for subgoal mode {args.subgoal_mode}")
            
            # Closest node index from VPR matching
            self.closest_node_idx = closest_node_idx 
            
            # How many nodes ahead from closest node to choose the subgoal
            subgoal_idx = min(closest_node_idx + args.lookahead, args.goal_node_idx)
            sg_img = self.topomap_images[subgoal_idx]

            print(f"Closest node index: {closest_node_idx}, subgoal index: {subgoal_idx}, goal index: {args.goal_node_idx}")

            # get current obs image & observation depth image - change for robot
            current_obs_path = self.all_image_paths[next_image_idx - 1]
            current_obs_img_raw = read_image(current_obs_path)
            if self.fm_method != 'reloc3r':
                current_obs_depth_np = read_depth_image(
                    self.all_image_paths[next_image_idx - 1].replace("/obs/", "/obs_depth/").replace(".jpg", ".png")
                )

            gt_idx = self.extract_gt_idx_from_path(current_obs_path)
            # gt_idx = self.extract_gt_idx_from_path(current_obs_path)
            # pdb.set_trace()

            # feature matching
            if self.fm_method == 'loftr':
                kp1, kp2, matches = feature_match.matching_features_loftr(
                    current_obs_img_raw,
                    sg_img,  # Convert to HWC format
                    self.fm_model
                )
            
            elif self.fm_method == 'roma':
                kp1, kp2, matches = feature_match.matching_features_roma(
                    self.all_image_paths[next_image_idx - 1],  # Current observation image path
                    self.topomap_img_dir / self.topomap_filenames[subgoal_idx],  # Subgoal image path
                    self.fm_model
                )

            elif self.fm_method == 'mast3r':
                kp1, kp2, matches = feature_match.matching_features_mast3r(
                    self.all_image_paths[next_image_idx - 1],  # Current observation image path
                    self.topomap_img_dir / self.topomap_filenames[subgoal_idx],  # Subgoal image path
                    self.fm_model
                )

            elif self.fm_method == 'liftfeat':
                kp1, kp2, matches = feature_match.matching_features_liftFeat(
                    current_obs_img_raw,
                    sg_img,  # Convert to HWC format
                    self.fm_model
                )
            elif self.fm_method == 'reloc3r':
                x,y,yaw = feature_match.matching_features_reloc3r_inv(
                    current_obs_img_raw,
                    sg_img,  # Convert to HWC format
                    self.fm_model, self.img_reso)

                x, y, yaw = self.smooth_pose(x,y,yaw)
                
                while x < 0:

                    subgoal_idx = subgoal_idx +1
                    print(f"subgoal from {subgoal_idx-1} to {subgoal_idx} due to negative x")
                    if subgoal_idx >= len(self.topomap_images):
                        print(f"[WARNING] Subgoal index {subgoal_idx} exceeds topomap image length. Stopping navigation.")
                        return
                    # Get the new subgoal image
                    sg_img = self.topomap_images[subgoal_idx]
                    x,y,yaw= feature_match.matching_features_reloc3r_inv(
                        current_obs_img_raw, sg_img, self.fm_model, self.img_reso)


                # REMOVE
                    # if x < 0:
                    #     x = 0.01
            else:
                raise ValueError(f"Feature matching method {self.fm_method} not recognized")
                    
            if self.fm_method == 'reloc3r':
                # x,y ,yaw = pose2to1[0, 3], pose2to1[1, 3], np.arctan2(pose2to1[1, 0], pose2to1[0, 0])
                if x is None or y is None or yaw is None:
                    print(f"[WARNING] Failed to estimate pose for frame {self.frame_counter}")
                    self.unvalidPnpCount += 1
                    return

                # Generate control commands
                v, w = control.vtr_controller(x, y, yaw, 
                                            self.robot_config['max_v'], 
                                            self.robot_config['max_w'])
                
                # Publish velocity command
                # self.publish_cmd_vel(v, w)
                
                print(f"Control Command: v={v:.3f} m/s, w={w:.3f} rad/s")
                print(f"Relative pose: x={x:.2f}m, y={y:.2f}m, yaw={yaw:.2f}deg")

                # Save visualization (optional - can be disabled for performance)
                # TODO: Need new plot saving 
                self.visualize_and_save_results_no_matching(
                    current_obs_img_raw, next_image_idx-1, None, None,
                    sg_img, subgoal_idx, x, y, yaw, v, w,
                    self.vis_dir
                )
            else:
                x, y, yaw = se2_estimate.pnpRansac(kp1, kp2, matches, current_obs_depth_np, self.K)
            
                if x is None or y is None or yaw is None:
                    print(f"[WARNING] Failed to estimate pose for frame {self.frame_counter}")
                    self.unvalidPnpCount += 1
                    return

                # Generate control commands
                v, w = control.vtr_controller(x, y, yaw, 
                                            self.robot_config['max_v'], 
                                            self.robot_config['max_w'])
                
                # Publish velocity command
                # self.publish_cmd_vel(v, w)
                
                print(f"Control Command: v={v:.3f} m/s, w={w:.3f} rad/s")
                print(f"Relative pose: x={x:.2f}m, y={y:.2f}m, yaw={yaw:.2f}deg")


                # self.visualize_and_save_results( 
                #     current_obs_img_raw,
                #     # current_obs, 
                #     next_image_idx - 1,  # Current image index
                #     gt_idx,
                #     # sg_img, 
                #     current_obs_depth_np,
                #     sg_img,  # Convert to HWC format
                #     subgoal_idx, 
                #     x, y, yaw,v,w,
                #     kp1, kp2, matches,
                #     self.unvalidPnpCount,
                #     self.vis_dir
                # )
            #####################################################################

            # Check if the goal has been reached
            reached_goal = closest_node_idx == args.goal_node_idx

            # Update the average runtime
            loop_duration = time.time() - start_time
            avg_runtime = (avg_runtime * avg_runtime_count + loop_duration) / (avg_runtime_count + 1)
            avg_runtime_count += 1

            if reached_goal:
                print("Reached goal. Stopping...")
            
            self.context_queue.pop(0)
            if next_image_idx < total_images:
                self.process_image(self.all_image_paths[next_image_idx])
                next_image_idx += 1
            
            obs_idx += 1
        
        # After navigation is complete, create a video from all saved images (offline)
        print("Creating video from saved images...")
        video_path = os.path.join(self.vis_dir, "place_recognition_results.mp4")
        self.create_video_from_images(self.vis_dir, video_path, fps=5)
        print(f"Video saved to {video_path}")

    def create_video_from_images(self, image_dir, output_video_path, fps=10):
        # Get all jpg files in the directory
        images = sorted(glob.glob(os.path.join(image_dir, "match_*.jpg")))
        
        if not images:
            print(f"No images found in {image_dir}")
            return
        
        # Read the first image to get dimensions
        frame = cv2.imread(images[0])
        height, width, _ = frame.shape
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Add each image to the video
        print(f"Creating video from {len(images)} images...")
        for image_path in tqdm(images):
            frame = cv2.imread(image_path)
            video_writer.write(frame)
        
        # Create a final summary plot
        final_plot_path = self.create_error_distribution_plot()
        
        # Add summary frame at the end for a few seconds if available
        if final_plot_path and os.path.exists(final_plot_path):
            # Create a frame with "Final Summary" text above the plot
            summary_img = cv2.imread(final_plot_path)
            summary_img = cv2.resize(summary_img, (width, height))
            
            # Add a title to the summary frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(summary_img, "Final Error Summary", (width//2 - 150, 50), 
                       font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add the summary frame multiple times (3 seconds worth)
            for _ in range(fps * 3):
                video_writer.write(summary_img)
        
        # Release the video writer
        video_writer.release()
        print(f"Video saved to {output_video_path}")

if __name__ == '__main__':
    args = parser.parse_args()

    guidenav_node = GuideNavNode(args)
    guidenav_node.navigate(args)