#!/usr/bin/env python3
"""
THIS IS NAIVE TOPOMAP BUILDER - Original HRI'26 uses DINOv3 embeddings (/topogen/gen_dinov3.py)
Keyframe extraction script for RGB-D data with odometry filtering.
Extracts aligned keyframes based on movement thresholds.

Usage:
    python build_topomap.py /path/to/input /path/to/output --distance 1.0
"""

import os
import csv
import shutil
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Optional

def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Convert quaternion to yaw angle in radians."""
    # Convert quaternion to yaw (rotation around z-axis)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def euclidean_distance_2d(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate 2D Euclidean distance between two positions."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def angle_difference(angle1: float, angle2: float) -> float:
    """Calculate the absolute difference between two angles in radians."""
    diff = abs(angle1 - angle2)
    return min(diff, 2 * np.pi - diff)  # Handle wrap-around

def find_closest_timestamp(target_timestamp: float, available_timestamps: List[str], 
                          max_time_diff: float = 0.1) -> Optional[str]:
    """Find the closest timestamp within max_time_diff seconds."""
    closest_timestamp = None
    min_diff = float('inf')
    
    for timestamp_str in available_timestamps:
        timestamp = float(timestamp_str)
        diff = abs(timestamp - target_timestamp)
        if diff < min_diff and diff <= max_time_diff:
            min_diff = diff
            closest_timestamp = timestamp_str
    
    return closest_timestamp

def load_odometry_data(odom_file: str) -> Dict[float, Dict]:
    """Load odometry data from CSV file."""
    odom_data = {}
    
    with open(odom_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = float(row['timestamp'])
            odom_data[timestamp] = {
                'pos_x': float(row['pos_x']),
                'pos_y': float(row['pos_y']),
                'pos_z': float(row['pos_z']),
                'ori_x': float(row['ori_x']),
                'ori_y': float(row['ori_y']),
                'ori_z': float(row['ori_z']),
                'ori_w': float(row['ori_w']),
                'lin_vel_x': float(row['lin_vel_x']),
                'lin_vel_y': float(row['lin_vel_y']),
                'lin_vel_z': float(row['lin_vel_z']),
                'ang_vel_x': float(row['ang_vel_x']),
                'ang_vel_y': float(row['ang_vel_y']),
                'ang_vel_z': float(row['ang_vel_z'])
            }
    
    return odom_data

def get_image_timestamps(image_dir: str) -> List[str]:
    """Extract timestamps from image filenames as strings to preserve precision."""
    timestamps = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            # Extract timestamp from filename (remove .png extension)
            timestamp_str = filename[:-4]
            try:
                # Validate it's a number but keep as string
                float(timestamp_str)
                timestamps.append(timestamp_str)
            except ValueError:
                print(f"Warning: Could not parse timestamp from {filename}")
    
    return sorted(timestamps, key=float)

def extract_keyframes(input_dir: str, output_dir: str, 
                     distance_threshold: float = 0.005,  # 0.5 cm in meters
                     yaw_threshold: float = np.radians(15),  # 15 degrees in radians
                     max_time_diff: float = 0.1):
    """
    Extract keyframes based on movement thresholds.
    
    Args:
        input_dir: Input directory containing /rgb, /depth, and odom.csv
        output_dir: Output directory to save keyframes
        distance_threshold: Minimum 2D distance in meters (default: 0.005m = 0.5cm)
        yaw_threshold: Minimum yaw difference in radians (default: 15 degrees)
        max_time_diff: Maximum time difference for timestamp matching (default: 0.1s)
    """
    
    # Setup paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    rgb_input_dir = input_path / 'color'
    depth_input_dir = input_path / 'depth'
    odom_input_file = input_path / 'odom.csv'
    
    rgb_output_dir = output_path / 'color'
    depth_output_dir = output_path / 'depth'
    topo_output_dir = output_path / 'topo'
    odom_output_file = output_path / 'odom.csv'
    
    # Create output directories
    for dir_path in [rgb_output_dir, depth_output_dir, topo_output_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading odometry data...")
    odom_data = load_odometry_data(str(odom_input_file))
    
    print("Getting image timestamps...")
    rgb_timestamps = get_image_timestamps(str(rgb_input_dir))
    depth_timestamps = get_image_timestamps(str(depth_input_dir))
    
    print(f"Found {len(rgb_timestamps)} RGB images and {len(depth_timestamps)} depth images")
    
    # Find aligned timestamps (RGB images that have corresponding depth and odometry)
    aligned_data = []
    
    for rgb_timestamp_str in rgb_timestamps:
        rgb_timestamp = float(rgb_timestamp_str)
        
        # Find closest depth timestamp
        depth_timestamp_str = find_closest_timestamp(rgb_timestamp, depth_timestamps, max_time_diff)
        if depth_timestamp_str is None:
            continue
            
        # Find closest odometry timestamp
        odom_timestamp = find_closest_timestamp(rgb_timestamp, list(odom_data.keys()), max_time_diff)
        if odom_timestamp is None:
            continue
            
        aligned_data.append({
            'rgb_timestamp': rgb_timestamp_str,
            'depth_timestamp': depth_timestamp_str,
            'odom_timestamp': odom_timestamp,
            'odom_data': odom_data[odom_timestamp]
        })
    
    print(f"Found {len(aligned_data)} aligned data points")
    
    if not aligned_data:
        print("No aligned data found. Check your input files and timestamps.")
        return
    
    # Filter keyframes based on movement thresholds
    keyframes = []
    last_keyframe_pos = None
    last_keyframe_yaw = None
    accumulated_distance = 0.0
    accumulated_yaw_diff = 0.0
    last_pos = None
    last_yaw = None
    topo_index = 0
    
    for data in aligned_data:
        odom = data['odom_data']
        current_pos = (odom['pos_x'], odom['pos_y'])
        current_yaw = quaternion_to_yaw(odom['ori_x'], odom['ori_y'], 
                                       odom['ori_z'], odom['ori_w'])
        
        # Always include the first frame
        if last_keyframe_pos is None:
            include_frame = True
        else:
            # Accumulate distance and rotation since last frame
            if last_pos is not None:
                frame_distance = euclidean_distance_2d(current_pos, last_pos)
                frame_yaw_diff = angle_difference(current_yaw, last_yaw)
                accumulated_distance += frame_distance
                accumulated_yaw_diff += frame_yaw_diff
            
            # Check if accumulated movement exceeds thresholds
            include_frame = (accumulated_distance >= distance_threshold or 
                           accumulated_yaw_diff >= yaw_threshold)
        
        if include_frame:
            keyframes.append({
                **data,
                'topo_index': topo_index
            })
            # Reset reference point and accumulators
            last_keyframe_pos = current_pos
            last_keyframe_yaw = current_yaw
            accumulated_distance = 0.0
            accumulated_yaw_diff = 0.0
            topo_index += 1
            print(f"Keyframe {topo_index-1}: pos=({current_pos[0]:.3f}, {current_pos[1]:.3f}), "
                  f"yaw={np.degrees(current_yaw):.1f}°")
        
        # Update last position for next iteration
        last_pos = current_pos
        last_yaw = current_yaw
    
    print(f"Selected {len(keyframes)} keyframes")
    
    # Save keyframes
    output_odom_data = []
    
    for i, keyframe in enumerate(keyframes):
        rgb_timestamp = keyframe['rgb_timestamp']
        depth_timestamp = keyframe['depth_timestamp']
        odom_timestamp = keyframe['odom_timestamp']
        topo_idx = keyframe['topo_index']
        
        # Copy RGB image
        rgb_src = rgb_input_dir / f"{rgb_timestamp}.png"
        rgb_dst = rgb_output_dir / f"{rgb_timestamp}.png"
        topo_dst = topo_output_dir / f"{topo_idx}.png"
        
        if rgb_src.exists():
            shutil.copy2(rgb_src, rgb_dst)
            shutil.copy2(rgb_src, topo_dst)
        else:
            print(f"Warning: RGB image {rgb_src} not found")
            continue
        
        # Copy depth image
        depth_src = depth_input_dir / f"{depth_timestamp}.png"
        depth_dst = depth_output_dir / f"{depth_timestamp}.png"
        
        if depth_src.exists():
            shutil.copy2(depth_src, depth_dst)
        else:
            print(f"Warning: Depth image {depth_src} not found")
            continue
        
        # Store odometry data for output CSV
        odom = keyframe['odom_data']
        output_odom_data.append({
            'timestamp': odom_timestamp,
            'pos_x': odom['pos_x'],
            'pos_y': odom['pos_y'],
            'pos_z': odom['pos_z'],
            'ori_x': odom['ori_x'],
            'ori_y': odom['ori_y'],
            'ori_z': odom['ori_z'],
            'ori_w': odom['ori_w'],
            'lin_vel_x': odom['lin_vel_x'],
            'lin_vel_y': odom['lin_vel_y'],
            'lin_vel_z': odom['lin_vel_z'],
            'ang_vel_x': odom['ang_vel_x'],
            'ang_vel_y': odom['ang_vel_y'],
            'ang_vel_z': odom['ang_vel_z']
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(keyframes)} keyframes")
    
    # Save filtered odometry data
    with open(odom_output_file, 'w', newline='') as f:
        fieldnames = ['timestamp', 'pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z', 'ori_w',
                     'lin_vel_x', 'lin_vel_y', 'lin_vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_odom_data)
    
    print(f"\nKeyframe extraction completed!")
    print(f"Saved {len(keyframes)} keyframes to {output_dir}")
    print(f"Output structure:")
    print(f"  - {rgb_output_dir}: Color images with original timestamps")
    print(f"  - {depth_output_dir}: Depth images with original timestamps") 
    print(f"  - {topo_output_dir}: Color images renamed as 0.png, 1.png, ...")
    print(f"  - {odom_output_file}: Filtered odometry data")

def main():
    parser = argparse.ArgumentParser(description='Extract keyframes from RGB-D data with odometry')
    parser.add_argument('input_dir', help='Input directory containing /color, /depth, and odom.csv')
    parser.add_argument('output_dir', help='Output directory for keyframes')
    parser.add_argument('--distance', type=float, default=0.5, 
                       help='Distance threshold in meters (default: 0.5m = 0.5cm)')
    parser.add_argument('--yaw', type=float, default=15, 
                       help='Yaw threshold in degrees (default: 15 degrees)')
    parser.add_argument('--max-time-diff', type=float, default=0.1,
                       help='Maximum time difference for timestamp matching in seconds (default: 0.1s)')
    
    args = parser.parse_args()
    
    # Convert yaw from degrees to radians
    yaw_threshold_rad = np.radians(args.yaw)
    
    extract_keyframes(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        distance_threshold=args.distance,
        yaw_threshold=yaw_threshold_rad,
        max_time_diff=args.max_time_diff
    )

if __name__ == '__main__':
    main()