"""
ROS2 node for extracting synchronized RGB and depth images with odometry.

Usage:
    ros2 run guidenav extract_data_two --output-dir /path/to/save
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import cv2
from datetime import datetime
import csv
import argparse
from nav_msgs.msg import Odometry


def parse_args():
    parser = argparse.ArgumentParser(description="Extract RGB-D data from ROS2 topics")
    parser.add_argument("--output-dir", "-o", type=str, default="./data_output",
                       help="Base output directory for extracted data")
    return parser.parse_args()


class SimpleImageSaver(Node):
    def __init__(self, output_base_dir):
        super().__init__('simple_image_saver')
        self.bridge = CvBridge()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = os.path.join(output_base_dir, timestamp)
        
        # Create directories
        self.d435_color_dir = os.path.join(self.base_dir, 'd435_color')
        self.d435i_color_dir = os.path.join(self.base_dir, 'd435i_color')
        self.depth_dir = os.path.join(self.base_dir, 'depth')
        self.odom_csv_path = os.path.join(self.base_dir, 'odom.csv')

        os.makedirs(self.d435_color_dir, exist_ok=True)
        os.makedirs(self.d435i_color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        # Simple individual subscribers - no synchronization
        self.create_subscription(Image, '/d435/color/image_raw', self.d435_callback, 10)
        self.create_subscription(Image, '/d435i/color/image_raw', self.d435i_callback, 10)
        self.create_subscription(Image, '/d435i/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self.odom_callback, 10)

        # Setup CSV for odometry
        self.odom_file = open(self.odom_csv_path, 'w', newline='')
        self.odom_writer = csv.writer(self.odom_file)
        self.odom_writer.writerow([
            'timestamp', 
            'pos_x', 'pos_y', 'pos_z',
            'ori_x', 'ori_y', 'ori_z', 'ori_w',
            'lin_vel_x', 'lin_vel_y', 'lin_vel_z',
            'ang_vel_x', 'ang_vel_y', 'ang_vel_z'
        ])

        # Counters
        self.d435_count = 0
        self.d435i_count = 0
        self.depth_count = 0
        self.odom_count = 0
        
        # Status timer
        self.create_timer(5.0, self.status_callback)
        
        self.get_logger().info(f"Simple extractor ready - saving to {self.base_dir}")

    def d435_callback(self, msg):
        try:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            ts_str = f"{timestamp:.9f}"
            
            # Convert and save image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            image_path = os.path.join(self.d435_color_dir, f"{ts_str}.png")
            cv2.imwrite(image_path, cv_image)
            
            self.d435_count += 1
            
        except Exception as e:
            self.get_logger().error(f"Error in d435_callback: {e}")

    def d435i_callback(self, msg):
        try:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            ts_str = f"{timestamp:.9f}"
            
            # Convert and save image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            image_path = os.path.join(self.d435i_color_dir, f"{ts_str}.png")
            cv2.imwrite(image_path, cv_image)
            
            self.d435i_count += 1
            
        except Exception as e:
            self.get_logger().error(f"Error in d435i_callback: {e}")

    def depth_callback(self, msg):
        try:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            ts_str = f"{timestamp:.9f}"
            
            # Convert and save depth image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_path = os.path.join(self.depth_dir, f"{ts_str}.png")
            cv2.imwrite(depth_path, depth_image)
            
            self.depth_count += 1
            
        except Exception as e:
            self.get_logger().error(f"Error in depth_callback: {e}")

    def odom_callback(self, msg):
        try:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            ts_str = f"{timestamp:.9f}"
            
            # Save odometry data
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            lv = msg.twist.twist.linear
            av = msg.twist.twist.angular
            
            self.odom_writer.writerow([
                ts_str,
                p.x, p.y, p.z,
                o.x, o.y, o.z, o.w,
                lv.x, lv.y, lv.z,
                av.x, av.y, av.z
            ])
            self.odom_file.flush()
            
            self.odom_count += 1
            
        except Exception as e:
            self.get_logger().error(f"Error in odom_callback: {e}")

    def status_callback(self):
        """Print status every 5 seconds"""
        total = self.d435_count + self.d435i_count + self.depth_count + self.odom_count
        self.get_logger().info(
            f"Saved - D435: {self.d435_count}, D435i: {self.d435i_count}, "
            f"Depth: {self.depth_count}, Odom: {self.odom_count} (Total: {total})"
        )

    def destroy_node(self):
        total = self.d435_count + self.d435i_count + self.depth_count + self.odom_count
        self.get_logger().info(f"Final count: {total} messages saved")
        if hasattr(self, 'odom_file'):
            self.odom_file.close()
        super().destroy_node()


def main():
    args = parse_args()
    rclpy.init()
    node = None
    try:
        node = SimpleImageSaver(args.output_dir)
        node.get_logger().info("Simple ImageSaver node started.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Exception in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()