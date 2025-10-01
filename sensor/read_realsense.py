import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import os

class RGBDLogger(Node):
    def __init__(self):
        super().__init__('rgbd_logger')

        self.bridge = CvBridge()
        self.rgb_sub = self.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/visual_slam/tracking/vo_pose', self.pose_callback, 10)

        self.rgb_dir = 'rgb'
        self.depth_dir = 'depth'
        self.pose_file = 'poses.txt'
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        self.frame_id = 0
        self.pose_fh = open(self.pose_file, 'w')

    def rgb_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb_path = os.path.join(self.rgb_dir, f'{self.frame_id:06d}.png')
        cv2.imwrite(rgb_path, img)

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')  # 16UC1
        depth_path = os.path.join(self.depth_dir, f'{self.frame_id:06d}.png')
        cv2.imwrite(depth_path, depth)

        # Increment frame only after depth (so both rgb and depth have same index)
        self.frame_id += 1

    def pose_callback(self, msg):
        pos = msg.pose.position
        ori = msg.pose.orientation
        self.pose_fh.write(f"{self.frame_id:06d},{pos.x:.6f},{pos.y:.6f},{pos.z:.6f},"
                           f"{ori.x:.6f},{ori.y:.6f},{ori.z:.6f},{ori.w:.6f}\n")

    def destroy_node(self):
        self.pose_fh.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = RGBDLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

