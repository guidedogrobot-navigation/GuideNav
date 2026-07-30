#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
import io
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from pathlib import Path
import time

class DebugSubscriber(Node):
    def __init__(self, topo_images_dir, output_dir="debug_results", timestamped=True):
        super().__init__('debug_subscriber')
        self.bridge = CvBridge()

        if timestamped:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(output_dir) / f"nav_run_{timestamp}"
        else:
            # Save directly into output_dir (no nav_run_<ts> subdir).
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Saving visualizations to: {self.output_dir}")
        
        # Load topo images
        self.topo_images = self.load_topo_images(topo_images_dir)
        print(f"[DEBUG] Loaded {len(self.topo_images)} topo images")
        
        # Storage for syncing image + nav data
        self.latest_nav_data = None
        
        # QoS matching robot
        fast_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.image_sub = self.create_subscription(
            CompressedImage, '/debug/image/compressed', 
            self.image_callback, fast_qos)
        self.nav_sub = self.create_subscription(
            Float64MultiArray, '/debug/nav_data',
            self.nav_callback, fast_qos)
        
        print("[DEBUG] Subscriber ready")
    
    def load_topo_images(self, topo_dir):
        """Load topological map images"""
        topo_path = Path(topo_dir)
        images = {}
        for suffix in ['png', 'jpg', 'jpeg']:
            for img_path in topo_path.glob(f"*.{suffix}"):
                idx = int(img_path.stem)
                images[idx] = cv2.imread(str(img_path))
        return images
    
    def nav_callback(self, msg):
        """Store latest navigation data"""
        # [frame_id, topo_idx, x, y, yaw, v, w]
        self.latest_nav_data = {
            'frame_id': int(msg.data[0]),
            'topo_idx': int(msg.data[1]),
            'x': msg.data[2], 'y': msg.data[3], 'yaw': msg.data[4],
            'v': msg.data[5], 'w': msg.data[6]
        }
    
    def image_callback(self, compressed_msg):
        """Process received debug image"""
        try:
            # Decompress image
            current_img = self.bridge.compressed_imgmsg_to_cv2(compressed_msg)
            
            # Get navigation data
            if self.latest_nav_data is None:
                return
            
            nav = self.latest_nav_data
            topo_idx = nav['topo_idx']
            
            # Get topo image
            if topo_idx not in self.topo_images:
                print(f"[DEBUG] Topo index {topo_idx} not found")
                return
            
            topo_img = self.topo_images[topo_idx]
            
            # Create debug visualization
            self.create_debug_visualization(current_img, topo_img, nav)
            
        except Exception as e:
            print(f"[DEBUG] Processing error: {e}")
    
    # ---- CVPR-demo visual style ---------------------------------------------
    # BGR colors (observation=azure blue, subgoal=green) used consistently on
    # the panels and in the pose plot.
    ACCENT_OBS = (255, 176, 0)     # azure blue for the live observation / robot
    ACCENT_SUB = (80, 220, 100)    # green for the subgoal / target
    PANEL_GAP = 16                 # px gutter between the two image panels

    def create_debug_visualization(self, current_img, topo_img, nav):
        """Create and save the demo visualization frame."""
        # Side-by-side observation | subgoal, each with a large corner label.
        combined_img = self.draw_images_side_by_side(
            current_img, topo_img,
            left_label=f"Observation: {nav['frame_id']}",
            right_label=f"Subgoal: {nav['topo_idx']}",
        )

        # Relative-target pose plot underneath.
        pose_plot = self.create_pose_plot(nav['x'], nav['y'], nav['yaw'],
                                          nav['v'], nav['w'])

        final_img = self.combine_images_vertically(combined_img, pose_plot)

        output_path = self.output_dir / f"debug_{nav['frame_id']:06d}_{nav['topo_idx']:04d}.jpg"
        cv2.imwrite(str(output_path), final_img)
        print(f"[DEBUG] Saved: {output_path.name}")

    def _panel_label(self, img, text, accent_bgr):
        """Draw a large, high-contrast label in the top-left of a panel.

        Style: a semi-transparent dark banner with a colored accent bar on the
        left and bold anti-aliased white text. Font scales with image height so
        it reads well at any resolution (demo-friendly)."""
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_DUPLEX
        # Scale font/margins to the panel height.
        fs = max(0.9, h / 620.0)
        thick = max(2, int(round(fs * 2)))
        pad = int(round(12 * fs))
        margin = int(round(18 * fs))

        (tw, th), base = cv2.getTextSize(text, font, fs, thick)
        bar_w = max(6, int(round(8 * fs)))
        x0, y0 = margin, margin
        box_w = bar_w + pad + tw + pad
        box_h = th + base + 2 * pad

        # Semi-transparent dark banner.
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        # Accent bar.
        cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + box_h), accent_bgr, -1)
        # Text (subtle shadow, then white).
        tx = x0 + bar_w + pad
        ty = y0 + pad + th
        cv2.putText(img, text, (tx + 1, ty + 1), font, fs, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(img, text, (tx, ty), font, fs, (255, 255, 255), thick, cv2.LINE_AA)
        return img

    def draw_images_side_by_side(self, img1, img2, left_label=None, right_label=None):
        """Observation (img1) and subgoal (img2) panels, side by side.

        Both are resized to a common height (they can arrive at different sizes
        -- the robot bridge downscales observations to 512px while topomap
        images are full-res -- so without this the smaller one gets pasted into
        a black canvas). Each panel gets a large corner label and a colored
        border; a neutral gutter separates them.
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        target_h = max(h1, h2)
        img1 = cv2.resize(img1, (int(round(w1 * target_h / h1)), target_h))
        img2 = cv2.resize(img2, (int(round(w2 * target_h / h2)), target_h))

        # Ensure 3-channel BGR (topomap/obs could be gray).
        if img1.ndim == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if img2.ndim == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        # Per-panel labels.
        if left_label:
            img1 = self._panel_label(img1, left_label, self.ACCENT_OBS)
        if right_label:
            img2 = self._panel_label(img2, right_label, self.ACCENT_SUB)

        # Colored border framing each panel (accent = same as its label).
        bt = max(3, target_h // 240)
        cv2.rectangle(img1, (0, 0), (img1.shape[1] - 1, target_h - 1), self.ACCENT_OBS, bt)
        cv2.rectangle(img2, (0, 0), (img2.shape[1] - 1, target_h - 1), self.ACCENT_SUB, bt)

        w1, w2 = img1.shape[1], img2.shape[1]
        gap = self.PANEL_GAP
        combined = np.full((target_h, w1 + gap + w2, 3), 18, dtype=np.uint8)  # dark gutter
        combined[:, 0:w1] = img1
        combined[:, w1 + gap:w1 + gap + w2] = img2
        return combined

    def create_pose_plot(self, x, y, yaw, v, w, margin=1.0):
        """Render the relative-target pose panel (CVPR-demo styling).

        Dark theme, bold labels, amber 'robot heading' and green 'target'
        matching the panel accent colors above. Returned as a BGR image.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import io

        # matplotlib uses RGB (0-1); mirror the BGR accents used on the panels.
        obs_rgb = tuple(c / 255.0 for c in self.ACCENT_OBS[::-1])   # azure blue
        sub_rgb = tuple(c / 255.0 for c in self.ACCENT_SUB[::-1])   # green

        yaw_rad = np.radians(yaw)
        arrow_length = 0.5
        dx = arrow_length * np.cos(yaw_rad)
        dy = arrow_length * np.sin(yaw_rad)

        x_min, x_max = min(0, x) - margin, max(0, x) + margin
        y_min, y_max = min(0, y) - margin, max(0, y) + margin

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
            fig.patch.set_facecolor("#141414")
            ax.set_facecolor("#141414")
            ax.set_aspect("equal")

            # direction robot->target (guidance vector)
            ax.annotate("", xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#888888",
                                        lw=2, alpha=0.6, shrinkA=0, shrinkB=0))
            # robot + heading (amber)
            ax.scatter([0], [0], s=120, color=obs_rgb, zorder=5,
                       edgecolors="white", linewidths=1.2, label="Robot")
            ax.quiver(0, 0, arrow_length, 0, angles="xy", scale_units="xy",
                      scale=1, color=obs_rgb, width=0.018, zorder=4)
            # target + its heading (green)
            ax.scatter([x], [y], s=120, color=sub_rgb, zorder=5,
                       edgecolors="white", linewidths=1.2, label="Target")
            ax.quiver(x, y, dx, dy, angles="xy", scale_units="xy",
                      scale=1, color=sub_rgb, width=0.018, zorder=4)

            ax.text(x, y + 0.12, f"({x:.2f}, {y:.2f}, {yaw:.1f}°)",
                    ha="center", va="bottom", fontsize=12, color="white",
                    fontweight="bold")

            # command readout as a boxed annotation (bottom-left)
            ax.text(0.02, 0.02, f"v = {v:.2f} m/s\nω = {w:.2f} rad/s",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=13, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#1f1f1f",
                              ec="#4a4a4a", alpha=0.9))

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("X  (forward, m)", fontsize=12, color="#cccccc")
            ax.set_ylabel("Y  (lateral, m)", fontsize=12, color="#cccccc")
            ax.set_title("Relative Target Pose", fontsize=15, color="white",
                         fontweight="bold", pad=10)
            ax.grid(True, alpha=0.18, color="#666666")
            ax.tick_params(colors="#aaaaaa", labelsize=10)
            for spine in ax.spines.values():
                spine.set_color("#3a3a3a")
            ax.legend(loc="upper left", fontsize=11, framealpha=0.3,
                      facecolor="#1f1f1f", edgecolor="#4a4a4a")

            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
            plt.close(fig)

        buf.seek(0)
        plot_img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        buf.close()
        return plot_img

    def combine_images_vertically(self, img1, img2):
        """Stack the image row (img1) above the pose plot (img2).

        The pose plot is centered on a matching dark background rather than
        stretched, so its aspect ratio (and text) stay undistorted.
        """
        w = img1.shape[1]
        h2, w2 = img2.shape[:2]

        # Scale the plot to fit within the composite width, keep aspect ratio.
        if w2 != w:
            scale = w / float(w2)
            img2 = cv2.resize(img2, (w, int(round(h2 * scale))),
                              interpolation=cv2.INTER_AREA)

        # Thin accent divider between the image row and the plot.
        divider = np.full((4, w, 3), 40, dtype=np.uint8)
        return np.vstack([img1, divider, img2])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--topo_dir', required=True, help='Path to topo images')
    parser.add_argument('--output_dir', default='debug_results', help='Output directory')
    parser.add_argument('--no-timestamp', action='store_true',
                        help='save directly into --output_dir (no nav_run_<timestamp> subdir)')
    args = parser.parse_args()

    rclpy.init()
    subscriber = DebugSubscriber(args.topo_dir, args.output_dir, timestamped=not args.no_timestamp)

    try:
        rclpy.spin(subscriber)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        # ExternalShutdownException: the process received a signal (e.g. from
        # tmux kill-session / Ctrl+C) and rclpy began shutting down the context.
        print("[DEBUG] Stopped by user")
    finally:
        subscriber.destroy_node()
        # Only shut down if the context is still valid -- avoids the
        # "rcl_shutdown already called" error on signal-driven exits.
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
