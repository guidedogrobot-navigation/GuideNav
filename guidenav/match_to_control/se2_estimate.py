import numpy as np
import cv2
from math import atan2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time


def pnpRansac(kp1, kp2, matches, depth_obs, K):
    """
        from matches, backproject 2D points to 3D using depth_obs
        and camera intrinsics K, then estimate relative pose using PnP.
    """
    
    # target image points from matches
    pts2d = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32) 
    # observation image points from matches
    pts2d_obs = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32) 

    # start_time = time.time()
    pts3d, valid = backproject_2d_to_3d(pts2d_obs, depth_obs, K)
    # end_time = time.time()
    # print(f"[INFO] Backprojection time: {end_time - start_time:.4f} seconds")

    pts2d_valid = pts2d[valid]

    if len(pts3d) < 6:
        # raise RuntimeError("Not enough valid 3D points (less than 6) for PnP")
        print("[WARN] Skipping frame: not enough valid 3D points for PnP.")
        return None

    # start_time_ransac = time.time()
    _, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d, pts2d_valid, K, None)
    # end_time_ransac = time.time()
    # print(f"[INFO] PnP RANSAC time: {end_time_ransac - start_time_ransac:.4f} seconds")
    R_mat, _ = cv2.Rodrigues(rvec)

    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = tvec.squeeze()

    # T is obs to target: how obs cam looks from target
    T_inv = np.linalg.inv(T) # target relative to obs
    # Convert to SE(2)
    x_rel = T_inv[2, 3]   # forward (Z in cam)
    y_rel = -T_inv[0, 3]  # left (X in cam = -Y in robot)
    yaw = np.arctan2(T_inv[0, 2], T_inv[2, 2])  # camera yaw

    # print(f"[INFO] Inliers (PnP): {len(inliers)} / {len(pts3d)}")
    return x_rel, y_rel, np.degrees(yaw)

def quat_to_yaw_deg(qx, qy, qz, qw):
    rot = R.from_quat([qx, qy, qz, qw])
    euler = rot.as_euler('xyz', degrees=True)
    return euler[2]  # Yaw in degrees

def backproject_2d_to_3d(pts_2d, depth, intrinsics):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = depth[pts_2d[:, 1].astype(int), pts_2d[:, 0].astype(int)]
    # print("Depth stats: min", np.min(z), "max", np.max(z), "mean", np.mean(z))
    valid = (z > 0) & np.isfinite(z)
    x = (pts_2d[:, 0] - cx) * z / fx
    y = (pts_2d[:, 1] - cy) * z / fy
    pts3d = np.stack([x, y, z], axis=-1)
    return pts3d[valid], valid