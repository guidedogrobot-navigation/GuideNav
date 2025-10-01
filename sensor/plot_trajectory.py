import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

def plot_se2(csv_path, save_path='pose_plots_lgrc/trajectory_with_yaw.png'):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Extract position
    x = df['pos_x'].values
    y = df['pos_y'].values

    quats = df[['ori_x', 'ori_y', 'ori_z', 'ori_w']].values
    yaws = R.from_quat(quats).as_euler('zyx', degrees=False)[:, 0]  # Extract yaw
    time = df['timestamp'].values
    time = time - time[0]  # Normalize time

     # Create output directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Color trajectory by time
    plt.figure(figsize=(10, 8))
    # sc = plt.scatter(x, y, c=time, cmap='viridis', s=8, label='Trajectory')
    # sc = plt.scatter(x, y, c=time, cmap='turbo', s=16, edgecolor='k', linewidth=0.2, label='Trajectory')
    sc = plt.scatter(x, y, c=time, cmap='plasma', s=16, edgecolors='none')



    # Add quiver arrows every N points to avoid clutter
    N = max(1, len(x) // 50)  # Adjust arrow density
    for i in range(0, len(x), N):
        dx = 0.1 * np.cos(yaws[i])
        dy = 0.1 * np.sin(yaws[i])
        plt.arrow(x[i], y[i], dx, dy, head_width=0.05, color='black')

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('2D Trajectory with Yaw (colored by time)')
    plt.axis('equal')
    plt.grid(True)
    plt.colorbar(sc, label='Time [s]')
    plt.savefig(save_path)
    plt.close()

    print(f"Saved trajectory plot with yaw to: {save_path}")

# Example usage:
plot_se2('../20250530_080826/odom.csv')
