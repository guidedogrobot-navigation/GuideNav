#!/usr/bin/env python3
"""
Trajectory plotter for odometry data.
Reads odom.csv and plots 2D trajectory with total travel distance.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import List, Tuple

def load_trajectory_data(odom_file: str) -> Tuple[List[float], List[float], List[float]]:
    """
    Load trajectory data from odometry CSV file.
    
    Returns:
        timestamps: List of timestamps
        x_positions: List of x coordinates
        y_positions: List of y coordinates
    """
    timestamps = []
    x_positions = []
    y_positions = []
    
    with open(odom_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp']))
            x_positions.append(float(row['pos_x']))
            y_positions.append(float(row['pos_y']))
    
    return timestamps, x_positions, y_positions

def calculate_total_distance(x_positions: List[float], y_positions: List[float]) -> float:
    """
    Calculate total travel distance from trajectory points.
    
    Args:
        x_positions: List of x coordinates
        y_positions: List of y coordinates
        
    Returns:
        Total distance traveled in meters
    """
    total_distance = 0.0
    
    for i in range(1, len(x_positions)):
        dx = x_positions[i] - x_positions[i-1]
        dy = y_positions[i] - y_positions[i-1]
        segment_distance = np.sqrt(dx**2 + dy**2)
        total_distance += segment_distance
    
    return total_distance

def plot_trajectory(timestamps: List[float], x_positions: List[float], y_positions: List[float], 
                   total_distance: float, output_file: str = None, show_plot: bool = True):
    """
    Plot 2D trajectory with distance information.
    
    Args:
        timestamps: List of timestamps
        x_positions: List of x coordinates  
        y_positions: List of y coordinates
        total_distance: Total travel distance
        output_file: Optional file to save the plot
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create subplot layout
    gs = plt.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
    
    # Main trajectory plot
    ax_main = plt.subplot(gs[0, 0])
    
    # Plot trajectory with color gradient based on time
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_positions)))
    
    # Plot trajectory line
    ax_main.plot(x_positions, y_positions, 'b-', alpha=0.6, linewidth=1, label='Trajectory')
    
    # Scatter plot with time-based coloring
    scatter = ax_main.scatter(x_positions, y_positions, c=timestamps, 
                             cmap='viridis', s=10, alpha=0.8, edgecolors='none')
    
    # Mark start and end points
    ax_main.plot(x_positions[0], y_positions[0], 'go', markersize=10, label='Start', zorder=5)
    ax_main.plot(x_positions[-1], y_positions[-1], 'ro', markersize=10, label='End', zorder=5)
    
    ax_main.set_xlabel('X Position (m)')
    ax_main.set_ylabel('Y Position (m)')
    ax_main.set_title(f'2D Trajectory\nTotal Distance: {total_distance:.2f} m')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()
    ax_main.set_aspect('equal', adjustable='box')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax_main)
    cbar.set_label('Timestamp')
    
    # X position over time
    ax_x = plt.subplot(gs[0, 1])
    ax_x.plot(timestamps, x_positions, 'r-', linewidth=1)
    ax_x.set_xlabel('Time')
    ax_x.set_ylabel('X Position (m)')
    ax_x.set_title('X vs Time')
    ax_x.grid(True, alpha=0.3)
    
    # Y position over time  
    ax_y = plt.subplot(gs[1, 0])
    ax_y.plot(timestamps, y_positions, 'g-', linewidth=1)
    ax_y.set_xlabel('Time')
    ax_y.set_ylabel('Y Position (m)')
    ax_y.set_title('Y vs Time')
    ax_y.grid(True, alpha=0.3)
    
    # Statistics
    ax_stats = plt.subplot(gs[1, 1])
    ax_stats.axis('off')
    
    # Calculate statistics
    duration = timestamps[-1] - timestamps[0]
    avg_speed = total_distance / duration if duration > 0 else 0
    x_range = max(x_positions) - min(x_positions)
    y_range = max(y_positions) - min(y_positions)
    
    stats_text = f"""Statistics:
Total Distance: {total_distance:.2f} m
Duration: {duration:.1f} s
Avg Speed: {avg_speed:.3f} m/s
Data Points: {len(x_positions)}
X Range: {x_range:.2f} m
Y Range: {y_range:.2f} m
Start: ({x_positions[0]:.2f}, {y_positions[0]:.2f})
End: ({x_positions[-1]:.2f}, {y_positions[-1]:.2f})"""
    
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, 
                 fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    # Show plot if requested
    if show_plot:
        plt.show()

def analyze_trajectory(odom_file: str, output_plot: str = None, show_plot: bool = True):
    """
    Main function to analyze trajectory from odometry file.
    
    Args:
        odom_file: Path to odometry CSV file
        output_plot: Optional path to save plot
        show_plot: Whether to display the plot
    """
    print(f"Loading trajectory data from: {odom_file}")
    
    # Load data
    timestamps, x_positions, y_positions = load_trajectory_data(odom_file)
    
    print(f"Loaded {len(timestamps)} trajectory points")
    
    if len(timestamps) < 2:
        print("Error: Need at least 2 points to calculate trajectory")
        return
    
    # Calculate total distance
    total_distance = calculate_total_distance(x_positions, y_positions)
    
    # Print summary
    duration = timestamps[-1] - timestamps[0]
    avg_speed = total_distance / duration if duration > 0 else 0
    
    print(f"\n=== Trajectory Analysis ===")
    print(f"Total Distance: {total_distance:.2f} m")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Average Speed: {avg_speed:.3f} m/s")
    print(f"Data Points: {len(timestamps)}")
    print(f"Start Position: ({x_positions[0]:.3f}, {y_positions[0]:.3f}) m")
    print(f"End Position: ({x_positions[-1]:.3f}, {y_positions[-1]:.3f}) m")
    
    # Plot trajectory
    plot_trajectory(timestamps, x_positions, y_positions, total_distance, 
                   output_plot, show_plot)

def main():
    parser = argparse.ArgumentParser(description='Plot 2D trajectory and calculate travel distance from odometry data')
    parser.add_argument('odom_file', help='Path to odometry CSV file')
    parser.add_argument('--output', '-o', help='Output file for plot (e.g., trajectory.png)')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display plot window')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.odom_file).exists():
        print(f"Error: File {args.odom_file} not found")
        return
    
    analyze_trajectory(
        odom_file=args.odom_file,
        output_plot=args.output,
        show_plot=not args.no_show
    )

if __name__ == '__main__':
    main()
