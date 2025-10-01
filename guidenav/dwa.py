import argparse
import parser
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
from matplotlib.patches import Rectangle, Circle
import math
import time

class DWA():
    def __init__(self, args: argparse.Namespace):
        with args.robot_config_path.open(mode="r", encoding="utf-8") as f:
            robot_configs = yaml.safe_load(f)
        self.robot_config = robot_configs[args.robot]
        # robot size
        self.robot_width = self.robot_config['size_width']
        self.robot_length = self.robot_config['size_length']
        
        # robot params
        self.v_max = self.robot_config['max_v']
        self.v_min = self.robot_config['min_v']
        self.w_max = self.robot_config['max_w']
        self.w_min = - self.w_max
        self.a_max = self.robot_config['max_a']
        self.alpha_max = self.robot_config['max_alpha']

        # cost function params
        self.cost_w_goal = self.robot_config['J_w_goal']
        self.cost_w_heading = self.robot_config['J_w_heading']
        self.cost_w_vPref = self.robot_config['J_w_vPref']
        self.cost_w_clearance = self.robot_config['J_w_clearance']
        self.cost_w_smoothness = self.robot_config['J_w_smoothness']

        # discretization
        self.dt_ctl = self.robot_config['control_dt']
        self.dt_pred = self.robot_config['predict_dt']
        self.t_horizon = self.robot_config['t_horizon']

        # Robot state [x, y, theta, vel, angular_vel]
        # TODO: use intiial state from cuvslam
        self.robot_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # DWA params
        self.dw_v_res = self.robot_config['dw_v']
        self.dw_w_res = self.robot_config['dw_w']

        self.goal = np.array([8.0, 3.0])
        self.goal_reached = False

        self.obstacle_builder()

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.setup_plot()

        # storage for traj visualization
        self.trajectory_history = []
        self.best_trajectory = []
        self.candidate_trajectories = []  # Store all candidate trajectories
        self.collision_trajectories = []  # Store collision trajectories
        
    def debug_reader(self):
        print("Reading robot_configs")
        print(f"v_max: {self.v_max}, v_min: {self.v_min}, w_max: {self.w_max}, w_min: {self.w_min}")
        print(f"robot_width: {self.robot_width}, robot_length: {self.robot_length}")
        
        
    def goal_updater(self):
        

        if self.goal_reached:
            print("Goal reached!")
            # give random goal a bit far from current position
            max_attempts = 100
            min_goal_distance = 2.0  # Minimum distance from current position
            safety_margin = 0.1      # Extra space around obstacles
            for attempt in range(max_attempts):
                # Generate random candidate goal
                goal_x = np.random.uniform(1.0, 9.0)
                goal_y = np.random.uniform(0.0, 5.0)
                candidate_goal = np.array([goal_x, goal_y])
                
                # Check if goal is far enough from current position
                current_pos = self.robot_state[:2]
                if np.linalg.norm(candidate_goal - current_pos) < min_goal_distance:
                    continue
                
                # Check if goal overlaps with any obstacle
                goal_is_safe = True
                for obs in self.obstacles:
                    obs_x, obs_y, obs_radius = obs
                    distance_to_obs = (goal_x - obs_x)**2 + (goal_y - obs_y)**2
                    
                    if distance_to_obs < (obs_radius + safety_margin) **2:
                        goal_is_safe = False
                        break
                
                if goal_is_safe:
                    self.goal = candidate_goal
                    break
            self.goal_circle.center = (self.goal[0], self.goal[1])
            self.goal_reached = False
        else:
            print(f"Current goal: {self.goal}")
    


    def obstacle_builder(self):
        self.obstacles = [
            [3.0, 1.0, 0.5],
            [5.0, 3.0, 0.7],
            [7.0, 0.5, 0.4],
            [2.0, 4.0, 0.6],
            [6.0, 5.0, 0.5]
        ]
    
    
    def update_state(self, v, w):
        print(f"Updating state with v: {v}, w: {w}")
        print(f"state before update: {self.robot_state}")
        
        self.robot_state[0] += v * math.cos(self.robot_state[2]) * self.dt_ctl
        self.robot_state[1] += v * math.sin(self.robot_state[2]) * self.dt_ctl
        self.robot_state[2] += w * self.dt_ctl
        self.robot_state[3] = v
        self.robot_state[4] = w

    def one_step_rollout(self, x,y,theta,v,w):
        updated_x = x + v * math.cos(theta) * self.dt_pred
        updated_y = y+ v * math.sin(theta) * self.dt_pred
        updated_theta = theta + w * self.dt_pred
        return [updated_x, updated_y, updated_theta]

    def compute_dw(self):
        v_min =  self.robot_state[3] - self.a_max * self.dt_ctl
        v_max = self.robot_state[3] + self.a_max * self.dt_ctl
        w_min = self.robot_state[4] - self.alpha_max * self.dt_ctl
        w_max = self.robot_state[4] + self.alpha_max * self.dt_ctl

        # add vel limits
        v_min = max(v_min, self.v_min)
        v_max = min(v_max, self.v_max)
        w_min = max(w_min, self.w_min)
        w_max = min(w_max, self.w_max)

        # dynamic window
        v_range = np.arange(v_min, v_max + self.dw_v_res, self.dw_v_res)
        w_range = np.arange(w_min, w_max + self.dw_w_res, self.dw_w_res)

        return v_range, w_range

    def traj_rollout(self, v, w):
        trajectory = []
        
        num_steps = int(self.t_horizon / self.dt_pred)
        # print(f"type of num_steps: {num_steps}")

        x, y , theta = self.robot_state[0], self.robot_state[1], self.robot_state[2]
        for step in range(num_steps):
            [x,y, theta] = self.one_step_rollout(x, y, theta, v, w)
            trajectory.append([x,y,theta])

        return np.array(trajectory)

    def check_collision(self, trajectory):
        for waypoint in trajectory:
            x, y, _ = waypoint
            for obs in self.obstacles:
                obs_x, obs_y, obs_r = obs
                distance = (x - obs_x) ** 2 + (y - obs_y) ** 2
                if distance < (max(self.robot_width, self.robot_length) / 2 + obs_r) ** 2:
                    return True
        return False
        

    def get_nearest_obstacle_dist(self):
        for obs in self.obstacles:
            obs_x, obs_y, obs_r = obs
            distance = math.sqrt((self.robot_state[0] - obs_x) ** 2 + (self.robot_state[1] - obs_y) ** 2)
            print(f"Distance to obstacle at ({obs_x}, {obs_y}): {distance}")
            print(f"Robot size: {max(self.robot_width, self.robot_length) / 2}, Obstacle radius: {obs_r}")  
            if distance < (max(self.robot_width, self.robot_length) / 2 + obs_r):
                return -1.0
            else:
                return distance
                

    def compute_cost(self, v, w, trajectory):
        final_pos = trajectory[-1]

        # cost (1) goal dist
        distance_err = (self.goal[0] - final_pos[0]) ** 2 + (self.goal[1] - final_pos[1]) ** 2 
        J_goal = self.cost_w_goal * distance_err

        # cost (2) heading
        current2goal_angle = math.atan2(self.goal[1]- final_pos[1], self.goal[0] - final_pos[0])
        heading_err = final_pos[2] - current2goal_angle
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))  # normalize angle
        J_heading = self.cost_w_heading * abs(heading_err)
        
        # cost (3) velocity preference
        vel_pref_err = abs(self.v_max - v)
        J_vPref = self.cost_w_vPref * vel_pref_err

        # cost (4) clearance
        d_min = self.get_nearest_obstacle_dist()
        clearance = 1.0 / d_min if d_min > 0 else float('inf')
        J_clearance = self.cost_w_clearance * clearance
        
        # cost (5) smoothness
        v_smoothness = abs(v - self.robot_state[3])
        w_smoothness = abs(w - self.robot_state[4])
        J_smoothness = self.cost_w_smoothness * (v_smoothness + w_smoothness)

        J = J_goal + J_heading + J_vPref + J_clearance + J_smoothness

        return J
        
        
    def dwa_planning(self):
        best_v, best_w = 0.0, 0.0 
        best_cost = float('inf')

        
        # Clear previous candidate trajectories (for visualization)
        best_trajectory = []
        self.candidate_trajectories = []
        self.collision_trajectories = []
        
        # (1) compute dynamic window
        dw_v, dw_w = self.compute_dw()
        print(f"Dynamic window: v_range={dw_v}, w_range={dw_w}")

        for v in dw_v:
            for w in dw_w:
                # (2) traj rollout
                trajectory = self.traj_rollout(v, w)

                # (3) check collision
                if self.check_collision(trajectory):
                    self.collision_trajectories.append(trajectory)
                    # compute cost J(v, w)
                    # (4) compute cost
                    continue
                cost = self.compute_cost(v, w, trajectory)
                # Store candidate trajectory for visualization
                self.candidate_trajectories.append((trajectory, cost))
                # else:
                #     cost = float('inf')
                #     continue

                if cost < best_cost:
                    best_cost = cost
                    best_v, best_w = v, w
                    best_trajectory = trajectory

        return best_v, best_w, best_trajectory
        

    def update_visualization(self):
        """Update robot visualization"""
        x, y, theta = self.robot_state[0], self.robot_state[1], self.robot_state[2]
        
        # Update robot rectangle
        # Calculate corner position considering rotation
        corner_x = x - self.robot_length/2 * math.cos(theta) + self.robot_width/2 * math.sin(theta)
        corner_y = y - self.robot_length/2 * math.sin(theta) - self.robot_width/2 * math.cos(theta)
        
        self.robot_patch.set_xy((corner_x, corner_y))
        self.robot_patch.set_angle(math.degrees(theta))
        
        # Update trajectory
        if len(self.trajectory_history) > 1:
            traj_array = np.array(self.trajectory_history)
            self.trajectory_line.set_data(traj_array[:, 0], traj_array[:, 1])
        
        # Update best trajectory
        if len(self.best_trajectory) > 0:
            self.best_traj_line.set_data(self.best_trajectory[:, 0], self.best_trajectory[:, 1])
        
        # Clear previous candidate trajectory lines
        for line in self.candidate_lines:
            line.remove()
        for line in self.collision_lines:
            line.remove()
        self.candidate_lines.clear()
        self.collision_lines.clear()
        
        # Draw all candidate trajectories (valid ones)
        for trajectory, cost in self.candidate_trajectories:
            if len(trajectory) > 0:
                # Color based on cost - darker = better cost
                max_cost = max([c for _, c in self.candidate_trajectories]) if self.candidate_trajectories else 1
                min_cost = min([c for _, c in self.candidate_trajectories]) if self.candidate_trajectories else 0
                
                if max_cost > min_cost:
                    normalized_cost = (cost - min_cost) / (max_cost - min_cost)
                else:
                    normalized_cost = 0.5
                
                # Green (good) to yellow (bad) color mapping
                color = plt.cm.YlGn(normalized_cost)
                
                line, = self.ax.plot(trajectory[:, 0], trajectory[:, 1], 
                                   color=color, linewidth=1, alpha=0.8)
                self.candidate_lines.append(line)
        
        # Draw collision trajectories in red
        for trajectory in self.collision_trajectories:
            if len(trajectory) > 0:
                line, = self.ax.plot(trajectory[:, 0], trajectory[:, 1], 
                                #    'r-', linewidth=0.5, alpha=0.3)
                                   'r-', linewidth=1, alpha=0.8)
                self.collision_lines.append(line)
        
    def setup_plot(self):
        """Setup the matplotlib plot"""
        self.ax.set_xlim(-1, 10)
        self.ax.set_ylim(-1, 6)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('DWA Local Planner - 2D Visualization')
        
        # Draw obstacles
        for obs in self.obstacles:
            circle = Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.7)
            self.ax.add_patch(circle)
        
        # Draw goal (goal may be updated)
        self.goal_circle = Circle((self.goal[0], self.goal[1]), 0.2, color='green', alpha=0.8)
        self.ax.add_patch(self.goal_circle)


        
        # Robot visualization (will be updated in animation)
        self.robot_patch = Rectangle((0, 0), self.robot_length, self.robot_width, 
                                   fill=True, color='blue', alpha=0.7)
        self.ax.add_patch(self.robot_patch)
        
        # Trajectory lines
        self.trajectory_line, = self.ax.plot([], [], 'b-', linewidth=2, alpha=0.8, label='Robot Path')
        self.best_traj_line, = self.ax.plot([], [], 'g--', linewidth=3, alpha=0.9, label='Best Trajectory')
        
        # Storage for candidate trajectories visualization
        self.candidate_lines = []
        self.collision_lines = []
        
        self.ax.legend()

    def animate(self, frame):
        dist2goal = (self.goal[0] - self.robot_state[0]) ** 2 + (self.goal[1] - self.robot_state[1]) ** 2
        if dist2goal < 0.7 ** 2:
            self.goal_reached = True
            print("Goal reached!")
        self.goal_updater()


        start_dwa_time = time.time()
        v_star, w_star, best_traj = self.dwa_planning()
        self.best_trajectory = best_traj
        end_dwa_time = time.time()

        # (5) update robot state
        start_update_time = time.time()
        self.update_state(v_star, w_star)
        end_update_time = time.time()

        print(f"DWA planning time: {end_dwa_time - start_dwa_time:.4f}s, ")
        print(f"State update time: {end_update_time - start_update_time:.4f}s")
        # Update visualization
        self.update_visualization()
        
        # Print current state
        print(f"Frame {frame}: pos=({self.robot_state[0]:.2f}, {self.robot_state[1]:.2f}), "
              f"vel=({v_star:.2f}, {w_star:.2f}), goal_dist={dist2goal:.2f}")
        
        # return [self.robot_patch] 
        return [self.robot_patch, self.trajectory_line, self.best_traj_line] + self.candidate_lines + self.collision_lines
    

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.animate, frames=1000, 
                                    interval=100, blit=False, repeat=False)

        plt.show()
        return ani

def main():
    args = parser.parse_args()

    local_planner = DWA(args)

    # (1) compute DW
    # (2) for (v,w) in DW:
    #   (1) trj rollout
    #   (2) check collision
    #       (a) if collision, skip this (v,w); J(v,w) = inf
    #       (b) if not collision, compute cost J(v,w)
    # (3) select (v*, w*) with argmin J(v,w)
    # (4) execute (v*, w*) for dt_ctl
    # (5) update robot state
    # (6) repeat until goal reached
    
    # local_planner.debug_reader()
    # local_planner.init_env()
    
    ani = local_planner.run()   
    plt.show()

if __name__ == '__main__':
    main()