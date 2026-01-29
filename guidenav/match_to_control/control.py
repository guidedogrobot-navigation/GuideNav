import numpy as np


def pController(x_rel, y_rel, yaw_rel,
                     max_v, max_w,
                     k_rho=0.3, k_alpha=1.0, k_beta=-0.3):
    rho = np.hypot(x_rel, y_rel)
    rho = np.hypot(x_rel, y_rel)  # distance to target
    alpha = np.arctan2(y_rel, x_rel)  # heading to target
    beta = yaw_rel - alpha  # orientation mismatch

    # Normalize angles to [-pi, pi]
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    beta = (beta + np.pi) % (2 * np.pi) - np.pi

    # Control law
    v = k_rho * rho
    w = k_alpha * alpha + k_beta * beta

    # Clamp
    v = np.clip(v, -max_v, max_v)
    w = np.clip(w, -max_w, max_w)
    return v, w

def normalize_angle(angle_rad):
    """
    Normalize angle to be within [-pi, pi].
    """
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def pose_to_velocity_p_controller(x_rel, y_rel, yaw_rel_deg,
                                   v_max, w_max,
                                   k_rho=0.6, k_alpha=1.2, k_beta=-0.3,
                                   min_dist=0.05):
    """
    Compute linear and angular velocity commands from a relative target pose.

    Parameters:
        x_rel (float): Relative x position (forward, meters)
        y_rel (float): Relative y position (left, meters)
        yaw_rel_deg (float): Relative yaw (degrees), positive = left turn
        v_max (float): Max linear velocity
        w_max (float): Max angular velocity
        k_rho (float): Gain for linear velocity
        k_alpha (float): Gain for heading correction
        k_beta (float): Gain for final orientation correction
        min_dist (float): Distance threshold to stop

    Returns:
        v (float): Linear velocity (m/s)
        w (float): Angular velocity (rad/s)
    """

    # Convert to radians
    theta_rel = np.radians(yaw_rel_deg)
    
    # Compute distance and bearing to goal
    rho = np.hypot(x_rel, y_rel)
    alpha = np.arctan2(y_rel, x_rel)  # angle from robot to target
    beta = normalize_angle(theta_rel - alpha)  # heading error at goal

    # Normalize alpha
    alpha = normalize_angle(alpha)

    # Stop condition if already very close
    if rho < min_dist:
        return 0.0, 0.0

    # Compute control commands
    v = k_rho * rho
    w = k_alpha * alpha + k_beta * beta

    # Clamp to limits
    v = np.clip(v, -v_max, v_max)
    w = np.clip(w, -w_max, w_max)

    return v, w
    
def ankermann_controller(x_rel, y_rel, theta_rel,
                         k_rho=1.0, k_alpha=1.0, k_theta=-0.2,
                         v_max=0.5, w_max=1.0):
    rho = np.hypot(x_rel, y_rel)
    alpha = np.arctan2(y_rel, x_rel)
    theta = theta_rel

    # Normalize angles
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    # Control law
    v = k_rho * rho
    w = k_alpha * alpha + k_theta * theta

    v = np.clip(v, -v_max, v_max)
    w = np.clip(w, -w_max, w_max)

    return v, w


def vtr_controller(x_rel, y_rel, yaw_rel_deg, 
                   v_max=0.3, w_max=0.8,
                   k_rho=0.8, k_alpha=0.8, k_beta=-0.4,
                   position_tol=0.08, heading_tol_deg=5.0):
    """
    VTR-optimized pose-to-velocity controller
    
    Args:
        x_rel (float): Relative x position (forward, meters)
        y_rel (float): Relative y position (left, meters) 
        yaw_rel_deg (float): Relative yaw (degrees), positive = left turn
        v_max (float): Max linear velocity (m/s)
        w_max (float): Max angular velocity (rad/s)
        k_rho (float): Linear velocity gain
        k_alpha (float): Heading correction gain
        k_beta (float): Final orientation gain (should be negative)
        position_tol (float): Position tolerance (meters)
        heading_tol_deg (float): Heading tolerance (degrees)
    
    Returns:
        tuple: (v, w) - linear and angular velocities
    """
    
    # Convert to radians and compute polar coordinates
    theta_rel = np.radians(yaw_rel_deg)
    heading_tol = np.radians(heading_tol_deg)
    rho = np.hypot(x_rel, y_rel)
    alpha = np.arctan2(y_rel, x_rel)
    beta = normalize_angle(theta_rel - alpha)
    alpha = normalize_angle(alpha)
    
    # Goal reached - stop
    if rho < position_tol and abs(theta_rel) < heading_tol:
        return 0.0, 0.0
    
    # Final alignment mode (close enough positionally)
    if rho < position_tol:
        w = k_beta * theta_rel * 1.5  # Enhanced final alignment
        return 0.0, np.clip(w, -w_max * 0.3, w_max * 0.3)
    
    # Compute base control commands
    # Exponential velocity shaping instead of linear
    v_raw = v_max * (1 - np.exp(-k_rho * rho))
    # v_raw = k_rho * rho
    w_raw = k_alpha * alpha + k_beta * beta
    
    # VTR-specific velocity shaping
    
    # 1. Slow approach zone (last 50cm)
    if rho < 0.5:
        slow_factor = max(0.4, rho / 0.5)
        v_raw *= slow_factor
    
    # 2. Reduce speed for large heading errors
    if abs(alpha) > np.radians(30):
        heading_factor = max(0.5, np.radians(30) / abs(alpha))
        v_raw *= heading_factor
    
    # 3. Enhanced final orientation correction
    if rho < 0.25:
        w_raw = k_alpha * alpha + k_beta * 1.3 * beta
    
    # 4. Coordinated velocity limiting (prevent saturation)
    v_scale = min(1.0, v_max / max(abs(v_raw), 1e-6))
    w_scale = min(1.0, w_max / max(abs(w_raw), 1e-6))
    scale = min(v_scale, w_scale, 0.9)  # Use 90% of max
    
    v = v_raw * scale
    w = w_raw * scale
    
    return v, w