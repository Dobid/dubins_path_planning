import numpy as np
from typing import List, Tuple

from scipy.interpolate import CubicSpline
from scipy.interpolate import BSpline, make_interp_spline


def is_collision(
    point1: Tuple[float, float, float],
    point2: Tuple[float, float, float],
    obstacles: List[Tuple[float, float, float, float, float, float]],
) -> bool:
    for obs in obstacles:
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        x_min, y_min, z_min, x_size, y_size, z_size = obs

        x_max = x_min + x_size + 1
        y_max = y_min + y_size + 1
        z_max = z_min + z_size + 1

        x_min -= 1
        y_min -= 1
        z_min -= 1

        # Check if line segment intersects with the faces of the obstacle
        for t in np.linspace(0, 1, 100):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            z = z1 + t * (z2 - z1)

            # If the point (x, y, z) lies within the obstacle, return True
            if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
                return True

    # No collision detected
    return False

def check_turning_radius_constraint(p0, p1, p2, min_turning_radius):
    def subtract(p1, p2):
        return (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    v1, v2 = subtract(p0, p1), subtract(p2, p1)
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    turning_radius = 0.5 * np.linalg.norm(subtract(p1, p0)) / np.sin(angle)

    return turning_radius >= min_turning_radius


def shortcut_path_radius(
    path: List[Tuple[float, float, float]],
    obstacles: List[Tuple[float, float, float, float, float, float]],
    min_turning_radius: float,
) -> List[Tuple[float, float, float]]:
    i = 0
    while i < len(path) - 2:
        j = i + 2
        while j < len(path):
            if not is_collision(path[i], path[j], obstacles) and check_turning_radius_constraint(path[i], path[i+1], path[j], min_turning_radius):
                path.pop(i + 1)
                j = (
                    i + 2
                )  # Reset j to the new direct neighbor of i after removing the intermediate point
            else:
                j += 1
        i += 1
    return path

def shortcut_path(
    path: List[Tuple[float, float, float]],
    obstacles: List[Tuple[float, float, float, float, float, float]],
) -> List[Tuple[float, float, float]]:
    i = 0
    while i < len(path) - 2:
        j = i + 2
        while j < len(path):
            if not is_collision(path[i], path[j], obstacles):
                path.pop(i + 1)
                j = (
                    i + 2
                )  # Reset j to the new direct neighbor of i after removing the intermediate point
            else:
                j += 1
        i += 1
    return path


def smooth_path_old(
    path: List[Tuple[float, float, float]], num_points: int = 100
) -> np.ndarray:
    path = np.array(path)
    t = np.linspace(0, 1, len(path))
    t_new = np.linspace(0, 1, num_points)

    # Compute cubic splines for x, y, and z coordinates
    x_spline = CubicSpline(t, path[:, 0])
    y_spline = CubicSpline(t, path[:, 1])
    z_spline = CubicSpline(t, path[:, 2])

    # Evaluate the splines at the new time points
    x_smooth = x_spline(t_new)
    y_smooth = y_spline(t_new)
    z_smooth = z_spline(t_new)

    smooth_path = np.column_stack((x_smooth, y_smooth, z_smooth))
    return smooth_path


def smooth_path(
    path: List[Tuple[float, float, float, float, float, float]], num_points: int = 100
) -> np.ndarray:
    path = np.array(path)
    t = np.linspace(0, 1, len(path))
    t_new = np.linspace(0, 1, num_points)

    # Compute cubic splines for x, y, z, roll, pitch, and yaw
    x_spline = CubicSpline(t, path[:, 0])
    y_spline = CubicSpline(t, path[:, 1])
    z_spline = CubicSpline(t, path[:, 2])
    roll_spline = CubicSpline(t, path[:, 3])
    pitch_spline = CubicSpline(t, path[:, 4])
    yaw_spline = CubicSpline(t, path[:, 5])

    # Evaluate the splines at the new time points
    x_smooth = x_spline(t_new)
    y_smooth = y_spline(t_new)
    z_smooth = z_spline(t_new)
    roll_smooth = roll_spline(t_new)
    pitch_smooth = pitch_spline(t_new)
    yaw_smooth = yaw_spline(t_new)

    smooth_path = np.column_stack(
        (x_smooth, y_smooth, z_smooth, roll_smooth, pitch_smooth, yaw_smooth)
    )
    return smooth_path

def smooth_trajectory(waypoints: List[Tuple[float, float, float]], k: int = 3, n_points: int = 100) -> np.ndarray:
    """Smooth a given set of 3D waypoints using a B-spline.
    
    Args:
        waypoints: A list of tuples representing the 3D waypoints.
        k: The order of the B-spline. Default is 3.
        n_points: The number of points to evaluate the B-spline at. Default is 100.

    Returns:
        A numpy array of the smoothed points.
    """
    waypoints = np.array(waypoints)
    t = np.linspace(0, 1, len(waypoints))
    t_new = np.linspace(0, 1, n_points)
    
    # Create a B-spline representation of the trajectory
    spline = make_interp_spline(t, waypoints, k=k)
    
    # Evaluate the B-spline at the new points
    smoothed_points = spline(t_new)
    
    return smoothed_points

def enforce_minimum_turning_radius(smoothed_points: np.ndarray, min_turning_radius: float) -> np.ndarray:
    """Enforce a minimum turning radius on a set of smoothed points.
    
    Args:
        smoothed_points: A numpy array of the smoothed points.
        min_turning_radius: The minimum allowed turning radius.

    Returns:
        A numpy array of the adjusted points.
    """
    adjusted_points = [smoothed_points[0]]
    
    for i in range(1, len(smoothed_points) - 1):
        p0, p1, p2 = smoothed_points[i-1], smoothed_points[i], smoothed_points[i+1]
        
        # Calculate the angle between two consecutive segments
        v1, v2 = p0 - p1, p2 - p1
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        
        # Calculate the required turning radius
        turning_radius = 0.5 * np.linalg.norm(p1 - p0) / np.sin(angle)
        
        # Adjust the waypoint if the required turning radius is greater than the minimum turning radius
        if turning_radius > min_turning_radius:
            t = min_turning_radius / turning_radius
            adjusted_point = (1 - t) * p0 + t * p1
            adjusted_points.append(adjusted_point)
        else:
            adjusted_points.append(p1)
    
    adjusted_points.append(smoothed_points[-1])
    
    return np.array(adjusted_points)

def verify_speed_constraints(adjusted_points: np.ndarray, min_speed: float, max_speed: float, dt: float) -> np.ndarray:
    """Verify that the speed constraints are satisfied for a given set of adjusted points.
    
    Args:
        adjusted_points: A numpy array of the adjusted points.
        min_speed: The minimum allowed speed.
        max_speed: The maximum allowed speed.
        dt: The time step between consecutive waypoints.

    Returns:
        A numpy array of the final trajectory.
    """
    # Compute the velocities between consecutive waypoints
    velocities = np.diff(adjusted_points, axis=0) / dt
    
    # Clip the velocities to stay within the speed constraints
    speeds = np.linalg.norm(velocities, axis=1)
    speeds_clipped = np.clip(speeds, min_speed, max_speed)
    
    # Scale the velocity vectors to match the clipped speeds
    velocities_clipped =velocities * (speeds_clipped / speeds)[:, np.newaxis]

    # Reconstruct the trajectory from the clipped velocities
    trajectory = np.vstack([adjusted_points[0], adjusted_points[:-1] + velocities_clipped * dt])

    return trajectory

def discretize_trajectory(trajectory, orientation, step_size_pos, step_size_vel, step_size_ori):
    discretized_segments = []
    trajectory = trajectory[1:,:]
    n_points = len(trajectory)
    for i in range(1, n_points):
        start_pos, end_pos = trajectory[i-1], trajectory[i]
        start_ori, end_ori = orientation[i-1], orientation[i]

        # Calculate the target position by linearly interpolating between the start and end points of the segment
        position_interp = np.linspace(start_pos, end_pos, num=step_size_pos, endpoint=False)

        # Calculate the target velocity by computing the first derivative of the position with respect to time
        delta_pos = (end_pos - start_pos) / step_size_pos
        velocity_interp = np.tile(delta_pos, (step_size_vel, 1))

        # Calculate the target orientation by linearly interpolating between the start and end orientations of the segment
        orientation_interp = np.linspace(start_ori, end_ori, num=step_size_ori, endpoint=False)

        segment = {
            "position": position_interp,
            "velocity": velocity_interp,
            "orientation": orientation_interp
        }
        discretized_segments.append(segment)

    return discretized_segments

def compute_orientations(trajectory):
    orientations = []
    for i in range(1, len(trajectory)):
        delta_pos = trajectory[i] - trajectory[i - 1]
        yaw = np.arctan2(delta_pos[1], delta_pos[0])
        pitch = np.arctan2(delta_pos[2], np.linalg.norm(delta_pos[:2]))
        orientations.append((yaw, pitch))
    return np.array(orientations)

def calculate_euler_angles(trajectory):
    # Compute the velocity vectors between consecutive points
    velocity_vectors = np.diff(trajectory, axis=0)
    
    # Compute the forward direction unit vectors (tangent to the trajectory)
    forward_directions = velocity_vectors / np.linalg.norm(velocity_vectors, axis=1, keepdims=True)
    
    # Calculate the change in forward direction between consecutive points
    delta_forward = np.diff(forward_directions, axis=0)
    
    # Compute the roll angle using the change in forward direction
    roll_angles = np.arctan2(delta_forward[:, 1], delta_forward[:, 0])
    
    # Initialize arrays for pitch and yaw angles (assuming a constant altitude)
    pitch_angles = np.zeros_like(roll_angles)
    yaw_angles = np.zeros_like(roll_angles)
    
    return np.column_stack((roll_angles, pitch_angles, yaw_angles))
