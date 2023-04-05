import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Tuple

def is_collision(point1: Tuple[float, float, float], point2: Tuple[float, float, float], obstacles: List[Tuple[float, float, float, float, float, float]]) -> bool:
    for obs in obstacles:
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        x_min, y_min, z_min, x_size, y_size, z_size = obs

        x_max = x_min + x_size  +1
        y_max = y_min + y_size  +1
        z_max = z_min + z_size  +1

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

def shortcut_path(path: List[Tuple[float, float, float]], obstacles: List[Tuple[float, float, float, float, float, float]]) -> List[Tuple[float, float, float]]:
    i = 0
    while i < len(path) - 2:
        j = i + 2
        while j < len(path):
            if not is_collision(path[i], path[j], obstacles):
                path.pop(i + 1)
                j = i + 2  # Reset j to the new direct neighbor of i after removing the intermediate point
            else:
                j += 1
        i += 1
    return path

def smooth_path_old(path: List[Tuple[float, float, float]], num_points: int = 20) -> np.ndarray:
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

def smooth_path(path: List[Tuple[float, float, float, float, float, float]], num_points: int = 100) -> np.ndarray:
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

    smooth_path = np.column_stack((x_smooth, y_smooth, z_smooth, roll_smooth, pitch_smooth, yaw_smooth))
    return smooth_path