import sys
import os
import numpy as np
# Get the current working directory
current_directory = os.getcwd()

# Append the current working directory to the system path
sys.path.append(current_directory)


import math
from trajectory_planning import (
    visualize_trajectory,
    smooth_trajectory,
    enforce_minimum_turning_radius, 
    verify_speed_constraints,
    discretize_trajectory,
    compute_orientations
)

# Example usage
waypoints = [(0, 0, 0), (1, 2, 3), (2, 4, 6), (4, 4, 4), (6, 2, 0)]
smoothed_points = smooth_trajectory(waypoints)
min_turning_radius = 5  # Replace with your desired value
adjusted_points = enforce_minimum_turning_radius(smoothed_points, min_turning_radius)
min_speed, max_speed, dt = 1, 10, 1  # Replace with your desired values
trajectory = verify_speed_constraints(adjusted_points, min_speed, max_speed, dt)


orientation = compute_orientations(trajectory)  # Replace with your desired orientation values
step_size_pos, step_size_vel, step_size_ori = 10, 10, 10  # Replace with your desired step sizes
discretized_segments = discretize_trajectory(trajectory, orientation, step_size_pos, step_size_vel, step_size_ori)

# Example usage
visualize_trajectory(np.array(waypoints), np.array(adjusted_points), np.array(trajectory))

