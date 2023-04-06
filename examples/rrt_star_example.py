import sys
import os

# Get the current working directory
current_directory = os.getcwd()

# Append the current working directory to the system path
sys.path.append(current_directory)


import math
from trajectory_planning import visualize_rrt_3d, rrt_star

if __name__ == "__main__":
    # Test the 3D RRT implementation
    start = (2, 2, 2, 0, 0, 0)
    goal = (24, 24, 24, math.pi / 2, 0, 0)
    obstacles = [
        (10, 15, 15, 20, 8, 2),
        (5, 6, 5, 6, 5, 6),
        (10, 11, 15, 16, 8, 7),
    ]

    # Set the parameters
    width, height, depth = 26, 26, 26
    delta = 0.5
    delta_angle = 0.25
    max_iter = 10000
    bias = 0.7
    timeout = 15
    robot_radius = 2.0
    optimize_path=False
    dynamic_generation=True
    verbose=True

    args = {
        "draw_obstacles": True,
        "draw_orientation": False,
        "draw_nodes": True,
    }
    # Run the RRT* algorithm
    path, path_orientation, node_list = rrt_star(
        start,
        goal,
        obstacles,
        width,
        height,
        depth,
        delta,
        delta_angle,
        max_iter,
        bias,
        timeout,
        robot_radius,
        optimize_path,
        dynamic_generation,
        verbose,
    )
    visualize_rrt_3d(
        node_list,
        path,
        path_orientation,
        start,
        goal,
        obstacles,
        width,
        height,
        depth,
        args
    )