import sys
import os

# Get the current working directory
current_directory = os.getcwd()

# Append the current working directory to the system path
sys.path.append(current_directory)


import math
from rrt_path_plannig.rrt import visualize_rrt_3d

if __name__ == '__main__':
    # Test the 3D RRT implementation
    start = (2, 2, 2, 0, 0, 0)
    goal = (24, 24, 24, math.pi/2, 0, 0)
    obstacles = [
        (10, 15, 15, 20, 8, 2),
        (5, 6, 5, 6, 5, 6),
        (10, 11, 15, 16, 8, 7),
    ]
    width, height, depth = 26, 26, 26
    delta = 0.5
    delta_angle = 0.5
    max_iter = 5000

    visualize_rrt_3d(start, goal, obstacles, width, height, depth, delta, delta_angle, max_iter, bias=0.5)
