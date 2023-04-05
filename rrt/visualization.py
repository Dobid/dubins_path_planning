import numpy as np
from typing import List, Tuple
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .node import Node
from .path_planning import rrt
from .post_processing import smooth_path, shortcut_path, smooth_path_old

def draw_obstacles(ax: Axes3D, obstacles: List[Tuple[float, float, float, float, float, float]]) -> None:
    for obs in obstacles:
        x, y, z, dx, dy, dz = obs
        vertices = [
            (x, y, z),
            (x + dx, y, z),
            (x + dx, y + dy, z),
            (x, y + dy, z),
            (x, y, z + dz),
            (x + dx, y, z + dz),
            (x + dx, y + dy, z + dz),
            (x, y + dy, z + dz),
        ]
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[3], vertices[0], vertices[4], vertices[7]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
        ]
        cube = Poly3DCollection(faces, linewidths=1, edgecolors='r', facecolors=(1, 0, 0, 0.1))
        ax.add_collection3d(cube)


def draw_nodes(ax: Axes3D, node_list: List[Node]) -> None:
    xs = [node.x for node in node_list]
    ys = [node.y for node in node_list]
    zs = [node.z for node in node_list]
    ax.scatter(xs, ys, zs, c='b', s=5, alpha=0.2)

def draw_path_old(ax: Axes3D, path: List[Tuple[float, float, float]]) -> None:
    xs, ys, zs = zip(*path)
    ax.plot(xs, ys, zs, 'g', linewidth=2)

def draw_path(ax: Axes3D, path: List[Tuple[float, float, float, float, float, float]]) -> None:
    xs, ys, zs, rolls, pitches, yaws = zip(*path)

    # Draw the path itself
    ax.plot(xs, ys, zs, 'g', linewidth=2)

    # Draw orientation arrows for each point
    for x, y, z, roll, pitch, yaw in path:
        # Calculate orientation vectors
        R = Rotation.from_euler('xyz', [roll, pitch, yaw])
        orientation = R.as_matrix() @ np.array([1, 1, 1])
        ax.quiver(
            x, y, z,
            orientation[0], orientation[1], orientation[2],
            length=1, color='r', arrow_length_ratio=0.1
        )

def visualize_rrt_3d(start: Tuple[float, float, float], goal: Tuple[float, float, float], obstacles: List[Tuple[float, float, float, float]], width: float, height: float, depth: float, delta: float, delta_angle: float, max_iter: int, bias: float) -> None:
    path, path_orientation, node_list = rrt(start, goal, obstacles, width, height, depth, delta, delta_angle, max_iter, bias)
    print('Path length:', len(path) - 1)
    shortcut = shortcut_path(path, obstacles)
    print('Shortcut path length:', len(path) - 1)

    smoothed_path = smooth_path_old(shortcut)
    smoothed_path_no_short = smooth_path(path_orientation)
    print('Smoothed path length:', len(smoothed_path) - 1)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_zlim(0, depth)

    draw_obstacles(ax, obstacles)
    draw_nodes(ax, node_list)
    # draw_path(ax, path_orientation)
    # draw_path(ax, smoothed_path_no_short)
    draw_path_old(ax, path)
    draw_path_old(ax, smoothed_path)  # Add this line to visualize the smoothed path

    ax.scatter(start[0], start[1], start[2], c='b', s=50, marker='s')
    ax.scatter(goal[0], goal[1], goal[2], c='r', s=50, marker='s')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()