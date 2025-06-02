import time
import math
import random
from tqdm import tqdm
from typing import List, Tuple

from .utils.node import Node
from .utils.plan_utils import (
    euclidean_distance,
    combined_distance,
    collision_free,
    local_density,
    path_cost,
    rewire,
    get_closest_node_to_goal,
    get_nearby_nodes,
    get_nearest_node,
    get_path
)



def rrt_star(
    start: Tuple[float, float, float, float, float, float],
    goal: Tuple[float, float, float, float, float, float],
    obstacles: List[Tuple[float, float, float, float, float, float]],
    width: float,
    height: float,
    depth: float,
    delta: float,
    delta_angle: float,
    max_iter: int,
    bias: float = 0.45,
    timeout: int = 25,
    robot_radius: float = 2.0,
    optimize_path: bool = False,
    dynamic_generation: bool = True,
    verbose: bool = False,
) -> Tuple[
    List[Tuple[float, float, float]],
    List[Tuple[float, float, float, float, float, float]],
    List[Node],
]:
    """
    Args:
        start (tuple): The start configuration (x, y, z, roll, pitch, yaw).
        goal (tuple): The goal configuration (x, y, z, roll, pitch, yaw).
        obstacles (list): A list of obstacle objects that the path must avoid.
        width (float): The width of the search space.
        height (float): The height of the search space.
        depth (float): The depth of the search space.
        delta (float): The maximum Euclidean distance between two nodes.
        delta_angle (float): The maximum angular distance between two nodes.
        max_iter (int): The maximum number of iterations to perform.
        bias (float, optional): The bias probability for generating a random node around the closest node to the goal. Defaults to 0.45.
        timeout (int, optional): The maximum duration of the search in seconds. Defaults to 25.

    Returns:
        path (list): The computed path as a list of (x, y, z) tuples.
        path_orientation (list): The computed path as a list of (x, y, z, roll, pitch, yaw) tuples.
        node_list (list): The list of nodes in the RRT.
    """

    if verbose:
        print(
            """
        RRT parameters:
        ---------------
        Start: {}
        Goal: {}
        Obstacles: {}
        Width: {}
        Height: {}
        Depth: {}
        Delta: {}
        Delta angle: {}
        Max iterations: {}
        Bias: {}
        Timeout: {}
        Optimize path: {}
        Dynamic generation: {}
        """.format(
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
                optimize_path,
                dynamic_generation,
            )
        )

    start_node = Node(start[0], start[1], start[2], start[3], start[4], start[5])
    goal_node = Node(goal[0], goal[1], goal[2], goal[3], goal[4], goal[5])
    node_list = [start_node]
    best_goal_node = None
    
    start_time = time.time()
    for _ in tqdm(range(max_iter), desc="RRT*"):

        if time.time() - start_time > timeout:
            break

        if best_goal_node is not None:
            dynamic_generation = False

        closest_node_to_goal = get_closest_node_to_goal(node_list, goal_node)
        density = local_density(closest_node_to_goal, node_list, delta)
        dynamic_bias = max(bias, density)

        if random.random() < dynamic_bias and dynamic_generation:
            # Generate a random node around the closest node to the goal
            random_node = Node(
                closest_node_to_goal.x + random.uniform(-delta, delta),
                closest_node_to_goal.y + random.uniform(-delta, delta),
                closest_node_to_goal.z + random.uniform(-delta, delta),
                closest_node_to_goal.roll + random.uniform(-delta_angle, delta_angle),
                closest_node_to_goal.pitch + random.uniform(-delta_angle, delta_angle),
                closest_node_to_goal.yaw + random.uniform(-delta_angle, delta_angle),
            )
        else:
            # Generate a completely random node
            random_node = Node(
                random.uniform(0, width),
                random.uniform(0, height),
                random.uniform(0, depth),
                random.uniform(-math.pi, math.pi),
                random.uniform(-math.pi / 2, math.pi / 2),
                random.uniform(-math.pi, math.pi),
            )

        nearest_node = get_nearest_node(node_list, random_node)

        # Calculate the position of the new node
        theta = math.atan2(
            random_node.y - nearest_node.y, random_node.x - nearest_node.x
        )
        phi = math.acos(
            (random_node.z - nearest_node.z)
            / euclidean_distance(random_node, nearest_node)
        )

        new_x = nearest_node.x + delta * math.cos(theta) * math.sin(phi)
        new_y = nearest_node.y + delta * math.sin(theta) * math.sin(phi)
        new_z = nearest_node.z + delta * math.cos(phi)

        # Calculate the orientation of the new node
        delta_roll = random_node.roll - nearest_node.roll
        delta_pitch = random_node.pitch - nearest_node.pitch
        delta_yaw = random_node.yaw - nearest_node.yaw

        # Normalize the angular differences to the range [-pi, pi]
        delta_roll = (delta_roll + math.pi) % (2 * math.pi) - math.pi
        delta_pitch = (delta_pitch + math.pi) % (2 * math.pi) - math.pi
        delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi

        new_roll = nearest_node.roll + delta_roll
        new_pitch = nearest_node.pitch + delta_pitch
        new_yaw = nearest_node.yaw + delta_yaw

        new_node = Node(new_x, new_y, new_z, new_roll, new_pitch, new_yaw)
        new_node.parent = nearest_node

        if collision_free(new_node, nearest_node, obstacles):
            new_node.cost = nearest_node.cost + combined_distance(nearest_node, new_node)
            node_list.append(new_node)

            # Find nearby nodes and rewire them
            nearby_nodes = get_nearby_nodes(node_list, new_node, delta)
            rewire(new_node, nearby_nodes, obstacles, robot_radius)

            if euclidean_distance(new_node, goal_node) < delta:
                new_path = []
                current_node = new_node
                while current_node is not None:
                    new_path.append(current_node)
                    current_node = current_node.parent

                if best_goal_node is None or path_cost(new_path) < path_cost(
                    get_path(best_goal_node)
                ):
                    if best_goal_node is None:
                        print("\nFEASIBLE PATH FOUND\nSWITCHED TO OPTIMIZATION MODE\n")
                    best_goal_node = new_node
                    if not optimize_path:
                        break
                    print(
                        "New best path found with cost: {}".format(
                            path_cost(get_path(best_goal_node))
                        )
                    )

    if best_goal_node is None:
        # Get the closest node to the goal if the goal is not found
        best_goal_node = get_closest_node_to_goal(node_list, goal_node)

    path = []
    path_orientation = []
    current_node = best_goal_node
    while current_node is not None:
        path.append((current_node.x, current_node.y, current_node.z))
        path_orientation.append(
            (
                current_node.x,
                current_node.y,
                current_node.z,
                current_node.roll,
                current_node.pitch,
                current_node.yaw,
            )
        )
        current_node = current_node.parent

    return path[::-1], path_orientation[::-1], node_list
