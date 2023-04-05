import math 
from typing import List, Tuple
import numpy as np

from .node import Node

def euclidean_distance(node1: Node, node2: Node) -> float:
    """
    Calculate the Euclidean distance between two nodes.

    Args:
    node1 (Node): The first node.
    node2 (Node): The second node.

    Returns:
    float: The Euclidean distance between the nodes.
    """
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2 + (node1.z - node2.z)**2)

def orientation_distance(node1: Node, node2: Node) -> float:
    """
    Calculate the orientation distance between two nodes.

    Args:
    node1 (Node): The first node.
    node2 (Node): The second node.

    Returns:
    float: The orientation distance between the nodes.
    """
    roll_diff = abs(node1.roll - node2.roll)
    pitch_diff = abs(node1.pitch - node2.pitch)
    yaw_diff = abs(node1.yaw - node2.yaw)

    return roll_diff + pitch_diff + yaw_diff

def combined_distance(node1: Node, node2: Node, alpha: float = 0.5) -> float:
    """
    Calculate the combined distance between two nodes based on position and orientation.

    Args:
    node1 (Node): The first node.
    node2 (Node): The second node.
    alpha (float, optional): The weight of position distance in the combined distance. Defaults to 0.5.

    Returns:
    float: The combined distance between the nodes.
    """
    position_distance = euclidean_distance(node1, node2)
    o_distance = orientation_distance(node1, node2)
    
    return alpha * position_distance + (1 - alpha) * o_distance

def get_closest_node_to_goal(node_list: List[Node], goal_node: Node) -> Node:
    """
    Find the closest node to the goal node in a list of nodes.

    Args:
    node_list (List[Node]): The list of nodes.
    goal_node (Node): The goal node.

    Returns:
    Node: The closest node to the goal node.
    """
    closest_node = node_list[0]
    min_distance = euclidean_distance(closest_node, goal_node)

    for node in node_list[1:]:
        distance = euclidean_distance(node, goal_node)
        if distance < min_distance:
            min_distance = distance
            closest_node = node

    return closest_node

def get_nearest_node(node_list: List[Node], random_node: Node) -> Node:
    """
    Find the nearest node to the random node in a list of nodes.
    Args:
    node_list (List[Node]): The list of nodes.
    random_node (Node): The random node.

    Returns:
    Node: The nearest node to the random node.
    """
    nearest_node = node_list[0]
    min_distance = combined_distance(nearest_node, random_node)

    for node in node_list[1:]:
        distance = combined_distance(node, random_node)
        if distance < min_distance:
            min_distance = distance
            nearest_node = node

    return nearest_node


def collision_free(new_node: Node, parent_node: Node, obstacles: List[Tuple[float, float, float, float, float, float]]) -> bool:
    """
    Check if the path between new_node and its parent_node is collision-free.
    Args:
    new_node (Node): The new node.
    parent_node (Node): The parent node.
    obstacles (List[Tuple[float, float, float, float, float, float]]): A list of obstacles, where each obstacle is represented
        by a tuple (x_min, y_min, z_min, x_len, y_len, z_len).

    Returns:
    bool: True if the path between new_node and parent_node is collision-free, False otherwise.
    """
    for obs in obstacles:
        x_min, y_min, z_min, x_len, y_len, z_len = obs
        x_max, y_max, z_max = x_min + x_len + 1, y_min + y_len + 1, z_min + z_len + 1

        if x_min-1 <= new_node.x <= x_max and y_min-1 <= new_node.y <= y_max and z_min-1 <= new_node.z <= z_max:
            return False
    return True

def local_density(closest_node: Node, node_list: List[Node], radius: float) -> float:
    """
    Calculate the local density of nodes around the closest_node.

    Args:
    closest_node (Node): The node around which to calculate the local density.
    node_list (List[Node]): The list of nodes.
    radius (float): The radius to consider around the closest_node.

    Returns:
    float: The local density of nodes around the closest_node.
    """
    count = 0
    for node in node_list:
        if euclidean_distance(node, closest_node) < radius:
            count += 1
    return count / len(node_list)

def path_cost(path: List[Node]) -> float:
    """
    Calculate the total cost of the path.

    Args:
    path (List[Node]): The list of nodes in the path.

    Returns:
    float: The total cost of the path.
    """
    cost = 0
    for i in range(len(path) - 1):
        cost += combined_distance(path[i], path[i + 1])
    return cost

def get_path(goal_node: Node) -> List[Node]:
    """
    Retrieve the path from the start node to the goal node.

    Args:
    goal_node (Node): The goal node.

    Returns:
    List[Node]: The path from the start node to the goal node.
    """
    path = []
    current_node = goal_node
    while current_node is not None:
        path.append(current_node)
        current_node = current_node.parent

    return path[::-1]