
class Node:
    def __init__(
        self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float, cost: float = 0
    ):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.cost = cost
        self.parent = None
