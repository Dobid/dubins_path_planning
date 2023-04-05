
class Node:
    def __init__(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.parent = None