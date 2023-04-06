import sys
import os
import numpy as np
# Get the current working directory
current_directory = os.getcwd()

# Append the current working directory to the system path
sys.path.append(current_directory)

import matplotlib.pyplot as plt
from trajectory_planning import Waypoint, calcDubinsPath, dubins_traj


if __name__ == '__main__':

    # User's waypoints: [x, y, heading (degrees)]
    pt1 = Waypoint(0,0,0)
    pt2 = Waypoint(6000,7000,260)
    pt3 = Waypoint(1000,15000,215)
    pt4 = Waypoint(0,0,0)
    Wptz = [pt1,pt2, pt3, pt4]
    # Run the code
    i = 0
    while i < len(Wptz)-1:
        param = calcDubinsPath(Wptz[i], Wptz[i+1], 90, 20)
        path = dubins_traj(param,1)

        # Plot the results
        plt.plot(Wptz[i].x,Wptz[i].y,'kx')
        plt.plot(Wptz[i+1].x,Wptz[i+1].y,'kx')
        plt.plot(path[:,0],path[:,1],'b-')
        i+=1
    plt.grid(True)
    plt.axis("equal")
    plt.title('Dubin\'s Curves Trajectory Generation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
