import sys
import os
import numpy as np
# Get the current working directory
current_directory = os.getcwd()

# Append the current working directory to the system path
sys.path.append(current_directory)
import math
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple


from trajectory_planning import DubinsManeuver3D_func, compute_sampling
import matplotlib.pyplot as plt


# Convert degrees to radians
def deg2rad(x):
    return np.pi * x / 180.

# Initial and final configurations [x, y, z, heading angle, pitch angle]
# qi = np.array([200., 500., 100., deg2rad(0.), deg2rad(-5.)])
# qf = np.array([400., 150., 250., deg2rad(0.), deg2rad(-5.)])
qi = np.array([0, 0, 600, deg2rad(0.), deg2rad(-0.)])
qf = np.array([61, 37, 589, deg2rad(0.), deg2rad(-0.)])
# Minimum turning radius
rhomin = 40.
# Pitch angle constraints [min_pitch, max_pitch]
pitchmax = deg2rad(np.array([-15., 20.]))


maneuver = DubinsManeuver3D_func(qi, qf, rhomin, pitchmax)
print(f"maneuver = {maneuver}")

# Sample the maneuver by 500 samples
samples = compute_sampling(maneuver, numberOfSamples=500)
# First and last samples - should be equal to qi and qf
print(f"samples[0] = {samples[0]}")
print(f"samples[-1] = {samples[-1]}")
print(f"qi = {qi}")
print(f"qf = {qf}")
np_samples = np.array(samples)

# Visualize the maneuver
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Dubins 3D Maneuver")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-200, 200])
ax.set_ylim([-200, 200])
ax.set_zlim([550, 650])
# ax.view_init(elev=20, azim=30)
# Plot the maneuver
ax.plot(np_samples[:, 0], np_samples[:, 1], np_samples[:, 2], color='b', label='Dubins Path')
# Plot the start and end points
ax.scatter(qi[0], qi[1], qi[2], color='g', s=100, label='Start Point')
ax.scatter(qf[0], qf[1], qf[2], color='r', s=100, label='End Point')
ax.legend()
plt.show()
