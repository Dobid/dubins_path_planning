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


# Convert degrees to radians
def deg2rad(x):
    return np.pi * x / 180.

# Initial and final configurations [x, y, z, heading angle, pitch angle]
qi = np.array([200., 500., 200., deg2rad(180.), deg2rad(-5.)])
qf = np.array([500., 350., 100., deg2rad(0.), deg2rad(-5.)])
# Minimum turning radius
rhomin = 40.
# Pitch angle constraints [min_pitch, max_pitch]
pitchmax = deg2rad(np.array([-15., 20.]))


maneuver = DubinsManeuver3D_func(qi, qf, rhomin, pitchmax)
# Length of the 3D Dubins path
print(f"maneuver.length = {maneuver.length}")

# Sample the maneuver by 500 samples
samples = compute_sampling(maneuver, numberOfSamples=500)
# First and last samples - should be equal to qi and qf
print(f"samples[0] = {samples[0]}")
print(f"samples[-1] = {samples[-1]}")
print(f"qi = {qi}")
print(f"qf = {qf}")

