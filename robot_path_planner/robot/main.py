from math import pi
import numpy as np
import matplotlib.pyplot as plt
from spatialmath import SE3
import roboticstoolbox as rtb
from roboticstoolbox.models.DH import Puma560  # or your robot
import sys
import os
import spatialmath.base as sb
from scipy.integrate import odeint

# Initialize robot
robot = rtb.models.DH.KR5()
# Visualize initial config
robot.plot(robot.q, backend="pyplot", block=False)

#Setting the link masses
robot.links[0].m = 6.5
robot.links[1].m = 6
robot.links[2].m = 3
robot.links[3].m = 2.5
robot.links[4].m = 1.5
robot.links[5].m = 1

# === Center of mass for each link in the local frame (adjusted slightly from original) ===
robot.links[0].r = [0, -0.03, 0.09]     # Shifted slightly more in y and z
robot.links[1].r = [0, 0.09, 0.01]      # Slightly reduced y, added small z component
robot.links[2].r = [0, 0, 0.045]        # Slightly shorter offset
robot.links[3].r = [0, 0, 0.035]        # Small adjustment for mass center
robot.links[4].r = [0, 0, 0.025]        # Slightly longer than before
robot.links[5].r = [0, 0, 0.012]        # Small extension to make model unique

# === Inertia tensors (Ix, Iy, Iz, Ixy, Ixz, Iyz) in local frame, slightly changed ===
robot.links[0].I = [0.09, 0.07, 0.045, 0, 0, 0]
robot.links[1].I = [0.075, 0.065, 0.038, 0, 0, 0]
robot.links[2].I = [0.018, 0.018, 0.009, 0, 0, 0]
robot.links[3].I = [0.013, 0.0013, 0.009, 0, 0, 0]
robot.links[4].I = [0.009, 0.009, 0.0045, 0, 0, 0]
robot.links[5].I = [0.0045, 0.0045, 0.0025, 0, 0, 0]

# === Motor inertia (Jm) - adjusted slightly while keeping mechanical plausibility ===
robot.links[0].Jm = 0.00038
robot.links[1].Jm = 0.00041
robot.links[2].Jm = 0.00036
robot.links[3].Jm = 0.00003
robot.links[4].Jm = 0.000035
robot.links[5].Jm = 0.000031

# === Viscous friction coefficient at the motor shaft (B) - modified slightly ===
robot.links[0].B = 0.0014
robot.links[1].B = 0.00095
robot.links[2].B = 0.00125
robot.links[3].B = 0.00007
robot.links[4].B = 0.000085
robot.links[5].B = 0.000034

# === Coulomb friction (positive and negative direction) - tweaked to add variance ===
robot.links[0].Tc = [0.385, -0.425]
robot.links[1].Tc = [0.121, -0.068]
robot.links[2].Tc = [0.128, -0.101]
robot.links[3].Tc = [0.0108, -0.0164]
robot.links[4].Tc = [0.0089, -0.0141]
robot.links[5].Tc = [0.0036, -0.0099]

# === Gear ratios (G) - kept close to original but slightly reduced/increased ===
robot.links[0].G = 60.1
robot.links[1].G = 105.2
robot.links[2].G = 51.7
robot.links[3].G = 74.0
robot.links[4].G = 70.2
robot.links[5].G = 75.3

# === Joint limits (min and max joint angles in radians) - small changes to ranges ===
robot.links[0].qlim = [-2.75, 2.75]            # Slightly tighter limits
robot.links[1].qlim = [-1.9, 1.9]
robot.links[2].qlim = [0, 0.14]                # Slightly shortened prismatic range
robot.links[3].qlim = [-4.6, 4.6]
robot.links[4].qlim = [-2.05, 2.05]
robot.links[5].qlim = [-6.2, 6.2]

q_start = [0, -pi/3, -pi/4, pi/3, -pi/3, 0]  # desired start pose
#q_start = [ pi/6, pi/4, -pi/4, pi/4, -pi/3, pi]
robot.qz = q_start
robot.plot(robot.qz, backend="pyplot", block=False)


# Add the parent directory to the Python path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))  # Go up one level
sys.path.append(project_root)
from vission.detector import get_barcode
from projectutils.pid_controller import control_function

pixel_center = get_barcode()  # returns ((x, y), "corners")
print("Pixel center:", pixel_center[0])

scale = 0.001  # 1 pixel = 1mm
offset = np.array([320, 240])  # image center (camera calibrated)

# Calculate displacement from center
diff = (np.array(pixel_center[0]) - offset) * scale

# Map to world coordinates, Z = 0.2m
target_world = SE3(0.5 + diff[0], diff[1], 0.2)
print("Target Pose (World):", target_world)


q_start = [0, -pi/3, -pi/4, pi/3, -pi/3, 0]

# q_end = robot.ikine_GN(T_end).q
solution = robot.ikine_LM(target_world)
q_end =solution.q
# Time parameters
N = 100
t_start = 0
t_stop = 5
time = np.linspace(t_start, t_stop, N)
# Generate trajectory (using quintic polynomial from Lab 2)
trajectory = rtb.mtraj(rtb.quintic, q_start, q_end, time)

# PD Controller Parameters
Kp = np.diag([100, 80, 50, 30, 30, 30])  # Proportional gains
Kd = np.diag([50, 40, 25, 10, 10, 10])  # Derivative gains

# Initial state (position and velocity)
q0 = np.concatenate((q_start, np.zeros(6)))

# Simulate the system
solution = odeint(control_function, q0, time, args=(robot, trajectory, Kp, Kd))
q_sim = solution[:, :6]
qd_sim = solution[:, 6:]

# Compute end-effector trajectory from desired trajectory (planned)
ee_positions = np.array([robot.fkine(q).t for q in trajectory.q])

# Plot 3D end-effector path from planned trajectory
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], label='Planned path')
ax.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2], c='green', label='Start')
ax.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2], c='red', label='Target')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Planned Trajectory of End-Effector')
ax.legend()
plt.tight_layout()
plt.show()

# === Animate the Simulated Motion ===
# Use the actual simulation result from the PD controller (q_sim)
print("Animating robot motion using simulated PD controller result...")

robot.plot(q_sim, block=True, backend="pyplot")

ee_sim_positions = np.array([robot.fkine(q).t for q in q_sim])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], label='Planned')
ax.plot(ee_sim_positions[:, 0], ee_sim_positions[:, 1], ee_sim_positions[:, 2], label='Simulated', linestyle='--')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Planned vs Simulated End-Effector Paths')
ax.legend()
plt.show()
