import numpy as np


def control_function(q_current, t, robot, traj, Kp, Kd):
    # Find the desired state at current time
    idx = np.argmin(np.abs(traj.t - t))
    q_d = traj.q[idx]
    qd_d = traj.qd[idx]
    
    # Calculate error
    q_error = q_d - q_current[:6]
    qd_error = qd_d - q_current[6:]
    
    # PD control law
    tau_pd = Kp @ q_error + Kd @ qd_error
    
    # Inverse dynamics feedforward
    qdd_d = traj.qdd[idx]
    tau_ff = robot.rne(q_d, qd_d, qdd_d)
    
    # Total torque
    tau = tau_ff + tau_pd
    
    # Robot dynamics (simplified for simulation)
    M = robot.inertia(q_current[:6])
    C = robot.coriolis(q_current[:6], q_current[6:])
    G = robot.gravload(q_current[:6])
    
    qdd = np.linalg.inv(M) @ (tau - C @ q_current[6:] - G)
    
    # Return derivatives (velocities and accelerations)
    return np.concatenate((q_current[6:], qdd))