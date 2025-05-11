import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

# Quintic trajectory for smooth altitude
def quintic_trajectory(time, t_final, target):
    tau = np.clip(time / t_final, 0, 1)
    pos = target * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
    vel = target * (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / t_final
    acc = target * (60 * tau - 180 * tau**2 + 120 * tau**3) / t_final**2
    return pos, vel, acc

# Extract roll, pitch, yaw from quaternion
def quat_to_euler(q):
    # MuJoCo quaternions: [w, x, y, z]
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    return r.as_euler('xyz', degrees=False)  # roll, pitch, yaw

# PD Controller class for general use
class PDController:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def compute(self, target, current, target_dot, current_dot):
        return self.kp * (target - current) + self.kd * (target_dot - current_dot)

# Load model and data
m = mujoco.MjModel.from_xml_path('mujoco_menagerie-main/skydio_x2/scene.xml')
d = mujoco.MjData(m)

# Control gains (tune as needed)
kp_z, kd_z = 18, 6
kp_r, kd_r = 10, 2
kp_p, kd_p = 10, 2
kp_y, kd_y = 4, 1

# Hover thrust from your XML keyframe
hover_thrust = 3.25

# Controllers
alt_controller = PDController(kp_z, kd_z)
roll_controller = PDController(kp_r, kd_r)
pitch_controller = PDController(kp_p, kd_p)
yaw_controller = PDController(kp_y, kd_y)

# Trajectory parameters
target_z = 0.5   # 50 cm
traj_time = 4.0  # seconds to reach target

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()
        sim_time = time.time() - start

        # Reference trajectory for altitude
        z_des, z_dot_des, _ = quintic_trajectory(sim_time, traj_time, target_z)

        # Get current state
        z = d.qpos[2]
        z_dot = d.qvel[2]

        # Orientation: quaternion in qpos[3:7]
        quat = d.qpos[3:7]
        roll, pitch, yaw = quat_to_euler(quat)
        # Angular velocities in qvel[3:6]
        roll_dot, pitch_dot, yaw_dot = d.qvel[3:6]

        # Altitude control (output: thrust adjustment)
        u_z = alt_controller.compute(z_des, z, z_dot_des, z_dot)

        # Attitude control (outputs: torque adjustments)
        u_roll = roll_controller.compute(0, roll, 0, roll_dot)
        u_pitch = pitch_controller.compute(0, pitch, 0, pitch_dot)
        u_yaw = yaw_controller.compute(0, yaw, 0, yaw_dot)

        # Motor mixing for "+" quadrotor (adjust if needed for "x" config)
        # [front left, front right, rear right, rear left]
        thrust_cmd = np.zeros(4)
        thrust_cmd[0] = hover_thrust + u_z + u_roll + u_pitch - u_yaw  # FL
        thrust_cmd[1] = hover_thrust + u_z - u_roll + u_pitch + u_yaw  # FR
        thrust_cmd[2] = hover_thrust + u_z - u_roll - u_pitch - u_yaw  # RR
        thrust_cmd[3] = hover_thrust + u_z + u_roll - u_pitch + u_yaw  # RL

        # Clamp thrusts to actuator range (from your XML: 0 to 13)
        thrust_cmd = np.clip(thrust_cmd, 0, 13)
        d.ctrl[0:4] = thrust_cmd

        mujoco.mj_step(m, d)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
        viewer.sync()

        # Timing
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
