import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R


def safe_quat_to_euler(quat):
    """Convert quaternion to Euler angles with zero-norm handling"""
    if np.linalg.norm(quat) < 1e-6:  # Tolerance for floating point errors
        return 0.0, 0.0, 0.0  # Default to neutral orientation
    return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz")


class PDController:
    def __init__(self, kp, kd, setpoint):
        self.kp = kp
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0

    def compute(
        self, measured_value, measured_velocity, desired_value, desired_velocity
    ):
        error = desired_value - measured_value
        derror = desired_velocity - measured_velocity
        output = (self.kp * error) + (self.kd * derror)
        return output

    def trajectory(self, time, t_final, C_final):
        a2 = 3 / t_final**2
        a3 = -2 / t_final**3
        Cposition = (a2 * time**2 + a3 * time**3) * C_final
        Cvelocity = (2 * a2 * time + 3 * time**2 * a3) * C_final
        Cacceleration = (2 * a2 + 6 * time * a3) * C_final
        return Cposition, Cvelocity, Cacceleration

    def quintic_trajectory(self, time, t_final, C_final):
        """
        Quintic (5th-order) trajectory for smooth position, velocity, and acceleration profiles.
        Moves from 0 to C_final in t_final seconds, with zero velocity and acceleration at endpoints.
        """
        tau = time / t_final
        if tau > 1:
            tau = 1  # Clamp to final value
        # Quintic polynomial coefficients for boundary conditions: pos/vel/accel = 0 at start/end
        position = C_final * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
        velocity = C_final * (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / t_final
        acceleration = C_final * (60 * tau - 180 * tau**2 + 120 * tau**3) / (t_final**2)
        return position, velocity, acceleration

    def altitudeThrust(self, zDesPos, zDesVel, zActPos, zActVel, mass):
        g = 9.81
        base_thrust = mass * g
        e = (zDesPos - zActPos) * 10
        edot = (zDesVel - zActVel) * 1
        return base_thrust + e + edot

    def attitude_control(
        self, measured_angle, measured_rate, desired_angle, desired_rate
    ):
        error = desired_angle - measured_angle
        rate_error = desired_rate - measured_rate
        return self.kp * error + self.kd * rate_error


m = mujoco.MjModel.from_xml_path("mujoco_menagerie-main/skydio_x2/scene.xml")
d = mujoco.MjData(m)
d.ctrl[0:4] = 4.0  # Set the thruster values to 0.5

# Parameters
kp = 80
kd = 5
t_final = 4.0  # Duration to reach 50 cm
C_final = 0.5  # Target height (50 cm)
pd_controller = PDController(kp, kd, setpoint=0.5)

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()

    # Initialize controllers OUTSIDE the loop
    roll_controller = PDController(kp=8, kd=0.5, setpoint=0)
    pitch_controller = PDController(kp=8, kd=0.5, setpoint=0)
    yaw_controller = PDController(kp=3, kd=0.1, setpoint=0)

    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()
        current_time = time.time() - start

        # Get desired position and velocity from your cubic trajectory
        zDesPos, zDesVel, _ = pd_controller.trajectory(
            min(current_time, t_final), t_final, C_final
        )
        # Get orientation FIRST
        quat = d.sensor("body_quat").data
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        roll, pitch, yaw = r.as_euler("xyz")

        # Desired angles (modify these for movement)
        desired_roll = 0.0  # radians
        desired_pitch = 0.0
        desired_yaw = 0.0

        # Initialize controllers
        roll_controller = PDController(kp=8, kd=0.5, setpoint=0)
        pitch_controller = PDController(kp=8, kd=0.5, setpoint=0)
        yaw_controller = PDController(kp=3, kd=0.1, setpoint=0)

        # NOW calculate moments
        roll_moment = roll_controller.attitude_control(roll, d.qvel[3], desired_roll, 0)
        pitch_moment = pitch_controller.attitude_control(
            pitch, d.qvel[4], desired_pitch, 0
        )
        yaw_moment = yaw_controller.attitude_control(yaw, d.qvel[5], desired_yaw, 0)

        # Get current position and velocity
        zActPos = d.qpos[2]
        zActVel = d.qvel[2]

        thrust = pd_controller.altitudeThrust(zDesPos, zDesVel, zActPos, zActVel, 0.027)
        print("altitude : ", zActPos)

        # Apply controls (indices match XML actuator order)
        d.ctrl[0] = thrust  # Body thrust (z-axis)
        d.ctrl[1] = roll_moment  # x_moment (roll)
        d.ctrl[2] = pitch_moment  # y_moment (pitch)
        d.ctrl[3] = yaw_moment  # z_moment (yaw)

        mujoco.mj_step(m, d)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
