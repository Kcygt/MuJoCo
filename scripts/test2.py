import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R


# Helper function to convert quaternion to Euler angles (ZYX convention)
def quat_to_euler(quat_wxyz):
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    r = R.from_quat(quat_xyzw)
    euler_rad = r.as_euler("xyz", degrees=False)  # roll (x), pitch (y), yaw (z)
    return euler_rad  # [roll, pitch, yaw]


# Generic PD Controller class
class PDController:
    def __init__(self, kp, kd, setpoint=0.0, trajectory_func=None, max_output=None):
        self.kp = kp
        self.kd = kd
        self.setpoint_pos = setpoint
        self.setpoint_vel = 0.0
        self.trajectory_func = trajectory_func
        self.max_output = max_output
        self.is_yaw_controller = False  # Default, can be overridden

    def update(self, measured_pos, measured_vel, current_time=None):
        if self.trajectory_func and current_time is not None:
            self.setpoint_pos, self.setpoint_vel = self.trajectory_func(
                current_sim_time=current_time
            )
        elif self.trajectory_func and current_time is None:
            # This case might happen if current_time is not passed for non-trajectory controllers
            # but they were initialized with one. For safety, let's assume static setpoint.
            pass

        pos_error = self.setpoint_pos - measured_pos
        vel_error = self.setpoint_vel - measured_vel

        if self.is_yaw_controller:
            pos_error = (pos_error + np.pi) % (2 * np.pi) - np.pi

        output = self.kp * pos_error + self.kd * vel_error

        if self.max_output is not None:
            output = np.clip(output, -self.max_output, self.max_output)
        return output


# Generic Minimum Jerk Trajectory
class MinJerkTrajectory:
    def __init__(self, initial_val, target_val, duration, start_time_offset=0.0):
        self.val0 = initial_val
        self.valf = target_val
        self.t_final = duration
        self.start_time = None  # Will be set on first call + offset
        self.start_time_offset = start_time_offset

    def __call__(self, current_sim_time):
        if self.start_time is None:
            self.start_time = current_sim_time + self.start_time_offset

        t = current_sim_time - self.start_time

        if t < 0:  # Before trajectory start
            return self.val0, 0.0
        if t > self.t_final:  # After trajectory end
            return self.valf, 0.0

        tau = t / self.t_final
        desired_pos = self.val0 + (self.valf - self.val0) * (
            10 * tau**3 - 15 * tau**4 + 6 * tau**5
        )
        desired_vel = (
            (self.valf - self.val0)
            / self.t_final
            * (30 * tau**2 - 60 * tau**3 + 30 * tau**4)
        )
        return desired_pos, desired_vel


m = mujoco.MjModel.from_xml_path("mujoco_menagerie-main/skydio_x2/scene.xml")
d = mujoco.MjData(m)

# --- Simulation Parameters ---
TARGET_POSITION_WORLD = np.array([0.0, 0.0, 0.5])  # Target X, Y, Z in world frame
TARGET_YAW_WORLD = 0.0  # Desired final yaw angle (radians)
TRAJECTORY_DURATION_XYZ = 8.0  # Duration for the XYZ trajectory
SIMULATION_DURATION = TRAJECTORY_DURATION_XYZ + 7.0  # Total simulation time

# --- PD Controller Gains ---
# Position (Outer Loop) - Output desired accelerations
kp_x_pos = 2.5  # Proportional gain for X position
kd_x_pos = 0.2  # Derivative gain for X position
kp_y_pos = 2.5  # Proportional gain for Y position
kd_y_pos = 0.2  # Derivative gain for Y position

# Altitude (Z Position) - Output is delta thrust force
kp_alt = 50.0
kd_alt = 2.0  # Slightly increased damping

# Attitude (Inner Loop) - Output is torque
kp_roll = 15.0  # Increased for responsiveness to XY commands
kd_roll = 0.5
kp_pitch = 15.0  # Increased for responsiveness to XY commands
kd_pitch = 0.5
kp_yaw = 3.0
kd_yaw = 0.8

# Drone and Environment Parameters
GRAVITY_MAGNITUDE = abs(m.opt.gravity[2])
MAX_TILT_ANGLE = np.deg2rad(20.0)  # Max allowed tilt for roll/pitch setpoints
MAX_ACCEL_XY = GRAVITY_MAGNITUDE * np.tan(
    MAX_TILT_ANGLE
)  # Max desired horizontal acceleration

# --- Initialize Trajectories ---
initial_drone_xyz = d.qpos[0:3].copy()  # Ensure it's a copy
print(f"Initial drone position: {initial_drone_xyz}")
print(f"Target drone position: {TARGET_POSITION_WORLD}")

x_trajectory = MinJerkTrajectory(
    initial_drone_xyz[0], TARGET_POSITION_WORLD[0], TRAJECTORY_DURATION_XYZ
)
y_trajectory = MinJerkTrajectory(
    initial_drone_xyz[1], TARGET_POSITION_WORLD[1], TRAJECTORY_DURATION_XYZ
)
z_trajectory = MinJerkTrajectory(
    initial_drone_xyz[2], TARGET_POSITION_WORLD[2], TRAJECTORY_DURATION_XYZ
)

# --- Initialize Controllers ---
# Position Controllers
x_pos_pd = PDController(
    kp_x_pos, kd_x_pos, trajectory_func=x_trajectory, max_output=MAX_ACCEL_XY
)
y_pos_pd = PDController(
    kp_y_pos, kd_y_pos, trajectory_func=y_trajectory, max_output=MAX_ACCEL_XY
)
altitude_pd = PDController(
    kp_alt, kd_alt, trajectory_func=z_trajectory
)  # Output is delta thrust

# Attitude Controllers
roll_pd = PDController(kp_roll, kd_roll, setpoint=0.0)  # Setpoint updated in loop
pitch_pd = PDController(kp_pitch, kd_pitch, setpoint=0.0)  # Setpoint updated in loop
yaw_pd = PDController(kp_yaw, kd_yaw, setpoint=TARGET_YAW_WORLD)
yaw_pd.is_yaw_controller = True

# Drone Mass and Hover Thrust
body_x2_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "x2")
if body_x2_id != -1 and m.body_mass[body_x2_id] > 0:
    total_mass = m.body_mass[body_x2_id]
else:
    total_mass = 4 * 0.25 + 0.325  # From XML geom masses
print(f"Estimated Total Mass: {total_mass:.3f} kg")
hover_thrust_total = total_mass * GRAVITY_MAGNITUDE

motor_ctrl_min, motor_ctrl_max = m.actuator_ctrlrange[0]
print(f"Hover thrust total: {hover_thrust_total:.2f} N")
print(f"Motor control range: {motor_ctrl_min} to {motor_ctrl_max}")

# Data logging
log_time = []
log_pos = []
log_vel = []
log_euler = []
log_setpoint_pos = []
log_setpoint_att = []
log_ctrl = []


with mujoco.viewer.launch_passive(m, d) as viewer:
    simulation_start_time = time.time()
    while (
        viewer.is_running() and d.time < SIMULATION_DURATION
    ):  # Use d.time for sim duration
        step_start_time = time.time()
        current_sim_time_in_loop = d.time

        # --- Read State ---
        current_pos_xyz = d.qpos[0:3]
        current_vel_xyz = d.qvel[0:3]
        current_x, current_y, current_z = current_pos_xyz
        current_vx, current_vy, current_vz = current_vel_xyz

        current_quat_wxyz = d.qpos[3:7]
        current_euler_rpy = quat_to_euler(current_quat_wxyz)
        current_roll, current_pitch, current_yaw = current_euler_rpy

        current_roll_rate, current_pitch_rate, current_yaw_rate = d.qvel[3:6]

        # --- Outer Loop: Position Control ---
        # Calculate desired world-frame accelerations (u_x, u_y)
        u_x_world = x_pos_pd.update(current_x, current_vx, current_sim_time_in_loop)
        u_y_world = y_pos_pd.update(current_y, current_vy, current_sim_time_in_loop)

        # Altitude control: PD output is delta thrust from hover
        delta_thrust_z = altitude_pd.update(
            current_z, current_vz, current_sim_time_in_loop
        )
        total_vertical_thrust_needed = hover_thrust_total + delta_thrust_z

        # --- Attitude Setpoint Calculation ---
        # Convert desired world accelerations (u_x_world, u_y_world) to desired roll/pitch angles
        # This uses the drone's current yaw to rotate desired accelerations into the body frame's perspective for tilt.
        # ax_body_desired = u_x_world * cos(yaw) + u_y_world * sin(yaw)
        # ay_body_desired = -u_x_world * sin(yaw) + u_y_world * cos(yaw)
        # pitch_setpoint ~ -ax_body_desired / g  (positive pitch = nose up = negative X-body acceleration)
        # roll_setpoint  ~  ay_body_desired / g   (positive roll = left wing down = positive Y-body acceleration (if Y_body is left))

        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)

        # Desired pitch to achieve u_x_world and part of u_y_world
        pitch_setpoint = (
            -(u_x_world * cos_yaw + u_y_world * sin_yaw) / GRAVITY_MAGNITUDE
        )
        # Desired roll to achieve u_y_world and part of u_x_world
        roll_setpoint = (
            -u_x_world * sin_yaw + u_y_world * cos_yaw
        ) / GRAVITY_MAGNITUDE  # Positive roll = right wing down

        # Clip desired angles
        pitch_setpoint = np.clip(pitch_setpoint, -MAX_TILT_ANGLE, MAX_TILT_ANGLE)
        roll_setpoint = np.clip(roll_setpoint, -MAX_TILT_ANGLE, MAX_TILT_ANGLE)

        # Update setpoints for inner-loop attitude controllers
        roll_pd.setpoint_pos = roll_setpoint
        pitch_pd.setpoint_pos = pitch_setpoint
        # yaw_pd.setpoint_pos is fixed (or could have its own trajectory)

        # --- Inner Loop: Attitude Control ---
        roll_torque_cmd = roll_pd.update(current_roll, current_roll_rate)
        pitch_torque_cmd = pitch_pd.update(current_pitch, current_pitch_rate)
        yaw_torque_cmd = yaw_pd.update(
            current_yaw, current_yaw_rate, current_sim_time_in_loop
        )  # Pass time for yaw_pd if it ever gets a trajectory

        # --- Mixer: Convert Thrust and Torques to Motor Commands ---
        # total_vertical_thrust_needed is thrust required in Z_world.
        # Convert to thrust along drone's body-Z axis:
        thrust_along_body_z = total_vertical_thrust_needed / (
            np.cos(current_roll) * np.cos(current_pitch)
            + 1e-6  # Add epsilon for stability
        )
        # Ensure thrust_along_body_z is not negative (e.g. if desired Z is far below and PD commands huge negative delta)
        thrust_along_body_z = max(0, thrust_along_body_z)

        # Standard X-quad mixer (d.ctrl[0] to d.ctrl[3] -> BL, FL, FR, BR)
        # Positive roll_torque_cmd: rolls right (increases left motors FL,BL; decreases right motors FR,BR)
        # Positive pitch_torque_cmd: pitches nose up (increases rear motors BL,BR; decreases front motors FL,FR)
        # Positive yaw_torque_cmd: yaws left/CCW (increases motors with CCW reaction torque BL,FR; decreases motors with CW reaction torque FL,BR)

        base_thrust_per_motor = thrust_along_body_z / 1.0

        # ctrl[0] is Back-Left (BL)
        # ctrl[1] is Front-Left (FL)
        # ctrl[2] is Front-Right (FR)
        # ctrl[3] is Back-Right (BR)

        m0_thrust = (
            base_thrust_per_motor - roll_torque_cmd + pitch_torque_cmd + yaw_torque_cmd
        )  # BL
        m1_thrust = (
            base_thrust_per_motor + roll_torque_cmd - pitch_torque_cmd - yaw_torque_cmd
        )  # FL
        m2_thrust = (
            base_thrust_per_motor + roll_torque_cmd - pitch_torque_cmd + yaw_torque_cmd
        )  # FR
        m3_thrust = (
            base_thrust_per_motor - roll_torque_cmd + pitch_torque_cmd - yaw_torque_cmd
        )  # BR

        d.ctrl[0] = np.clip(m0_thrust, motor_ctrl_min, motor_ctrl_max)
        d.ctrl[1] = np.clip(m1_thrust, motor_ctrl_min, motor_ctrl_max)
        d.ctrl[2] = np.clip(m2_thrust, motor_ctrl_min, motor_ctrl_max)
        d.ctrl[3] = np.clip(m3_thrust, motor_ctrl_min, motor_ctrl_max)

        # Logging data
        log_time.append(d.time)
        log_pos.append(current_pos_xyz.copy())
        log_vel.append(current_vel_xyz.copy())
        log_euler.append(np.array([current_roll, current_pitch, current_yaw]))
        log_setpoint_pos.append(
            np.array(
                [x_pos_pd.setpoint_pos, y_pos_pd.setpoint_pos, altitude_pd.setpoint_pos]
            )
        )
        log_setpoint_att.append(
            np.array([roll_pd.setpoint_pos, pitch_pd.setpoint_pos, yaw_pd.setpoint_pos])
        )
        log_ctrl.append(d.ctrl[:4].copy())

        # Print some info (optional)
        if int(d.time * 100) % 50 == 0:  # Print every 0.5 seconds
            print(
                f"T:{d.time:0.2f} Pos:[{current_x:0.2f},{current_y:0.2f},{current_z:0.2f}]m DesPos:[{x_pos_pd.setpoint_pos:0.2f},{y_pos_pd.setpoint_pos:0.2f},{altitude_pd.setpoint_pos:0.2f}]m"
            )
            print(
                f"  Roll:{np.rad2deg(current_roll):0.1f}({np.rad2deg(roll_setpoint):0.1f}) Pitch:{np.rad2deg(current_pitch):0.1f}({np.rad2deg(pitch_setpoint):0.1f}) Yaw:{np.rad2deg(current_yaw):0.1f}({np.rad2deg(yaw_pd.setpoint_pos):0.1f}) deg"
            )
            # print(f"  Ux_w:{u_x_world:0.2f} Uy_w:{u_y_world:0.2f} dThrustZ:{delta_thrust_z:0.2f} TotalBodyZThrust:{thrust_along_body_z:0.2f}")
            # print(f"  RollCmd:{roll_torque_cmd:0.2f} PitchCmd:{pitch_torque_cmd:0.2f} YawCmd:{yaw_torque_cmd:0.2f}")
            # print(f"  Motors: {d.ctrl[0]:0.2f} {d.ctrl[1]:0.2f} {d.ctrl[2]:0.2f} {d.ctrl[3]:0.2f}\n")

        mujoco.mj_step(m, d)
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
        viewer.sync()

        time_spent_in_step = time.time() - step_start_time
        time_to_wait = m.opt.timestep - time_spent_in_step
        if time_to_wait > 0:
            time.sleep(time_to_wait)

# After simulation, plot results
import matplotlib.pyplot as plt

log_time = np.array(log_time)
log_pos = np.array(log_pos)
log_vel = np.array(log_vel)
log_euler = np.rad2deg(np.array(log_euler))
log_setpoint_pos = np.array(log_setpoint_pos)
log_setpoint_att = np.rad2deg(np.array(log_setpoint_att))
log_ctrl = np.array(log_ctrl)

fig, axs = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle("Quadcopter Trajectory Control Performance", fontsize=16)

# X Position
axs[0, 0].plot(log_time, log_pos[:, 0], label="Actual X")
axs[0, 0].plot(log_time, log_setpoint_pos[:, 0], label="Desired X", linestyle="--")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("X Position (m)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Y Position
axs[0, 1].plot(log_time, log_pos[:, 1], label="Actual Y")
axs[0, 1].plot(log_time, log_setpoint_pos[:, 1], label="Desired Y", linestyle="--")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Y Position (m)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Z Position (Altitude)
axs[1, 0].plot(log_time, log_pos[:, 2], label="Actual Z (Altitude)")
axs[1, 0].plot(log_time, log_setpoint_pos[:, 2], label="Desired Z", linestyle="--")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Z Position (m)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Roll and Pitch
axs[1, 1].plot(log_time, log_euler[:, 0], label="Actual Roll")
axs[1, 1].plot(log_time, log_setpoint_att[:, 0], label="Desired Roll", linestyle="--")
axs[1, 1].plot(log_time, log_euler[:, 1], label="Actual Pitch")
axs[1, 1].plot(log_time, log_setpoint_att[:, 1], label="Desired Pitch", linestyle="--")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Angle (degrees)")
axs[1, 1].legend()
axs[1, 1].grid(True)

# Yaw
axs[2, 0].plot(log_time, log_euler[:, 2], label="Actual Yaw")
axs[2, 0].plot(log_time, log_setpoint_att[:, 2], label="Desired Yaw", linestyle="--")
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel("Yaw Angle (degrees)")
axs[2, 0].legend()
axs[2, 0].grid(True)

# Motor Controls
axs[2, 1].plot(log_time, log_ctrl[:, 0], label="Motor 0 (BL)")
axs[2, 1].plot(log_time, log_ctrl[:, 1], label="Motor 1 (FL)")
axs[2, 1].plot(log_time, log_ctrl[:, 2], label="Motor 2 (FR)")
axs[2, 1].plot(log_time, log_ctrl[:, 3], label="Motor 3 (BR)")
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel("Motor Command")
axs[2, 1].legend()
axs[2, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
