import time
import numpy as np
import mujoco
import mujoco.viewer


# Helper function to convert quaternion to Euler angles (ZYX convention)
# MuJoCo's mju_quat2euler uses ZYX convention: result is [roll, pitch, yaw]
# where roll is about new X, pitch about new Y, yaw about new Z.
# For typical aerospace, roll is about body X, pitch about body Y, yaw about body Z.
# mj_quat2euler expects q = [w, x, y, z]
# d.qpos[3:7] is [w, x, y, z]
from scipy.spatial.transform import Rotation as R


def quat_to_euler(quat_wxyz):
    # Convert [w, x, y, z] to [x, y, z, w] for scipy
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    r = R.from_quat(quat_xyzw)
    euler_rad = r.as_euler("xyz", degrees=False)  # roll (x), pitch (y), yaw (z)
    return euler_rad  # [roll, pitch, yaw]


# Generic PD Controller class
class PDController:
    def __init__(self, kp, kd, setpoint=0.0, trajectory_func=None, max_output=None):
        self.kp = kp
        self.kd = kd
        self.setpoint_pos = setpoint  # Desired position (angle or altitude)
        self.setpoint_vel = 0.0  # Desired velocity (angular rate or vertical speed)
        self.trajectory_func = trajectory_func
        self.max_output = max_output

    def update(self, measured_pos, measured_vel, current_time=None):
        if self.trajectory_func:
            self.setpoint_pos, self.setpoint_vel = self.trajectory_func(current_time)

        pos_error = self.setpoint_pos - measured_pos
        vel_error = self.setpoint_vel - measured_vel

        # Special handling for yaw error to keep it in [-pi, pi]
        if hasattr(self, "is_yaw_controller") and self.is_yaw_controller:
            pos_error = (pos_error + np.pi) % (2 * np.pi) - np.pi

        output = self.kp * pos_error + self.kd * vel_error

        if self.max_output is not None:
            output = np.clip(output, -self.max_output, self.max_output)
        return output


# Altitude Trajectory (Minimum Jerk)
class AltitudeTrajectory:
    def __init__(self, initial_height, target_height, duration):
        self.h0 = initial_height
        self.hf = target_height
        self.t_final = duration
        self.start_time = None  # Will be set on first call

    def __call__(self, current_sim_time):
        if self.start_time is None:
            self.start_time = current_sim_time

        t = current_sim_time - self.start_time

        if t < 0:
            t = 0  # Ensure time is not negative if called before start
        if t > self.t_final:
            return self.hf, 0.0  # Maintain position after reaching target

        tau = t / self.t_final
        # Position: h(t) = h0 + (hf - h0) * (10*tau^3 - 15*tau^4 + 6*tau^5)
        desired_pos = self.h0 + (self.hf - self.h0) * (
            10 * tau**3 - 15 * tau**4 + 6 * tau**5
        )
        # Velocity: v(t) = (hf - h0)/t_final * (30*tau^2 - 60*tau^3 + 30*tau^4)
        desired_vel = (
            (self.hf - self.h0)
            / self.t_final
            * (30 * tau**2 - 60 * tau**3 + 30 * tau**4)
        )
        return desired_pos, desired_vel


m = mujoco.MjModel.from_xml_path("mujoco_menagerie-main/skydio_x2/scene.xml")
d = mujoco.MjData(m)

# --- PD Controller Gains ---
# Altitude
kp_alt = 150.0  # Proportional gain for height control
kd_alt = 20.0  # Derivative gain for damping for height control (critically damped: 2*sqrt(kp_alt*mass_eff))
# For mass_eff around 1.3kg, kd_alt ~ 2*sqrt(150*1.3) ~ 2*sqrt(195) ~ 2*14 ~ 28. Let's try 20-30.

# Roll
kp_roll = 10.0  # Proportional gain for roll stabilization
kd_roll = 2.0  # Derivative gain for roll damping

# Pitch
kp_pitch = 10.0  # Proportional gain for pitch stabilization
kd_pitch = 2.0  # Derivative gain for pitch damping

# Yaw
kp_yaw = 2.5  # Proportional gain for yaw stabilization
kd_yaw = 0.5  # Derivative gain for yaw damping

# --- Initialize Controllers ---
# Initial height from qpos or a known start
initial_drone_z = d.qpos[
    2
]  # or m.key_qpos[m.key_name2id('hover')][2] if starting from hover
altitude_trajectory = AltitudeTrajectory(
    initial_height=initial_drone_z, target_height=0.8, duration=5.0
)
altitude_pd = PDController(kp_alt, kd_alt, trajectory_func=altitude_trajectory)

# Attitude controllers (setpoint is 0 for stabilization, no trajectory needed for now)
roll_pd = PDController(kp_roll, kd_roll, setpoint=0.0)
pitch_pd = PDController(kp_pitch, kd_pitch, setpoint=0.0)
yaw_pd = PDController(kp_yaw, kd_yaw, setpoint=0.0)
yaw_pd.is_yaw_controller = True  # For angle wrapping

# Drone parameters
total_mass = sum(m.body_mass[1:])  # Sum mass of all bodies except world
# Or, more accurately from the XML if bodies have sub-components and no explicit inertial tag:
# rotor_mass = 0.25
# visual_ellipsoid_mass = 0.325
# total_mass = 4 * rotor_mass + visual_ellipsoid_mass # Approx 1.325 kg
# For Skydio X2, from your XML, sum of geom masses is 4*0.25 (rotors) + 0.325 (ellipsoid) = 1.325 kg
# This is a bit of a hack, MuJoCo computes inertial properties.
# A more robust way to get total mass if the root body has an inertial tag or if all sub-bodies have masses:
body_x2_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "x2")
if body_x2_id != -1 and m.body_mass[body_x2_id] > 0:  # if x2 body itself has mass
    total_mass = m.body_mass[body_x2_id]
else:  # sum components
    total_mass = 4 * 0.25 + 0.325  # From your XML geom masses
print(f"Estimated Total Mass: {total_mass:.3f} kg")

gravity = m.opt.gravity[2]  # Should be -9.81
hover_thrust_total = -total_mass * gravity  # Total thrust to counteract gravity

# Motor control limits from XML
motor_ctrl_min, motor_ctrl_max = m.actuator_ctrlrange[
    0
]  # Assuming all motors have same range

print(f"Hover thrust total: {hover_thrust_total:.2f} N")
print(f"Motor control range: {motor_ctrl_min} to {motor_ctrl_max}")

with mujoco.viewer.launch_passive(m, d) as viewer:
    simulation_start_time = time.time()
    while viewer.is_running() and time.time() - simulation_start_time < 100:
        step_start_time = time.time()
        current_sim_time_in_loop = (
            d.time
        )  # Use MuJoCo's simulation time for controllers

        # --- Read State ---
        # Position and Linear Velocity (World Frame)
        # qpos[0:3] = [x, y, z]
        # qvel[0:3] = [vx, vy, vz]
        current_z = d.qpos[2]
        current_z_vel = d.qvel[2]

        # Orientation (Quaternion w,x,y,z) and Angular Velocity (Body Frame)
        # qpos[3:7] = [qw, qx, qy, qz]
        # qvel[3:6] = [omega_x, omega_y, omega_z] (body frame angular velocities)
        current_quat_wxyz = d.qpos[3:7]
        current_euler_rpy = quat_to_euler(current_quat_wxyz)  # [roll, pitch, yaw]
        current_roll, current_pitch, current_yaw = (
            current_euler_rpy[0],
            current_euler_rpy[1],
            current_euler_rpy[2],
        )

        current_roll_rate = d.qvel[3]
        current_pitch_rate = d.qvel[4]
        current_yaw_rate = d.qvel[5]

        # --- PD Control Calculations ---
        # Altitude
        # The altitude PD controller directly outputs the desired total *vertical* thrust force.
        thrust_cmd_z = altitude_pd.update(
            current_z, current_z_vel, current_sim_time_in_loop
        )
        # Add gravity compensation. The PD output is an adjustment around hover.
        # Or, the PD controller should be tuned to output the total required thrust.
        # If PD setpoint is absolute height, its output should be total force.
        # Let's assume PD output is total desired vertical force.
        total_thrust_command = (
            thrust_cmd_z + hover_thrust_total
        )  # PD output is delta from hover_thrust

        # Attitude (Roll, Pitch, Yaw) - these output torque commands
        roll_torque_cmd = roll_pd.update(current_roll, current_roll_rate)
        pitch_torque_cmd = pitch_pd.update(current_pitch, current_pitch_rate)
        yaw_torque_cmd = yaw_pd.update(current_yaw, current_yaw_rate)

        # --- Mixer: Convert Thrust and Torques to Motor Commands ---
        # This basic mixer assumes torque commands are somewhat scaled to thrust differences.
        # You might need to add scaling factors (k_roll, k_pitch, k_yaw_rate_to_thrust_diff)
        # if the PD torque outputs are not directly comparable to thrust units.
        # For now, let's assume PD gains are tuned such that outputs are "thrust-like".

        # base_thrust_per_motor is the thrust each motor would provide if only vertical motion.
        # However, total_thrust_command is along the drone's body-Z, which might be tilted.
        # For small angles, cos(roll)*cos(pitch) ~ 1.
        # More accurately: thrust_along_body_z = total_vertical_thrust / (cos(roll)*cos(pitch))
        # This makes the controller more robust to tilts.
        thrust_along_body_z = total_thrust_command / (
            np.cos(current_roll) * np.cos(current_pitch) + 1e-6
        )  # add epsilon to avoid div by zero

        # Mixer based on standard X-configuration:
        # Motor 1: Back-Left (-.14, -.18) (corresponds to d.ctrl[0])
        # Motor 2: Front-Left (-.14, .18) (corresponds to d.ctrl[1])
        # Motor 3: Front-Right (.14, .18) (corresponds to d.ctrl[2])
        # Motor 4: Back-Right (.14, -.18) (corresponds to d.ctrl[3])

        # Positive roll torque (roll_torque_cmd > 0) should roll drone to the right (increase left motors, decrease right)
        # Positive pitch torque (pitch_torque_cmd > 0) should pitch drone nose up (increase rear motors, decrease front)
        # Positive yaw torque (yaw_torque_cmd > 0) should yaw drone left (CCW from top)

        # Mixer (check signs carefully based on motor numbering and desired torque directions)
        # m1 (Back-Left, CW from XML gear .0201)
        # m2 (Front-Left, CCW from XML gear -.0201)
        # m3 (Front-Right, CW from XML gear .0201)
        # m4 (Back-Right, CCW from XML gear -.0201)

        # To roll right (positive roll command): m2,m1 increase, m3,m4 decrease.
        #   roll_effect_m1 = -roll_torque_cmd
        #   roll_effect_m2 = +roll_torque_cmd
        #   roll_effect_m3 = +roll_torque_cmd
        #   roll_effect_m4 = -roll_torque_cmd

        # To pitch nose up (positive pitch command): m1,m4 increase, m2,m3 decrease
        #   pitch_effect_m1 = +pitch_torque_cmd
        #   pitch_effect_m2 = -pitch_torque_cmd
        #   pitch_effect_m3 = -pitch_torque_cmd
        #   pitch_effect_m4 = +pitch_torque_cmd

        # To yaw left/CCW (positive yaw command, using reaction torques from gear):
        #   Increase thrust of motors that spin CW (1&3, reaction torque CCW)
        #   Decrease thrust of motors that spin CCW (2&4, reaction torque CW)
        #   yaw_effect_m1 = +yaw_torque_cmd
        #   yaw_effect_m2 = -yaw_torque_cmd
        #   yaw_effect_m3 = +yaw_torque_cmd
        #   yaw_effect_m4 = -yaw_torque_cmd

        # Combining:
        m1_thrust = (
            thrust_along_body_z / 4.0
            - roll_torque_cmd
            + pitch_torque_cmd
            + yaw_torque_cmd
        )
        m2_thrust = (
            thrust_along_body_z / 4.0
            + roll_torque_cmd
            - pitch_torque_cmd
            - yaw_torque_cmd
        )
        m3_thrust = (
            thrust_along_body_z / 4.0
            + roll_torque_cmd
            + pitch_torque_cmd
            + yaw_torque_cmd
        )
        m4_thrust = (
            thrust_along_body_z / 4.0
            - roll_torque_cmd
            - pitch_torque_cmd
            - yaw_torque_cmd
        )

        # Apply control commands, clamping to actuator limits
        d.ctrl[0] = np.clip(m1_thrust, motor_ctrl_min, motor_ctrl_max)
        d.ctrl[1] = np.clip(m2_thrust, motor_ctrl_min, motor_ctrl_max)
        d.ctrl[2] = np.clip(m3_thrust, motor_ctrl_min, motor_ctrl_max)
        d.ctrl[3] = np.clip(m4_thrust, motor_ctrl_min, motor_ctrl_max)

        # Print some info (optional)
        if int(d.time * 100) % 20 == 0:  # Print every 0.2 seconds
            print(
                f"T:{d.time:0.2f} Alt:{current_z:0.2f}m DesAlt:{altitude_pd.setpoint_pos:0.2f}m ThrustZ:{thrust_cmd_z:0.2f}N TotalT:{thrust_along_body_z:0.2f}N"
            )
            print(
                f"  Roll:{np.rad2deg(current_roll):0.1f} Pitch:{np.rad2deg(current_pitch):0.1f} Yaw:{np.rad2deg(current_yaw):0.1f} deg"
            )
            print(
                f"  RollCmd:{roll_torque_cmd:0.2f} PitchCmd:{pitch_torque_cmd:0.2f} YawCmd:{yaw_torque_cmd:0.2f}"
            )
            print(
                f"  Motors: {d.ctrl[0]:0.2f} {d.ctrl[1]:0.2f} {d.ctrl[2]:0.2f} {d.ctrl[3]:0.2f}\n"
            )

        # Physics step
        mujoco.mj_step(m, d)

        # Visualization sync
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
        viewer.sync()

        # Timing control to match simulation timestep
        time_spent_in_step = time.time() - step_start_time
        time_to_wait = m.opt.timestep - time_spent_in_step
        if time_to_wait > 0:
            time.sleep(time_to_wait)
