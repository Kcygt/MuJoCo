import time

import mujoco
import mujoco.viewer





class PDController:
    def __init__(self, kp, kd, setpoint):
        self.kp = kp
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0

    def compute(self, measured_value, measured_velocity, desired_value, desired_velocity):
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

    def quintic_trajectory(self,time, t_final, C_final):
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


m = mujoco.MjModel.from_xml_path('mujoco_menagerie-main/skydio_x2/scene.xml')
d = mujoco.MjData(m)
d.ctrl[0:4] = 4.0  # Set the thruster values to 0.5

# Parameters
kp = 80
kd = 5
t_final = 4.0    # Duration to reach 50 cm
C_final = 0.5    # Target height (50 cm)
pd_controller = PDController(kp, kd, setpoint=0.5)

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()
        current_time = time.time() - start

        # Get desired position and velocity from your cubic trajectory
        desired_pos, desired_vel, _ = pd_controller.quintic_trajectory(
            min(current_time, t_final), t_final, C_final
        )

        # Get current position and velocity
        measured_pos = d.qpos[2]
        measured_vel = d.qvel[2]

        # PD control using desired and measured values
        control_signal = pd_controller.compute(
            measured_pos, measured_vel, desired_pos, desired_vel
        )

        # Apply control (adjust base thrust as needed for your drone)
        d.ctrl[0:4] = 5. + control_signal
        print("Altitude : ", measured_pos)
        mujoco.mj_step(m, d)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
