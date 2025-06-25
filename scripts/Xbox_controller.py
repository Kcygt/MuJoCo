import time
import mujoco
import mujoco.viewer
import inputs  # For Xbox controller

# --- Xbox Controller Setup ---
MAX_ABS_VAL = 32768.0  # Max value for Xbox controller analog sticks
controller_present = False
gamepad = None
try:
    gamepads = inputs.devices.gamepads
    if gamepads:
        gamepad = gamepads[0]
        controller_present = True
        print("Xbox controller found.")
    else:
        print("No Xbox controller found. Running without controller input.")
except Exception as e:
    print(f"Error initializing gamepad: {e}. Running without controller input.")

# Store the state of relevant controller axes
controller_state = {
    "ABS_X": 0,  # Left stick X (Yaw)
    "ABS_Y": 0,  # Left stick Y (Thrust)
    "ABS_RX": 0,  # Right stick X (Roll)
    "ABS_RY": 0,  # Right stick Y (Pitch)
    # Add more if needed, e.g., 'ABS_Z', 'ABS_RZ' for triggers
}


def apply_deadzone(value, deadzone_threshold=0.08):
    """Applies a deadzone to an analog stick value."""
    if abs(value) < deadzone_threshold:
        return 0.0
    # Scale to fill the range after deadzone
    return (value - deadzone_threshold * (1 if value > 0 else -1)) / (
        1.0 - deadzone_threshold
    )


# --- PD Controller Class (can be kept for future use, but not actively controlling motors here) ---
class PDController:
    def __init__(self, kp, kd, t_final=30.0, setpoint=0.5):
        self.kp = kp
        self.kd = kd
        self.setpoint = setpoint
        self.t_final = t_final
        self.start_time = time.time()  # This should be simulation start time

    def update_start_time(self, sim_start_time):
        self.start_time = sim_start_time

    def trajectory(self, current_sim_time):
        current_traj_time = current_sim_time - self.start_time
        if (
            current_traj_time > self.t_final or current_traj_time < 0
        ):  # Check for < 0 if start_time updated late
            return self.setpoint, 0.0

        tau = current_traj_time / self.t_final
        desired_pos = self.setpoint * (
            10 * tau**3 - 15 * tau**4 + 6 * tau**5
        )  # Corrected min jerk
        desired_vel = (
            self.setpoint * (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.t_final
        )
        return desired_pos, desired_vel


# --- MuJoCo Setup ---
try:
    m = mujoco.MjModel.from_xml_path("mujoco_menagerie-main/skydio_x2/scene.xml")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure 'mujoco_menagerie-main/skydio_x2/scene.xml' path is correct.")
    exit()

d = mujoco.MjData(m)

# --- Control Parameters (Adjust these for desired responsiveness) ---
# This is an approximate thrust needed per motor to hover.
# You might need to find this experimentally or calculate based on drone mass and gravity.
# The original code used PD controller output + 4.2. Let's use 4.2 as a neutral point.
HOVER_THRUST_PER_MOTOR = 4.2  # from original PD code's base thrust
THRUST_CONTROL_SCALE = 3.0  # How much left stick Y affects thrust variation
ROLL_PITCH_SCALE = 1.8  # How much right stick X/Y affects roll/pitch
YAW_SCALE = 1.0  # How much left stick X affects yaw

# The PD controller for trajectory (not directly used for motor control here, but can provide targets)
# kp_pd = 600
# kd_pd = 200
# pd_controller = PDController(kp_pd, kd_pd, t_final=4, setpoint=0.5)

with mujoco.viewer.launch_passive(m, d) as viewer:
    sim_start_time = time.time()
    # pd_controller.update_start_time(sim_start_time) # If using PD trajectory

    while viewer.is_running() and time.time() - sim_start_time < 100:
        step_start_time = time.time()
        current_sim_time_elapsed = time.time() - sim_start_time

        # --- Read Controller Inputs ---
        if controller_present and gamepad:
            try:
                events = gamepad.read()  # Read all available events
                for event in events:
                    if event.ev_type == "Absolute" and event.code in controller_state:
                        controller_state[event.code] = event.state
            except (EOFError, inputs.UnpluggedError):
                print("Controller disconnected.")
                controller_present = False
                # Optionally, implement a failsafe (e.g., zero thrust)
                d.ctrl[:4] = 0.0  # Example: cut motors
            except BlockingIOError:
                # No new events, which is fine
                pass
            except Exception as e:
                # Catch other potential errors from inputs lib like 'struct.error' on some systems if no event
                # print(f"Gamepad read error: {e}") # Can be noisy
                pass

        # --- Get Desired Control from Xbox Controller ---
        # Normalize and apply deadzone
        # Left Stick Y (Thrust): Up is negative, so invert.
        raw_thrust = controller_state.get("ABS_Y", 0) / MAX_ABS_VAL
        thrust_input = apply_deadzone(-raw_thrust)  # Inverted

        # Left Stick X (Yaw):
        raw_yaw = controller_state.get("ABS_X", 0) / MAX_ABS_VAL
        yaw_input = apply_deadzone(raw_yaw)

        # Right Stick Y (Pitch): Up is negative, so invert.
        raw_pitch = controller_state.get("ABS_RY", 0) / MAX_ABS_VAL
        pitch_input = apply_deadzone(-raw_pitch)  # Inverted

        # Right Stick X (Roll):
        raw_roll = controller_state.get("ABS_RX", 0) / MAX_ABS_VAL
        roll_input = apply_deadzone(raw_roll)

        # --- Mixer: Convert controller inputs to motor commands ---
        # This is a common X-quadcopter mixer configuration.
        # Motor order in d.ctrl might need verification for skydio_x2.xml
        # Assuming:
        # d.ctrl[0]: Front-Left motor
        # d.ctrl[1]: Front-Right motor
        # d.ctrl[2]: Rear-Right motor
        # d.ctrl[3]: Rear-Left motor
        # (This is a common order, e.g. for Betaflight motor mapping if you remap)

        # Let's use a standard X-frame mixer.
        # Positive pitch_input = nose down.
        # Positive roll_input = roll right.
        # Positive yaw_input = yaw right (clockwise).
        # Positive thrust_input = more upward thrust.

        # Total base thrust commanded by the stick
        current_base_thrust = (
            HOVER_THRUST_PER_MOTOR + thrust_input * THRUST_CONTROL_SCALE
        )

        # Calculate motor commands
        # Signs depend on propeller rotation direction and motor placement.
        # This is a common "plus" or "X" configuration mixer:
        # motor_fr = base + Kp*pitch + Kr*roll - Ky*yaw
        # motor_fl = base + Kp*pitch - Kr*roll + Ky*yaw
        # motor_rr = base - Kp*pitch + Kr*roll + Ky*yaw
        # motor_rl = base - Kp*pitch - Kr*roll - Ky*yaw
        # NOTE: The signs for roll/pitch/yaw in the mixer must be correct for your drone's
        # specific motor numbering and prop directions. The one below is common for X-frames.

        # Assuming d.ctrl indexes: 0:FL, 1:FR, 2:RR, 3:RL
        # Motor commands (adjust signs experimentally if drone behaves unexpectedly)
        m_fl = (
            current_base_thrust
            - pitch_input * ROLL_PITCH_SCALE
            - roll_input * ROLL_PITCH_SCALE
            + yaw_input * YAW_SCALE
        )
        m_fr = (
            current_base_thrust
            - pitch_input * ROLL_PITCH_SCALE
            + roll_input * ROLL_PITCH_SCALE
            - yaw_input * YAW_SCALE
        )
        m_rr = (
            current_base_thrust
            + pitch_input * ROLL_PITCH_SCALE
            + roll_input * ROLL_PITCH_SCALE
            + yaw_input * YAW_SCALE
        )
        m_rl = (
            current_base_thrust
            + pitch_input * ROLL_PITCH_SCALE
            - roll_input * ROLL_PITCH_SCALE
            - yaw_input * YAW_SCALE
        )

        if not controller_present:  # If no controller, just set to hover (or zero)
            d.ctrl[:4] = (
                0.0  # Or HOVER_THRUST_PER_MOTOR if you want it to try and hover by default
            )
        else:
            d.ctrl[0] = max(0, m_fl)  # Ensure non-negative thrust
            d.ctrl[1] = max(0, m_fr)
            d.ctrl[2] = max(0, m_rr)
            d.ctrl[3] = max(0, m_rl)

        # --- Sensor Data (as before) ---
        # gyro_data = d.sensordata[0:3]
        # accel_data = d.sensordata[3:6]
        # quat_data = d.sensordata[6:10]
        # measured_pos_z = d.qpos[2] # Altitude

        # print(f"Thrust:{thrust_input:.2f} Roll:{roll_input:.2f} Pitch:{pitch_input:.2f} Yaw:{yaw_input:.2f}")
        # print(f"Motors: FL:{d.ctrl[0]:.2f} FR:{d.ctrl[1]:.2f} RR:{d.ctrl[2]:.2f} RL:{d.ctrl[3]:.2f}")
        # print(f"Altitude : {measured_pos_z:.2f}")

        # --- Physics step ---
        mujoco.mj_step(m, d)

        # --- Visualization sync ---
        if viewer.is_running():
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
            viewer.sync()

        # --- Timing control ---
        time_until_next_step = m.opt.timestep - (time.time() - step_start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
