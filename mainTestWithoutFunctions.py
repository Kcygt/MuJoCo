# import time
# import numpy as np
# import mujoco
# import mujoco.viewer
# from simple_pid import PID

# # ----------------------- Initialization -----------------------

# target = np.array([0, 0, 1])
# model = mujoco.MjModel.from_xml_path("mujoco_menagerie-main/skydio_x2/scene.xml")
# data = mujoco.MjData(model)

# # Sensor lambdas
# get_position = lambda: data.qpos
# get_velocity = lambda: data.qvel
# get_acceleration = lambda: data.qacc

# # Planner PIDs
# pid_x = PID(2, 0.15, 1.5, setpoint=target[0], output_limits=(-2, 2))
# pid_y = PID(2, 0.15, 1.5, setpoint=target[1], output_limits=(-2, 2))

# # Inner control PIDs
# pid_alt = PID(10.50844, 1.57871, 1.2, setpoint=0)
# pid_roll = PID(2.6785, 0.56871, 1.2508, setpoint=0, output_limits=(-1, 1))
# pid_pitch = PID(2.6785, 0.56871, 1.2508, setpoint=0, output_limits=(-1, 1))
# pid_yaw = PID(0.54, 0, 5.358333, setpoint=1, output_limits=(-3, 3))

# # Outer control PIDs
# pid_v_x = PID(0.1, 0.003, 0.02, setpoint=0, output_limits=(-0.1, 0.1))
# pid_v_y = PID(0.1, 0.003, 0.02, setpoint=0, output_limits=(-0.1, 0.1))

# # ----------------------- Viewer and Loop -----------------------

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     time.sleep(1)
#     start = time.time()
#     step = 1

#     while viewer.is_running() and time.time() - start < 5:
#         step_start = time.time()

#         # Update target
#         target = np.array([0.6, 0.6, 0.5])
#         pid_x.setpoint = target[0]
#         pid_y.setpoint = target[1]

#         # --- Outer control ---
#         v = get_velocity()
#         loc = get_position()[:3]

#         vx = pid_x(loc[0])
#         vy = pid_y(loc[1])
#         velocities = np.array([vx, vy, 0])

#         # Altitude control
#         distance = target[2] - loc[2]
#         vel_limit = 0.5

#         if abs(distance) > 0.5:
#             time_sample = 1 / 4
#             time_to_target = abs(distance) / vel_limit
#             number_steps = int(time_to_target / time_sample)
#             delta_alt = distance / number_steps
#             alt_setpoint = loc[2] + 2 * delta_alt
#         else:
#             alt_setpoint = target[2]

#         pid_alt.setpoint = alt_setpoint
#         pid_v_x.setpoint = velocities[0]
#         pid_v_y.setpoint = velocities[1]

#         angle_pitch = pid_v_x(v[0])
#         angle_roll = -pid_v_y(v[1])

#         pid_pitch.setpoint = angle_pitch
#         pid_roll.setpoint = angle_roll

#         # --- Inner control ---
#         pos = get_position()
#         angles = pos[3:]
#         alt = pos[2]

#         cmd_thrust = pid_alt(alt) + 3.2495
#         cmd_roll = -pid_roll(angles[1])
#         cmd_pitch = pid_pitch(angles[2])
#         cmd_yaw = -pid_yaw(angles[0])

#         motor_output = [
#             cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw,
#             cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw,
#             cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw,
#             cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw,
#         ]
#         data.ctrl[:4] = motor_output

#         print(data.qpos)
#         mujoco.mj_step(model, data)

#         with viewer.lock():
#             viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

#         viewer.sync()

#         time_until_next_step = model.opt.timestep - (time.time() - step_start)
#         if time_until_next_step > 0:
#             time.sleep(time_until_next_step)

#         step += 1
import time
import numpy as np
import mujoco
import mujoco.viewer

# -------------------- Load Mujoco model --------------------
model = mujoco.MjModel.from_xml_path("mujoco_menagerie-main/skydio_x2/scene.xml")
data = mujoco.MjData(model)

# -------------------- Target and Init --------------------
target = np.array([0.6, 0.6, 0.5])

# PID state: [previous_error, integral]
pid_state = {}

def init_pid(name, kp, ki, kd, setpoint=0.0, out_min=None, out_max=None):
    pid_state[name] = {
        "kp": kp,
        "ki": ki,
        "kd": kd,
        "setpoint": setpoint,
        "prev_error": 0.0,
        "integral": 0.0,
        "out_min": out_min,
        "out_max": out_max,
    }

def compute_pid(name, measurement, dt=0.01):
    p = pid_state[name]
    error = p["setpoint"] - measurement
    p["integral"] += error * dt
    derivative = (error - p["prev_error"]) / dt
    p["prev_error"] = error

    output = p["kp"] * error + p["ki"] * p["integral"] + p["kd"] * derivative

    if p["out_min"] is not None:
        output = max(p["out_min"], output)
    if p["out_max"] is not None:
        output = min(p["out_max"], output)

    return output

# -------------------- PID Controllers Init --------------------
# Planner PIDs
init_pid("x", 20, 0.15, 1.5, setpoint=target[0], out_min=-2, out_max=2)
init_pid("y", 20, 0.15, 1.5, setpoint=target[1], out_min=-2, out_max=2)

# Outer control PIDs
init_pid("vx", 0.1, 0.003, 0.02, setpoint=0, out_min=-0.1, out_max=0.1)
init_pid("vy", 0.1, 0.003, 0.02, setpoint=0, out_min=-0.1, out_max=0.1)

# Inner control PIDs
init_pid("alt", 10.50844, 1.57871, 1.2, setpoint=0)
init_pid("roll", 2.6785, 0.56871, 1.2508, setpoint=0, out_min=-1, out_max=1)
init_pid("pitch", 2.6785, 0.56871, 1.2508, setpoint=0, out_min=-1, out_max=1)
init_pid("yaw", 0.54, 0.0, 5.358333, setpoint=1, out_min=-3, out_max=3)

# -------------------- Simulation Loop --------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    time.sleep(1)
    start = time.time()
    dt = model.opt.timestep
    vel_limit = 0.5

    while viewer.is_running() and time.time() - start < 5:
        step_start = time.time()

        # Sensor values
        pos = data.qpos
        vel = data.qvel
        angles = pos[3:]  # roll, yaw, pitch
        alt = pos[2]

        # --------- Planner Output ---------
        vx_target = compute_pid("x", pos[0], dt)
        vy_target = compute_pid("y", pos[1], dt)

        # Altitude setpoint
        distance = target[2] - pos[2]
        if abs(distance) > 0.5:
            time_sample = 1 / 4
            time_to_target = abs(distance) / vel_limit
            number_steps = int(time_to_target / time_sample)
            delta_alt = distance / number_steps
            alt_setpoint = pos[2] + 2 * delta_alt
        else:
            alt_setpoint = target[2]
        pid_state["alt"]["setpoint"] = alt_setpoint

        # Outer loop setpoints
        pid_state["vx"]["setpoint"] = vx_target
        pid_state["vy"]["setpoint"] = vy_target

        angle_pitch = compute_pid("vx", vel[0], dt)
        angle_roll = -compute_pid("vy", vel[1], dt)

        pid_state["pitch"]["setpoint"] = angle_pitch
        pid_state["roll"]["setpoint"] = angle_roll

        # --------- Inner Control Loop ---------
        thrust = compute_pid("alt", alt, dt) + 3.2495
        roll_cmd = -compute_pid("roll", angles[1], dt)
        pitch_cmd = compute_pid("pitch", angles[2], dt)
        yaw_cmd = -compute_pid("yaw", angles[0], dt)

        # --------- Mixer to Motor Outputs ---------
        motor_out = [
            thrust + roll_cmd + pitch_cmd - yaw_cmd,
            thrust - roll_cmd + pitch_cmd + yaw_cmd,
            thrust - roll_cmd - pitch_cmd - yaw_cmd,
            thrust + roll_cmd - pitch_cmd + yaw_cmd,
        ]
        data.ctrl[:4] = motor_out

        # --------- Step Simulation ---------
        mujoco.mj_step(model, data)
        print(data.qpos)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
        viewer.sync()

        # Wait for next timestep
        elapsed = time.time() - step_start
        if model.opt.timestep - elapsed > 0:
            time.sleep(model.opt.timestep - elapsed)
