import time
import numpy as np
import mujoco
import mujoco.viewer

# -------------------- Load Mujoco model --------------------
model = mujoco.MjModel.from_xml_path("mujoco_menagerie-main/skydio_x2/scene.xml")
data = mujoco.MjData(model)


# -------------------- Trajectory Function --------------------
def trajectory(time, t_final, C_final, home_pos):
    """Generates trajectory position, velocity, and acceleration."""
    a2 = 3 / t_final**2
    a3 = -2 / t_final**3
    Cposition = (a2 * time**2 + a3 * time**3) * C_final
    Cvelocity = (2 * a2 * time + 3 * time**2 * a3) * C_final
    Cacceleration = (2 * a2 + 6 * time * a3) * C_final
    return Cposition, Cvelocity, Cacceleration


# -------------------- PID Controller Setup --------------------
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


# -------------------- PID Initialization --------------------
# These will be updated dynamically based on trajectory
init_pid("x", 20, 0.15, 1.5, out_min=-2, out_max=2)
init_pid("y", 20, 0.15, 1.5, out_min=-2, out_max=2)

init_pid("vx", 1, 0.003, 0.02, setpoint=0, out_min=-0.1, out_max=0.1)
init_pid("vy", 1, 0.003, 0.02, setpoint=0, out_min=-0.1, out_max=0.1)

init_pid("alt", 20.50844, 1.57871, 1.2, setpoint=0)
init_pid("roll", 20.6785, 0.56871, 1.2508, setpoint=0, out_min=-1, out_max=1)
init_pid("pitch", 20.6785, 0.56871, 1.2508, setpoint=0, out_min=-1, out_max=1)
init_pid("yaw", 10.54, 0.0, 5.358333, setpoint=1, out_min=-3, out_max=3)

# -------------------- Target Setup --------------------
target = np.array([0.0, 0.0, 1])
t_final = 5.0  # seconds
vel_limit = 0.5

# -------------------- Simulation Loop --------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    time.sleep(1)
    dt = model.opt.timestep
    start = time.time()

    # Save initial position to compute relative trajectory
    initial_pos = np.array(data.qpos[:3])  # x, y, z
    C_final = target - initial_pos

    while viewer.is_running() and data.time < t_final + 5:
        step_start = time.time()

        # Sensor values
        pos = data.qpos
        vel = data.qvel
        angles = pos[3:]  # roll, yaw, pitch
        alt = pos[2]

        # --------- Trajectory Following ---------
        traj_time = data.time
        traj_pos, traj_vel, traj_acc = trajectory(
            traj_time, t_final, C_final, initial_pos
        )
        target_pos = initial_pos + traj_pos
        target_vel = traj_vel

        # Set planner PID setpoints
        pid_state["x"]["setpoint"] = target_pos[0]
        pid_state["y"]["setpoint"] = target_pos[1]
        pid_state["alt"]["setpoint"] = target_pos[2]

        # Outer velocity setpoints
        vx_target = compute_pid("x", pos[0], dt)
        vy_target = compute_pid("y", pos[1], dt)
        pid_state["vx"]["setpoint"] = vx_target
        pid_state["vy"]["setpoint"] = vy_target

        # Compute pitch/roll setpoints
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
        print(f"Time: {data.time:.2f}, Pos: {pos[:3]}")

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
        viewer.sync()

        # Wait for next timestep
        elapsed = time.time() - step_start
        if model.opt.timestep - elapsed > 0:
            time.sleep(model.opt.timestep - elapsed)
