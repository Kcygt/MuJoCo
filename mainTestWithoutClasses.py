import time
import numpy as np
import mujoco
import mujoco.viewer
from simple_pid import PID

# ------------------------ State Initialization ------------------------

def init_drone(target):
    model = mujoco.MjModel.from_xml_path("mujoco_menagerie-main/skydio_x2/scene.xml")
    data = mujoco.MjData(model)

    drone_state = {
        "model": model,
        "data": data,
        "target": target,

        # Sensor proxies
        "get_position": lambda: data.qpos,
        "get_velocity": lambda: data.qvel,
        "get_acceleration": lambda: data.qacc,

        # Planner PIDs
        "pid_x": PID(2, 0.15, 1.5, setpoint=target[0], output_limits=(-2, 2)),
        "pid_y": PID(2, 0.15, 1.5, setpoint=target[1], output_limits=(-2, 2)),

        # Inner control
        "pid_alt": PID(10.50844, 1.57871, 1.2, setpoint=0),
        "pid_roll": PID(2.6785, 0.56871, 1.2508, setpoint=0, output_limits=(-1, 1)),
        "pid_pitch": PID(2.6785, 0.56871, 1.2508, setpoint=0, output_limits=(-1, 1)),
        "pid_yaw": PID(0.54, 0, 5.358333, setpoint=1, output_limits=(-3, 3)),

        # Outer control
        "pid_v_x": PID(0.1, 0.003, 0.02, setpoint=0, output_limits=(-0.1, 0.1)),
        "pid_v_y": PID(0.1, 0.003, 0.02, setpoint=0, output_limits=(-0.1, 0.1)),
    }

    return drone_state


# ---------------------------- Control Logic ----------------------------

def compute_velocities(drone_state, loc):
    pid_x = drone_state["pid_x"]
    pid_y = drone_state["pid_y"]
    vx = pid_x(loc[0])
    vy = pid_y(loc[1])
    return np.array([vx, vy, 0])


def get_alt_setpoint(drone_state, loc):
    target = drone_state["target"]
    distance = target[2] - loc[2]
    vel_limit = 0.5  # You can expose this as a configurable param

    if abs(distance) > 0.5:
        time_sample = 1 / 4
        time_to_target = abs(distance) / vel_limit
        number_steps = int(time_to_target / time_sample)
        delta_alt = distance / number_steps
        return loc[2] + 2 * delta_alt
    else:
        return target[2]


def update_outer_control(drone_state):
    v = drone_state["get_velocity"]()
    loc = drone_state["get_position"]()[:3]

    velocities = compute_velocities(drone_state, loc)

    drone_state["pid_alt"].setpoint = get_alt_setpoint(drone_state, loc)
    drone_state["pid_v_x"].setpoint = velocities[0]
    drone_state["pid_v_y"].setpoint = velocities[1]

    angle_pitch = drone_state["pid_v_x"](v[0])
    angle_roll = -drone_state["pid_v_y"](v[1])

    drone_state["pid_pitch"].setpoint = angle_pitch
    drone_state["pid_roll"].setpoint = angle_roll


def update_inner_control(drone_state):
    pos = drone_state["get_position"]()
    angles = pos[3:]  # roll, yaw, pitch
    alt = pos[2]

    cmd_thrust = drone_state["pid_alt"](alt) + 3.2495
    cmd_roll = -drone_state["pid_roll"](angles[1])
    cmd_pitch = drone_state["pid_pitch"](angles[2])
    cmd_yaw = -drone_state["pid_yaw"](angles[0])

    motor_out = compute_motor_control(cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw)
    drone_state["data"].ctrl[:4] = motor_out


def compute_motor_control(thrust, roll, pitch, yaw):
    return [
        thrust + roll + pitch - yaw,
        thrust - roll + pitch + yaw,
        thrust - roll - pitch - yaw,
        thrust + roll - pitch + yaw,
    ]


def update_target(drone_state, new_target):
    drone_state["target"] = new_target
    drone_state["pid_x"].setpoint = new_target[0]
    drone_state["pid_y"].setpoint = new_target[1]


# --------------------------- Main Loop -------------------------------

if __name__ == "__main__":
    drone_state = init_drone(np.array([0, 0, 1]))

    with mujoco.viewer.launch_passive(drone_state["model"], drone_state["data"]) as viewer:
        time.sleep(1)
        start = time.time()
        step = 1

        while viewer.is_running() and time.time() - start < 5:
            step_start = time.time()

            update_target(drone_state, np.array([0.6, 0.6, 0.5]))
            update_outer_control(drone_state)
            update_inner_control(drone_state)

            print(drone_state["data"].qpos)
            mujoco.mj_step(drone_state["model"], drone_state["data"])

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                    drone_state["data"].time % 2
                )

            viewer.sync()

            time_until_next_step = (
                drone_state["model"].opt.timestep - (time.time() - step_start)
            )
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            step += 1
