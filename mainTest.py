import time
import numpy as np
import json

import mujoco
import mujoco.viewer
import pickle

from simple_pid import PID


class dummyPlanner:
    """Generate Path from 1 point directly to another"""

    def __init__(self, target, ) -> None:
        self.target = target
        # setpoint target location, controller output: desired velocity.
        self.pid_x = PID(2,0.15,1.5,setpoint=self.target[0],output_limits=(-2, 2),)
        self.pid_y = PID(2, 0.15, 1.5, setpoint=self.target[1], output_limits=(-2, 2))

    def __call__(self, loc: np.array):
        """Calls planner at timestep to update cmd_vel"""
        velocites = np.array([0, 0, 0])
        velocites[0] = self.pid_x(loc[0])
        velocites[1] = self.pid_y(loc[1])
        return velocites

    def get_velocities(
        self,
        loc: np.array,
        target: np.array,
        time_to_target: float = None,
        flight_speed: float = 0.5,
    ) -> np.array:
        """Compute

        Args:
            loc (np.array): Current location in world coordinates.
            target (np.array): Desired location in world coordinates
            time_to_target (float): If set, adpats length of velocity vector.

        Returns:
            np.array: returns velocity vector in world coordinates.
        """

        direction = target - loc
        distance = np.linalg.norm(direction)
        # maps drone velocities to one.
        if distance > 0.5:
            velocities = flight_speed * direction / distance

        else:
            velocities = direction * distance

        return velocities

    def get_alt_setpoint(self, loc: np.array) -> float:

        target = self.target
        distance = target[2] - loc[2]

        # maps drone velocities to one.
        if distance > 0.5:
            time_sample = 1 / 4
            time_to_target = distance / self.vel_limit
            number_steps = int(time_to_target / time_sample)
            # compute distance for next update
            delta_alt = distance / number_steps

            # 2 times for smoothing
            alt_set = loc[2] + 2 * delta_alt

        else:
            alt_set = target[2]

        return alt_set

    def update_target(self, target):
        """Update targets"""
        self.target = target
        # setpoint target location, controller output: desired velocity.
        self.pid_x.setpoint = self.target[0]
        self.pid_y.setpoint = self.target[1]


class dummySensor:
    """Dummy sensor data. So the control code remains intact."""

    def __init__(self, d):
        self.position = d.qpos
        self.velocity = d.qvel
        self.acceleration = d.qacc

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_acceleration(self):
        return self.acceleration


class drone:
    """Simple drone classe."""

    def __init__(self, target=np.array((0, 0, 0))):
        self.m = mujoco.MjModel.from_xml_path(
            "mujoco_menagerie-main/skydio_x2/scene.xml"
        )
        self.d = mujoco.MjData(self.m)

        self.planner = dummyPlanner(target=target)
        self.sensor = dummySensor(self.d)

        # instantiate controllers

        # inner control to stabalize inflight dynamics
        self.pid_alt = PID(10.50844,1.57871,1.2,setpoint=0,)  
        self.pid_roll = PID(2.6785, 0.56871, 1.2508, setpoint=0, output_limits=(-1, 1)) 
        self.pid_pitch = PID(2.6785, 0.56871, 1.2508, setpoint=0, output_limits=(-1, 1))
        self.pid_yaw = PID(0.54, 0, 5.358333, setpoint=1, output_limits=(-3, 3)) 

        # outer control loops
        self.pid_v_x = PID(0.1, 0.003, 0.02, setpoint=0, output_limits=(-0.1, 0.1))
        self.pid_v_y = PID(0.1, 0.003, 0.02, setpoint=0, output_limits=(-0.1, 0.1))

    def update_outer_conrol(self):
        """Updates outer control loop for trajectory planning"""
        v = self.sensor.get_velocity()
        location = self.sensor.get_position()[:3]

        # Compute velocites to target
        velocites = self.planner(loc=location)

        # In this example the altitude is directly controlled by a PID
        self.pid_alt.setpoint = self.planner.get_alt_setpoint(location)
        self.pid_v_x.setpoint = velocites[0]
        self.pid_v_y.setpoint = velocites[1]

        # Compute angles and set inner controllers accordingly
        angle_pitch = self.pid_v_x(v[0])
        angle_roll = -self.pid_v_y(v[1])

        self.pid_pitch.setpoint = angle_pitch
        self.pid_roll.setpoint = angle_roll

    def update_inner_control(self):
        """Upates inner control loop and sets actuators to stabilize flight
        dynamics"""
        alt = self.sensor.get_position()[2]
        angles = self.sensor.get_position()[3:]  # roll, yaw, pitch

        # apply PID
        cmd_thrust = self.pid_alt(alt) + 3.2495
        cmd_roll = -self.pid_roll(angles[1])
        cmd_pitch = self.pid_pitch(angles[2])
        cmd_yaw = -self.pid_yaw(angles[0])

        # transfer to motor control
        out = self.compute_motor_control(cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw)
        self.d.ctrl[:4] = out

    #  as the drone is underactuated we set
    def compute_motor_control(self, thrust, roll, pitch, yaw):
        motor_control = [
            thrust + roll + pitch - yaw,
            thrust - roll + pitch + yaw,
            thrust - roll - pitch - yaw,
            thrust + roll - pitch + yaw,
        ]
        return motor_control


# -------------------------- Initialization ----------------------------------
my_drone = drone(target=np.array((0, 0, 1)))

with mujoco.viewer.launch_passive(my_drone.m, my_drone.d) as viewer:
    time.sleep(1)
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    step = 1

    while viewer.is_running() and time.time() - start < 5:
        step_start = time.time()

        my_drone.planner.update_target(np.array((0.6, 0.6, 0.5)))
            
        # outer control loop
        my_drone.update_outer_conrol()
        # Inner control loop
        my_drone.update_inner_control()
        print(my_drone.d.qpos)
        mujoco.mj_step(my_drone.m, my_drone.d)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                my_drone.d.time % 2
            )

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Increment to time slower outer control loop
        step += 1

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = my_drone.m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
