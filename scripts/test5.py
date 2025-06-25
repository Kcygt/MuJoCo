import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from getSensordata import get_sensor_data
from complementary_filter import complementary_filter


class PDController:
    def __init__(self, kp, kd, t_final=30.0, setpoint=0.5):
        self.kp = kp
        self.kd = kd
        self.setpoint = setpoint
        self.t_final = t_final
        self.start_time = time.time()

    def trajectory(self, current_time):
        if current_time > self.t_final:
            return self.setpoint, 0.0
        tau = current_time / self.t_final
        desired_pos = self.setpoint * (6 * tau**5 - 15 * tau**4 + 10 * tau**3)
        desired_vel = (
            self.setpoint * (30 * tau**4 - 60 * tau**3 + 30 * tau**2) / self.t_final
        )
        return desired_pos, desired_vel


def quat_to_euler(quat):
    # Convert quaternion to Euler angles (roll, pitch, yaw)
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # xyzw -> wxyz
    return r.as_euler("xyz", degrees=False)  # returns roll, pitch, yaw


m = mujoco.MjModel.from_xml_path("mujoco_menagerie-main/skydio_x2/scene.xml")
d = mujoco.MjData(m)

# Gains
alt_kp, alt_kd = 600, 200
att_kp, att_kd = 100, 10  # Orientation control gains

pd_controller = PDController(alt_kp, alt_kd, t_final=4, setpoint=0.5)

roll, pitch = 0.0, 0.0

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 100:
        step_start = time.time()
        current_time = time.time() - start

        desired_pos, desired_vel = pd_controller.trajectory(current_time)

        # Sensor data
        quat_data = d.sensordata[6:10]
        gyro_data = d.sensordata[0:3]  # Angular velocity: [wx, wy, wz]

        gyro = get_sensor_data(m, d, "body_gyro")
        accel = get_sensor_data(m, d, "body_linacc")
        quat = get_sensor_data(m, d, "body_quat")

        roll, pitch = complementary_filter(gyro, accel, (roll, pitch), 0.98, 0.001)

        print("Filtered Roll:", np.degrees(roll), "Pitch:", np.degrees(pitch))
        # Get altitude
        measured_pos = d.qpos[2]
        measured_vel = d.qvel[2]

        # Altitude PD control
        pos_error = desired_pos - measured_pos
        vel_error = desired_vel - measured_vel
        thrust = pd_controller.kp * pos_error + pd_controller.kd * vel_error

        # Orientation control
        roll, pitch, yaw = quat_to_euler(quat_data)
        roll_rate, pitch_rate, yaw_rate = gyro_data

        roll_torque = -att_kp * roll - att_kd * roll_rate
        pitch_torque = -att_kp * pitch - att_kd * pitch_rate
        yaw_torque = (
            -att_kp * yaw - att_kd * yaw_rate
        )  # Optional: yaw can be loosely controlled

        # Mixer (simplified): Distribute thrust and torques to 4 motors
        base_thrust = thrust + 3.2496  # Hover bias
        d.ctrl[0] = base_thrust + roll_torque + pitch_torque - yaw_torque
        d.ctrl[1] = base_thrust - roll_torque + pitch_torque + yaw_torque
        d.ctrl[2] = base_thrust - roll_torque - pitch_torque - yaw_torque
        d.ctrl[3] = base_thrust + roll_torque - pitch_torque + yaw_torque

        mujoco.mj_step(m, d)

        # Visualization
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
