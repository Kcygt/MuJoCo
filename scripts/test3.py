import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R


class PDController:
    def __init__(self, kp, kd, t_final=30.0, setpoint=0.5):
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint  # Final target height (0.5m = 50cm)
        self.t_final = t_final  # Time to reach target height
        self.start_time = time.time()

    def trajectory(self, current_time):
        """Generates smooth trajectory from current height to target"""
        if current_time > self.t_final:
            return self.setpoint, 0.0  # Maintain position after reaching target

        # Minimum jerk trajectory calculations
        tau = current_time / self.t_final
        desired_pos = self.setpoint * (6 * tau**5 - 15 * tau**4 + 10 * tau**3)
        desired_vel = (
            self.setpoint * (30 * tau**4 - 60 * tau**3 + 30 * tau**2) / self.t_final
        )
        return desired_pos, desired_vel


def get_roll_pitch_from_accel(accel):
    ax, ay, az = accel
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    return roll, pitch


def integrate_gyro_yaw(yaw_prev, gyro_z, dt):
    return yaw_prev + gyro_z * dt


def estimate_rpy(accel, gyro, yaw_prev, dt):
    roll, pitch = get_roll_pitch_from_accel(accel)
    yaw = integrate_gyro_yaw(yaw_prev, gyro[2], dt)
    return roll, pitch, yaw


m = mujoco.MjModel.from_xml_path("mujoco_menagerie-main/skydio_x2/scene.xml")
d = mujoco.MjData(m)

# Modified control section
kp = 600  # Increased proportional gain for height control
kd = 300  # Increased derivative gain for damping
pd_controller = PDController(kp, kd, t_final=4, setpoint=0.5)
yaw_est = 0.0
dt = d.model.opt.timestep
kp_roll = 10.0
kd_roll = 0.5
kp_pitch = 10.0
kd_pitch = 0.5
kp_yaw = 10.0
kd_yaw = 0.5


with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 100:
        # Time step
        step_start = time.time()
        current_time = time.time() - start
        # sensor data
        gyro = d.sensordata[0:3]
        accel = d.sensordata[3:6]
        quat = d.sensordata[6:10]
        # Orientation from quaternion
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        roll, pitch, yaw = r.as_euler("xyz")

        yaw_target = 3 * np.sin(current_time)  # Example: oscillating yaw
        # Estimate roll, pitch, yaw from accelerometer and gyroscope
        roll_cmd = kp_roll * (0 - roll) - kd_roll * gyro[0]
        pitch_cmd = kp_pitch * (0 - pitch) - kd_pitch * gyro[1]
        yaw_cmd = kp_yaw * (yaw_target - yaw) - kd_yaw * gyro[2]
        # Get trajectory targets
        desired_pos, desired_vel = pd_controller.trajectory(current_time)

        # Get current state
        measured_pos = d.qpos[2]
        measured_vel = d.qvel[2]

        # PD control calculation
        pos_error = desired_pos - measured_pos
        vel_error = desired_vel - measured_vel
        thrust = pd_controller.kp * pos_error + pd_controller.kd * vel_error
        print(f"Thrust: {measured_pos}")
        motor_out = [
            thrust + roll_cmd + pitch_cmd - yaw_cmd,
            thrust - roll_cmd + pitch_cmd + yaw_cmd,
            thrust - roll_cmd - pitch_cmd - yaw_cmd,
            thrust + roll_cmd - pitch_cmd + yaw_cmd,
        ]
        d.ctrl[:4] = thrust  # Adjust base thrust as needed

        thrust = 3.24955
        # Apply control with base thrust (adjust 4.2 if needed)

        # Sensor values (assuming the order is as declared in XML)
        # 3 values for gyro, 3 for accelerometer, 4 for quaternion

        gyro_data = d.sensordata[0:3]
        accel_data = d.sensordata[3:6]
        quat_data = d.sensordata[6:10]

        # print(f"Gyro       : {gyro_data}")
        # print(f"Accelerometer: {accel_data}")
        # print(f"Quaternion : {quat_data}")
        # print("Altitude : ", measured_pos)
        # Physics step
        mujoco.mj_step(m, d)

        # Visualization sync
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
        viewer.sync()

        # Timing control
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
