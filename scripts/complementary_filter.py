import numpy as np


def acc_to_euler(acc):
    ax, ay, az = acc
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    return roll, pitch


def integrate_gyro(gyro, prev_angles, dt):
    gx, gy, gz = gyro  # Assuming body frame
    prev_roll, prev_pitch = prev_angles
    new_roll = prev_roll + gx * dt
    new_pitch = prev_pitch + gy * dt
    return new_roll, new_pitch


def complementary_filter(gyro, accel, prev_angles, alpha, dt):
    # Step 1: estimate orientation from gyro
    gyro_angles = integrate_gyro(gyro, prev_angles, dt)

    # Step 2: estimate orientation from accel
    acc_angles = acc_to_euler(accel)

    # Step 3: blend the two
    roll = alpha * gyro_angles[0] + (1 - alpha) * acc_angles[0]
    pitch = alpha * gyro_angles[1] + (1 - alpha) * acc_angles[1]

    return roll, pitch
