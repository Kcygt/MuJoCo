import time

import mujoco
import mujoco.viewer




class PDController:
    def __init__(self, kp, kd, t_final=30.0, setpoint=0.2):
        self.kp = kp          # Proportional gain
        self.kd = kd          # Derivative gain
        self.setpoint = setpoint  # Final target height (0.5m = 50cm)
        self.t_final = t_final    # Time to reach target height
        self.start_time = time.time()
        
    def trajectory(self, current_time):
        """Generates smooth trajectory from current height to target"""
        if current_time > self.t_final:
            return self.setpoint, 0.0  # Maintain position after reaching target
            
        # Minimum jerk trajectory calculations
        tau = current_time / self.t_final
        desired_pos = self.setpoint * (6*tau**5 - 15*tau**4 + 10*tau**3)
        desired_vel = self.setpoint * (30*tau**4 - 60*tau**3 + 30*tau**2) / self.t_final
        return desired_pos, desired_vel


m = mujoco.MjModel.from_xml_path('mujoco_menagerie-main/skydio_x2/scene.xml')
d = mujoco.MjData(m)

# Modified control section
kp = 15    # Increased proportional gain for height control
kd = 5    # Increased derivative gain for damping
pd_controller = PDController(kp, kd, t_final=4, setpoint=.5)

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()
        current_time = time.time() - start
        
        # Get trajectory targets
        desired_pos, desired_vel = pd_controller.trajectory(current_time)
        
        # Get current state
        measured_pos = d.qpos[2]
        measured_vel = d.qvel[2]
        
        # PD control calculation
        pos_error = desired_pos - measured_pos
        vel_error = desired_vel - measured_vel
        control_signal = pd_controller.kp * pos_error + pd_controller.kd * vel_error
        
        # Apply control with base thrust (adjust 4.2 if needed)
        d.ctrl = 4.2 + control_signal  # Base thrust + PD adjustment
        print("Altitude : ",measured_pos)
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


