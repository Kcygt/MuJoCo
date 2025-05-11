import time

import mujoco
import mujoco.viewer




class PDController:
    def __init__(self, kp, kd, setpoint):
        self.kp = kp
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0

    def compute(self, measured_value):
        error = self.setpoint - measured_value
        derivative = error - self.prev_error
        output = (self.kp * error) + (self.kd * derivative)
        self.prev_error = error
        return output

    def trajectory(tself,time, t_final, C_final):
        """Generates trajectory position, velocity, and acceleration."""
        a2 = 3 / t_final**2
        a3 = -2 / t_final**3
        Cposition = (a2 * time**2 + a3 * time**3) * C_final 
        Cvelocity = (2 * a2 * time + 3 * time**2 * a3) * C_final
        Cacceleration = (2 * a2 + 6 * time * a3) * C_final
        
        return Cposition, Cvelocity, Cacceleration


m = mujoco.MjModel.from_xml_path('mujoco_menagerie-main/skydio_x2/scene.xml')
d = mujoco.MjData(m)
d.ctrl = 4  # Set the thruster values to 0.5

kp = 10  # Proportional gain
kd = 1  #
setpoint = 0 # vertical speed should be 0

pd_controller = PDController(kp, kd, setpoint)

with mujoco.viewer.launch_passive(m, d) as viewer:
  
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()
    
    ### Control loop to stabelize drone height.
    measured_position = d.qpos[2]
    #print (measured_speed)
    control_signal = pd_controller.compute(measured_position)
    print(d.qpos[2])
    d.ctrl = d.ctrl + control_signal
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)