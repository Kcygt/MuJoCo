import mujoco

m = mujoco.MjModel.from_xml_path("mujoco_menagerie-main/skydio_x2/scene.xml")
d = mujoco.MjData(m)


def get_sensor_data(model, data, sensor_name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    start = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return data.sensordata[start : start + dim]
