import numpy as np
import mujoco
import math
import yaml
from scipy.spatial.transform import Rotation as R
import struct 
import os

ROBOT_JOINTS = 7

def count_subdirectories(path):
    if not os.path.isdir(path):
        print(f"Error: '{path}' is not a valid directory.")
        return -1

    count = 0
    for entry_name in os.listdir(path):
        full_path = os.path.join(path, entry_name)
        if os.path.isdir(full_path):
            count += 1
    return count

def params_from_yaml(yaml_path : str):

    with open(yaml_path, 'r') as stream:
        params = yaml.safe_load(stream)

    return params

def fetch_body_id(mj_model, data, body_name, obj_type = "body"):

    body_id = None

    if obj_type == "geom":
        body_id = data.geom(body_name).id
    elif obj_type == "body":
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    return body_id

def fetch_body_pos_orient(data, body_id, point = None, frame = None, quat = False):

    pos = data.xpos[body_id]
    orient = data.xmat[body_id].reshape(3,3)

    if quat:
        orient = data.xquat[body_id]
        orient = R.from_quat(np.array([orient[1], orient[2], orient[3], orient[0]])).as_matrix()

    if point is None or frame is None:
        return pos, orient
    
    return frame.T @ (pos - point), frame.T @ orient

def extract_pos_orient_keypoints(key_points: np.ndarray):

    pos = np.mean(key_points, axis = 1)

    orient = np.zeros((3,3))

    orient[:,0] = key_points[:,0] - key_points[:,1]
    orient[:, 0] = orient[:, 0]/np.linalg.norm(orient[:,0])
    orient[:,1] = key_points[:,2] - key_points[:,1]
    orient[:,1] = orient[:,1]/np.linalg.norm(orient[:,1])
    orient[:, 2] = np.cross(orient[:,0], orient[:,1]) 

    return pos, orient

def generate_random_sphere_point():
    random_vector = np.random.randn(3)

    norm_val = np.linalg.norm(random_vector)

    if norm_val == 0:
        return np.array([0.0, 0.0, 0.0])

    unit_vector = random_vector / norm_val
    return unit_vector

def get_force_data(model, data, frame_orient = None, force_dim = 1):

    if force_dim == 0:
        return np.zeros(3), np.zeros(3)
    # Get sensor id
    force_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor")
    torque_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "torque_sensor")
    force_adr = model.sensor_adr[force_sensor_id]
    torque_adr = model.sensor_adr[torque_sensor_id]

    # Force in local site frame
    f_local = data.sensordata[force_adr:force_adr+3]
    t_local = data.sensordata[torque_adr:torque_adr+3]

    # Get the site the sensor is attached to
    site_id = model.sensor_objid[force_sensor_id]  
    site_xmat = data.site_xmat[site_id].reshape(3, 3)  # rotation matrix from site to world

    # Rotate force into world frame
    f_world = site_xmat @ f_local
    t_world = site_xmat @ t_local

    if frame_orient is not None:
        return frame_orient.T @ f_world, frame_orient.T @ t_world
    return f_world, t_world


def get_velocity_data(model, data, frame = None):

    tool_id = fetch_body_id(model, data, "tool")
    vel = data.cvel[tool_id, :3]

    if frame is None:
        return vel
    
    return frame.T @ vel

def realize_points_in_frame(points, pos, frame):
    return frame.T @(points - pos.reshape(3,1))

#assume that the key points come in the world frame
def switch_frame_key_points(key_points, pos, frame):
    key_points_transformed = frame.T @ (key_points - pos.reshape(3,1))

    return key_points_transformed

def compute_geodesic_distance(orient1, orient2):

    arg = (np.trace(orient1 @ orient2.T) -1)/2
    arg_clamped = np.clip(arg, -1.0, 1.0)
    angle_rad = np.arccos(arg_clamped) 

    return angle_rad

def continuous_reward_func(a: float, b: float, x: float) -> float:
    exp_ax = math.exp(a * x)
    exp_neg_ax = math.exp(-a * x)

    denominator = b + exp_neg_ax + exp_ax

    r = 2 * (b + 2) * (denominator**-1) - 1

    return r


class RewardsModuleV2():

    def __init__(self):

        pass

    def select_reward(self, reward_type, tool_key_points, task_key_points, metadata):
        if reward_type == "6d":
            return self.sixd_orient_error(tool_key_points, task_key_points, metadata)
        elif reward_type == "dist":
            return self.dist(tool_key_points, task_key_points)
        
        return 0

    def sixd_orient_error(self, tool_key_points, task_key_points, metadata):

        dist_weight = metadata["dist_weight"]
        orient_weight = metadata["orient_weight"]
        sat_dist = metadata["dist_scale"]
        sat_theta = metadata["orient_scale"]
        alpha = metadata["alpha"]
        beta = metadata["beta"]
        success_thresh = metadata["success_thresh"]

        pos_tool, orient_tool = extract_pos_orient_keypoints(tool_key_points)
        pos_task, orient_task = extract_pos_orient_keypoints(task_key_points)

        angle_rad = compute_geodesic_distance(orient_tool, orient_task)

        dist = np.linalg.norm(pos_tool - pos_task)

        dist_norm = dist/sat_dist
        theta_norm = angle_rad/sat_theta

        total_error = dist_norm * dist_weight + theta_norm * orient_weight
        
        cont_reward = continuous_reward_func(alpha, beta, total_error)

        disc_reward= 0

        if metadata["out_of_bounds"]:
            disc_reward -= 100

        if dist < success_thresh:
            disc_reward += 100

        return cont_reward + disc_reward
    
    def dist(self, tool_key_points, task_key_points):

        pos_tool, _ = extract_pos_orient_keypoints(tool_key_points)
        pos_task, _ = extract_pos_orient_keypoints(task_key_points)

        dist = np.linalg.norm(pos_tool - pos_task)

        return dist

def params_from_yaml(yaml_path : str):

    with open(yaml_path, 'r') as stream:
        params = yaml.safe_load(stream)

    return params

def send_and_recieve_to_server(stacked_data: np.ndarray, socket) -> np.ndarray:#clean
    packed_data = struct.pack('f' * stacked_data.shape[-1], *stacked_data)

    socket.send(packed_data)
    reply = socket.recv()
    joint_torques = np.frombuffer(reply, dtype = np.float32) 

    joint_torques = joint_torques.reshape(-1, ROBOT_JOINTS) 
    return joint_torques

def get_joint_torques(data, des_pos , des_orient, jt_socket):

    des_orient_quat = R.from_matrix(des_orient).as_quat()

    qpos = data.qpos.copy()      
    qvel = data.qvel.copy() 

    stacked_data = np.concatenate([des_pos, des_orient_quat, qpos, qvel])
    joint_torques = send_and_recieve_to_server(stacked_data, jt_socket)
    return joint_torques

def get_mft_torques(des_pos, des_orient, qpos, qvel, motion_force_axis, force_dim, force_magnitude, dx_world, sigma_force, mft_socket):

    # print(f"des pos: {des_pos}")
    # print(f"des orient: {des_orient}")
    # print(f"motion force axis: {motion_force_axis}")
    # print(f"force dim: {force_dim}")
    # print(f"dx_world: {dx_world}")


    dx = dx_world.reshape((3,1))

    desired_force = sigma_force @ dx
    if np.linalg.norm(desired_force) < 1e-5:
        desired_force = np.zeros((3,))
    else:
        desired_force = force_magnitude * desired_force/np.linalg.norm(desired_force)

    desired_force = desired_force.reshape(3,)
    
    stacked_data = np.concatenate([des_pos, des_orient, qpos, qvel, motion_force_axis.flatten(), np.array([force_dim]), desired_force])

    joint_torques = send_and_recieve_to_server(stacked_data, mft_socket)
    return joint_torques

def check_out_of_bounds(male_point, zone):

    if zone["shape"] == "cyl":
        in_z_bound = np.abs(male_point[2] - zone["center"][2]) < zone["height"]/2

        in_base = np.linalg.norm(male_point[:2] - zone["center"][:2]) < zone["radius"]

        if not in_z_bound or not in_base:
            return True

    elif zone["shape"] == "sphere":
        if np.linalg.norm(np.array(zone["center"]) - zone) > zone["radius"]:
            return True
    
    return False

def show_key_points(model, data, tool_points, task_points): 

    tool_id = fetch_body_id(model, data, "tool", obj_type = "body")
    tool_pos, tool_orient = fetch_body_pos_orient(data, tool_id)

    tool_points_world = tool_orient @ tool_points + tool_pos.reshape(3,1)

    task_id = fetch_body_id(model, data, "task", obj_type = "body")
    task_pos, task_orient = fetch_body_pos_orient(data, task_id)

    task_points_world = task_orient @ task_points + task_pos.reshape(3,1)
    
    for i in range(4):
        tool_point_id = fetch_body_id(model, data, f"tool_keypoint_{i+1}", "geom")
        task_point_id = fetch_body_id(model, data, f"task_keypoint_{i+1}", "geom")
        data.geom_xpos[tool_point_id] = tool_points_world[:, i].reshape(3,)
        data.geom_xpos[task_point_id] = task_points_world[:, i].reshape(3,)

    control_point_id = fetch_body_id(model, data, "control_point", "geom")
    tool_pos, _ = extract_pos_orient_keypoints(tool_points_world)
    data.geom_xpos[control_point_id] = tool_pos.reshape(3,)

    return data

class TaskModule():

    def __init__(self):

        self.task_points_ph = np.array([[0.005, 0.005, 0], [-0.005, 0.005, 0], [-0.005, -0.005, 0], [0.005, -0.005, 0]]).T
        self.tool_points_ph = np.array([[0.005, -0.005, 0.025], [-0.005, -0.005, 0.025], [-0.005, 0.005, 0.025], [0.005, 0.005, 0.025]]).T

        self.task_points_wc = np.array([
            [-0.007,  0.0007, 0.0],
            [ 0.007,  0.0007, 0.0],
            [ 0.007,  0.01,   0.0],
            [-0.007,  0.01,   0.0]
            
        ]).T
        
        self.tool_points_wc = np.array([[0.007, 0, 0],[-0.007, 0, 0], [-0.007, 0, 0.01], [0.007, 0, 0.01] ]).T
        
    def get_task_tool_points(self, task_name):

        if task_name == "wc":
            return self.task_points_wc, self.tool_points_wc
        elif task_name == "ph":
            return self.task_points_ph, self.tool_points_ph
        
        return None, None

