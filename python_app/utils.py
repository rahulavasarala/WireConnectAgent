import os
import yaml
from scipy.spatial.transform import Rotation as R
import numpy as np
import mujoco
import mujoco
import math
import struct 

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

def compute_theta_distance(orient1, orient2):

    quat1 = R.from_matrix(orient1).as_quat()
    quat2 = R.from_matrix(orient2).as_quat()

    dot = np.abs(np.dot(quat1, quat2))
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = 2 * np.arccos(dot) 

    return angle_rad

def params_from_yaml(yaml_path : str):

    with open(yaml_path, 'r') as stream:
        params = yaml.safe_load(stream)

    return params

class MjObserver():

    def __init__(self, mj_model, mj_data):
        self.id_data = {}

        self.id_data["force_sensor"] = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor")
        self.id_data["torque_sensor"] = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torque_sensor")
        self.id_data["site"] = mj_model.sensor_objid[self.id_data["force_sensor"]]

        self.id_data["male"] = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "male-connector-minimal")
        self.id_data["female"] = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "female-connector-truncated")
        self.id_data["link_7"] = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "link7")

        self.id_data["male_1"] = mj_data.geom('keypoint_male_1').id
        self.id_data["male_2"] = mj_data.geom('keypoint_male_2').id
        self.id_data["male_3"] = mj_data.geom('keypoint_male_3').id
        self.id_data["male_4"] = mj_data.geom('keypoint_male_4').id

        self.id_data["female_1"] = mj_data.geom('keypoint_female_1').id
        self.id_data["female_2"] = mj_data.geom('keypoint_female_2').id
        self.id_data["female_3"] = mj_data.geom('keypoint_female_3').id
        self.id_data["female_4"] = mj_data.geom('keypoint_female_4').id

        self.id_data["control_point"] = mj_data.geom('keypoint_control').id

        self.id_data["force_sensor_adr"] = mj_model.sensor_adr[self.id_data["force_sensor"]]
        self.id_data["torque_sensor_adr"] = mj_model.sensor_adr[self.id_data["torque_sensor"]]

    def get(self, id):
        return self.id_data[id]
    
    #These functions are unsafe for certain inputs
    def get_pos_orient(self, data , body_name, pos = None, frame = None, quat = False):

        if (pos is not None and frame is None) or (pos is None and frame is not None):
            raise ValueError("get_pos_orientation error: both pos and frame have to have values or be empty!")

        pos_world = data.xpos[self.id_data[body_name]]
        
        orient_world = data.xmat[self.id_data[body_name]].reshape(3,3)

        if quat:
            orient_world = data.xquat[self.id_data[body_name]]
            orient_world = R.from_quat(orient_world).as_matrix()

        if pos is None:
            return pos_world, orient_world
        
        if frame.shape[-1] == 4:
            frame = R.from_quat(frame).as_matrix()
        
        pos_frame = frame.T @(pos_world - pos)
        orient_frame = frame.T @ orient_world

        return pos_frame, orient_frame
    
def get_key_points_male(observer: MjObserver, data, mating_offset, pos = None, frame = None):
    key_points_male = np.array([[0.007, 0.007, -0.007, -0.007],
                                        [0.0, 0.0 , 0.0, 0.0], 
                                        [0.0, 0.01, 0.01, 0.0]])
    
    key_points_male -= (mating_offset * np.array([0, 1, 0])).reshape(3,1)

    male_pos, male_orient = observer.get_pos_orient(data, "male")
    key_points_world = male_orient @ key_points_male + male_pos.reshape(3,1)

    if pos is None:
        return key_points_world
    
    return frame.T @(key_points_world - pos.reshape(3,1))

# def get_key_points_female(observer: MjObserver, data, pos = None, frame = None): 
#     key_points_female = np.array([[-0.007, -0.007, 0.007, 0.007],
#                                     [0.0, 0.008 , 0.008, 0.0], 
#                                     [0.0, 0.0, 0.0, 0.0]])
    
#     female_pos, female_orient = observer.get_pos_orient(data, "female")
#     key_points_world = female_orient @ key_points_female + female_pos.reshape(3,1)

#     if pos is None:
#         return key_points_world
    
#     return frame.T @ (key_points_world - pos.reshape(3,1))
        
def get_velocity_reading(observer: MjObserver ,data, frame = None):
    vel = data.cvel[observer.get("male_body_id"), :3]

    if frame is None:
        return vel
    
    return frame.T @ vel

def get_force_torque_sensor_reading(observer: MjObserver, data, frame = None):
    fs_adr = observer.get("force_sensor_adr")
    ts_adr = observer.get("torque_sensor_adr")

    force = data.sensordata[ fs_adr : fs_adr + 3]
    _, force_sensor_orient = observer.get_pos_orient(data, "force_sensor")
    
    torque = data.sensordata[ts_adr : ts_adr + 3]

    force_world = force_sensor_orient @ force
    torque_world = force_sensor_orient @ torque

    if frame is None:
        return force_world, torque_world
    
    return frame.T @ force_world, frame.T @ torque_world

def get_male_pos_orient(observer: MjObserver, data, pos= None, frame = None, quat = False):

    _, male_orient = observer.get_pos_orient(data, "link_7", pos, frame, quat)
    male_pos , _ = observer.get_pos_orient(data, "male", pos, frame)

    return male_pos, male_orient

def realize_points_in_frame(points, pos, frame):
    return frame.T @(points - pos.reshape(3,1))

def show_keypoints(observer, data, params, metadata):
    male_key_points_world = get_key_points_male(observer, data, params["mating_offset"])

    female_key_points_world = metadata["female_key_points"]

    male_pos , _ = get_male_pos_orient(observer, data)

    for i in range(4):
        data.geom_xpos[observer.get(f"male_{i+1}")] = male_key_points_world[:, i].reshape(3,)
        data.geom_xpos[observer.get(f"female_{i+1}")] = female_key_points_world[:,i].reshape(3,)

    data.geom_xpos[observer.get("control_point")] = male_pos.reshape(3,)

    return data
    

class RewardModule():
    def __init__(self, debug):

        self.debug = debug

    def custom_pos_orientation_reward(self, observer: MjObserver, data, params, metadata, telemetry):

        key_points_male = get_key_points_male(observer, data, params["mating_offset"])
        key_points_female = metadata["female_points_world"]

        average_male_point = np.mean(key_points_male, axis = 1)
        average_female_point = np.mean(key_points_female, axis = 1)

        dist = np.linalg.norm(average_male_point - average_female_point)
        telemetry["min_dist"] = min(telemetry["min_dist"], dist)
    
        _, male_orientation = get_male_pos_orient(observer, data , quat = True)
        angle_rad = compute_theta_distance(male_orientation, metadata["init_orient"])

        norm_angle_rad = angle_rad/params["saturation_theta"]
        norm_distance = dist/(params["saturation_radius"] - params["mating_offset"])

        total_error = params["orient_weight"] * norm_angle_rad+params["dist_weight"] * norm_distance

        continuous_reward = continuous_reward_func(params["alpha"], params["beta"],total_error)

        if self.debug:
            print(f"norm_distance is: {norm_distance}, norm angle: {angle_rad}, total_error : {total_error}, cont reward: {continuous_reward}")


        discrete_reward = 0

        if dist < params["success_dist_thresh"]:
            discrete_reward += params["success_reward"]
        
        if telemetry["out_of_bounds"]:
            discrete_reward -= params["out_of_bounds_penalty"]

        return continuous_reward + discrete_reward


    def distance_reward(self, observer: MjObserver, data, params, metadata, telemetry):
        key_points_male = get_key_points_male(observer, data, params["mating_offset"])
        key_points_female = metadata["female_points_world"]

        average_male_point = np.mean(key_points_male, axis = 1)
        average_female_point = np.mean(key_points_female, axis = 1)

        dist = np.linalg.norm(average_male_point - average_female_point)

        telemetry["min_dist"] = min(dist, telemetry["min_dist"])

        norm_distance = dist/(params["saturation_radius"] - params["mating_offset"])
        
        continuous_reward = continuous_reward_func(params["alpha"], params["beta"], norm_distance)

        if self.debug:
            print(f"norm_distance is: {norm_distance}, cont reward: {continuous_reward}")

        discrete_reward = 0

        if dist < params["success_dist_thresh"]:
            discrete_reward += params["success_reward"]
        
        if telemetry["out_of_bounds"]:
            discrete_reward -= params["out_of_bounds_penalty"]

        return continuous_reward + discrete_reward

    def call(self, observer: MjObserver, data, params, metadata,telemetry, reward_type):
        if reward_type == "distance":
            return self.distance_reward(observer, data, params, metadata, telemetry)
        elif reward_type == "custom_pos_orient":
            return self.custom_pos_orientation_reward(observer, data, params, metadata, telemetry)
        
        return -1

def continuous_reward_func(a: float, b: float, x: float) -> float:
    exp_ax = math.exp(a * x)
    exp_neg_ax = math.exp(-a * x)

    denominator = b + exp_neg_ax + exp_ax

    r = 2 * (b + 2) * (denominator**-1) - 1

    return r

#helper function which communicates to the zmq server


        



    



    




