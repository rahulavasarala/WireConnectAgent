import numpy as np
import struct
import mujoco
from mujoco import viewer
import os
import time
import zmq
import math
from scipy.spatial.transform import Rotation as R
import random

from utils import compute_minimum_angle_rotation_between_orientations

#------------Global Constants ------------------------------------------------------------------
mj_xml_path = "/Users/rahulavasarala/Desktop/OpenSai/WireConnectAgent/models/scenes/rizon4smaleconjl.xml"
mj_model = mujoco.MjModel.from_xml_path(mj_xml_path)
mj_data = mujoco.MjData(mj_model)
ROBOT_JOINTS = 7

#Global variables related to the position of the object
force_sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor")
force_sensor_adr = mj_model.sensor_adr[force_sensor_id]

torque_sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torque_sensor")
torque_sensor_adr = mj_model.sensor_adr[torque_sensor_id]

site_id = mj_model.sensor_objid[force_sensor_id]
male_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "male-connector-minimal")

female_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "female-connector-truncated")
link_7_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "link7")

FILTER_FREQ = 50.0
CONTROL_FREQ = 1000.0

male_1_id = mj_data.geom('keypoint_male_1').id
male_2_id = mj_data.geom('keypoint_male_2').id
male_3_id = mj_data.geom('keypoint_male_3').id
male_4_id = mj_data.geom('keypoint_male_4').id

male_key_point_ids = [male_1_id, male_2_id, male_3_id, male_4_id]

female_1_id = mj_data.geom('keypoint_female_1').id
female_2_id = mj_data.geom('keypoint_female_2').id
female_3_id = mj_data.geom('keypoint_female_3').id
female_4_id = mj_data.geom('keypoint_female_4').id

female_key_point_ids = [female_1_id, female_2_id, female_3_id, female_4_id]

control_point_id = mj_data.geom('keypoint_control').id
#------------Global Constants ------------------------------------------------------------------

class ZMQEnv():

    def __init__(self, params, mj_model, mj_data, jt_socket, mft_socket, fspf_socket, action_space_dim = 7, observation_space_dim = 25, eval_pos_orient = None):#clean

        self.params = params
        self.target_pos = np.array([0.2, 0.2,0.2])
        self.target_orient = np.array([0.2,0.2,0.2])
        self.mft_socket = mft_socket
        self.fspf_socket = fspf_socket
        self.jt_socket = jt_socket
        self.mj_model = mj_model

        mujoco.mj_step(mj_model, mj_data)

        self.quick_reset_joint_pos = None

        #dimensions for action space and observation space
        self.action_space_dim = action_space_dim
        self.observation_space_dim = observation_space_dim

        self.motion_or_force_axis = np.zeros(3)
        self.force_dim = 0

        self.key_points_male = np.array([[0.007, 0.007, -0.007, -0.007],
                                        [0.0, 0.0 , 0.0, 0.0], 
                                        [0.0, 0.01, 0.01, 0.0]])
        
        self.key_points_male -= (self.params["mating_offset"] * np.array([0, 1, 0])).reshape(3,1)
        
        self.key_points_female = np.array([[-0.007, -0.007, 0.007, 0.007],
                                    [0.0, 0.008 , 0.008, 0.0], 
                                    [0.0, 0.0, 0.0, 0.0]])
        
        self.out_of_bounds = False

        self.mode = "train"
        self.telemetry = {"step_count": 0, "no_contact_count": 0, "min_dist": 0, "reward_sum": 0, "out_of_bounds": False}
       
        if eval_pos_orient is not None:
            self.on_init_given_eval_pos(mj_data, eval_pos_orient)
        else:
            self.on_init(mj_data)

    def reset(self, data):#clean
        mujoco.mj_resetData(self.mj_model, data)

        if self.quick_reset_joint_pos is not None:
            data.qpos[:] = self.quick_reset_joint_pos
            data.qvel[:] = np.zeros(ROBOT_JOINTS)
            mujoco.mj_step(self.mj_model, data)
        else:
            self.target_pos = self.male_init_point
            self.target_orient = R.from_matrix(self.male_init_orient).as_quat()
            mujoco.mj_step(self.mj_model, data)
            self.move_to_targets(data, iterations = 10000)
            self.quick_reset_joint_pos = data.qpos.copy()

        self.sigmaMotion = np.identity(3)
        self.sigmaForce = np.zeros((3,3))
        self.reset_fspf_data()

        obs, _, _, _ = self._get_obs(data)

        self.telemetry["step_count"] = 0
        self.telemetry["no_contact_count"] = 0
        self.telemetry["out_of_bounds"] = False
        self.telemetry["reward_sum"] = 0

        return data, obs

    def on_init(self, data):
        z_space = 0.005
        
        euler = self.params["orient_noise"] * np.array([0, random.random(), random.random()])
        R_female_to_tilt = R.from_euler('xyz', euler).as_matrix()
        translation = self.params["trans_noise"] * np.array([random.random(), random.random(), random.random()])

        R_world_to_female = data.xmat[female_id].reshape((3,3))
        female_pos = data.xpos[female_id]
        R_total = R_world_to_female @ R_female_to_tilt
        female_points_world = R_total @ self.key_points_female
        female_points_world += (female_pos + translation).reshape(3,1)
        self.female_points_world = female_points_world

        target_point = np.mean(female_points_world, axis=1)
        target_point = target_point + z_space * R_total[:,2]
        target_orient = np.array([[1, 0, 0], [0, -1, 0], [0,0,-1]]) @ R_total

        self.male_init_point = target_point
        self.male_init_orient = target_orient
        self.female_points_frame = self.male_init_orient.T @ (female_points_world - self.male_init_point.reshape(3,1))

        ee_start = np.concatenate([self.male_init_point, R.from_matrix(self.male_init_orient).as_quat()])
        print("Saving Orientations and positions of the male connector frame before training!")
        np.savetxt(f"./runs/run_{self.params["name"]}/start_pos_orient.txt", ee_start)

    def on_init_given_eval_pos(self, data, eval_pos_orient):
        self.male_init_point = eval_pos_orient[:3]
        self.male_init_orient = R.from_quat(eval_pos_orient[3:]).as_matrix()
        R_world_to_female = data.xmat[female_id].reshape((3,3))
        female_pos = data.xpos[female_id]
        female_points_world = R_world_to_female @ self.key_points_female
        female_points_world += female_pos.reshape(3,1)
        self.female_points_world = female_points_world
        self.female_points_frame = self.male_init_orient.T @ (female_points_world - self.male_init_point.reshape(3,1))      

    def move_to_targets(self, data, iterations =1): #clean
        #step the simulation iterations times with normal joint torque server
        for _ in range(iterations):
            joint_torques = self.get_joint_torques(data)
            data.ctrl[:] = joint_torques
            mujoco.mj_step(self.mj_model, data)

        return data

    def get_joint_torques(self, data) -> np.ndarray:#clean
        #get the cart_pos, cart_orient, qpos, qvel
        stacked_data = np.concatenate([self.target_pos, self.target_orient])

        qpos = data.qpos.copy()      
        qvel = data.qvel.copy() 

        stacked_data = np.concatenate([stacked_data, qpos, qvel])
        joint_torques = self.send_and_recieve_joint_torque_server(stacked_data)

        return joint_torques

    def send_and_recieve_joint_torque_server(self, stacked_data: np.ndarray) -> np.ndarray:#clean
        packed_data = struct.pack('f' * stacked_data.shape[-1], *stacked_data)

        self.jt_socket.send(packed_data)
        reply = self.jt_socket.recv()
        joint_torques = np.frombuffer(reply, dtype = np.float32) 

        joint_torques = joint_torques.reshape(-1, ROBOT_JOINTS) 
        # print("joint_torques: {}".format(joint_torques))
        return joint_torques
    
    def continuous_reward_function(self, x):

        r = 2*(self.params["beta"] + 2)*(self.params["beta"] + math.e**(-self.params["alpha"] * x) + math.e**(self.params["alpha"] * x))**-1 -1
        return r
    
    def compute_reward(self, data, debug = False):
        key_points_male = self.get_key_points_male(data, "world")
        key_points_female = self.get_key_points_female("world")

        average_male_point = np.mean(key_points_male, axis = 1)
        average_female_point = np.mean(key_points_female, axis = 1)

        distance = np.linalg.norm(average_male_point - average_female_point)
        self.telemetry["min_dist"] = min(self.telemetry["min_dist"], distance)
    
        _, male_orientation = self.get_male_connector_pos_orient(data , "world")
        angle_rad = compute_minimum_angle_rotation_between_orientations(male_orientation, self.male_init_orient)

        norm_angle_rad = angle_rad/self.params["saturation_theta"]
        norm_distance = distance/(self.params["saturation_radius"] - self.params["mating_offset"])

        if debug:
            print("norm angle rad : {}, norm_distance : {},  distance: {}".format(norm_angle_rad, norm_distance, distance)) 

        total_error = self.params["orient_weight"] * norm_angle_rad+self.params["dist_weight"] * norm_distance
        continuous_reward = self.continuous_reward_function(total_error)

        discrete_reward = 0

        if distance < self.params["success_dist_thresh"]:
            discrete_reward += self.params["success_reward"]
        
        if self.out_of_bounds:
            discrete_reward -= self.params["out_of_bounds_penalty"]

        return continuous_reward + discrete_reward
    
    def sample_obs(self, data):#clean
        force_frame, torque_frame = self.get_force_torque_sensor_reading(data, "frame")
        male_pos_frame, male_orient_frame = self.get_male_connector_pos_orient(data, "frame")

        male_orient_frame_quat = R.from_matrix(male_orient_frame).as_quat()
        flattened_point_cloud = self.female_points_frame.flatten()

        obs = np.concatenate([force_frame, torque_frame, male_pos_frame, male_orient_frame_quat, flattened_point_cloud])

        return obs
    
    def update_no_contact_count(self, data):

        force = data.sensordata[force_sensor_adr : force_sensor_adr + 3]
        
        if np.linalg.norm(force) < 0.01:
            self.telemetry["no_contact_count"] += 1
        else:
            self.telemetry["no_contact_count"] = 0

    
    def _get_obs(self, data):

        rewards = np.zeros(1)
        reward = self.compute_reward(data)
        rewards[0] = reward

        dones = np.zeros(1)

        self.update_no_contact_count(data)
        if self.telemetry["no_contact_count"] > 15 or self.telemetry["step_count"] > self.params["actions_per_episode"] * self.params["num_phys_steps"] or self.telemetry["out_of_bounds"]:
            dones[0] = 1

        obs = np.zeros((1, self.observation_space_dim))
        obs[0] = self.sample_obs(data)

        info = {
            "reward": rewards,
            "terminated": dones,
            "TimeLimit.truncated": np.zeros_like(dones)
        }

        return (obs, rewards, dones, info)
    
    def reset_fspf_data(self):
        self.motion_or_force_axis = np.zeros(3)
        self.force_dim = 0

        data = np.ones(1)
        packed_data = struct.pack('f' * data.shape[-1], *data)

        self.fspf_socket.send(packed_data)
        reply = self.fspf_socket.recv()

    def update_fspf_server(self, data, desired_dx):

        motion_control = (FILTER_FREQ/CONTROL_FREQ) * self.sigmaMotion @ desired_dx #correct
        force_control = (FILTER_FREQ/CONTROL_FREQ) * self.sigmaForce @ desired_dx #correct 

        measured_force, _ = self.get_force_torque_sensor_reading(data, "world")
        measured_vel = self.get_velocity_reading(data, "world")

        stacked_data = np.concatenate([motion_control, force_control, measured_force, measured_vel])
        packed_data = struct.pack('f' * stacked_data.shape[-1], *stacked_data)

        self.fspf_socket.send(packed_data)
        reply = self.fspf_socket.recv()
        motion_or_force_axis_and_fdim  = np.frombuffer(reply, dtype = np.float32) 

        motion_or_force_axis = motion_or_force_axis_and_fdim[:3]
        fdim = motion_or_force_axis_and_fdim[3]

        #selection matrix inference logic
        if fdim == 0:
            self.sigmaForce = np.zeros((3,3))
            self.sigmaMotion = np.identity(3)
        elif fdim == 1:
            self.sigmaForce = motion_or_force_axis @ motion_or_force_axis.T
            self.sigmaMotion = np.identity(3) - self.sigmaForce
        elif fdim == 2:
            self.sigmaMotion = motion_or_force_axis @ motion_or_force_axis.T
            self.sigmaForce = np.identity(3) - self.sigmaForce
        elif fdim == 3:
            self.sigmaForce = np.identity(3)
            self.sigmaMotion = np.zeros((3,3))

        return motion_or_force_axis, fdim
    
    def get_joint_torques_mft_server(self, des_pos, des_orient, qpos, qvel, force_or_motion_axis, fdim,  desired_force):#clean

        stacked_data = np.concatenate([des_pos, des_orient, qpos, qvel, force_or_motion_axis, fdim, desired_force])
        packed_data = struct.pack('f' * stacked_data.shape[-1], *stacked_data)

        self.mft_socket.send(packed_data)
        reply = self.mft_socket.recv()
        joint_torques = np.frombuffer(reply, dtype = np.float32) 

        joint_torques = joint_torques.reshape(-1, ROBOT_JOINTS) 
        # print("joint_torques: {}".format(joint_torques))
        return joint_torques
    
    def move_to_targets_mft(self, data):#clean

        des_pos = np.array([0.2,0.2,0.2])
        des_orient = np.array([0,1,0,0])

        qpos = data.qpos.copy()
        qvel = data.qvel.copy() 
        force_or_motion_axis = np.array([0,0,0])
        fdim = np.array([0])
        desired_force = np.array([0, 0 ,0])

        joint_torques = self.get_joint_torques_mft_server(des_pos, des_orient, qpos, qvel, force_or_motion_axis, fdim,  desired_force)

        data.ctrl[:] = joint_torques
        mujoco.mj_step(self.mj_model, data)

        return data

    def step(self, data, action = None):#clean

        if action is not None:
            action = action.reshape(-1)
            self.apply_action(data, action)

        iterations = 1 if self.mode == "eval" else self.params["num_phys_steps"]

        for _ in range(iterations):
            joint_torques = self.get_joint_torques(data)
            data.ctrl[:] = joint_torques
            mujoco.mj_step(self.mj_model, data)
            self.telemetry["step_count"] += 1

        obs, reward, dones, info = self._get_obs(data)
        self.telemetry["reward_sum"] += reward[0]

        if dones[0] == 1:
            print(f"telemetry: {self.telemetry}")
            self.reset(data)

        return data, (obs, reward, dones, info)
    
    #What apply action does is set the target pos and quat for the robot
    def apply_action(self, data, action):
        self.telemetry["out_of_bounds"] = False

        dx_frame = action[:3]
        dtheta_frame = action[3:6]
        magnitude_force = action[6]

        #scaling the dx world, dtheta_frame, and magnitude force created by the actor network
        if np.linalg.norm(dx_frame) > self.params["max_displacement_thresh"]:
            dx_frame = self.params["max_displacement_thresh"] * dx_frame/np.linalg.norm(dx_frame)

        if np.linalg.norm(dtheta_frame) > self.params["max_rotation_thresh"]:
            dtheta_frame = self.params["max_rotation_thresh"] * dtheta_frame/np.linalg.norm(dtheta_frame)

        if np.linalg.norm(magnitude_force) > self.params["max_force_thresh"]:
            magnitude_force = self.params["max_force_thresh"] * magnitude_force/np.linalg.norm(magnitude_force)

        R_dtheta_frame = R.from_euler("xyz", dtheta_frame).as_matrix()

        dx_world = self.male_init_orient @ dx_frame
        male_pos_world, _ = self.get_male_connector_pos_orient(data, "world")
        _ , male_orient_frame = self.get_male_connector_pos_orient(data, "frame")

        male_pos_world_prime = male_pos_world + dx_world
        male_orient_prime = self.male_init_orient @ R_dtheta_frame @ male_orient_frame

        # #find the orientation angle distance between the target orientation and frame as angle_rad
        angle_rad = compute_minimum_angle_rotation_between_orientations(male_orient_prime, self.male_init_orient)

        #if the distance between the target orientation or target position and the frame is too much, send out_of_bounds to true
        if np.linalg.norm(male_pos_world_prime - self.male_init_point) > self.params["saturation_radius"]:
            print("distance violation!")
            self.telemetry["out_of_bounds"] = True
        elif angle_rad > self.params["saturation_theta"]:
            print("angle violation!")
            self.telemetry["out_of_bounds"] = True

        self.target_pos = male_pos_world_prime
        self.target_orient = R.from_matrix(male_orient_prime).as_quat()

        return dx_world

    # The function below is a utility function
    def get_male_connector_pos_orient(self, data, frame_of_choice):
        male_con_pos_world = np.array(data.xpos[male_body_id])
        xquat = data.xquat[link_7_id] 
        male_orient_world = R.from_quat(xquat).as_matrix()  

        if frame_of_choice == "world":
            return male_con_pos_world, male_orient_world
        elif frame_of_choice == "frame":
            return self.male_init_orient.T @ (male_con_pos_world - self.male_init_point), self.male_init_orient.T @ male_orient_world
        
        return male_con_pos_world, male_orient_world
    
    #The function below is a utility function
    def get_key_points_male(self, data, frame_of_choice):

        if frame_of_choice == "male":
            return self.key_points_male
        #base male
        male_con_pos_world = np.array(data.xpos[male_body_id])
        xmat = data.xmat[male_body_id]   
        male_orient_world = xmat.reshape(3, 3) 
       
        key_points_male_world = male_orient_world @ self.key_points_male + male_con_pos_world.reshape(3,1)
        return key_points_male_world
    
    #The function below is a utility function
    def get_key_points_female(self, frame_of_choice):

        if frame_of_choice == "female":
            return self.key_points_female
        
        return self.female_points_world
    
    def get_force_torque_sensor_reading(self, data, frame_of_choice):

        force = data.sensordata[force_sensor_adr : force_sensor_adr + 3]
        xquat = data.xquat[force_sensor_id] 
        force_sensor_orient_world = R.from_quat(xquat).as_matrix() 
        
        torque = data.sensordata[torque_sensor_adr : torque_sensor_adr + 3]

        force_world = force_sensor_orient_world @ force
        torque_world = force_sensor_orient_world @ torque

        if frame_of_choice == "world":
            return force_world, torque_world
        elif frame_of_choice == "frame":
            return self.male_init_orient.T @ force_world, self.male_init_orient.T @ torque_world
        
        return force, torque
    
    def get_velocity_reading(self,data, frame_of_choice):

        vel = data.cvel[male_body_id, :3]

        if frame_of_choice == "world":
            return vel
        elif frame_of_choice == "frame":
            return self.male_init_orient.T @ vel
        
        return vel
    
    #The function below is a utility function
    def show_keypoints(self, data):
        male_key_points_world = self.get_key_points_male(data, "world")

        female_key_points_world = self.get_key_points_female("world")

        male_pos , _ = self.get_male_connector_pos_orient(data, "world")

        for i in range(4):
            data.geom_xpos[male_key_point_ids[i]] = male_key_points_world[:, i].reshape(3,)
            data.geom_xpos[female_key_point_ids[i]] = female_key_points_world[:,i].reshape(3,)

        data.geom_xpos[control_point_id] = male_pos.reshape(3,)

        return data

    #The function below is a utility function
    def set_target_pos_orient(self, pos , orient):
        self.target_pos = pos
        self.target_orient = orient

    def set_mode(self, mode):
        if mode == "eval":
            self.mode = "eval"
            return

        self.mode = "train"




    
