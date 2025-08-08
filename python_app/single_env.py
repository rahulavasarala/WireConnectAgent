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
import glfw
from flax.core import FrozenDict
import orbax.checkpoint as ocp
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import flax
from distrax import Normal

from policy_network import Agent
import torch

#-------------------------------------------------#
# from policy_network import Network, Critic, Actor
#--------------------------------------------------#

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

ctx = zmq.Context()

jt_socket = ctx.socket(zmq.REQ)
jt_socket.connect("ipc:///tmp/zmq_torque_server")

mft_socket = ctx.socket(zmq.REQ)
mft_socket.connect("ipc:///tmp/zmq_motion_force_server")

fspf_socket = ctx.socket(zmq.REQ)
fspf_socket.connect("ipc:///tmp/zmq_fspf_server")

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

class SingleZMQEnv():

    def __init__(self, mj_model, mj_data, target_pos: np.ndarray, target_orient: np.ndarray, jt_socket, mft_socket, fspf_socket, action_space_dim = 7, observation_space_dim = 25, eval_pos_orient = None):#clean

        self.target_pos = target_pos
        self.target_orient = target_orient

        self.mft_socket = mft_socket
        self.fspf_socket = fspf_socket
        self.jt_socket = jt_socket

        self.mj_model = mj_model

        mujoco.mj_step(mj_model, mj_data)

        self.quick_reset_joint_pos = None

        #dimensions for action space and observation space
        self.action_space_dim = action_space_dim
        self.observation_space_dim = observation_space_dim

        self.step_count = 0
        self.no_contact_count = 0

        #Thresholds for the outputs of the agent
        self.max_force_thresh = 0.1
        self.max_rotation_thresh = 0.01
        self.max_displacement_thresh = 0.002

        self.saturation_radius = 0.05
        self.saturation_theta = math.pi/18

        self.motion_or_force_axis = np.zeros(3)
        self.force_dim = 0

        self.key_points_male = np.array([[0.007, 0.007, -0.007, -0.007],
                                    [0.0, 0.0 , 0.0, 0.0], 
                                    [0.0, 0.01, 0.01, 0.0]])
        
        self.key_points_female = np.array([[-0.007, -0.007, 0.007, 0.007],
                                    [0.0, 0.008 , 0.008, 0.0], 
                                    [0.0, 0.0, 0.0, 0.0]])
        
        self.out_of_bounds = False

        if eval_pos_orient is not None:
            self.male_init_point = eval_pos_orient[:3]
            self.male_init_orient = R.from_quat(eval_pos_orient[3:]).as_matrix()
            R_world_to_female = mj_data.xmat[female_id].reshape((3,3))
            female_pos = mj_data.xpos[female_id]
            female_points_world = R_world_to_female @ self.key_points_female
            female_points_world += female_pos.reshape(3,1)
            self.female_points_world = female_points_world
            self.female_points_frame = self.male_init_orient.T @ (female_points_world - self.male_init_point.reshape(3,1))
        else:
            self.on_init(mj_data) 

        self.mode = "train"
        self.train_physics_steps = 170
        
        self.num_envs = 1

    def reset(self, data):#clean
        mujoco.mj_resetData(self.mj_model, data)

        if self.quick_reset_joint_pos is not None:
            data.qpos[:] = self.quick_reset_joint_pos
            data.qvel[:] = np.zeros(7)
            mujoco.mj_step(self.mj_model, data)
        else:
            self.target_pos = self.male_init_point
            self.target_orient = R.from_matrix(self.male_init_orient).as_quat()
            mujoco.mj_step(self.mj_model, data)
            self.move_to_targets(data, iterations = 10000)
            self.quick_reset_joint_pos = data.qpos.copy()

        # self.sigmaMotion = np.identity(3)
        # self.sigmaForce = np.zeros((3,3))
        # self.reset_fspf_data()

        obs, _, _, _ = self._get_obs(data)

        self.step_count = 0
        self.no_contact_count = 0
        self.out_of_bounds = False
        self.reward_sum = 0

        return data, obs

    def on_init(self, data):
        z_space = 0.005
        translation_noise_magnitude = 0.005
        rotation_noise_magnitude = 0.05
        
        euler = rotation_noise_magnitude * np.array([0, random.random(), random.random()])
        R_female_to_tilt = R.from_euler('xyz', euler).as_matrix()
        translation = translation_noise_magnitude* np.array([random.random(), random.random(), random.random()])

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
        np.savetxt("start_pos_orient.txt", ee_start)

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

        joint_torques = joint_torques.reshape(-1, 7) 
        # print("joint_torques: {}".format(joint_torques))
        return joint_torques
    
    def compute_reward(self, data):#clean
        
        key_points_male = self.get_key_points_male(data, "world")
        key_points_female = self.get_key_points_female("world")

        distance = np.linalg.norm(key_points_male - key_points_female)

        bias = 0

        if distance < 0.002:
            bias += 1
        
        if self.out_of_bounds:
            bias -= 3

        return -1*distance + bias
    
    def sample_obs(self, data):#clean
        force = data.sensordata[force_sensor_adr : force_sensor_adr + 3]
        torque = data.sensordata[torque_sensor_adr: torque_sensor_adr + 3]

        R_flat = data.xmat[site_id]
        R_world_sensor = R_flat.reshape((3,3))

        force_world = R_world_sensor @ force
        torque_world = R_world_sensor @ torque

        force_frame = self.male_init_orient.T @ force_world
        torque_frame = self.male_init_orient.T @ torque_world

        xmat = data.xmat[male_body_id]   
        male_orient_world = xmat.reshape(3, 3)  

        male_orient_frame = self.male_init_orient.T @ male_orient_world
        male_pos_world = data.xpos[male_body_id]
        male_pos_frame = self.male_init_orient.T @(male_pos_world - self.male_init_point)

        male_frame_quat = R.from_matrix(male_orient_frame).as_quat()

        flattened_point_cloud = self.female_points_frame.flatten()
        obs = np.concatenate([force_frame, torque_frame, male_frame_quat, male_pos_frame, flattened_point_cloud])

        return obs
    
    def update_no_contact_count(self, data):

        force = data.sensordata[force_sensor_adr : force_sensor_adr + 3]
        
        if np.linalg.norm(force) < 0.01:
            self.no_contact_count += 1
        else:
            self.no_contact_count = 0

    
    def _get_obs(self, data):

        rewards = np.zeros(self.num_envs)
        reward = self.compute_reward(data)
        rewards[0] = reward

        dones = np.zeros(self.num_envs)

        self.update_no_contact_count(data)
        if self.no_contact_count > 15 or self.step_count > 20 * self.train_physics_steps or self.out_of_bounds:
            dones[0] = 1

        obs = np.zeros((self.num_envs, self.observation_space_dim))
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

        force = data.sensordata[force_sensor_adr : force_sensor_adr + 3]

        R_flat = data.xmat[site_id]
        R_world_sensor = R_flat.reshape((3,3))
        measured_force = R_world_sensor @ force
        measured_vel = data.cvel[male_body_id, :3]

        stacked_data = np.concatenate([motion_control, force_control, measured_force, measured_vel])
        # print("stacked_data : {}".format(stacked_data))
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

        joint_torques = joint_torques.reshape(-1, 7) 
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

        iterations = 1 if self.mode == "eval" else self.train_physics_steps

        for _ in range(iterations):
            joint_torques = self.get_joint_torques(data)
            data.ctrl[:] = joint_torques
            mujoco.mj_step(self.mj_model, data)
            self.step_count += 1

        obs, reward, dones, info = self._get_obs(data)
        self.reward_sum += reward[0]

        if dones[0] == 1:
            print("reset stats: no contact count : {}, step count: {}, out of bounds: {}, reward_sum: {}".format(self.no_contact_count, self.step_count, self.out_of_bounds, self.reward_sum))
            self.reset(data)

        return data, (obs, reward, dones, info)
    
    #What apply action does is set the target pos and quat for the robot
    def apply_action(self, data, action):
        self.out_of_bounds = False

        dx_frame = action[:3]
        dtheta_frame = action[3:6]
        magnitude_force = action[6]

        #scaling the dx world, dtheta_frame, and magnitude force created by the actor network
        if np.linalg.norm(dx_frame) > self.max_displacement_thresh:
            dx_frame = self.max_displacement_thresh * dx_frame/np.linalg.norm(dx_frame)

        if np.linalg.norm(dtheta_frame) > self.max_rotation_thresh:
            dtheta_frame = self.max_rotation_thresh * dtheta_frame/np.linalg.norm(dtheta_frame)

        if np.linalg.norm(magnitude_force) > self.max_force_thresh:
            magnitude_force = self.max_force_thresh * magnitude_force/np.linalg.norm(magnitude_force)

        R_dtheta_frame = R.from_euler("xyz", dtheta_frame).as_matrix()

        # print("R_dtheta_frame: {}".format(R_dtheta_frame))

        dx_world = self.male_init_orient @ dx_frame
        male_pos_world, _ = self.get_male_connector_pos_orient(data, "world")
        _ , male_orient_frame = self.get_male_connector_pos_orient(data, "frame")

        male_pos_world_prime = male_pos_world + dx_world
        male_orient_prime = self.male_init_orient @ R_dtheta_frame @ male_orient_frame

        # #find the orientation angle distance between the target orientation and frame as angle_rad
        male_orient_prime_quat = R.from_matrix(male_orient_prime).as_quat()
        male_orient_init_quat = R.from_matrix(self.male_init_orient).as_quat()

        dot = np.abs(np.dot(male_orient_prime_quat, male_orient_init_quat))
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = 2 * np.arccos(dot) 

        #if the distance between the target orientation or target position and the frame is too much, send out_of_bounds to true
        if np.linalg.norm(male_pos_world_prime - self.male_init_point) > self.saturation_radius:
            print("Distance violation!")
            self.out_of_bounds = True
        elif angle_rad > self.saturation_theta:
            print("angle violation!")
            self.out_of_bounds = True

        self.target_pos = male_pos_world_prime
        self.target_orient = R.from_matrix(male_orient_prime).as_quat()

    # The function below is a utility function
    def get_male_connector_pos_orient(self, data, frame_of_choice):
        male_con_pos_world = np.array(data.xpos[male_body_id])
        xquat = data.xquat[link_7_id] 
        male_orient_world = R.from_quat(xquat).as_matrix()  

        if frame_of_choice == "world":
            return male_con_pos_world, male_orient_world
        elif frame_of_choice == "frame":
            return self.male_init_orient.T @ (male_con_pos_world - self.male_init_orient), self.male_init_orient.T @ male_orient_world
        
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
    
class AgentRolloutTest():

    def __init__(self, weights_path, eval_pos_orient):

        self.step_count = 0
        self.env = SingleZMQEnv(mj_model,mj_data, np.zeros(3), np.array([0,1,0,0]), jt_socket, mft_socket, fspf_socket, eval_pos_orient=eval_pos_orient)
        self.env.set_mode("eval")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #lets create some init code
        self.model = Agent(self.env)
        state_dict = torch.load(weights_path)

        self.model.load_state_dict(state_dict)

        self.model.eval()

    def sample_action(self, obs):
        action, _, _, _ = self.model.get_action_and_value(obs)
        return action
    
    def reset(self, data):
        self.env.reset(data)

    def step(self, data):

        if self.step_count >= 10:
            obs, _, _ , _ = self.env._get_obs(data)
            
            obs = torch.Tensor(obs).to(self.device)
            action = self.sample_action(obs)
            action = action.cpu().numpy()
            action = action.reshape(7,)
            self.step_count = 0
            print("action taken: {}".format(action))
            self.env.apply_action(data, action)

        for _ in range(17):
            data , _ = self.env.step(data)

        self.step_count += 1

        return data, 1
    
#Random step test with pause on outofbounds
class RandomStepTest():

    def __init__(self):

        self.step_count = 0
        self.env = SingleZMQEnv(mj_model,mj_data, np.zeros(3), np.array([0,1,0,0]), jt_socket, mft_socket, fspf_socket)
        self.env.set_mode("eval")
        self.action = np.zeros(7)
        self.out_of_bounds = False

        # eval_pos_orient=np.array([0.2,0.2,0.2, 0, 1,0,0])

    def sample_action(self):
        action = np.random.uniform(low = -0.1, high = 0.1, size = 7)
        return action

    def reset(self, data):
        self.env.reset(data)
        _, orient = self.env.get_male_connector_pos_orient(data, "world")
        print(self.env.male_init_orient, orient)

    def step(self, data):

        if self.step_count >= 10:
            self.action = self.sample_action()
            self.action = self.action.reshape(7,)
            self.step_count = 0
            print("action taken: {}".format(self.action))
            data, _ = self.env.step(data, self.action)

        for _ in range(17):
            data , _ = self.env.step(data)

        self.step_count += 1

        return data, 1
    
class ShowPointsTest():

    def __init__(self):
        self.step_count = 0
        self.env = SingleZMQEnv(mj_model,mj_data, 1, np.zeros(3), np.array([0,1,0,0]), jt_socket, mft_socket, fspf_socket)
        self.target_point = np.array([0.3,0.3, 0.2])
        self.target_orient = np.array([0,1,0,0])

        self.action = np.zeros(3)

    def sample_action(self):
        action = np.random.uniform(low = -0.1, high = 0.1, size = 3)
        return action

    def reset(self, data):
        self.env.reset(data)
        self.env.set_target_pos_orient(self.target_point, self.target_orient)

    def step(self, data):

        if self.step_count >= 10: 
            # self.env.set_target_pos_orient(self.target_point + 0.1 * self.sample_action(), self.target_orient)
            self.step_count = 0

        for _ in range(17):
            data = self.env.move_to_targets(data)

        data = self.env.show_keypoints(data)

        male_pos, _ = self.env.get_male_connector_pos_orient(data, "world")

        print("difference between target position and actual male connector position: {}".format(np.linalg.norm(self.target_point - male_pos)))

        self.step_count += 1

        return data, _
    
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
scene = mujoco.MjvScene(mj_model, maxgeom=1000)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)

def scroll_callback(window, xoffset, yoffset):
        mujoco.mjv_moveCamera(mj_model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, yoffset * 0.1, scene, cam)

# Global state
def mouse_button_callback(window, button, action, mods):
    global button_left, button_middle, button_right, lastx, lasty
    # Update button state
    button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

    # Update mouse position
    lastx, lasty = glfw.get_cursor_pos(window)

def cursor_position_callback(window, xpos, ypos):
        global button_left, button_middle, button_right, lastx, lasty

        if not button_left and not button_middle and not button_right:
            return  # No mouse buttons pressed

        dx = xpos - lastx
        dy = ypos - lasty
        lastx = xpos
        lasty = ypos

        width, height = glfw.get_window_size(window)

        # Check for Shift modifier
        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        # Determine camera action based on button
        if button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        # Perform the camera movement
        mujoco.mjv_moveCamera(mj_model, action, dx / height, dy / height, scene, cam)

        
#This is the main loop for
def main():
    global mj_data
    RHZ = 60.0
    test_to_run = "Random"
    test = None
    weights_path = "/Users/rahulavasarala/Desktop/OpenSai/WireConnectAgent/python_app/runs/run2/model300.cleanrl_model"
    
    if test_to_run == "Agent":
        loaded_pos_orient = np.loadtxt('start_pos_orient.txt')
        test = AgentRolloutTest(weights_path, loaded_pos_orient)
    elif test_to_run == "Random":
        test = RandomStepTest()
    elif test_to_run == "ShowPoints":
        test = ShowPointsTest()

    # Reset simulation
    test.reset(mj_data)

    # Initialize GLFW
    glfw.init()
    window = glfw.create_window(800, 600, "WireBot Simulation Python", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # Enable vsync

    # Create context for rendering
    context = mujoco.MjrContext(mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw_loop_count = 0

    print("Starting WireBot Simulation!")

    # Physics + Graphics loop
    while not glfw.window_should_close(window):
        #step test
        start_time = time.time()
        mj_data, _ = test.step(mj_data)
        time_elapsed = time.time() - start_time

        if time_elapsed < 1.0/RHZ:
            time.sleep(1.0/RHZ - time_elapsed)
            
        # Update scene
        mujoco.mjv_updateScene(mj_model, mj_data, scene_option, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

        # Get framebuffer size
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)

        # Render scene
        mujoco.mjr_render(viewport, scene, context)

        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

        glfw_loop_count += 1

    # Cleanup
    glfw.terminate()

if __name__ == "__main__":
    main()
   


