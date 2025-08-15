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

from utils import compute_theta_distance, MjObserver, RewardModule, get_force_torque_sensor_reading
from utils import get_key_points_male, get_velocity_reading, get_male_pos_orient, realize_points_in_frame

#------------Global Constants ------------------------------------------------------------------
ROBOT_JOINTS = 7
FILTER_FREQ = 50.0
CONTROL_FREQ = 1000.0
#------------Global Constants ------------------------------------------------------------------

class ZMQEnv():

    def __init__(self, params, mj_model, mj_data, jt_socket, mft_socket, fspf_socket, action_space_dim = 7, observation_space_dim = 25, eval_pos_orient = None):#clean

        self.observer = MjObserver(mj_model, mj_data)
        self.reward_module = RewardModule(params["debug"])
        print(f"Debug mode is: {params["debug"]}")

        self.params = params
        self.target_pos = np.array([0.2,0.2,0.2])
        self.target_orient = np.array([0.2,0.2,0.2])
        self.mft_socket = mft_socket
        self.fspf_socket = fspf_socket
        self.jt_socket = jt_socket
        self.mj_model = mj_model

        mujoco.mj_step(mj_model, mj_data)

        self.quick_reset_joint_pos = None
        self.action_space_dim = action_space_dim
        self.observation_space_dim = observation_space_dim

        self.motion_or_force_axis = np.zeros(3)
        self.force_dim = 0
        
        self.key_points_female = np.array([[-0.007, -0.007, 0.007, 0.007],
                                    [0.0007, 0.01 , 0.01, 0.0007], 
                                    [0.0, 0.0, 0.0, 0.0]])
        
        self.key_points_female += np.array([0,0.001, 0]).reshape(3,1)
        
        self.out_of_bounds = False

        self.mode = "train"
        self.telemetry = {"step_count": 0, "no_contact_count": 0, "min_dist": 1, "reward_sum": 0, "out_of_bounds": False}
       
        if eval_pos_orient is not None:
            self.on_init_given_eval_pos(eval_pos_orient)
        else:
            self.on_init(mj_data)

    #This function resets the state of the robot to the init position, and resets the values in the telemetry
    def reset(self, data):
        mujoco.mj_resetData(self.mj_model, data)

        if self.quick_reset_joint_pos is not None:
            data.qpos[:] = self.quick_reset_joint_pos
            data.qvel[:] = np.zeros(ROBOT_JOINTS)
            mujoco.mj_step(self.mj_model, data)
            self.target_point = self.male_init_point
            self.target_orient = R.from_matrix(self.male_init_orient).as_quat()
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

    #This creates some noise in the estimate of the female connector's position, and initializes the end effectors starting
    #position from the noisy estimate
    def on_init(self, data):
        z_space = 0.005
        
        euler = self.params["orient_noise"] * np.array([0, random.random(), random.random()])
        R_female_to_tilt = R.from_euler('xyz', euler).as_matrix()
        translation = self.params["trans_noise"] * np.array([random.random(), random.random(), random.random()])

        female_pos, female_orient = self.observer.get_pos_orient(data, "female")
        R_total = female_orient @ R_female_to_tilt
        female_points_world = R_total @ self.key_points_female
        female_points_world += (female_pos + translation).reshape(3,1)
        self.female_points_world = female_points_world

        target_point = np.mean(female_points_world, axis=1)
        target_point = target_point + z_space * R_total[:,2]
        target_orient = np.array([[1, 0, 0], [0, -1, 0], [0,0,-1]]) @ R_total

        self.male_init_point = target_point
        self.male_init_orient = target_orient
        self.female_points_frame = realize_points_in_frame(self.female_points_world, self.male_init_point, self.male_init_orient)

        ee_start = np.concatenate([self.male_init_point, R.from_matrix(self.male_init_orient).as_quat(), self.female_points_world.flatten()])
        print("Saving orientation and position of end effector and female world points before training!")
        np.savetxt(f"./runs/run_{self.params["name"]}/start_pos_orient.txt", ee_start)

    #This initializes the end effector starting points in the simulation during evaluation
    def on_init_given_eval_pos(self, meta_data):
        self.male_init_point = meta_data[:3]
        self.male_init_orient = R.from_quat(meta_data[3:7]).as_matrix()
        self.female_points_world = meta_data[7:].reshape(3,4)
        self.female_points_frame = realize_points_in_frame(self.female_points_world, self.male_init_point, self.male_init_orient)

    # This function steps the robot towards the target positions and orientations
    def move_to_targets(self, data, iterations =1): 
        for _ in range(iterations):
            joint_torques = self.get_joint_torques(data)
            data.ctrl[:] = joint_torques
            mujoco.mj_step(self.mj_model, data)

        return data

    #This function will get the joint_torques needed to move to the target position, sends to zmq, and gets 
    #the joint torques back
    def get_joint_torques(self, data) -> np.ndarray:#clean
        stacked_data = np.concatenate([self.target_pos, self.target_orient])

        qpos = data.qpos.copy()      
        qvel = data.qvel.copy() 

        stacked_data = np.concatenate([stacked_data, qpos, qvel])
        joint_torques = send_and_recieve_joint_torque_server(stacked_data, self.jt_socket)

        return joint_torques
    
    #This function takes the reward type from params, and calls the reward module on the specified reward type
    # the reward module has the ability to update the telemetry, take in metadata from the environment, and take in 
    # the params themselves which specify how the reward function will behave    
    def compute_reward(self, data):

        reward_type = self.params["reward"]
        metadata = {"init_orient": self.male_init_orient, "init_point": self.male_init_point, "female_points_world": self.female_points_world}

        return self.reward_module.call(self.observer, data, self.params, metadata, self.telemetry, reward_type)
    
    # This function uses many helper functions to come up with an observation from the environment
    # Later down the road, we will make an observation module as well for multiple different types of observations
    def sample_obs(self, data):
        force_frame, torque_frame = get_force_torque_sensor_reading(self.observer, data, self.male_init_orient)
        male_pos_frame, male_orient_frame = get_male_pos_orient(self.observer, data, self.male_init_point, self.male_init_orient)
        male_orient_frame_quat = R.from_matrix(male_orient_frame).as_quat()
        flattened_point_cloud = self.female_points_frame.flatten()
        obs = np.concatenate([force_frame, torque_frame, male_pos_frame, male_orient_frame_quat, flattened_point_cloud])

        return obs
    
    # This is a simple helper function that allows us to update the no_contact_count, or the running sum of the amount of actions where
    # the robot is not in contact with the female connector
    def update_no_contact_count(self, data):
        force , _ = get_force_torque_sensor_reading(self.observer, data)

        if np.linalg.norm(force) < 0.01:
            self.telemetry["no_contact_count"] += 1
        else:
            self.telemetry["no_contact_count"] = 0

    # This function gets the observation
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
    
    # This function resets the fspf data
    def reset_fspf_data(self):
        self.motion_or_force_axis = np.zeros(3)
        self.force_dim = 0

        data = np.ones(1)
        packed_data = struct.pack('f' * data.shape[-1], *data)

        self.fspf_socket.send(packed_data)
        reply = self.fspf_socket.recv()

    # def update_fspf_server(self, data, desired_dx):

    #     motion_control = (FILTER_FREQ/CONTROL_FREQ) * self.sigmaMotion @ desired_dx #correct
    #     force_control = (FILTER_FREQ/CONTROL_FREQ) * self.sigmaForce @ desired_dx #correct 

    #     measured_force, _ = self.get_force_torque_sensor_reading(data, "world")
    #     measured_vel = self.get_velocity_reading(data, "world")

    #     stacked_data = np.concatenate([motion_control, force_control, measured_force, measured_vel])
    #     packed_data = struct.pack('f' * stacked_data.shape[-1], *stacked_data)

    #     self.fspf_socket.send(packed_data)
    #     reply = self.fspf_socket.recv()
    #     motion_or_force_axis_and_fdim  = np.frombuffer(reply, dtype = np.float32) 

    #     motion_or_force_axis = motion_or_force_axis_and_fdim[:3]
    #     fdim = motion_or_force_axis_and_fdim[3]

    #     #selection matrix inference logic
    #     if fdim == 0:
    #         self.sigmaForce = np.zeros((3,3))
    #         self.sigmaMotion = np.identity(3)
    #     elif fdim == 1:
    #         self.sigmaForce = motion_or_force_axis @ motion_or_force_axis.T
    #         self.sigmaMotion = np.identity(3) - self.sigmaForce
    #     elif fdim == 2:
    #         self.sigmaMotion = motion_or_force_axis @ motion_or_force_axis.T
    #         self.sigmaForce = np.identity(3) - self.sigmaForce
    #     elif fdim == 3:
    #         self.sigmaForce = np.identity(3)
    #         self.sigmaMotion = np.zeros((3,3))

    #     return motion_or_force_axis, fdim
    
    # def get_joint_torques_mft_server(self, des_pos, des_orient, qpos, qvel, force_or_motion_axis, fdim,  desired_force):#clean

    #     stacked_data = np.concatenate([des_pos, des_orient, qpos, qvel, force_or_motion_axis, fdim, desired_force])
    #     packed_data = struct.pack('f' * stacked_data.shape[-1], *stacked_data)

    #     self.mft_socket.send(packed_data)
    #     reply = self.mft_socket.recv()
    #     joint_torques = np.frombuffer(reply, dtype = np.float32) 

    #     joint_torques = joint_torques.reshape(-1, ROBOT_JOINTS) 
    #     return joint_torques
    
    # def move_to_targets_mft(self, data):#clean

    #     des_pos = np.array([0.2,0.2,0.2])
    #     des_orient = np.array([0,1,0,0])

    #     qpos = data.qpos.copy()
    #     qvel = data.qvel.copy() 
    #     force_or_motion_axis = np.array([0,0,0])
    #     fdim = np.array([0])
    #     desired_force = np.array([0, 0 ,0])

    #     joint_torques = self.get_joint_torques_mft_server(des_pos, des_orient, qpos, qvel, force_or_motion_axis, fdim,  desired_force)

    #     data.ctrl[:] = joint_torques
    #     mujoco.mj_step(self.mj_model, data)

    #     return data

    #The step function has the following functionality: if an action is provided, it will apply the action and step num_physics_steps
    # times towards the new target positions, if no action is provided, it will simply step the physics simulation with the current
    # parameters
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
    
    # this function takes the action that is outputted by the RL agent, and comes up with a continuous action for the robot
    # to perform
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
        male_pos_world, _ = get_male_pos_orient(self.observer, data)
        _ , male_orient_frame = get_male_pos_orient(self.observer, data, self.male_init_point, self.male_init_orient, quat = True)

        male_pos_world_prime = male_pos_world + dx_world
        male_orient_prime = self.male_init_orient @ R_dtheta_frame @ male_orient_frame

        # #find the orientation angle distance between the target orientation and frame as angle_rad
        angle_rad = compute_theta_distance(male_orient_prime, self.male_init_orient)

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

    #The function below is a utility function
    def set_target_pos_orient(self, pos , orient):
        self.target_pos = pos
        self.target_orient = orient

    def set_mode(self, mode):
        if mode == "eval":
            self.mode = "eval"
            return

        self.mode = "train"

#Helper functions related to the environment stepping: 
def send_and_recieve_joint_torque_server(stacked_data: np.ndarray, jt_socket) -> np.ndarray:#clean
    packed_data = struct.pack('f' * stacked_data.shape[-1], *stacked_data)

    jt_socket.send(packed_data)
    reply = jt_socket.recv()
    joint_torques = np.frombuffer(reply, dtype = np.float32) 

    joint_torques = joint_torques.reshape(-1, ROBOT_JOINTS) 
    return joint_torques






    
