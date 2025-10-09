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
import torch
import yaml
from enum import Enum
import redis
import json

from utils import compute_geodesic_distance, RewardsModuleV2, get_force_data, get_velocity_data, fetch_body_id, fetch_body_pos_orient
from utils import extract_pos_orient_keypoints, generate_random_sphere_point, switch_frame_key_points, params_from_yaml
from utils import send_and_recieve_to_server, get_joint_torques, get_mft_torques, realize_points_in_frame, check_out_of_bounds

redis_client = redis.Redis()

class RedisKeys(Enum):

    ROBOT_QPOS = "robotqpos"
    ROBOT_QVEL = "robotqvel"
    FORCE_MOTION_AXIS = "forcemotionaxis"
    FORCE_DIM = "forcedim"
    DESIRED_FORCE = "desiredforce"
    DESIRED_CARTESIAN_POSITION = "descartpos"
    DESIRED_CARTESIAN_ORIENTATION = "descartorient"
    CONTROL_POINT_POSITION = "controlpointposition"
    CONTROL_POINT_ORIENTATION = "controlpointorientation"
    MEASURED_FORCE = "measuredforce"
    MEASURED_TORQUE = "measuredtorque"

    

#This guy only has eval mode
class RealRobotEnv():

    def __init__(self,servers, model, data, task_base_points, tool_base_points, params, tool_start_info):

        self.step_count = 0
        self.model = model

        self.fspf_server = servers[0]

        self.task_base_points = task_base_points
        self.tool_base_points = tool_base_points

        self.reward_module = RewardsModuleV2()

        self.initialize_parameters(params)
        self.load_tool_start_info(tool_start_info)
        

        self.count_observation_size(data)

    def initialize_parameters(self, params):

        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.dist_weight = params["dist_weight"]
        self.orient_weight = params["orient_weight"]
        self.num_phys_steps = 1000//params["hz"]
        self.actions_per_episode = params["actions_per_episode"]

        self.trans_noise = params["trans_noise"]
        self.orient_noise = params["orient_noise"]

        self.z_space = params["z_space"]
        self.max_force_thresh = params["max_force_thresh"]
        self.max_rotation_thresh = params["max_rotation_thresh"]
        self.max_displacement_thresh = params["max_displacement_thresh"]
        self.success_reward = params["success_reward"]
        
        self.obs_key = params["obs_key"]
        self.reward_type = params["reward"]

        self.zone = params["zone"] #all the information encoded by the zone is inside params["zone"]
        self.out_of_bounds_penalty = params["out_of_bounds_penalty"]
        self.orient_scale = params["orient_scale"]
        self.dist_scale = 1

        self.action_horizon = params["action_horizon"]
        self.exec_actions = params["exec_actions"]
        self.obs_stack = params["obs_stack"]
        self.debug = params["debug"]
        self.nominal = params["nominal"]

        self.run_name = params["name"]
        self.min_dist = 1

    def load_tool_start_info(self, tool_start_info):
        self.tool_frame_pos = tool_start_info[:3]
        self.tool_frame_orient = R.from_quat(tool_start_info[3:7]).as_matrix()
        self.task_points_noise = tool_start_info[7:].reshape(3,4)

    def set_mft_redis_keys(self):

        #This function takes all of the internal state of the environment and sets the redis keys to reflect that

        redis_client.set(RedisKeys.DESIRED_CARTESIAN_POSITION.value, json.dumps(self.target_pos.tolist()))
        redis_client.set(RedisKeys.DESIRED_CARTESIAN_ORIENTATION.value, json.dumps(self.target_orient.tolist()))
        redis_client.set(RedisKeys.FORCE_MOTION_AXIS.value, json.dumps(self.motion_or_force_axis.tolist()))
        redis_client.set(RedisKeys.FORCE_DIM.value, self.force_dim)
        redis_client.set(RedisKeys.DESIRED_FORCE.value, json.dumps(self.desired_force.tolist()))



    #This function will make the real robot probe to see what the success distance is, such that it knows when to stop the policy
    def find_success_thresh(self, iterations = 500):
        #This is needed to be implemented, this is a real robot function, where the robot goes down until it feels a force that is high

        self.reset()

        z_axis = self.tool_frame_orient[:, 2]
        target_pos = self.tool_frame_pos + 0.3 * z_axis
        self.target_pos = target_pos

        self.set_mft_redis_keys()
        measured_force = np.array(json.loads(redis_client.get(RedisKeys.MEASURED_FORCE.value)))

        while np.linalg.norm(measured_force) < 3:
            time.sleep(0.01)

        task_points_pos, _ = extract_pos_orient_keypoints(self.task_points_noise)
        control_point_pos = np.array(json.loads(redis_client.get(RedisKeys.CONTROL_POINT_POSITION.value)))

        self.success_thresh = np.linalg.norm(task_points_pos - control_point_pos)
        self.dist_scale = 1.5 * self.success_thresh

    #This function will tell the real robot to reset to it's desired position
    def reset(self, iterations = 500):

        # find a good universal position to set the robot too with the motion force task

        self.motion_or_force_axis = np.zeros(3)
        self.force_dim = 0
        self.desired_force_magnitude = 0
        self.dx_world = np.zeros(3)
        self.sigmaMotion = np.eye(3)
        self.sigmaForce = np.zeros((3,3))

        self.target_pos = self.tool_frame_pos
        self.target_orient = self.tool_frame_orient

        self.motion_or_force_axis = np.zeros(3)
        self.force_dim = 0
        self.desired_force = np.zeros(3)

        self.set_mft_redis_keys()

        for _ in range(iterations):
            time.sleep(0.01)

    def reset_obs_list(self, data):
        obs = self.sample_observation(data)
        self.obs_list = [None] * self.obs_stack

        for i in range(self.obs_stack):
            self.obs_list[i] = obs.copy()

    def get_tool_points(self):

        tool_pos = np.array(json.loads(redis_client.get(RedisKeys.CONTROL_POINT_POSITION.value)))
        tool_orient = np.array(json.loads(redis_client.get(RedisKeys.CONTROL_POINT_ORIENTATION.value)))

        tool_points = tool_orient @ self.tool_base_points + tool_pos.reshape(3,1)

        return tool_points

    def get_tool_pos_orient(self):
        
        tool_pos = np.array(json.loads(redis_client.get(RedisKeys.CONTROL_POINT_POSITION.value)))
        tool_orient = np.array(json.loads(redis_client.get(RedisKeys.CONTROL_POINT_ORIENTATION.value)))

        return tool_pos, tool_orient

    def get_task_points(self):
        return self.task_points_noise

    def get_force_data(self):
        #get the real force data from the robot, and we should be good

        force_data = np.array(json.loads(redis_client.get(RedisKeys.MEASURED_FORCE.value)))
        torque_data = np.array(json.loads(redis_client.get(RedisKeys.MEASURED_TORQUE.value)))


        return force_data, torque_data

    def sample_observation(self, data):

        obs_list = []

        for obs_command in self.obs_key:
            if obs_command == "task_points":
                points = realize_points_in_frame(self.get_task_points(), self.tool_frame_pos, self.tool_frame_orient)
                obs_list.append(points.flatten())
            elif obs_command == "tool":
                tool_pos, tool_orient = self.get_tool_pos_orient(data)
                tool_orient = self.tool_frame_orient.T @ tool_orient
                tool_orient_frame_quat = R.from_matrix(tool_orient).as_quat()
                tool_pos_frame = realize_points_in_frame(tool_pos.reshape(3,1) , self.tool_frame_pos, self.tool_frame_orient)
                obs_list.append(tool_pos_frame.flatten())
                obs_list.append(tool_orient_frame_quat.flatten())
            elif obs_command == "force":
                force_frame, torque_frame = self.get_force_data(self.tool_frame_orient)
                obs_list.append(force_frame.flatten())
                obs_list.append(torque_frame.flatten())

        obs = np.concatenate(obs_list)
        obs = obs.reshape(1, -1)

        return obs
    
    def count_observation_size(self, data):
        
        obs_size = 0

        for obs_command in self.obs_key:
            if obs_command == "task_points":
                obs_size += 12
            elif obs_command == "tool":
                obs_size += 7
            elif obs_command == "force":
                obs_size += 6

        self.obs_size = obs_size

    def get_full_observation(self):
        full_obs = np.hstack(self.obs_list)
        full_obs = full_obs.flatten()

        return full_obs
    
    def call_add_observation_to_stack(self, obs):

        if self.step_count % self.num_phys_steps * self.exec_actions:
            self.add_observation_to_stack(obs)

    def add_observation_to_stack(self, obs):

        if len(self.obs_list) >= self.obs_stack:
            self.obs_list.pop(0)

        self.obs_list.append(obs)

    def get_current_distance(self):
        # returns the distance between the task points and the tool points.

        curr_dist = self.reward_module.select_reward("dist", self.get_tool_points(), self.task_points_noise, {})

        return curr_dist

    def _get_obs(self, data):

        dones = np.array([self.check_dones(data)])

        new_obs = self.sample_observation(data)
        self.call_add_observation_to_stack(new_obs)

        obs = self.get_full_observation()

        obs = np.array([obs])
        obs = obs.reshape(1,-1)

        return obs, dones
    
    def check_dones(self):

        #Check whether the step count is past a certain amount, or the robot escapes the boundary zone it was trained on

        pass
    
    def apply_action(self, data, action):

        dx_tool = action[:3]
        dtheta_tool = action[3:6]
        magnitude_force = action[6]

        if self.nominal:
            dx_tool[2] = 1

        #scaling the dx world, dtheta_frame, and magnitude force created by the actor network
        if np.linalg.norm(dx_tool) > self.max_displacement_thresh:
            dx_tool = self.max_displacement_thresh * dx_tool/np.linalg.norm(dx_tool)

        if np.linalg.norm(dtheta_tool) > self.max_rotation_thresh:
            dtheta_tool = self.max_rotation_thresh * dtheta_tool/np.linalg.norm(dtheta_tool)

        if np.linalg.norm(magnitude_force) > self.max_force_thresh:
            magnitude_force = self.max_force_thresh * magnitude_force/np.linalg.norm(magnitude_force)

        R_dtheta_tool = R.from_euler("xyz", dtheta_tool).as_matrix()

        dx_world = self.tool_frame_orient @ dx_tool
        #we want the change in orientation to be wrst the frame, so we need to f

        tool_pos, orient_tool = self.get_tool_pos_orient(data)
        tool_pos_prime = tool_pos + dx_world
        orient_tool = self.tool_frame_orient.T @ orient_tool
        orient_tool = R_dtheta_tool @ orient_tool
        orient_tool_prime = self.tool_frame_orient @ orient_tool

        tool_pos_in_frame = self.tool_frame_orient.T @ (tool_pos - self.tool_frame_pos)
        theta_dist = compute_geodesic_distance(self.tool_frame_orient, orient_tool_prime)

        #if the distance between the target orientation or target position and the frame is too much, send out_of_bounds to true
        if check_out_of_bounds(tool_pos_in_frame, self.zone):
            print("distance violation!")
            self.out_of_bounds = True
        elif theta_dist > self.orient_scale:
            print("angle violation!")
            self.out_of_bounds = True

        self.target_pos = tool_pos_prime
        self.target_orient = R.from_matrix(orient_tool_prime).as_quat()
        self.desired_force_magnitude = magnitude_force
        self.dx_world = dx_world

    def step(self, data):
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        joint_torques = get_mft_torques(self.target_pos, self.target_orient, qpos, qvel, self.motion_or_force_axis, self.force_dim, self.desired_force_magnitude, self.dx_world, self.sigmaForce, self.mft_server)

        #just publish all these keys to redis, and have the c++ files consume them, which is great!

        data.ctrl[:] = joint_torques
        mujoco.mj_step(self.model, data)
        self.step_count += 1

        obs, reward, dones, info = self._get_obs(data)

        if dones[0] == 1:
            print("episode has terminated!")
            self.print_telemetry()
            self.reset(data)

        return data, (obs, reward, dones, info)
    
    def print_telemetry(self):
        #Print whatever telemetry you want t
        pass



    