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

from utils import compute_geodesic_distance, RewardsModuleV2, get_force_data, get_velocity_data, fetch_body_id, fetch_body_pos_orient
from utils import extract_pos_orient_keypoints, generate_random_sphere_point, switch_frame_key_points, params_from_yaml
from utils import send_and_recieve_to_server, get_joint_torques, get_mft_torques, realize_points_in_frame, check_out_of_bounds, TaskModule

ROBOT_JOINTS = 7

class ContactRichEnv:

    def __init__(self ,servers, model, data, params, tool_start_info = None):

        self.step_count = 0
        self.model = model
        self.init_position_robot(data)

        self.jt_server = servers[0]
        self.mft_server= servers[1]
        self.fspf_server = servers[2]

        self.reward_module = RewardsModuleV2()

        self.initialize_parameters(params)

        if tool_start_info is not None:
            self.load_tool_start_info(tool_start_info)
        else:
            self.create_noisy_task_points(data)
            self.create_start_frame(data)
            self.save_tool_start_info()

        self.initialize_quick_reset_positions(data)
        self.find_success_thresh(data)
        self.show_zones(data)
        self.count_observation_size(data)

    def init_position_robot(self, data):
        home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        home_qpos = self.model.key_qpos[home_key_id]

        data.qpos[:] = home_qpos
        data.qvel[:] = np.zeros(7)

        print(f"set the robot's joint positions to: {home_qpos}")

        mujoco.mj_step(self.model, data)

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

        task_module = TaskModule()
        self.task_base_points, self.tool_base_points = task_module.get_task_tool_points(params["task"])

    def find_success_thresh(self, data, iterations = 500):
        self.reset(data)
        self.success_thresh = 0

        #Move in the nominal direction until the force spikes
        z_axis = self.tool_frame_orient[:, 2]
        target_pos = self.tool_frame_pos + 0.3 * z_axis

        for _ in range(iterations):
            self.move_to_targets(data, target_pos, self.tool_frame_orient, iterations = 1)

            force, _ = get_force_data(self.model, data)

            tool_pos, _ = extract_pos_orient_keypoints(self.get_tool_points(data))
            print(f"tool position is : {tool_pos}, {force}")

            if np.linalg.norm(force) > 3:
                

                dist = self.reward_module.select_reward("dist", self.get_tool_points(data), self.get_noisy_task_points(data), {})

                self.success_thresh = 0.75 * dist
                self.dist_scale = 1.5 * dist
                print(f"Set the success threshold!!! {dist} {self.success_thresh} , set dist scale: {self.dist_scale}")
                

                break

        self.reset(data)
            
    def reset(self , data):
        data.qpos[:] = self.quick_reset_positions
        data.qvel[:] = np.zeros(ROBOT_JOINTS)
        mujoco.mj_step(self.model, data)
        self.move_to_targets(data, self.tool_frame_pos, self.tool_frame_orient, iterations = 250)

        self.step_count = 0
        self.out_of_bounds = False
        self.curr_dist = 1
        self.success = False
        self.reward_sum = 0

        self.reset_fspf_data()
        self.reset_obs_list(data)
        #set the force data and the

        return data, self.get_full_observation()

    def reset_obs_list(self, data):
        obs = self.sample_observation(data)
        self.obs_list = [None] * self.obs_stack

        for i in range(self.obs_stack):
            self.obs_list[i] = obs.copy()

    def create_noisy_task_points(self, data):

        task_id = fetch_body_id(self.model, data, "task", obj_type = "body")
        t_p, t_o = fetch_body_pos_orient(data, task_id)

        task_points = t_o @ self.task_base_points + t_p.reshape(3,1)

        task_pos, _ = extract_pos_orient_keypoints(task_points)

        print(f"task pos: {task_pos}")

        random_euler_angle = self.orient_noise * generate_random_sphere_point()
        rand_rotation = R.from_euler("xyz", random_euler_angle).as_matrix()

        task_points_noise = rand_rotation @ (task_points - task_pos.reshape(3,1)) + task_pos.reshape(3,1)

        rand_trans_noise = self.trans_noise * generate_random_sphere_point()
        task_points_noise = rand_trans_noise.reshape(3,1) + task_points_noise

        self.task_points_noise = task_points_noise
        print(f"created noisy task points: {self.task_points_noise}")

    def create_start_frame(self, data):
        task_pos, task_orient = extract_pos_orient_keypoints(self.task_points_noise)
        z_axis = task_orient[: , 2]

        self.tool_frame_pos = task_pos - self.z_space * z_axis.reshape(3)
        self.tool_frame_orient = task_orient

        print(f"created tool_frame_pos: {self.tool_frame_pos}, created tool frame orient: {self.tool_frame_orient}")

    def load_tool_start_info(self, tool_start_info):
        self.tool_frame_pos = tool_start_info[:3]
        self.tool_frame_orient = R.from_quat(tool_start_info[3:7]).as_matrix()
        self.task_points_noise = tool_start_info[7:].reshape(3,4)

    def save_tool_start_info(self):
        tool_start_info = np.concatenate([self.tool_frame_pos, R.from_matrix(self.tool_frame_orient).as_quat(), self.task_points_noise.flatten()])
        print("Saving tool start info!")
        np.savetxt(f"./runs/run_{self.run_name}/tool_start_info.txt", tool_start_info)

    def initialize_quick_reset_positions(self, data):
        print(f"tool_frame_pos: {self.tool_frame_pos}, tool_frame_orient: {self.tool_frame_orient}".format())
        self.move_to_targets(data, self.tool_frame_pos, self.tool_frame_orient, iterations = 20000)
        tool_pos, _ = self.get_tool_pos_orient(data)
        print(f"robot position after reset is: {tool_pos}")
        self.quick_reset_positions = data.qpos.copy()

    def move_to_targets(self, data, target_pos, target_orient, iterations = 500):

        for _ in range(iterations):
            joint_torques = get_joint_torques(data, target_pos , target_orient, self.jt_server)
            data.ctrl[:] = joint_torques
            mujoco.mj_step(self.model, data)

        return data
 
    def get_tool_points(self, data):
        #get the orientation of the body named tool
        tool_id = fetch_body_id(self.model, data, "tool", obj_type = "body")
        tool_pos, tool_orient = fetch_body_pos_orient(data, tool_id)

        tool_points_world = tool_orient @ self.tool_base_points + tool_pos.reshape(3,1)

        return tool_points_world
    
    def get_tool_pos_orient(self, data):
        tool_points_world = self.get_tool_points(data)
        tool_pos, tool_orient = extract_pos_orient_keypoints(tool_points_world)

        return tool_pos, tool_orient

    def get_task_points(self, data):
        return self.task_base_points
    
    def get_noisy_task_points(self, data):
        return self.task_points_noise

    def get_reward(self, data):
        metadata = {"dist_weight": self.dist_weight, "orient_weight": self.orient_weight, "dist_scale": self.dist_scale, "orient_scale": self.orient_scale, "alpha": self.alpha, "beta": self.beta, "success_thresh": self.success_thresh, "out_of_bounds": self.out_of_bounds}
        reward = self.reward_module.select_reward(self.reward_type, self.get_tool_points(data), self.task_points_noise, metadata)
        curr_dist = self.reward_module.select_reward("dist", self.get_tool_points(data), self.task_points_noise, {})

        self.curr_dist = curr_dist
        self.min_dist = min(self.min_dist, curr_dist)

        if self.mode == "eval":
            if self.step_count % self.num_phys_steps == 0:
                self.reward_sum += reward
        
        if self.mode == "train":
            self.reward_sum += reward

        return reward

    def check_dones(self, data):

        done = 0

        if self.out_of_bounds:
            return 1
        
        if self.step_count > self.num_phys_steps * self.actions_per_episode:
            return 1

        if self.curr_dist < self.success_thresh:
            self.success = True
            return 1

        return 0

    #The sample observation will be based on an engine where we can decipher what types of observations will be there
    def sample_observation(self, data):

        obs_list = []

        for obs_command in self.obs_key:
            if obs_command == "task_points":
                points = realize_points_in_frame(self.get_noisy_task_points(data), self.tool_frame_pos, self.tool_frame_orient)
                obs_list.append(points.flatten())
            elif obs_command == "tool":
                tool_pos, tool_orient = self.get_tool_pos_orient(data)
                tool_orient = self.tool_frame_orient.T @ tool_orient
                tool_orient_frame_quat = R.from_matrix(tool_orient).as_quat()
                tool_pos_frame = realize_points_in_frame(tool_pos.reshape(3,1) , self.tool_frame_pos, self.tool_frame_orient)
                obs_list.append(tool_pos_frame.flatten())
                obs_list.append(tool_orient_frame_quat.flatten())
            elif obs_command == "force":
                force_frame, torque_frame = get_force_data(self.model, data,  self.tool_frame_orient)
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

        if self.mode == "train":
            self.add_observation_to_stack(obs)
        elif self.mode == "eval" and self.step_count % self.num_phys_steps * self.exec_actions:
            self.add_observation_to_stack(obs)
    
    def add_observation_to_stack(self, obs):

        if len(self.obs_list) >= self.obs_stack:
            self.obs_list.pop(0)

        self.obs_list.append(obs)
    
    def _get_obs(self, data):

        dones = np.array([self.check_dones(data)])

        reward = self.get_reward(data)
        rewards = np.array([reward])

        new_obs = self.sample_observation(data)
        self.call_add_observation_to_stack(new_obs)

        obs = self.get_full_observation()

        obs = np.array([obs])
        obs = obs.reshape(1,-1)

        info = {
            "reward": rewards,
            "terminated": dones,
            "TimeLimit.truncated": np.zeros_like(dones)
        }

        return obs, rewards, dones, info

    def reset_fspf_data(self):
        self.motion_or_force_axis = np.zeros(3)
        self.force_dim = 0
        self.desired_force_magnitude = 0
        self.dx_world = np.zeros(3)
        self.sigmaMotion = np.eye(3)
        self.sigmaForce = np.zeros((3,3))

        data = np.ones(1)
        packed_data = struct.pack('f' * data.shape[-1], *data)

        self.fspf_server.send(packed_data)
        reply = self.fspf_server.recv()

    def show_zones(self, data):
        shape = self.zone["shape"]

        if shape == "cyl":
            cyl_id = fetch_body_id(self.model, data, "zone_cyl", obj_type="geom")
            # update geom parameters
            self.model.geom_size[cyl_id, 0] = self.zone["radius"]
            self.model.geom_size[cyl_id, 1] = self.zone["height"]/2  # half-length
            self.model.geom_rgba[cyl_id, 3] = 0.1
        elif shape == "sphere":

            sphere_id = fetch_body_id(self.model, data, "zone_sphere", obj_type="geom")
            self.model.geom_size[sphere_id, 0] = self.zone["radius"]
            self.model.geom_rgba[sphere_id, 3] = 0.1

    def show_key_points(self, data): 
        tool_points = self.get_tool_points(data)
        
        for i in range(4):
            tool_point_id = fetch_body_id(self.model, data, f"tool_keypoint_{i+1}", "geom")
            task_point_id = fetch_body_id(self.model, data, f"task_keypoint_{i+1}", "geom")
            data.geom_xpos[tool_point_id] = tool_points[:, i].reshape(3,)
            data.geom_xpos[task_point_id] = self.task_points_noise[:,i].reshape(3,)

        control_point_id = fetch_body_id(self.model, data, "control_point", "geom")
        tool_pos, _ = self.get_tool_pos_orient(data)
        data.geom_xpos[control_point_id] = tool_pos.reshape(3,)

        shape = self.zone["shape"]

        if shape == "cyl":
            cyl_id = fetch_body_id(self.model, data, "zone_cyl", obj_type="geom")
            data.geom_xpos[cyl_id] = self.tool_frame_orient @ np.array(self.zone["center"]) + self.tool_frame_pos
            data.geom_xmat[cyl_id] = self.tool_frame_orient.flatten()
        elif shape == "sphere":
            sphere_id = fetch_body_id(self.model, data, "zone_sphere", obj_type="geom")
            data.geom_xpos[sphere_id] = self.tool_frame_orient @ np.array(self.zone["center"]) + self.tool_frame_pos



        return data

    def update_fspf_data(self,  data):#This is clean, verified from visual_tests.py

        dx = self.dx_world.reshape((3,1))

        motion_control = (50/1000) * self.sigmaMotion @ dx #correct
        force_control = (50/1000) * self.sigmaForce @ dx #correct 

        motion_control = motion_control.reshape(3,)
        force_control = force_control.reshape(3,)

        measured_force, _ = get_force_data(self.model, data)
        measured_force = measured_force * -1
        measured_vel = get_velocity_data(self.model, data)

        target_data = np.hstack((motion_control, force_control, measured_vel, measured_force))
        target_data = np.tile(target_data, (1,))
        target_data_packed = struct.pack('f' * 12 * 1, *(target_data))
        self.fspf_server.send(target_data_packed)
        reply = self.fspf_server.recv()

        motion_or_force_axis_and_fdim  = np.frombuffer(reply, dtype = np.float32)

        motion_or_force_axis = motion_or_force_axis_and_fdim[1:4]
        fdim = motion_or_force_axis_and_fdim[0]

        motion_or_force_axis = motion_or_force_axis.reshape(3,1)

        #selection matrix inference logic
        if fdim == 0.0:
            self.sigmaForce = np.zeros((3,3))
            self.sigmaMotion = np.eye(3)
        elif fdim == 1.0:
            self.sigmaForce = motion_or_force_axis @ motion_or_force_axis.T
            self.sigmaMotion = np.eye(3) - self.sigmaForce
        elif fdim == 2.0:
            self.sigmaMotion = motion_or_force_axis @ motion_or_force_axis.T
            self.sigmaForce = np.eye(3) - self.sigmaForce
        elif fdim == 3.0:
            self.sigmaForce = np.eye(3)
            self.sigmaMotion = np.zeros((3,3))

        self.motion_or_force_axis = motion_or_force_axis
        self.force_dim = fdim

        return motion_or_force_axis, fdim
    
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

    def step(self, data, horizon_action = None):

        horizon_iterations = 1 if self.mode == "eval" else self.exec_actions

        for i in range(horizon_iterations):

            if horizon_action is not None:
                horizon_action = horizon_action.reshape(-1)
                self.apply_action(data, horizon_action[7*i:7*(i+1)])

            iterations = 1 if self.mode == "eval" else self.num_phys_steps

            for _ in range(iterations):
                if self.step_count % 50 == 0:
                    self.update_fspf_data(data)

                qpos = data.qpos.copy()
                qvel = data.qvel.copy()

                joint_torques = get_mft_torques(self.target_pos, self.target_orient, qpos, qvel, self.motion_or_force_axis, self.force_dim, self.desired_force_magnitude, self.dx_world, self.sigmaForce, self.mft_server)

                data.ctrl[:] = joint_torques
                mujoco.mj_step(self.model, data)
                self.step_count += 1

        obs, reward, dones, info = self._get_obs(data)

        if dones[0] == 1:
            print("episode has terminated!")
            self.print_telemetry()
            self.reset(data)

        return data, (obs, reward, dones, info)
    
    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"

    def print_telemetry(self):
        
        print(f"self out of bounds: {self.out_of_bounds}")
        print(f"self.step_count: {self.step_count} self.num_phys_steps: {self.num_phys_steps} greater: {self.step_count > self.num_phys_steps* self.actions_per_episode}")
        print(f"self.curr_dist: {self.curr_dist}, self.success_thresh: {self.success_thresh}, success indication: {self.curr_dist < self.success_thresh}")
        print(f"self.reward_sum: {self.reward_sum}, self.min_dist: {self.min_dist}")


    



    

