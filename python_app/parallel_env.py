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
import jax
from mujoco import mjx
import jax.numpy as jnp
import jax.tree_util as tree_util
from jax.scipy.spatial.transform import Rotation as Rj

from utils import send_and_recieve_to_server, TaskModule, params_from_yaml
from jax_utils import extract_pos_orient_keypoints, generate_random_sphere_point_jax, dist, realize_points_in_frame
from jax_utils import compute_geodesic_distance_batch, check_out_of_bounds_batch, RewardModuleJax, create_random_rotations
from jax_utils import compute_sigma_masked

ROBOT_JOINTS = 7
NUM_ENVS = 2
mj_model = mujoco.MjModel.from_xml_path("../models/scenes/fr3squarehole.xml")
mj_data = mujoco.MjData(mj_model)
tool_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "tool")
task_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "task")
link7_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "link7")
home_key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
force_sid = mj_model.sensor("force_sensor").id          # sensor id
force_adr = mj_model.sensor_adr[force_sid]              # start index in sensordata
torque_sid = mj_model.sensor("torque_sensor").id          # sensor id
torque_adr = mj_model.sensor_adr[torque_sid]      
home_qpos = mj_model.key_qpos[home_key_id] 
HOME_QPOS = jnp.array(home_qpos)

#Master key for all random number generation
master_key = jax.random.PRNGKey(42)
key_list = jax.random.split(master_key, num = 10)

#This is finalized for the mft server - you cant jax jit this function since it requires sending values to a zmq server
def get_mft_torques(targets_and_values: jnp.array, mft_server) -> jnp.array:

    targets_and_values_cpu = np.array(targets_and_values) 
    stacked_data = targets_and_values_cpu.flatten()
    joint_torques = send_and_recieve_to_server(stacked_data, mft_server)

    joint_torques = joint_torques.reshape(NUM_ENVS, ROBOT_JOINTS)

    joint_torques = jnp.array(joint_torques)
    return joint_torques

def step_mjx_env(mjx_model, data: mjx.Data, joint_torques: jnp.array):
    data = data.replace(ctrl=joint_torques)
    new_data = mjx.step(mjx_model, data)

    return new_data

step_mjx_env_batch = jax.jit(jax.vmap(step_mjx_env, in_axes=(None, 0, 0)))

naive_step = jax.jit(jax.vmap(mjx.step, in_axes = (None, 0)))

def get_joint_torque_targets_and_values(data: mjx.Data, target_pos: jnp.array , target_orient:jnp.array):

    qpos = data.qpos
    qvel = data.qvel

    target_orient_quat = Rj.from_matrix(target_orient).as_quat()
    target_orient_quat = target_orient_quat.flatten()

    # print(f"qpos size: {qpos.shape}")

    return jnp.concatenate([target_pos, target_orient_quat, qpos, qvel])

get_joint_torque_targets_and_values_batch =  jax.jit(jax.vmap(get_joint_torque_targets_and_values, in_axes = (0,0,0)))

def get_qpos(data: mjx.Data):
    return data.qpos

get_qpos_batch = jax.jit(jax.vmap(get_qpos))

def quick_reset(data: mjx.Data, qpos: jnp.array, mask):
   
    new_qpos = jnp.where(mask, qpos, data.qpos)
    new_qvel = jnp.where(mask, jnp.zeros(7), data.qvel)

    data = data.replace(qpos=new_qpos, qvel=new_qvel)

    return data

quick_reset_batch = jax.jit(jax.vmap(quick_reset, in_axes = (0,0,0)))

def get_tool_pos_ori(data: mjx.Data):
    R_link7 = data.xmat[link7_id].reshape(3,3)
    return data.xpos[link7_id] + R_link7 @ jnp.array([0.0, 0.0, 0.1765]), R_link7

get_tool_pos_ori_batch = jax.jit(jax.vmap(get_tool_pos_ori))


def get_qvel(data: mjx.Data):
    return data.qvel

get_qvel_batch = jax.jit(jax.vmap(get_qvel))

def get_force(data: mjx.Data):
    return data.sensordata[force_adr: force_adr + 3]

get_force_batch = jax.jit(jax.vmap(get_force, in_axes= (0)))

def get_torque(data: mjx.Data):
    return data.sensordata[torque_adr: torque_adr + 3]

get_torque_batch = jax.jit(jax.vmap(get_torque, in_axes= (0)))

def get_ee_vel(data: mjx.Data):
    return data.cvel[link7_id, 3:6]

get_ee_vel_batch = jax.jit(jax.vmap(get_ee_vel))

def reset_to_home(data: mjx.Data): 
    data = data.replace(qpos = HOME_QPOS, qvel = jnp.zeros(7))

reset_to_home_batch = jax.jit(jax.vmap(reset_to_home, in_axes = (0)))

def masked_update(mask, old_tree, new_tree):
    def update_leaf(old, new):
        m = mask.reshape((mask.shape[0],) + (1,) * (old.ndim - 1))
        return jnp.where(m, new, old)
    return jax.tree.map(update_leaf, old_tree, new_tree)

masked_update_jit = jax.jit(masked_update)

class ParallelEnv:

    def __init__(self ,servers, mjx_model, model, params):
        self.mjx_model = mjx_model
        self.mft_server = servers[0] 
        self.fspf_server = servers[1]

        self.target_pos = jnp.tile(jnp.array([0.4, 0.4, 0.4]), (NUM_ENVS, 1))
        self.target_orient = jnp.tile(jnp.array([[1,0,0], [0, -1, 0], [0,0,-1]]), (NUM_ENVS, 1,1))

        self.quick_reset_positions = jnp.zeros((NUM_ENVS, ROBOT_JOINTS))

        self.model = model

        self.init_params(params)
        self.count_observation_size()
        self.obs_list = jnp.zeros((NUM_ENVS, self.obs_stack * self.obs_size))

        self.force_dim = jnp.zeros(NUM_ENVS)
        self.motion_or_force_axis = jnp.zeros((NUM_ENVS, 3))
        self.desired_force_magnitude = jnp.zeros(NUM_ENVS)
        self.dx_world = jnp.zeros((NUM_ENVS, 3))

        self.step_count = jnp.zeros(NUM_ENVS)

        self.sigmaMotion = jnp.tile(jnp.eye(3), (NUM_ENVS, 1,1))
        self.sigmaForce = jnp.tile(jnp.zeros((3,3)), (NUM_ENVS, 1,1))

        self.success_thresh = jnp.ones((NUM_ENVS))
        self.dist_scale = jnp.ones((NUM_ENVS))

        self.min_dists = jnp.ones((NUM_ENVS))


    def dry_start(self, data): #tested
        data = step_mjx_env_batch(self.mjx_model, data, jnp.zeros((NUM_ENVS, 7)))
        return data

    def init_env_params(self, data): #tested
        
        self.create_noisy_task_points(data)
        self.create_start_frames()
        self.init_quick_reset_pos(data)

        self.find_success_thresh(data, iterations = 500)

        self.reward_module = RewardModuleJax(self.dist_weight, self.orient_weight, 
                                             self.alpha, self.beta, self.dist_scale, self.orient_scale, 
                                             self.success_thresh, self.task_points_noise)
        

    def init_params(self, params): #tested

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
        self.probe = params["probe"]

        task_module = TaskModule()
        self.task_base_points, self.tool_base_points = task_module.get_task_tool_points(params["task"])
        self.task_base_points = jnp.array(self.task_base_points)
        self.tool_base_points = jnp.array(self.tool_base_points)

    #the size of mask should be num_envs -- this is pretty important
    def move_to_targets(self, data,mask, iterations = 2000): #tested

        for _ in range(iterations):
            targets_and_values = get_joint_torque_targets_and_values_batch(data, self.target_pos, self.target_orient)
    
            trivial_mft_vals = jnp.zeros((NUM_ENVS, 7))
            targets_and_values = jnp.concatenate((targets_and_values, trivial_mft_vals), axis = 1)
        
            joint_torques = get_mft_torques(targets_and_values, self.mft_server)

            new_data = step_mjx_env_batch(self.mjx_model, data, joint_torques)

            data = masked_update_jit(mask, data, new_data)

        return data
    
    
    def reset(self, data, mask = jnp.ones((NUM_ENVS,))): #tested

        data = quick_reset_batch(data, self.quick_reset_positions, mask)
        mask3 = mask[:, None]  
        mask33 = mask[:, None, None]
        self.target_pos = jnp.where(mask3, self.tool_frame_pos, self.target_pos)
        self.target_orient = jnp.where(mask33, self.tool_frame_orient, self.target_orient)

        data = self.move_to_targets(data, mask, iterations = 250)

        #Everytime I reset, I need to reset the observation list based on the map, which is sick

        new_observation = self.sample_observation(data)
        # print(f"new observation size: {new_observation.shape}")
        new_observation = np.tile(new_observation, (1,3))
        # print(f"new observation size: {new_observation.shape}")

        self.obs_list = jnp.where(mask3, new_observation, self.obs_list)
        self.step_count = jnp.where(mask, jnp.zeros(NUM_ENVS), self.step_count)

        self.reset_fspf_data(mask)
        #for the reset for the force space particle filter, you just call a repetitive selective reset on each of the axes

        dists = dist(self.get_tool_points(data), self.task_points_noise, debug = False)

        self.min_dists = jnp.where(mask, dists, self.min_dists)

        return data, self.get_full_observation()

    def init_quick_reset_pos(self, data): #tested

        #what do we need to do for this function
        mask = jnp.ones(NUM_ENVS)
        self.target_pos = self.tool_frame_pos.copy()
        self.target_orient = self.tool_frame_orient.copy()

        data = self.move_to_targets(data, mask, iterations = 2000)
        self.quick_reset_positions = get_qpos_batch(data)

        return data

    def reset_fspf_data(self, mask): #tested

        self.force_dim = jnp.where(mask, jnp.zeros(NUM_ENVS), self.force_dim)

        mask3 = mask[:, None]  
        self.motion_or_force_axis = jnp.where(mask3, jnp.zeros((NUM_ENVS, 3)), self.motion_or_force_axis)
        self.desired_force_magnitude = jnp.where(mask3, jnp.zeros(NUM_ENVS), self.desired_force_magnitude)
        self.dx_world = jnp.where(mask3, jnp.zeros((NUM_ENVS, 3)), self.dx_world)

        mask33 = mask[:, None, None]

        self.sigmaMotion = jnp.where(mask33, jnp.tile(jnp.eye(3), (NUM_ENVS, 1,1)), self.sigmaMotion)
        self.sigmaForce = jnp.where(mask33 , jnp.tile(jnp.zeros((3,3)), (NUM_ENVS, 1,1)), self.sigmaForce)

        indices = jnp.where(mask == 1)[0]
        data = jnp.concatenate([jnp.array([9999]), indices]).flatten()

        data = np.asarray(data)

        packed_data = struct.pack('f' * data.shape[-1], *data)

        self.fspf_server.send(packed_data)
        reply = self.fspf_server.recv()

    def update_fspf_data(self, data):

        dx = self.dx_world

        motion_control = (200 / 1000) * jnp.einsum("bij,bj->bi", self.sigmaMotion, dx)
        force_control = (200 / 1000) * jnp.einsum("bij,bj->bi", self.sigmaForce, dx)

        measured_force = get_force_batch(data)
        measured_force *= -1
        measured_vel = get_ee_vel_batch(data)

        target_data = jnp.hstack((motion_control, force_control, measured_vel, measured_force))
        target_data = target_data.flatten()

        target_data = np.asarray(target_data)
        target_data_packed = struct.pack('f' * target_data.shape[-1], *(target_data))
        self.fspf_server.send(target_data_packed)
        reply = self.fspf_server.recv()

        motion_or_force_axis_and_fdim  = np.frombuffer(reply, dtype = np.float32)

        motion_or_force_axis_and_fdim = jnp.asarray(motion_or_force_axis_and_fdim)
        motion_or_force_axis_and_fdim = motion_or_force_axis_and_fdim.reshape(NUM_ENVS, 4)

        motion_or_force_axis = motion_or_force_axis_and_fdim[:, 1:4]
        fdim = motion_or_force_axis_and_fdim[:,0]

        motion_or_force_axis = motion_or_force_axis.reshape((NUM_ENVS, 3))
        fdim = fdim.reshape((NUM_ENVS))

        self.sigmaMotion, self.sigmaForce = compute_sigma_masked(fdim, motion_or_force_axis)

        self.motion_or_force_axis = motion_or_force_axis
        self.force_dim = fdim

        return motion_or_force_axis, fdim

    def create_noisy_task_points(self, data): #tested
        #all the logic for creating noisy task points should be in here

        #lets create a num_envs, 3,3 rotation matrices

        task_points_base = self.task_base_points
        task_points_base = jnp.array(task_points_base)

        task_points_base = jnp.tile(task_points_base, (NUM_ENVS, 1,1))

        task_pos = data.xpos[0][task_id].reshape(3,1)
        task_orient = data.xmat[0][task_id].reshape(3,3)

        print(f"task pos found: {task_pos}, task_orient found: {task_orient}")

        #I have to tile the task_pos and add a little bit of random noise to it
        task_pos = jnp.tile(task_pos, (NUM_ENVS, 1,1))
        task_orient = jnp.tile(task_orient, (NUM_ENVS, 1,1))

        random_noise = jax.random.normal(key_list[0], (NUM_ENVS, 3))
        trans_noise = random_noise / jnp.linalg.norm(random_noise, axis=1, keepdims=True)

        trans_noise = self.trans_noise * trans_noise.reshape((NUM_ENVS, 3, 1))

        rotation_noise = create_random_rotations(key_list[1], NUM_ENVS, self.orient_noise)

        task_orient_prime = rotation_noise @ task_orient
        task_pos_prime = task_pos + trans_noise

        self.task_points_noise = task_orient_prime @ task_points_base + task_pos_prime

    def create_start_frames(self): #tested
        # all the logic for creating noisy task points should be in here

        task_pos, task_orient = extract_pos_orient_keypoints(self.task_points_noise)

        z_axis = task_orient[:, :, 2:3]

        task_pos = task_pos.reshape(NUM_ENVS, 3, 1)

        task_pos = task_pos - self.z_space * z_axis

        task_pos = task_pos.reshape(NUM_ENVS, 3)

        self.tool_frame_pos = task_pos
        self.tool_frame_orient = task_orient

    def save_start_info(self):

        tool_start_info = np.concatenate([self.tool_frame_pos, Rj.from_matrix(self.tool_frame_orient).as_quat(), self.task_points_noise.reshape(NUM_ENVS, 12)], axis = 1)
        print("Saving tool start info!")
        np.savetxt(f"./runs_jax/run_{self.run_name}/tool_start_info.txt", tool_start_info)

    def get_tool_points(self, data): #tested
        tool_pos, tool_ori = get_tool_pos_ori_batch(data)

        return tool_ori @ self.tool_base_points + tool_pos.reshape((NUM_ENVS, 3,1))
    
    def get_tool_pos_orient(self, data): #tested
        tool_points_world = self.get_tool_points(data)
        tool_pos, tool_orient = extract_pos_orient_keypoints(tool_points_world)

        return tool_pos, tool_orient

    #looks like a good algorithm, let us test it out to see if it works
    def find_success_thresh(self, data, iterations = 2000): #tested
        #move the targets down until you feel a force for all of them
        #we can do this one at a time or with a batch, it doesnt matter because it is a one time thing

        mask = jnp.ones((NUM_ENVS))

        data , _ = self.reset(data, mask)
        print("starting dists are: ", dist(self.get_tool_points(data), self.task_points_noise, debug = True))

        #This is just a vector operation
        z_axis = self.tool_frame_orient[:,:,2]
        z_axis = z_axis.reshape((NUM_ENVS, 3))

        #This is the best we can do
        self.target_pos = self.tool_frame_pos + 0.3*z_axis
        self.target_orient = self.tool_frame_orient

        for i in range(iterations):
            completed = jnp.all(mask == 0)

            if completed:
                print("Finding the success threshold has completed!")
                return
            
            data = self.move_to_targets(data, mask, iterations = 1)
            force_data = get_force_batch(data) 

            print(f"force data: {force_data}")

            force_magnitudes = jnp.linalg.norm(force_data, axis = 1)

            force_bools = force_magnitudes < 3.0
            force_bools = force_bools.astype(jnp.int32)

            update_mask = (mask == 1) & force_bools == 0

            dists = dist(self.get_tool_points(data), self.task_points_noise)
            self.success_thresh = jnp.where(update_mask == 1, 0.75 * dists, self.success_thresh)
            self.dist_scale = jnp.where(update_mask == 1, 1.5 * dists, self.dist_scale)

            mask = jnp.logical_and(mask, force_bools)

        return
    
    def _get_obs(self, data): #tested

        rewards = self.get_reward(data)
        dones = self.get_dones()
        

        new_obs = self.sample_observation(data)
        self.call_add_observation_to_stack(new_obs)

        obs = self.get_full_observation()

        info = {
            "reward": rewards,
            "terminated": dones,
            "TimeLimit.truncated": np.zeros_like(dones)
        }

        return obs, rewards, dones, info

    def get_reward(self, data): #tested

        tool_points = self.get_tool_points(data)

        rewards = self.reward_module.batched_reward(tool_points, self.out_of_bounds)
        self.curr_dist = self.reward_module.dist(tool_points)

        return rewards

    def get_dones(self): #tested

        ended = self.step_count > self.num_phys_steps * self.actions_per_episode

        success_reached = self.curr_dist < self.success_thresh

        dones = success_reached | ended | self.out_of_bounds

        return dones

    #The shape of action is (NUM_ENVS, 7)
    def apply_action(self, data, action):

        dx_tool = action[:, 0:3]
        dtheta_tool = action[:, 3:6]
        magnitude_force = action[:, 6]

        if self.nominal:
            dx_tool = dx_tool.at[:, 2].set(1.0)

        dx_norm = jnp.linalg.norm(dx_tool, axis=1, keepdims=True)
        dtheta_norm = jnp.linalg.norm(dtheta_tool, axis=1, keepdims=True)

        scale_dx = jnp.minimum(1.0, self.max_displacement_thresh / (dx_norm + 1e-8))
        scale_dtheta = jnp.minimum(1.0, self.max_rotation_thresh / (dtheta_norm + 1e-8))

        dx_tool = dx_tool * scale_dx
        dtheta_tool = dtheta_tool * scale_dtheta

        magnitude_force = jnp.clip(
            magnitude_force,
            -self.max_force_thresh,
            self.max_force_thresh,
        )

        R_dtheta_tool = Rj.from_euler("xyz", dtheta_tool).as_matrix()

        dx_world = jnp.einsum("bij,bj->bi", self.tool_frame_orient, dx_tool) 
        # print(f"dx world is: {dx_world}")               
        tool_pos, orient_tool = self.get_tool_pos_orient(data)  
        tool_pos_prime = tool_pos + dx_world                   

        orient_tool_frame = jnp.swapaxes(self.tool_frame_orient, -1, -2) @ orient_tool       
        orient_tool_frame = R_dtheta_tool @ orient_tool_frame                        
        orient_tool_prime = self.tool_frame_orient @ orient_tool_frame  


        tool_pos_in_frame = jnp.swapaxes(self.tool_frame_orient, -1, -2) @ (tool_pos[...,None] - self.tool_frame_pos[...,None])
        tool_pos_in_frame = tool_pos_in_frame[..., 0]   
        theta_dist = compute_geodesic_distance_batch(self.tool_frame_orient, orient_tool_prime)

        out_of_bounds = check_out_of_bounds_batch(tool_pos_in_frame, self.zone)

        theta_violation = theta_dist > self.orient_scale
        out_of_bounds = out_of_bounds | theta_violation

        self.out_of_bounds = out_of_bounds

        self.target_pos = tool_pos_prime
        self.target_orient = orient_tool_prime

        self.desired_force_magnitude = magnitude_force
        self.dx_world = dx_world

    def sample_observation(self, data): #tested

        task_points_in_frame = realize_points_in_frame(self.task_points_noise, self.tool_frame_pos, self.tool_frame_orient)
        task_points_in_frame = task_points_in_frame.reshape((NUM_ENVS, -1))

        tool_pos, tool_ori = self.get_tool_pos_orient(data)
        tool_pos_frame = jnp.swapaxes(self.tool_frame_orient, -1, -2) @ (tool_pos[...,None] - self.tool_frame_pos[...,None])
        tool_pos_frame = tool_pos_frame[..., 0]
        tool_ori_frame = jnp.swapaxes(self.tool_frame_orient, -1, -2) @ tool_ori
        tool_ori_quat = Rj.from_matrix(tool_ori_frame).as_quat()

    
        force_data = self.get_force_data(data)

        force_data = force_data.reshape(NUM_ENVS, 3, 1)
        force_data_in_frame = realize_points_in_frame(force_data, self.tool_frame_pos, self.tool_frame_orient)
        force_data_in_frame = force_data_in_frame.reshape((NUM_ENVS ,-1))

        torque_data = self.get_torque_data(data)

        torque_data = torque_data.reshape(NUM_ENVS, 3, 1)
        torque_data_in_frame = realize_points_in_frame(torque_data, self.tool_frame_pos, self.tool_frame_orient)
        torque_data_in_frame = torque_data_in_frame.reshape((NUM_ENVS ,-1))

        stacked_data = jnp.hstack((task_points_in_frame, tool_pos_frame, tool_ori_quat, force_data_in_frame, torque_data_in_frame))

        return stacked_data
    
    def count_observation_size(self):
        
        obs_size = 0

        for obs_command in self.obs_key:
            if obs_command == "task_points":
                obs_size += 12
            elif obs_command == "tool":
                obs_size += 7
            elif obs_command == "force":
                obs_size += 6

        self.obs_size = obs_size
    
    def get_full_observation(self): #tested
        self.obs_list = self.obs_list.reshape((NUM_ENVS, self.obs_stack * self.obs_size))
        return self.obs_list
    
    def call_add_observation_to_stack(self, obs): #tested
        self.obs_list = np.hstack((self.obs_list[:, self.obs_size:], obs))

    def get_force_data(self, data): #tested
        force_data = get_force_batch(data)
        condition = self.force_dim == 0 
        condition = condition[:, None]
        force_data_zeros = jnp.zeros_like(force_data)
        force_data = jnp.where(condition, force_data_zeros, force_data)

        return force_data * -1
    
    def get_torque_data(self, data): #tested
        torque_data = get_torque_batch(data)
        condition = self.force_dim == 0 
        condition = condition[:, None]
        torque_data_zeros = jnp.zeros_like(torque_data)
        torque_data = jnp.where(condition, torque_data_zeros, torque_data)

        return torque_data * -1
    
    def get_targets_and_values_mft(self, data): #tested

        targets_and_values = get_joint_torque_targets_and_values_batch(data, self.target_pos, self.target_orient)

        forces = jnp.einsum("bij,bj->bi", self.sigmaForce, self.dx_world) 
        norms = jnp.linalg.norm(forces, axis=1, keepdims=True) 
        normalized = forces / (norms + 1e-12)

        scaled_forces = normalized * self.desired_force_magnitude[:, None]  

        mask = self.desired_force_magnitude > 1e-5                  
        scaled_forces = jnp.where(mask[:, None], scaled_forces, jnp.zeros_like(scaled_forces))

        targets_and_values = jnp.hstack([targets_and_values, self.motion_or_force_axis.reshape(NUM_ENVS, 3), self.force_dim.reshape(NUM_ENVS,1), scaled_forces])

        return targets_and_values
    
    def step(self, data, action):

        self.apply_action(data, action)

        for i in range(self.num_phys_steps):

            if i % 50 == 0:
                self.update_fspf_data(data)

            targets_and_values = self.get_targets_and_values_mft(data)
            joint_torques = get_mft_torques(targets_and_values, self.mft_server)
            data = step_mjx_env_batch(self.mjx_model, data, joint_torques)

            self.step_count += jnp.ones(NUM_ENVS)

        self.update_min_dists(data)

        obs, reward, dones, info = self._get_obs(data)

        self.print_telemetry()

        data, _ = self.reset(data, dones)

        return data, (obs, reward, dones, info)
    
    def update_min_dists(self, data):
        dists = dist(self.get_tool_points(data), self.task_points_noise, debug = False)
        lower = dists < self.min_dists
        self.min_dists = jnp.where(lower, dists, self.min_dists)

    def print_telemetry(self):
        print(f"min dists of completed environments : {self.min_dists}")
        print(f"step count of envs: {self.step_count}")

    
class DummyParallelEnv():

    def __init__(self, obs_stack, obs_size):

        self.obs_stack = obs_stack
        self.obs_size = obs_size

    def reset(self, data):
        
        obs, _, _, _ = self._get_obs(data)
        return data, obs
    
    def _get_obs(self, data):

        reward = jnp.zeros((NUM_ENVS,))
        dones = jnp.zeros((NUM_ENVS,))
        obs = jnp.zeros((NUM_ENVS, self.obs_stack * self.obs_size))

        info = {
            "reward": reward,
            "terminated": dones,
            "TimeLimit.truncated": jnp.zeros_like(dones)
        }

        return obs, reward, dones, info
    
    

    def step(self, data, action):   

        return data, self._get_obs(data)


    


















    



        
        


    



    

