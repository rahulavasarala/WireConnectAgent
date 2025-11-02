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
import argparse

from utils import compute_geodesic_distance, RewardsModuleV2, get_force_data, get_velocity_data, fetch_body_id, fetch_body_pos_orient
from utils import extract_pos_orient_keypoints, generate_random_sphere_point, switch_frame_key_points, params_from_yaml
from utils import send_and_recieve_to_server, get_joint_torques, get_mft_torques, realize_points_in_frame, check_out_of_bounds
from utils import TaskModule
from policy_network import TransformerAgent

redis_client = redis.Redis()

class RedisKeys(Enum):

    JOINT_ANGLES_KEY = "sai::sensors::FrankaRobot::joint_positions"
    JOINT_VELOCITIES_KEY = "sai::sensors::FrankaRobot::joint_velocities"
    JOINT_TORQUES_COMMANDED_KEY = "sai::commands::FrankaRobot::control_torques"

    FORCE_SENSOR_KEY = "sai::sensors::FrankaRobot::ft_sensor::end-effector::force"
    MOMENT_SENSOR_KEY = "sai::sensors::FrankaRobot::ft_sensor::end-effector::moment"

    SMOOTHED_FORCE = "sai::sensors::FrankaRobot::ft_sensor::smoothed_force"
    SMOOTHED_TORQUE = "sai::sensors::FrankaRobot::ft_sensor::smoothed_torque"
    MOTION_FORCE_AXIS = "sai::sensors::FrankaRobot::motion_force_axis" #Type Eigen VectorXD x , y, z, dim
    DESIRED_FORCE = "sai::sensors::FrankaRobot::desired_force"

    TARGET_POS = "sai::sensors::FrankaRobot::target_pos"
    TARGET_ORIENT = "sai::sensors::FrankaRobot::target_orient"

    CONTROL_POINT_ORIENT = "sai::sensors::FrankaRobot::control_point_orient"
    CONTROL_POINT_POS = "sai::sensors::FrankaRobot::control_point_pos"
    CONTROL_POINT_VEL = "sai::sensors::FrankaRobot::control_point_vel"

class RealRobotEnv():

    def __init__(self, fspf_server, params, tool_start_info):

        self.step_count = 0

        self.fspf_server = fspf_server

        self.reward_module = RewardsModuleV2()

        self.initialize_parameters(params)
        self.load_tool_start_info(tool_start_info)
        

        self.count_observation_size()

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

        self.force_bias = np.zeros(3)
        self.torque_bias = np.zeros(3)

    def load_tool_start_info(self, tool_start_info):
        self.tool_frame_pos = tool_start_info[:3]
        self.tool_frame_orient = R.from_quat(tool_start_info[3:7]).as_matrix()
        self.task_points_noise = tool_start_info[7:].reshape(3,4)

    def set_mft_redis_keys(self):
        redis_client.set(RedisKeys.TARGET_POS.value, json.dumps(self.target_pos.tolist()))
        redis_client.set(RedisKeys.TARGET_ORIENT.value, json.dumps(self.target_orient.tolist()))
        motion_force_axis = np.concatenate([self.motion_or_force_axis.reshape(3,), [self.force_dim]])
        
        redis_client.set(RedisKeys.MOTION_FORCE_AXIS.value, json.dumps(motion_force_axis.tolist()))
        redis_client.set(RedisKeys.DESIRED_FORCE.value, json.dumps(self.desired_force.tolist()))

    #This function will make the real robot probe to see what the success distance is, such that it knows when to stop the policy
    def find_success_thresh(self, iterations = 700):
        #This is needed to be implemented, this is a real robot function, where the robot goes down until it feels a force that is high

        self.reset()

        z_axis = self.tool_frame_orient[:, 2]
        target_pos = self.tool_frame_pos + 0.3 * z_axis
        self.target_pos = target_pos
        self.target_orient = R.from_matrix(self.tool_frame_orient).as_quat()

        self.set_mft_redis_keys()
        measured_force = np.array(json.loads(redis_client.get(RedisKeys.SMOOTHED_FORCE.value)))

        while np.linalg.norm(measured_force) < 3:
            time.sleep(0.01)

        task_points_pos, _ = extract_pos_orient_keypoints(self.task_points_noise)
        control_point_pos = np.array(json.loads(redis_client.get(RedisKeys.CONTROL_POINT_POS.value)))

        self.success_thresh = np.linalg.norm(task_points_pos - control_point_pos)
        self.dist_scale = 1.5 * self.success_thresh

    #This function will tell the real robot to reset to it's desired position
    def reset(self, iterations = 2000):

        # find a good universal position to set the robot too with the motion force task
        self.reset_fspf_data()
        self.reset_obs_list()

        self.target_pos = self.tool_frame_pos
        self.target_orient = R.from_matrix(self.tool_frame_orient).as_quat()

        self.set_mft_redis_keys()

        for _ in range(iterations):
            time.sleep(0.001)

    def reset_obs_list(self):
        obs = self.sample_observation()
        self.obs_list = [None] * self.obs_stack

        for i in range(self.obs_stack):
            self.obs_list[i] = obs.copy()

    def move_to_targets(self, target_pos,target_orient =np.array([[1,0,0],[0,-1,0],[0,0,-1]]),  iterations = 700):

        self.target_pos = target_pos
        self.target_orient = R.from_matrix(target_orient).as_quat()

        self.motion_or_force_axis = np.zeros(3)
        self.force_dim = 0
        self.desired_force = np.zeros(3)

        self.set_mft_redis_keys()

        for _ in range(iterations):
            time.sleep(0.001)

    def find_force_sensor_bias(self):
        self.move_to_targets(target_pos =np.array([0.3, 0, 0.3]),  iterations= 4000)
        print("moved to targets!")

        polling_length = 2000

        force_sensor_data = np.zeros(shape = (6, 2000))

        for i in range(polling_length):
            force_world, torque_world = self.get_force_data() 

            force_torque = np.concatenate([force_world, torque_world]).reshape(6,)

            force_sensor_data[:, i] = force_torque

            time.sleep(0.001)

        average_force_torque = np.mean(force_sensor_data, axis = 1) 

        self.force_bias = average_force_torque[0:3]
        self.torque_bias = average_force_torque[3:6]

        print(f"average force torque is: {average_force_torque}, force_bias : {self.force_bias} torque_bias: {self.torque_bias}")

    def get_tool_points(self):

        tool_pos = np.array(json.loads(redis_client.get(RedisKeys.CONTROL_POINT_POS.value)))
        tool_orient = np.array(json.loads(redis_client.get(RedisKeys.CONTROL_POINT_ORIENT.value)))

        tool_points = tool_orient @ self.tool_base_points + tool_pos.reshape(3,1)

        return tool_points

    def get_tool_pos_orient(self):
        
        tool_pos = np.array(json.loads(redis_client.get(RedisKeys.CONTROL_POINT_POS.value)))
        tool_orient = np.array(json.loads(redis_client.get(RedisKeys.CONTROL_POINT_ORIENT.value)))

        return tool_pos, tool_orient

    def get_noisy_task_points(self):
        return self.task_points_noise
    
    def get_task_points(self):
        return self.task_base_points

    def get_force_data(self, frame = None):
        #get the real force data from the robot, and we should be good

        force_data = np.array(json.loads(redis_client.get(RedisKeys.SMOOTHED_FORCE.value)))
        torque_data = np.array(json.loads(redis_client.get(RedisKeys.SMOOTHED_TORQUE.value)))

        #get the force data in the world frame

        tool_orient = np.array(json.loads(redis_client.get(RedisKeys.CONTROL_POINT_ORIENT.value)))
        print("tool_orient")

        force_world = tool_orient @ force_data
        torque_world = tool_orient @ torque_data

        force_world -= self.force_bias
        torque_world -= self.torque_bias

        if frame is not None:

            return frame.T @ force_world , frame.T @ torque_world
        
        return force_world, torque_world
    
    def get_velocity_data(self):
        velocity = np.array(json.loads(redis_client.get(RedisKeys.CONTROL_POINT_VEL.value)))

        return velocity

    def sample_observation(self):

        obs_list = []

        for obs_command in self.obs_key:
            if obs_command == "task_points":
                points = realize_points_in_frame(self.get_noisy_task_points(), self.tool_frame_pos, self.tool_frame_orient)
                obs_list.append(points.flatten())
            elif obs_command == "tool":
                tool_pos, tool_orient = self.get_tool_pos_orient()
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


    def _get_obs(self, data):

        dones = np.array([self.check_dones(data)])

        new_obs = self.sample_observation(data)
        self.call_add_observation_to_stack(new_obs)

        obs = self.get_full_observation()

        obs = np.array([obs])
        obs = obs.reshape(1,-1)

        return obs, dones
    
    def check_dones(self):

        done = 0

        if self.out_of_bounds:
            return 1
        
        if self.step_count > self.num_phys_steps * self.actions_per_episode:
            return 1

        if self.curr_dist < self.success_thresh:
            self.success = True
            return 1

        return 0
    
    def apply_action(self, action):

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

        tool_pos, orient_tool = self.get_tool_pos_orient()
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

        dx = dx_world.reshape((3,1))

        self.desired_force = self.sigmaForce @ dx
        if np.linalg.norm(self.desired_force) < 1e-5:
            self.desired_force = np.zeros((3,))
        else:
            self.desired_force = self.desired_force_magnitude * self.desired_force/np.linalg.norm(self.desired_force)

    def step(self, action):
       
        #apply action, and step 67 times at 0.001 seconds, if the steps actually pass 50, then we call an
        self.apply_action(action)
        self.set_mft_redis_keys()

        for i in range(self.num_phys_steps):
            time.sleep(0.001)
            self.step_count += 1

            # if self.step_count % 50 == 0:
            #     self.update_fspf_data()
    
    def reset_fspf_data(self):
        self.motion_or_force_axis = np.zeros(3)
        self.force_dim = 0
        self.desired_force_magnitude = 0
        self.dx_world = np.zeros(3)
        self.sigmaMotion = np.eye(3)
        self.sigmaForce = np.zeros((3,3))
        self.desired_force = np.zeros(3)

        data = np.ones(1)
        packed_data = struct.pack('f' * data.shape[-1], *data)

        self.fspf_server.send(packed_data)
        reply = self.fspf_server.recv()

    def update_fspf_data(self):#This is clean, verified from visual_tests.py

        dx = self.dx_world.reshape((3,1))

        motion_control = (50/1000) * self.sigmaMotion @ dx #correct
        force_control = (50/1000) * self.sigmaForce @ dx #correct 

        motion_control = motion_control.reshape(3,)
        force_control = force_control.reshape(3,)

        measured_force, _ = self.get_force_data()
        measured_force = measured_force * -1
        measured_vel = self.get_velocity_data()

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

    def print_telemetry(self):
        print(f"self out of bounds: {self.out_of_bounds}")
        print(f"self.step_count: {self.step_count} self.num_phys_steps: {self.num_phys_steps} greater: {self.step_count > self.num_phys_steps* self.actions_per_episode}")


def deploy_model(real_robot_env: RealRobotEnv, agent: TransformerAgent, action_list = None, random = True):

    print("finding force sensor bias...")
    # real_robot_env.find_force_sensor_bias()

    time.sleep(2)

    print("Resetting the robot!")

    real_robot_env.reset()

    obs = real_robot_env.sample_observation()

    print(f"first observation is: {obs}")

    for i in range(10):

        action = np.zeros(7)

        if random:
            action = np.random.uniform(low=-1.0, high=1.0, size=(7,))
        elif action_list is not None:
            real_robot_env.step(action_list[i])
        else:

            obs = real_robot_env.get_full_observation()
            

            obs = torch.tensor(obs, dtype = torch.float32)
            action, _, _, _ = agent.get_action_and_value(obs)
            action = action.cpu().numpy()
            action = action.reshape(7 * real_robot_env.action_horizon,)

            if i == 0:
                print(f"full obs: {obs}, action : {action}")

        real_robot_env.step(action)

#In this test, the robot will be moved to 7 random locations, and the position error will be recorded
def calibration_test(real_robot_env: RealRobotEnv):

    pos_err = np.zeros(7)

    for i in range(7):
        target_pos = np.array([0.4,0, 0.4]) + 0.1* generate_random_sphere_point()

        real_robot_env.move_to_targets(target_pos, iterations = 1000)
        print("moved to the correct point!")

        tool_pos, _ = real_robot_env.get_tool_pos_orient()

        pos_err[i] = np.linalg.norm(target_pos - tool_pos)

        print(f"Pos error for pos: {pos_err[i]} is {tool_pos}")
        time.sleep(1)

    print(f"Avg err: {np.mean(pos_err)}")

def fspf_test(real_robot_env: RealRobotEnv):

    real_robot_env.find_force_sensor_bias()

    real_robot_env.reset_fspf_data()

    count = 0
    while count < 20000:

        if count%50 == 0:
            real_robot_env.update_fspf_data()

        count+= 1

        time.sleep(0.001)

        if count % 200 == 0:
            print(f"motion_force_axis: {real_robot_env.motion_or_force_axis}, force_dim : {real_robot_env.force_dim}")



def main():
    #Parse arguments for type of mode run by real robot: 

    parser = argparse.ArgumentParser(description="Example of parsing flags in Python.")
    
    # Define a flag called --mode that takes a string argument
    parser.add_argument(
        '--mode',
        type=str,
        required=True,     # make it required (optional)
        help='Mode to run the program(fspf,calibration,deploy)'
    )
    
    args = parser.parse_args()

    ctx = zmq.Context()

    fspf_socket = ctx.socket(zmq.REQ)
    fspf_socket.connect("ipc:///tmp/zmq_fspf_server")
    visual_config_file = "./sim_tests_config.yaml"
    visual_config = params_from_yaml(visual_config_file)
    run_name = visual_config["model"]["run_name"]
    params = params_from_yaml(f"{visual_config["model"]["run_dir"]}/run_{run_name}/experiment_{run_name}.yaml")

    tool_start_info = np.loadtxt(f"./runs/run_{run_name}/tool_start_info.txt")

    real_robot_env = RealRobotEnv(fspf_socket, params = params, tool_start_info=tool_start_info)
    weights_path = f"{visual_config["model"]["run_dir"]}/run_{run_name}/model{visual_config["model"]["iter"]}.cleanrl_model"

    state_dict = torch.load(weights_path)
    agent = TransformerAgent(real_robot_env.obs_size, real_robot_env.obs_size, real_robot_env.obs_stack, 7 * real_robot_env.action_horizon)
    agent.load_state_dict(state_dict)

    # action_list = np.zeros((10 , 7))

    # action_list[:, 0] = 0
    # action_list[:,2] = 1

    if args.mode == "deploy":
        deploy_model(real_robot_env,agent, action_list = None, random = False)
    elif args.mode == "random":
        deploy_model(real_robot_env,agent, random = True)
    elif args.mode == "calibration":
        calibration_test(real_robot_env)
    elif args.mode == "fspf":
        fspf_test(real_robot_env)
    else:
        print("Enter a valid option for mode!")




if __name__ == "__main__":
    main()











    











    