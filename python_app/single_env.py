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

ctx = zmq.Context()

jt_socket = ctx.socket(zmq.REQ)
jt_socket.connect("ipc:///tmp/zmq_torque_server")

mft_socket = ctx.socket(zmq.REQ)
mft_socket.connect("ipc:///tmp/zmq_motion_force_server")

fspf_socket = ctx.socket(zmq.REQ)
fspf_socket.connect("ipc:///tmp/zmq_fspf_server")

FILTER_FREQ = 50.0
CONTROL_FREQ = 1000.0

#------------Global Constants ------------------------------------------------------------------

class SingleZMQEnv():

    def __init__(self, mj_model, target_pos: np.ndarray, target_orient: np.ndarray, jt_socket, mft_socket, fspf_socket, action_space_dim = 7, observation_space_dim = 13):#Looks good

        self.target_pos = target_pos
        self.target_orient = target_orient

        self.mft_socket = mft_socket
        self.fspf_socket = fspf_socket
        self.jt_socket = jt_socket

        self.mj_model = mj_model

        self.quick_reset = False
        self.quick_reset_joint_pos = np.zeros(7)

        #dimensions for action space and observation space
        self.action_space_dim = action_space_dim
        self.observation_space_dim = observation_space_dim

        self.num_physics_steps = 100
        self.step_count = 0
        self.no_contact_count = 0

        #Thresholds for the outputs of the agent
        self.max_force_thresh = 0.01
        self.max_rotation_thresh = 0.01
        self.max_displacement_thresh = 0.01

        self.motion_or_force_axis = np.zeros(3)
        self.force_dim = 0

        self.female_percieved_orientation = np.array([1,0,0,0])
        self.female_percieved_position = np.zeros(3)

        self.num_envs = 1



    def reset(self, data):
        mujoco.mj_resetData(self.mj_model, data)

        if self.quick_reset:
            data.qpos[:] = self.quick_reset_joint_pos
            data.qvel[:] = np.zeros(7)
            mujoco.mj_step(self.mj_model, data)
        else:
            mujoco.mj_step(self.mj_model, data)
            move_wait_steps = 10000 #We can change this 

            self.init_starting_position_orient(data)

            self.move_to_targets(data, move_wait_steps)

            self.init_quick_reset_joint_positions(data)
            self.quick_reset = True

            self.target_pos = np.array([0,0.4,0.1])
            self.target_orient = np.array([0,1,0,0])

        self.sigmaMotion = np.identity(3)
        self.sigmaForce = np.zeros((3,3))
        self.reset_fspf_data()

        obs = np.zeros((self.num_envs, self.observation_space_dim))

        self.step_count = 0

        return data, obs

    def init_starting_position_orient(self, data):#Assume that a test step is done already

        z_space = 0.02
        translation_noise_magnitude = 0.005
        rotation_noise_magnitude = 0.01
        
        euler = rotation_noise_magnitude * np.array([0, random.random(), random.random()])
        R_female_to_tilt = R.from_euler('xyz', euler).as_matrix()
        translation = translation_noise_magnitude* np.array([random.random(), random.random(), random.random()])

        R_world_to_female = data.xmat[female_id].reshape((3,3))
        female_pos = data.xpos[female_id]

        key_points_female_tilt = np.array([[-0.007, -0.007, 0.007, 0.007],
                                    [0.0, 0.008 , 0.008, 0.0], 
                                    [0.0, 0.0, 0.0, 0.0]])
        
        R_total = R_world_to_female @ R_female_to_tilt
        female_points_world = R_total @ key_points_female_tilt
        female_points_world += (female_pos + translation).reshape(3,1)

        #Figuring out the target quaternions for the female orientation
        self.female_percieved_orientation = R.from_matrix(R_total).as_quat()
        target_point = np.mean(female_points_world, axis=1)
        
        self.female_percieved_position = female_pos + translation

        target_point = target_point + z_space * R_total[:,2]

        target_orientation = np.array([[1, 0, 0], [0, -1, 0], [0,0,-1]]) @ R_total
        target_quat = R.from_matrix(target_orientation).as_quat() 
        self.target_pos =target_point
        self.target_orient = target_quat

        self.R_female_to_world = R_total.T
        self.R_world_to_female = R_total

    def init_quick_reset_joint_positions(self, data):
        self.quick_reset_joint_pos = data.qpos.copy()

    def move_to_targets(self, data, iterations =1 ): #This method will run 1 physics simulation step of mujoco
        #step the simulation iterations times with normal joint torque server
        for _ in range(iterations):
            joint_torques = self.get_joint_torques(data)
            data.ctrl[:] = joint_torques
            mujoco.mj_step(self.mj_model, data)

        return data


    def get_joint_torques(self, data) -> np.ndarray:
        #get the cart_pos, cart_orient, qpos, qvel
        stacked_data = np.concatenate([self.target_pos, self.target_orient])

        qpos = data.qpos.copy()      
        qvel = data.qvel.copy() 

        stacked_data = np.concatenate([stacked_data, qpos, qvel])
        joint_torques = self.send_and_recieve_joint_torque_server(stacked_data)

        return joint_torques

    def send_and_recieve_joint_torque_server(self, stacked_data: np.ndarray) -> np.ndarray:
        packed_data = struct.pack('f' * stacked_data.shape[-1], *stacked_data)

        self.jt_socket.send(packed_data)
        reply = self.jt_socket.recv()
        joint_torques = np.frombuffer(reply, dtype = np.float32) 

        joint_torques = joint_torques.reshape(-1, 7) 
        # print("joint_torques: {}".format(joint_torques))
        return joint_torques
    
    def compute_reward(self, data):#sanity checked this function
        key_points_female = np.array([[-0.007, -0.007, 0.007, 0.007],
                                    [0.0, 0.008 , 0.008, 0.0], 
                                    [0.0, 0.0, 0.0, 0.0]])
        
        world_key_points_female = self.R_world_to_female @ key_points_female + self.female_percieved_position.reshape(3,1)

        key_points_male = np.array([[0.007, 0.007, -0.007, -0.007],
                                    [0.0, 0.0 , 0.0, 0.0], 
                                    [0.0, 0.01, 0.01, 0.0]])
        
        male_pos = np.array(data.xpos[male_body_id]) 
        R_world_male = np.reshape(data.xmat[male_body_id], (3, 3))
        world_key_points_male = R_world_male @ key_points_male + male_pos.reshape(3,1)

        reward = np.linalg.norm(world_key_points_female - world_key_points_male)

        return reward
    
    def sample_obs(self, data):
        #conglomerate the observations together/for now let us not worry about the points

        force = data.sensordata[force_sensor_adr : force_sensor_adr + 3]
        torque = data.sensordata[torque_sensor_adr: torque_sensor_adr + 3]

        R_flat = data.xmat[site_id]
        R_world_sensor = R_flat.reshape((3,3))

        force_world = R_world_sensor @ force
        torque_world = R_world_sensor @ torque

        force_female_percieved = self.R_female_to_world @ force_world
        torque_female_percieved = self.R_female_to_world @ torque_world

        xmat = data.xmat[male_body_id]   
        male_orientation_world = xmat.reshape(3, 3)  

        male_orient_relative_female = self.R_female_to_world * male_orientation_world
        male_pos_world = data.xpos[male_body_id]
        male_pos_relative_female = self.R_female_to_world @ (male_pos_world - self.female_percieved_position)

        male_orientation_relative_quat = R.from_matrix(male_orient_relative_female).as_quat()

        obs = np.concatenate([force_female_percieved, torque_female_percieved, male_orientation_relative_quat, male_pos_relative_female])

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
        if self.no_contact_count > 15:
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

    def update_fspf_server(self, data, dx_female):

        #Obviously, we need to find the
        dx_world = self.R_world_to_female @ dx_female

        motion_control = (FILTER_FREQ/CONTROL_FREQ) * self.sigmaMotion @ dx_world
        force_control = (FILTER_FREQ/CONTROL_FREQ) * self.sigmaForce @ dx_world

        force = data.sensordata[force_sensor_adr : force_sensor_adr + 3]

        R_flat = data.xmat[site_id]
        R_world_sensor = R_flat.reshape((3,3))
        measured_force = R_world_sensor @ force
        measured_vel = data.cvel[male_body_id, :3]

        stacked_data = np.concatenate([motion_control, force_control, measured_force, measured_vel])
        packed_data = struct.pack('f' * stacked_data.shape[-1], *stacked_data)

        self.fspf_socket.send(packed_data)
        reply = self.fspf_socket.recv()
        motion_or_force_axis_and_fdim  = np.frombuffer(reply, dtype = np.float32) 

        motion_or_force_axis = motion_or_force_axis_and_fdim[:3]
        fdim = motion_or_force_axis_and_fdim[3]

        #let us write logic to create our selection matrices

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
    
    def get_joint_torques_mft_server(self, des_pos, des_orient, qpos, qvel, force_or_motion_axis, fdim,  desired_force):#This function works

        stacked_data = np.concatenate([des_pos, des_orient, qpos, qvel, force_or_motion_axis, fdim, desired_force])
        packed_data = struct.pack('f' * stacked_data.shape[-1], *stacked_data)

        self.mft_socket.send(packed_data)
        reply = self.mft_socket.recv()
        joint_torques = np.frombuffer(reply, dtype = np.float32) 

        joint_torques = joint_torques.reshape(-1, 7) 
        # print("joint_torques: {}".format(joint_torques))
        return joint_torques
    
    def move_to_targets_mft(self, data):#This function works, is a sanity check

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

    def step(self, data, action):

        dx_female = action[:3]
        dx_world = self.R_world_to_female @ dx_female

        if np.linalg.norm(dx_female) > self.max_displacement_thresh:
            dx_female = self.max_displacement_thresh * dx_female/np.linalg.norm(dx_female)

        dtheta_female = action[3:6]

        if np.linalg.norm(dtheta_female) > self.max_displacement_thresh:
            dtheta_female = self.max_rotation_thresh * dtheta_female/np.linalg.norm(dtheta_female)

        R_dtheta_female = R.from_euler("xyz", dtheta_female).as_matrix()

        magnitude_force = action[6]

        if np.linalg.norm(magnitude_force) > self.max_force_thresh:
            magnitude_force = self.max_force_thresh * magnitude_force/np.linalg.norm(magnitude_force)

        xmat = data.xmat[male_body_id]   
        male_orientation_world = xmat.reshape(3, 3)  

        male_orientation_female = self.R_female_to_world @ male_orientation_world
        male_orientation_female_prime = R_dtheta_female @ male_orientation_female

        male_orientation_world_prime = self.R_world_to_female @ male_orientation_female_prime
        male_orientation_world_prime = R.from_matrix(male_orientation_world_prime).as_quat()
        
        male_pos_world = data.xpos[male_body_id]
        male_pos_relative_female = self.R_female_to_world @ (male_pos_world - self.female_percieved_position)

        male_pos_female_prime = male_pos_relative_female + dx_female
        male_pos_world_prime = self.R_world_to_female @ (male_pos_female_prime + self.female_percieved_position)


        for _ in range(self.num_physics_steps):
            if self.step_count % 50 == 0:
                self.motion_or_force_axis, self.force_dim = self.update_fspf_server(data, dx_female)
            qpos = data.qpos.copy()
            qvel = data.qvel.copy()
            desired_force = self.sigmaForce @ dx_world
            if np.linalg.norm(desired_force) < 0.001:
                desired_force = np.array([0.1,0.1,0.1])
            else:
                desired_force = magnitude_force * desired_force/np.linalg.norm(desired_force)
            joint_torques = self.get_joint_torques_mft_server(male_pos_world_prime, male_orientation_world_prime,qpos, qvel, self.motion_or_force_axis, np.array([self.force_dim]), desired_force)
            data.ctrl[:] = joint_torques
            mujoco.mj_step(self.mj_model, data)

            self.step_count += 1

        return data, self._get_obs(data)


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
    #This value is the hertz in which the simulation will be rendered
    RHZ = 60.0

    base_position = np.array([0,0.4,0.1])

    #load model
    env = SingleZMQEnv(mj_model=mj_model, target_pos=base_position, target_orient=np.array([0,1,0,0]), jt_socket=jt_socket, mft_socket=mft_socket, fspf_socket=fspf_socket)

    # Initialize GLFW
    glfw.init()
    window = glfw.create_window(800, 600, "Python Test Bench", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # Enable vsync

    # Create context for rendering
    context = mujoco.MjrContext(env.mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)

    # Reset simulation
    mj_data , _ = env.reset(mj_data)
    glfw_loop_count = 0

    print("starting the simulation!!")

    # Physics + Graphics loop
    while not glfw.window_should_close(window):

        start_time = time.time()

        while time.time() < start_time + 1.0/RHZ:
            action = np.random.rand(7)
            mj_data, _ = env.step(mj_data, action)

        if glfw_loop_count == 500:
            _, reward, _ ,_ = env._get_obs(mj_data)
            print("reward: {}".format(reward))
            print("reset was called!!!!!!")
            mj_data , _  = env.reset(mj_data)
            _, reward, _ ,_ = env._get_obs(mj_data)
            print("reward: {}".format(reward))
            

        # Update scene
        mujoco.mjv_updateScene(env.mj_model, mj_data, scene_option, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

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
   


