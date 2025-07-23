#This environment will be strictly for debugging, testing out whether algorithms are stepping properly
#First, we will need to compile the zmq environment to take in 1 argument, number of envs it should be expecting

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

stacked_data_dim = 7 * 3 
mj_xml_path = "/Users/rahulavasarala/Desktop/OpenSai/WireConnectAgent/models/scenes/rizon4smaleconjl.xml"
SERVER_ENDPOINT = "ipc:///tmp/zmq_torque_server"
ctx = zmq.Context()
socket = ctx.socket(zmq.REQ)
socket.connect(SERVER_ENDPOINT)

button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

#Wrapper class of the mujoco step
class SingleZMQEnv():

    def __init__(self, mj_xml: str, target_pos: np.ndarray, target_orient: np.ndarray, zmq_socket, num_envs= 1):

        self.target_pos = target_pos
        self.target_orient = target_orient
        self.num_envs = num_envs

        self.socket = zmq_socket

        self.mj_model = mujoco.MjModel.from_xml_path(mj_xml)
        self.mj_data = mujoco.MjData(self.mj_model)


    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)

    def step(self): #This method will run 1 physics simulation step of mujoco

        joint_torques = self.get_joint_torques()
        self.mj_data.ctrl[:] = joint_torques

        mujoco.mj_step(self.mj_model, self.mj_data)

    def get_joint_torques(self) -> np.ndarray:
        #get the cart_pos, cart_orient, qpos, qvel

        stacked_data = np.concatenate([self.target_pos, self.target_orient])

        qpos = self.mj_data.qpos.copy()      
        qvel = self.mj_data.qvel.copy() 

        stacked_data = np.concatenate([stacked_data, qpos, qvel])
        joint_torques = self.send_and_recieve_joint_torques(stacked_data, self.socket)

        return joint_torques

    @staticmethod
    def send_and_recieve_joint_torques(stacked_data: np.ndarray, socket) -> np.ndarray:
        packed_data = struct.pack('f' * stacked_data.shape[-1], *stacked_data)

        socket.send(packed_data)
        reply = socket.recv()
        joint_torques = np.frombuffer(reply, dtype = np.float32) 

        joint_torques = joint_torques.reshape(-1, 7) 
        # print("joint_torques: {}".format(joint_torques))
        return joint_torques
    
def main():

    #Define Hyper parameters for rendering
    RHZ = 60.0

    base_position = np.array([0,0.4,0.1])

    #load model
    env = SingleZMQEnv(mj_xml=mj_xml_path, target_pos=base_position, target_orient=np.array([0,1,0,0]), zmq_socket=socket)

    env.step()
    female_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "female-connector-truncated")

    euler = 0.1 * np.array([0, random.random(), random.random()])
    R_female_to_tilt = R.from_euler('xyz', euler).as_matrix()

    translation = 0.005* np.array([random.random(), random.random(), random.random()])

    R_world_to_female = env.mj_data.xmat[female_id].reshape((3,3))
    female_pos = env.mj_data.xpos[female_id]
    print("female pos: {}".format(female_pos))

    key_points_female_tilt = np.array([[-0.007, -0.007, 0.007, 0.007],
                                [0.0, 0.008 , 0.008, 0.0], 
                                [0.0, 0.0, 0.0, 0.0]])
    
    R_total = R_world_to_female @ R_female_to_tilt
    
    female_points_world = R_total @ key_points_female_tilt

    female_points_world += (female_pos + translation).reshape(3,1)

    #Then what I should do is take the average of these points in the world frame
    #add height * z axis of R_total, and we should tell the control point to move there with that axis

    target_point = np.mean(female_points_world, axis=1)
    z_space = 0.01
    target_point = target_point + z_space * R_total[:,2]

    target_orientation = np.array([[1, 0, 0], [0, -1, 0], [0,0,-1]]) @ R_total

    target_quat = R.from_matrix(target_orientation).as_quat() 

    env.target_pos =target_point
    env.target_orient = target_quat
    
    # Init visualization
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    scene = mujoco.MjvScene(env.mj_model, maxgeom=1000)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    # Initialize GLFW
    glfw.init()
    window = glfw.create_window(800, 600, "Python Test Bench", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # Enable vsync

    # Create context for rendering
    context = mujoco.MjrContext(env.mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def scroll_callback(window, xoffset, yoffset):
        mujoco.mjv_moveCamera(env.mj_model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, yoffset * 0.1, scene, cam)

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
        mujoco.mjv_moveCamera(env.mj_model, action, dx / height, dy / height, scene, cam)

    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)

    # Reset simulation
    env.reset()

    # Physics + Graphics loop
    while not glfw.window_should_close(window):

        start_time = time.time()

        while time.time() < start_time + 1.0/RHZ:
            env.step() 

        # Update scene
        mujoco.mjv_updateScene(env.mj_model, env.mj_data, scene_option, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

        # Get framebuffer size
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)

        # Render scene
        mujoco.mjr_render(viewport, scene, context)

        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

    # Cleanup
    glfw.terminate()

if __name__ == "__main__":
    main()
   


