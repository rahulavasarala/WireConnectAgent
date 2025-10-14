from random import Random
from sys import implementation
from chex import params_product
import torch
import mujoco
import zmq
import numpy as np
import math
import glfw
import time
from utils import params_from_yaml
import struct
from scipy.spatial.transform import Rotation as R
import argparse
from mujoco import mjx
from utils import show_key_points, TaskModule, fetch_body_id, fetch_body_pos_orient, extract_pos_orient_keypoints, get_joint_torques

parser = argparse.ArgumentParser(
        description="A script that processes a file specified by a path."
)
    
parser.add_argument(
    '--file', 
    type=str,
    default= "fr3peghole_modified.xml",
    required=False,
    help='The path to the input file to be processed.'
)

TEST_MODE = True

ctx = zmq.Context()
jt_socket = ctx.socket(zmq.REQ)
jt_socket.connect("ipc:///tmp/zmq_torque_server")

    # 3. Parse the arguments
args = parser.parse_args()

mj_xml_path = f"../models/scenes/{args.file}"


mj_model = mujoco.MjModel.from_xml_path(mj_xml_path)
mj_data = mujoco.MjData(mj_model)

# mjx_model = mjx.put_model(mj_model, impl='warp')
# mjx_data = mjx.put_data(mj_model, impl='warp')

home_key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
HOME_QPOS = mj_model.key_qpos[home_key_id]

task_tool_module = TaskModule()
task_points, tool_points = task_tool_module.get_task_tool_points("wc")


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

def reset_mj_data(data):

    mujoco.mj_resetData(mj_model, data)

    data.qpos[:] = HOME_QPOS
    data.qvel[:] = np.zeros(7)

    mujoco.mj_step(mj_model, data)
    data = show_key_points(mj_model, data, tool_points, task_points)

    return data

#checks the offset between the control point(Opensai) and the tool points
def check_offset(data):

    #get the position of link 7 + 0.36, and the control point data.

    tool_id = fetch_body_id(mj_model, data, "tool", obj_type = "body")
    tool_pos, tool_orient = fetch_body_pos_orient(data, tool_id)

    tool_points_world = tool_orient @ tool_points + tool_pos.reshape(3,1)

    print(f"tool points x axis: {tool_points_world[:,0] - tool_points_world[:,1]}")
    print(f"tool points y axis: {tool_points_world[:,2] - tool_points_world[:,1]}")

    tool_pos, tool_orient = extract_pos_orient_keypoints(tool_points_world)

    link7_id = tool_id = fetch_body_id(mj_model, data, "fr3_link7", obj_type = "body")
    link7_pos, link7_orient = fetch_body_pos_orient(data, link7_id)

    print(f"tool orient: {tool_orient} , link 7 orient: {link7_orient}")

    task_id = fetch_body_id(mj_model, data, "task", obj_type = "body")
    task_pos, task_orient = fetch_body_pos_orient(data, task_id)

    task_points_world = task_orient @ task_points + task_pos.reshape(3,1)

    task_pos, task_orient = extract_pos_orient_keypoints(task_points_world)

    print(f"task orient: {task_orient}")

def get_position(data):

    tool_id = fetch_body_id(mj_model, data, "tool", obj_type = "body")
    tool_pos, tool_orient = fetch_body_pos_orient(data, tool_id)

    tool_points_world = tool_orient @ tool_points + tool_pos.reshape(3,1)
    tool_pos, tool_orient = extract_pos_orient_keypoints(tool_points_world)

    print(tool_pos)

def get_task_position(data):

    task_id = fetch_body_id(mj_model, data, "task", obj_type = "body")
    task_pos, task_orient = fetch_body_pos_orient(data, task_id)

    print(task_pos)

def move_to_targets(data, target_pos, target_orient, iterations = 17):

    for _ in range(iterations):
        joint_torques = get_joint_torques(data, target_pos , target_orient, jt_socket)
        data.ctrl[:] = joint_torques
        mujoco.mj_step(mj_model, data)

    get_task_position(data)

    return data


target_pos = np.array([0.3, 0.3, 0.3])
target_orient = np.array([[1,0,0], [0,-1,0], [0,0,-1]])

    
def main():
    global mj_data
    RHZ = 60.0
    
    # Reset simulation
    mj_data = reset_mj_data(mj_data)

    check_offset(mj_data)

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
        if TEST_MODE:
            mj_data = move_to_targets(mj_data, target_pos, target_orient)
        # mj_data, _ = test.step(mj_data)
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