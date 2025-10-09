from random import Random
from chex import params_product
from contact_rich_env import ContactRichEnv
from policy_network import TransformerAgent
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


launch_params = params_from_yaml("./sim_tests_config.yaml")
task_points = np.array([[0.005, 0.005, 0], [-0.005, 0.005, 0], [-0.005, -0.005, 0], [0.005, -0.005, 0]]).T
tool_points = np.array([[0.005, -0.005, 0.2], [-0.005, -0.005, 0.2], [-0.005, 0.005, 0.2], [0.005, 0.005, 0.2]]).T


mj_xml_path = launch_params["xml_path"]
mj_model = mujoco.MjModel.from_xml_path(mj_xml_path)
mj_data = mujoco.MjData(mj_model)

params = params_from_yaml("./experiment.yaml")

ctx = zmq.Context()

jt_socket = ctx.socket(zmq.REQ)
jt_socket.connect("ipc:///tmp/zmq_torque_server")

mft_socket = ctx.socket(zmq.REQ)
mft_socket.connect("ipc:///tmp/zmq_motion_force_server")

fspf_socket = ctx.socket(zmq.REQ)
fspf_socket.connect("ipc:///tmp/zmq_fspf_server")

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
    
class Simulation():
    def __init__(self, visual_config_file, tool_points, task_points):
        visual_config = params_from_yaml(visual_config_file)
        self.action_type = visual_config["action_type"]
        run_name = visual_config["model"]["run_name"]

        params = params_from_yaml(f"{visual_config["model"]["run_dir"]}/run_{run_name}/experiment_{run_name}.yaml")
        tool_start_info = np.loadtxt(f"./runs/run_{run_name}/tool_start_info.txt")

        self.env = ContactRichEnv([jt_socket, mft_socket, fspf_socket], model=mj_model, data= mj_data, task_base_points=task_points, tool_base_points=tool_points, params = params, tool_start_info=tool_start_info)
        self.debug = visual_config["debug"]

        self.env.eval()
        weights_path = f"{visual_config["model"]["run_dir"]}/run_{run_name}/model{visual_config["model"]["iter"]}.cleanrl_model"

        self.load_model(weights_path = weights_path)
        self.target_pos = None
        self.target_orient = None
        self.replay = visual_config["replay"]

    def reset(self, data):
        self.env.reset(data)
        self.action = None
        self.action_index = self.env.exec_actions
        self.switch_to_nominal = False
        self.nominal_step_count = 0

    def generate_action(self, obs):
        action = None

        if self.action_type == "random":
            action = np.random.uniform(-1, 1, (7 * self.env.action_horizon))
        elif self.action_type == "model":
            obs = torch.tensor(obs, dtype = torch.float32)
            action, _, _, _ = self.model.get_action_and_value(obs)
            action = action.cpu().numpy()
            action = action.reshape(7 * self.env.action_horizon,)
            
        return action

    def load_model(self, weights_path):
        state_dict = torch.load(weights_path)
        self.model = TransformerAgent(self.env.obs_size, self.env.obs_size, self.env.obs_stack, 7 * self.env.action_horizon)
        self.model.load_state_dict(state_dict)

    def step(self, data):

        if not self.switch_to_nominal:
            data = self.policy_step(data)
        else:
            data= self.nominal_step(data)

        if self.debug:
            self.env.show_key_points(data)
            

        return data, 1
    
    def policy_step(self, data):
        #Create a policy step, and then create a 

        for _ in range(17): 

            if self.action_index == self.env.exec_actions:
                obs = self.env.get_full_observation()
                self.action = self.generate_action(obs)
                self.action_index = 0
                self.env.apply_action(data, self.action[0:7])
            elif self.env.step_count % self.env.num_phys_steps == 0:
                self.action_index += 1
                if self.action_index == self.env.exec_actions:
                    continue
                self.env.apply_action(data, self.action[7*self.action_index:7*(self.action_index + 1)])

            data , tup = self.env.step(data)

            if tup[2][0] == 1 and self.env.success:
                print("Switching to nominal!")
                self.switch_to_nominal = True
            elif tup[2][0] == 1:
                self.reset(data)
                return data

        return data

    def nominal_step(self, data):

        if self.nominal_step_count > 500:
            if self.replay:
                self.reset(data)
                return data
            else: 
                return data

        if self.target_pos is None:
            tool_pos, tool_orient = self.env.get_tool_pos_orient(data)
            z_axis = tool_orient[:, 2]
            self.target_pos = tool_pos + self.env.success_thresh * 0.50 *z_axis.reshape(3,)
            self.target_orient = tool_orient

        #For now, lets just move to targets

        self.env.move_to_targets(data, self.target_pos, self.target_orient, iterations = 17)

        self.nominal_step_count += 1

        return data

def main():
    global mj_data
    RHZ = 60.0
    
    test = Simulation("./visual_tests_config.yaml", tool_points, task_points)
    
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
   


