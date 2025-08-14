from chex import params_product
from env import ZMQEnv
from policy_network import Agent
import torch
import mujoco
import zmq
import numpy as np
import math
import glfw
import time
from utils import params_from_yaml

mj_xml_path = "/Users/rahulavasarala/Desktop/OpenSai/WireConnectAgent/models/scenes/rizon4smaleconjl.xml"
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

class AgentRolloutTest():

    def __init__(self, weights_path, eval_pos_orient):

        self.step_count = 0
        self.wait_count = 0
        self.env = ZMQEnv(params, mj_model,mj_data, jt_socket, mft_socket, fspf_socket, eval_pos_orient=eval_pos_orient)
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

        self.wait_count += 1

        if self.wait_count < 120:
            print("waiting")
            return data, None

        if self.step_count >= 10:
            obs, _, _ , _ = self.env._get_obs(data)
            obs = torch.Tensor(obs).to(self.device)
            action = self.sample_action(obs)
            action = action.cpu().numpy()
            action = action.reshape(7,)
            print("action taken: {}".format(action))
            self.env.apply_action(data, action)

            self.step_count = 0

        for _ in range(17):
            data , _ = self.env.step(data)

        self.step_count += 1

        return data, 1
    
class RandomStepTest():

    def __init__(self):

        self.step_count = 0
        self.env = ZMQEnv(params, mj_model,mj_data, jt_socket, mft_socket, fspf_socket)
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
        self.env = ZMQEnv(params, mj_model,mj_data, jt_socket, mft_socket, fspf_socket, eval_pos_orient=np.array([0.3, 0.3, 0.1, 0,1,0,0]))
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

        for _ in range(17):
            data = self.env.move_to_targets(data)

        data = self.env.show_keypoints(data)

        male_pos, _ = self.env.get_male_connector_pos_orient(data, "world")

        print("difference between target position and actual male connector position: {}".format(np.linalg.norm(self.target_point - male_pos)))

        self.step_count += 1

        return data, _
    
class FSPFTest():

    def __init__(self):
        self.step_count = 0
        self.env = ZMQEnv(params, mj_model,mj_data, jt_socket, mft_socket, fspf_socket, eval_pos_orient = np.array([0.3, 0.3, 0.1, 0,1,0,0]))
        self.env.set_mode("eval")

        self.action = np.zeros(3)

        self.env.saturation_radius = 1
        self.env.saturation_theta = math.pi

    def reset(self, data):
        self.env.reset(data)
        self.env.set_target_pos_orient(np.array([0.3, 0.4, -0.03]), np.array([0,1,0,0]))

    def step(self, data):

        curr_pos_male, _ = self.env.get_male_connector_pos_orient(data, "frame")
        self.dx = self.env.target_pos - curr_pos_male
    
        for _ in range(17):
            data = self.env.move_to_targets(data)
            if self.env.step_count % 50 == 0:
                _, fdim = self.env.update_fspf_server(data, self.dx)
                print("Fdim : {}".format(fdim))

        force_world, _ = self.env.get_force_torque_sensor_reading(data, "world")

        print("Force sensor data in world frame is: {}".format(force_world))
        print("Velocity reading is: {}".format(self.env.get_velocity_reading(data, "world")))

        self.step_count += 1

        return data, _
    
def main():
    global mj_data
    RHZ = 60.0
    launch_params = params_from_yaml("./visual_tests_launch.yaml")
    test_to_run = launch_params["test"]
    run_folder = f"{launch_params["run_dir"]}/run_{launch_params["run_name"]}"
    
    if test_to_run == "Agent":
        loaded_pos_orient = np.loadtxt(f'{run_folder}/start_pos_orient.txt')
        test = AgentRolloutTest(f"{run_folder}/model{launch_params["iter"]}.cleanrl_model", loaded_pos_orient)
    elif test_to_run == "Random":
        test = RandomStepTest()
    elif test_to_run == "ShowPoints":
        test = ShowPointsTest()
    elif test_to_run == "FSPF":
        test = FSPFTest()

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
   


