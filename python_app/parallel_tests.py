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
from parallel_env import ParallelEnv, NUM_ENVS, get_force_batch

from parallel_env import step_mjx_env_batch


mj_model = mujoco.MjModel.from_xml_path("../models/scenes/fr3squarehole.xml")
mj_data = mujoco.MjData(mj_model)
home_key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
home_qpos = mj_model.key_qpos[home_key_id]
HOME_QPOS = jnp.array(home_qpos)

naive_step = jax.jit(jax.vmap(mjx.step, in_axes = (None, 0)))

#First I need to figure out how to unit test the quick reset --- That would be good 

def generate_parallel_env():

    params = params_from_yaml("./experiment.yaml")

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, )

    batched_data = jax.vmap(lambda _: mjx_data.replace(qpos=HOME_QPOS))(rng)

    ctx = zmq.Context()
    mft_socket = ctx.socket(zmq.REQ)
    mft_socket.connect("ipc:///tmp/zmq_motion_force_server")

    fspf_socket = ctx.socket(zmq.REQ)
    fspf_socket.connect("ipc:///tmp/zmq_fspf_server")

    env = ParallelEnv([mft_socket, fspf_socket], mjx_model, mj_model, params)

    return env, batched_data

# contactRichEnv, batched_data = generate_parallel_env()

def test_move_to_targets(env: ParallelEnv, batched_data):

    mask = jnp.zeros(NUM_ENVS)

    mask = mask.at[1].set(1)

    env.target_pos = jnp.tile(jnp.array([0.4, 0.4, 0.5]), (NUM_ENVS, 1))
    env.target_orient = jnp.tile(jnp.array([[1,0,0], [0, -1, 0], [0,0,-1]]), (NUM_ENVS, 1,1))

    batched_data = env.move_to_targets(batched_data, mask,  iterations = 500)

    print("contact rich env pos ori end effector: ", env.get_tool_pos_orient(batched_data))
    print("tool points and positions -------- ")

    pass

def test_sample_observation(env: ParallelEnv, batched_data):

    env.target_pos = jnp.tile(jnp.array([0.4, 0.4, 0.5]), (NUM_ENVS, 1))
    env.target_orient = jnp.tile(jnp.array([[1,0,0], [0, -1, 0], [0,0,-1]]), (NUM_ENVS, 1,1))

    mask = jnp.zeros(NUM_ENVS)
    batched_data = env.move_to_targets(batched_data, mask, iterations = 200)

    obs = env.sample_observation(batched_data)

    print(f"The observation sampled is : {obs}")

def test_init_quick_reset_pos(env: ParallelEnv, batched_data):

    batched_data = env.init_quick_reset_pos(batched_data)

    #let us check the positions and orientations of the batched_data to see what it is like

    print("target positions are: {}, {}, tool positions are: {}".format(env.tool_frame_pos, env.tool_frame_orient, env.get_tool_pos_orient(batched_data)))

def test_find_success_thresh(env: ParallelEnv, batched_data):

    env.find_success_thresh(batched_data) #have to run this test again

    print("Contact rich success thresh is: ", env.success_thresh)
    print(f"dist scale for each env is: {env.dist_scale}")


def test_get_first_obs():

    pass

#This test works, I am accurately extracting pos and orientations with the extract_pos_orient_keypoints method,
#and the method of adding noise is working as well
def test_extract_orientations():

    master_key = jax.random.PRNGKey(42)
    key_list = jax.random.split(master_key, num = 10)

    tn = 0.05
    on = 0.02

    points = np.array([[1,1, 0], [-1, 1, 0], [-1,-1, 0], [1, -1, 0]]).T
    points = np.tile(points, (NUM_ENVS, 1,1))

    random_noise = jax.random.normal(key_list[0], (NUM_ENVS, 3))
    trans_noise = random_noise / jnp.linalg.norm(random_noise, axis=1, keepdims=True)

    trans_noise = tn* trans_noise.reshape((NUM_ENVS, 3, 1))

    rotation_noise = create_random_rotations(key_list[1], NUM_ENVS, on)

    points_pos, points_orient = extract_pos_orient_keypoints(points)

    print(f"points pos : {points_pos}, points_orient : {points_orient}")

    points_orient_prime = rotation_noise @ points_orient
    points_pos_prime = trans_noise + points_pos.reshape(NUM_ENVS, 3,1)

    print(f"points pos prime: {points_pos_prime}, orient_prime: {points_orient_prime}")

    new_points = points_orient_prime @ points + points_pos_prime

    print(f"The new points we have is : {new_points}")

def test_create_noisy_task_points(env: ParallelEnv, batched_data):
    env.create_noisy_task_points(batched_data)
    print(f"task points noise generated are: {env.task_points_noise}")
    pos_task, ori_task = extract_pos_orient_keypoints(env.task_points_noise)
    print(f"task point noise position and orientation are , {pos_task} , {ori_task}")

def test_create_start_frames(env: ParallelEnv, batched_data):

    env.create_start_frames()

    print(f"ENV start frames: {env.tool_frame_pos}, {env.tool_frame_orient}")

def test_check_out_of_bounds_batch(env: ParallelEnv, batched_data):

    zone = {"shape": "cyl", "center": [0,0,0], "radius": 0.03, "height": 0.5}

    tool_pos_in_frame = jnp.array([[0.02, 0.02 , 0.1], [0.2, 0.2, 0.2]])

    results = check_out_of_bounds_batch(tool_pos_in_frame, zone)

    print(f"Out of bounds results are : {results}")

def test_update_fspf(env: ParallelEnv, batched_data):

    mask = jnp.ones(NUM_ENVS)

    batched_data, _ = env.reset(batched_data, mask) 

    env.target_pos = jnp.tile(jnp.array([0.4, 0, 0.2]), (NUM_ENVS, 1))
    env.target_orient = jnp.tile(jnp.array([[1,0,0], [0, -1, 0], [0,0,-1]]), (NUM_ENVS, 1,1))

    iter = 0

    while iter < 2000:

        if iter % 50 == 0:
            env.dx_world = jnp.array([[0, 0,-0.05], [0, 0,-0.05]])
            mofa, fdim = env.update_fspf_data(batched_data)
            print(f"fdim we are getting is: {fdim}, force reading : {env.get_force_data(batched_data)}")
            print(f"position of tool: {env.get_tool_pos_orient(batched_data)[0]}")

        batched_data = env.move_to_targets(batched_data, mask , iterations = 1)

        iter += 1

def test_apply_action(env: ParallelEnv, batched_data):


    action = jnp.array([[0, 0, 0.01, 0,0, 0, 0],[0, -0.01, -0.01,0,0,0,0]])

    mask = jnp.ones(NUM_ENVS)

    batched_data, _ = env.reset(batched_data, mask) 

    print(f"env tool points pos: {env.get_tool_pos_orient(batched_data)[0]}")

    env.apply_action(batched_data, action)

    batched_data = env.move_to_targets(batched_data, mask, iterations = 500)

    print(f"env tool points pos prime: {env.get_tool_pos_orient(batched_data)[0]}")

def test_init_env_params(env: ParallelEnv, batched_data):

    env.init_env_params(batched_data)


def run_tests():
    env, batched_data = generate_parallel_env()

    batched_data = env.dry_start(batched_data)

    # print("Successfully generated contactRichEnv and the data to iterate on!")

    # test_move_to_targets(contactRichEnv, batched_data)

    # test_extract_orientations()
    # test_create_noisy_task_points(env, batched_data)
    
    # test_create_start_frames(env, batched_data)

    # test_sample_observation(env, batched_data)

    # test_check_out_of_bounds_batch(env, batched_data)

    # test_move_to_targets(env, batched_data)

    # test_init_quick_reset_pos(env, batched_data)

    # test_apply_action(env, batched_data)

    test_init_env_params(env, batched_data)

    # test_update_fspf(env, batched_data)

    # test_find_success_thresh(env , batched_data)

run_tests()

    
