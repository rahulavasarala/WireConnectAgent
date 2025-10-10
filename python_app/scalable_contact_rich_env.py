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

from utils import send_and_recieve_to_server

ROBOT_JOINTS = 7
NUM_ENVS = 1
mj_model = mujoco.MjModel.from_xml_path("../models/scenes/fr3peghole_modified.xml")
mj_data = mujoco.MjData(mj_model)
link7_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link7")
home_key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
home_qpos = mj_model.key_qpos[home_key_id]
HOME_QPOS = jnp.array(home_qpos)

def rotmat_to_quat(R: jnp.ndarray) -> jnp.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    trace = jnp.trace(R)

    def case1(_):
        qw = jnp.sqrt(1.0 + trace) / 2.0
        qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
        return jnp.array([qx, qy, qz, qw])

    def case2(_):
        cond_x = (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2])
        cond_y = R[1, 1] > R[2, 2]

        def branch_x(_):
            s = jnp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
            return jnp.array([qx, qy, qz, qw])

        def branch_y(_):
            s = jnp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
            return jnp.array([qx, qy, qz, qw])

        def branch_z(_):
            s = jnp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
            return jnp.array([qx, qy, qz, qw])

        return jax.lax.cond(cond_x, branch_x, 
                            lambda _: jax.lax.cond(cond_y, branch_y, branch_z, None),
                            None)

    return jax.lax.cond(trace > 0.0, case1, case2, None)


def get_joint_torques(targets_and_values: jnp.array, jt_server) -> jnp.array:

    targets_and_values_cpu = np.array(targets_and_values) 
    stacked_data = targets_and_values_cpu.flatten()
    joint_torques = send_and_recieve_to_server(stacked_data, jt_server)

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

    target_orient_quat = rotmat_to_quat(target_orient)
    target_orient_quat = target_orient_quat.flatten()

    # print(f"qpos size: {qpos.shape}")

    return jnp.concatenate([target_pos, target_orient_quat, qpos, qvel])


get_joint_torque_targets_and_values_batch =  jax.jit(jax.vmap(get_joint_torque_targets_and_values, in_axes = (0,0,0)))

def get_qpos(data: mjx.Data):
    return data.qpos

get_qpos_batch = jax.jit(jax.vmap(get_qpos))

def get_eepos(data: mjx.Data):
    return data.xpos[link7_id]

get_eepos_batch = jax.jit(jax.vmap(get_eepos))

def get_qvel(data: mjx.Data):
    return data.qvel

get_qvel_batch = jax.jit(jax.vmap(get_qvel))

class ParallelContactRichEnv:

    def __init__(self ,servers, mjx_model, model):

        self.mjx_model = mjx_model

        self.jt_server = servers[0]

        self.target_pos = jnp.tile(jnp.array([0.3, 0.4, 0.3]), (NUM_ENVS, 1))
        self.target_orient = jnp.tile(jnp.array([[1,0,0], [0, -1, 0], [0,0,-1]]), (NUM_ENVS, 1,1))

    def move_to_targets(self, data, iterations = 5000):

        for i in range(iterations):
            targets_and_values = get_joint_torque_targets_and_values_batch(data, self.target_pos, self.target_orient)
            # print(f"size targets and values: {targets_and_values} {i}")
            # # q_vel = self.get_qvel(data)
            # # print(f"qvel : {q_vel}")
            joint_torques = get_joint_torques(targets_and_values, self.jt_server)
            # print(f"Joint torques: {joint_torques}")
            # joint_torques = jnp.zeros((NUM_ENVS, ROBOT_JOINTS))
            data = step_mjx_env_batch(self.mjx_model, data, joint_torques)
            # data = naive_step(self.mjx_model, data)
            
            # print(f"ee pos at iter {i} = {ee_pos}")

        return data
    
    def get_qpos(self, data):
        return get_qpos_batch(data)
    
    def get_eepos(self, data):
        return get_eepos_batch(data)
    
    def get_qvel(self,data):
        return get_qvel_batch(data)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

rng = jax.random.PRNGKey(0)
rng = jax.random.split(rng, NUM_ENVS)
# batched_data = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (ROBOT_JOINTS,))))(rng)

batched_data = jax.vmap(lambda _: mjx_data.replace(qpos=HOME_QPOS))(rng)

ctx = zmq.Context()
jt_socket = ctx.socket(zmq.REQ)
jt_socket.connect("ipc:///tmp/zmq_torque_server")

contactRichEnv = ParallelContactRichEnv([jt_socket], mjx_model, mj_model)
q_pos = contactRichEnv.get_qpos(batched_data)
ee_pos = contactRichEnv.get_eepos(batched_data)

print(f"ee pos beginning = {ee_pos}")
print(f"q pos beginning = {q_pos}")

batched_data = contactRichEnv.move_to_targets(batched_data)

q_pos = contactRichEnv.get_qpos(batched_data)
ee_pos = contactRichEnv.get_eepos(batched_data)

print(f"ee pos final = {ee_pos}")

print("Finished!!!!!")

















    



        
        


    



    

