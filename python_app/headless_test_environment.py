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

ROBOT_JOINTS = 7
NUM_ENVS = 1
mj_model = mujoco.MjModel.from_xml_path("../models/scenes/fr3peghole_modified.xml")
mj_data = mujoco.MjData(mj_model)
mujoco.mj_step(mj_model, mj_data)
link7_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link7")

#Then run this same test with the set q pos home joints

from utils import get_joint_torques

target_pos = np.array([0.3,0.4,0.3])
target_orient = np.array([[1,0,0], [0, -1, 0], [0, 0, -1]])

def move_to_targets(mj_model, data, jt_server, iter = 5000):

    for _ in range(iter):
        data.ctrl[:] = get_joint_torques(data, target_pos, target_orient, jt_server)
        mujoco.mj_step(mj_model, data)

    return data



ctx = zmq.Context()
jt_socket = ctx.socket(zmq.REQ)
jt_socket.connect("ipc:///tmp/zmq_torque_server")

mj_data = move_to_targets(mj_model, mj_data, jt_socket)

print(f"Link 7 position: {mj_data.xpos[link7_id]}")

