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

#define the fspf test over here

def send_data(fspf_server, motion_control, force_control, measured_vel, measured_force):

    print("resetting the fspf data: ")

    data = np.ones(1)
    packed_data = struct.pack('f' * data.shape[-1], *data)

    fspf_server.send(packed_data)
    reply = fspf_server.recv()

    print("reset the fspf server!!!")

    for i in range(1000):
    
        target_data = np.hstack((motion_control, force_control, measured_vel, measured_force))
        target_data = np.tile(target_data, (1,))
        target_data_packed = struct.pack('f' * 12 * 1, *(target_data))
        fspf_server.send(target_data_packed)
        reply = fspf_server.recv()

        reply_np  = np.frombuffer(reply, dtype = np.float32)

        print(f"Here is the reply: {reply_np}")

    print("Fspf test is now done")

ctx = zmq.Context()

fspf_socket = ctx.socket(zmq.REQ)
fspf_socket.connect("ipc:///tmp/zmq_fspf_server")

# measured force: [  5.71212318   5.1516309  -36.97502073], measured_velocity: [-0.00328178  0.0040331  -0.00038066], dx_world: [ 0.     0.    -0.005], force_control: [0. 0. 0.]
# motion or force axis: [[0.]

send_data(fspf_socket, np.array([0,0,-0.5]), np.array([0,0,0]), np.random.uniform(low = -0.001, high = 0.001, size = 3), measured_force= np.array([5,5, -36]))

