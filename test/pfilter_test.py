import zmq
import struct
import numpy as np
import threading
import time
import random

NUM_WORKERS = 1
FREQ_HZ = 50
INTERVAL = 1.0 / FREQ_HZ  # seconds per request
DURATION_SEC = 10         # test duration per worker
SERVER_ENDPOINT = "ipc:///tmp/zmq_pf_server"  # or "ipc:///tmp/zmq_torque_server"
num_envs = 5

"""
    Test particle filter server
    - Set commanded motion and force data, and measured velocity and force data
    - Get force dimension size and force_or_motion_axis
    - Reset filter after set duration
"""
NOMINAL_FORCE_DIRECTION = np.array([0, 0, 1])
NOMINAL_FORCE_MAGNITUDE = 5
NOMINAL_VELOCITY_DIRECTION = np.array([1, 0, 0]) 

def makeSimulatedData():
    force_noise = [random.gauss(mu=0, sigma=0.1) for i in range(3)]
    vel_noise = [random.gauss(mu=0, sigma=0.01) for i in range(3)]

    commanded_motion = NOMINAL_FORCE_DIRECTION.astype(np.float32)
    commanded_force = NOMINAL_FORCE_DIRECTION.astype(np.float32)
    measured_velocity = (NOMINAL_VELOCITY_DIRECTION + vel_noise).astype(np.float32)
    measured_force = (NOMINAL_FORCE_MAGNITUDE * NOMINAL_FORCE_DIRECTION + force_noise).astype(np.float32)    

    target_data = np.hstack((commanded_motion, commanded_force, measured_velocity, measured_force))
    target_data = np.tile(target_data, (num_envs,))

    # print(target_data)

    return struct.pack('f' * 12 * num_envs, *(target_data))

def main():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect(SERVER_ENDPOINT)

    # Reset
    socket.send(struct.pack('f' * 1, *[1]))
    reply = socket.recv()
    print('Reset filter')

    count = 0
    start = time.time()
    end_time = start + DURATION_SEC
    while time.time() < end_time:
        st = time.time()
        packed = makeSimulatedData()
        socket.send(packed)
        reply = socket.recv()
        count += 1

        filter_output = np.frombuffer(reply, dtype=np.float32)
        print(filter_output[0:4])  # extra stuff at the end of reply?

    print("Total calls in 10 secs: {}, Avg calls per second: {}".format(count, count/10.0))

    # Reset
    socket.send(struct.pack('f' * 1, *[1]))
    print('Reset filter')

if __name__ == "__main__":
    main()