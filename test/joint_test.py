import zmq
import struct
import numpy as np
import threading
import time

NUM_WORKERS = 1
FREQ_HZ = 1000
INTERVAL = 1.0 / FREQ_HZ  # seconds per request
DURATION_SEC = 10         # test duration per worker
SERVER_ENDPOINT = "ipc:///tmp/zmq_torque_server"  # or "ipc:///tmp/zmq_torque_server"
num_envs = 5

#in the zero mq test, I will launch requests that are Identical to the. mj envs
def make_joint_data():
    target_data = np.array([0.1,0.1,0.1, 1,0,0,0]).astype(np.float32)
    target_data = np.tile(target_data, (num_envs,))
    joint_pos = np.random.rand(7 * num_envs).astype(np.float32)
    joint_vel = np.random.rand(7 * num_envs).astype(np.float32)
    return struct.pack('f' * 21 * num_envs, *(np.concatenate([target_data, joint_pos, joint_vel])))
def main():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect(SERVER_ENDPOINT)
    count = 0
    start = time.time()
    end_time = start + DURATION_SEC
    while time.time() < end_time:
        st = time.time()
        packed = make_joint_data()
        socket.send(packed)
        reply = socket.recv()
        count += 1
    print("Total calls in 10 secs: {}, Avg calls per second: {}".format(count, count/10.0))
if __name__ == "__main__":
    main()