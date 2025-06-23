import numpy as np
import redis
import time
from enum import Enum
import json

redis_client = redis.Redis()

class RedisKeys(Enum):
    CURRENT_CARTESIAN_POSITION = "rizon4s::current_cartesian_position"
    CURRENT_CARTESIAN_ORIENTATION = "rizon4s::current_cartesian_orientation"
    CURRENT_GRIPPER_POSITION = "rizon4s::current_gripper_position"

    DESIRED_CARTESIAN_POSITION = "rizon4s::desired_cartesian_position"
    DESIRED_CARTESIAN_ORIENTATION = "rizon4s::desired_cartesian_orientation"
    DESIRED_GRIPPER_POSITION = "rizon4s::desired_gripper_position"


#There will be 4 states, idle, moving to grasp, grasp, 

def main():

    # print("Starting the control simulation on the python end...")

    STATE = "IDLE"

    while STATE != "FINISH":
        if STATE == "IDLE":
            moveToPos(np.array([0.5, 0.5, 0.5]))
            setGripper(150)
            STATE = "GRASP"
        elif STATE == "GRASP":
            moveToPos(np.array([0.4, 0.4, 0.4]))
            setGripper(0)
            STATE = "MOVE_TO_FEMALE"
        elif STATE == "MOVE_TO_FEMALE":
            moveToPos(np.array([0.3,0.3,0.3]))
            STATE = "FINISH"


def moveToPos(desired_pos: np.array, desired_orient = np.array([[1,0,0], [0, -1, 0], [0, 0, -1]]), max_iters = 100):

    redis_client.set(RedisKeys.DESIRED_CARTESIAN_POSITION.value, json.dumps(desired_pos.tolist()))
    redis_client.set(RedisKeys.DESIRED_CARTESIAN_ORIENTATION.value, json.dumps(desired_orient.tolist()))

    for i in range(max_iters):
        curr_pos = np.array(json.loads(redis_client.get(RedisKeys.CURRENT_CARTESIAN_POSITION.value)))
        curr_orient = np.array(json.loads(redis_client.get(RedisKeys.CURRENT_CARTESIAN_ORIENTATION.value)))

        if np.linalg.norm(curr_pos - desired_pos) < 1e-2:
            return
        
        time.sleep(0.01)

def setGripper(gripperVal: int):
    redis_client.set(RedisKeys.DESIRED_GRIPPER_POSITION.value, gripperVal)

if __name__ == "__main__":
    main()