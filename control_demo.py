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
    RESET = "rizon4s::reset"
    VEL_SATURATION = "rizon4s::velocity_saturation"


#There will be 4 states, idle, moving to grasp, grasp, 

def main():

    # print("Starting the control simulation on the python end...")

    reset()

    time.sleep(1)

    STATE = "IDLE"



    # setVelocitySaturation(np.array([0.2, 0.5]))

    moveToPos(np.array([0.4,0.4,0.12]))
    # time.sleep(1)
    # moveToPos(np.array([0,0.5,0.08]))
    # time.sleep(1)
    # closeGripperSmooth()
    # time.sleep(1)
    # moveToPos(np.array([0, 0.5, 0.5]))
   
    # calibration_test()

    # while STATE != "FINISH":
    #     if STATE == "IDLE":
    #         moveToPos(np.array([0.3, 0.5, 0.1]))
    #         moveToPos(np.array([0.3, 0.5, 0.07]))
    #         setGripper(225)
    #         time.sleep(1)
    #         STATE = "GRASP"
    #     elif STATE == "GRASP":
    #         # moveToPos(np.array([0.3, 0.5, 0.1]))
    #         # moveToPos(np.array([0.395, 0.495, 0.1]))
    #         # moveToPos(np.array([0.395, 0.495, 0.03]))
    #         # setGripper(0)
    #         STATE = "MOVE_TO_FEMALE"
    #     elif STATE == "MOVE_TO_FEMALE":
    #         # moveToPos(np.array([0.3,0.3,0.3]))
    #         STATE = "FINISH"


def moveToPos(desired_pos: np.array, desired_orient = np.array([[1,0,0], [0, -1, 0], [0, 0, -1]]), max_iters = 200):

    redis_client.set(RedisKeys.DESIRED_CARTESIAN_POSITION.value, json.dumps(desired_pos.tolist()))
    redis_client.set(RedisKeys.DESIRED_CARTESIAN_ORIENTATION.value, json.dumps(desired_orient.tolist()))

    for i in range(max_iters):
        curr_pos = getCurrentPosition()

        if np.linalg.norm(curr_pos - desired_pos) < 1e-2:
            return
        
        time.sleep(0.01)

def getCurrentPosition() -> np.array:
    return np.array(json.loads(redis_client.get(RedisKeys.CURRENT_CARTESIAN_POSITION.value)))

def closeGripperSmooth():

    for i in range(225):
        redis_client.set(RedisKeys.DESIRED_GRIPPER_POSITION.value, i)
        time.sleep(0.01)

def openGripperSmooth():

    for i in range(225,0, -1):
        redis_client.set(RedisKeys.DESIRED_GRIPPER_POSITION.value, i)
        time.sleep(0.01)

def setGripper(gripperVal: int):

    redis_client.set(RedisKeys.DESIRED_GRIPPER_POSITION.value, gripperVal)

def reset():
    redis_client.set(RedisKeys.RESET.value, 1)

def setVelocitySaturation(vel_saturation: np.array):
    redis_client.set(RedisKeys.VEL_SATURATION.value, json.dumps(vel_saturation.tolist()))

def calibration_test():
    positions = [np.array([0.4,0.4,0.4]), np.array([0.2,0.2,0.2]), np.array([0.3,0.3,0.3])]

    error = 0
    setGripper(225)
    for pos in positions:
        moveToPos(pos)
        
        time.sleep(2)

        local_error = np.linalg.norm(pos - getCurrentPosition())

        print("local error is : {}".format(error))

        error = error + local_error

    print("total Error is : {}".format(error))



if __name__ == "__main__":
    main()