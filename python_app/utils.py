import os
import yaml
from scipy.spatial.transform import Rotation as R
import numpy as np

def count_subdirectories(path):
    if not os.path.isdir(path):
        print(f"Error: '{path}' is not a valid directory.")
        return -1

    count = 0
    for entry_name in os.listdir(path):
        full_path = os.path.join(path, entry_name)
        if os.path.isdir(full_path):
            count += 1
    return count

def compute_minimum_angle_rotation_between_orientations(orient1, orient2):

    quat1 = R.from_matrix(orient1).as_quat()
    quat2 = R.from_matrix(orient2).as_quat()

    dot = np.abs(np.dot(quat1, quat2))
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = 2 * np.arccos(dot) 

    return angle_rad

def params_from_yaml(yaml_path : str):

    with open(yaml_path, 'r') as stream:
        params = yaml.safe_load(stream)

    return params

    



    




