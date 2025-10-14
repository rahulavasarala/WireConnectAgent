import jax.numpy as jnp
import jax

#This will be the folder with all the utils for jnp

def extract_pos_orient_keypoints(key_points: jnp.ndarray):

    pos = jnp.mean(key_points, axis = 1)

    orient = jnp.zeros((3,3))

    orient[:,0] = key_points[:,0] - key_points[:,1]
    orient[:, 0] = orient[:, 0]/jnp.linalg.norm(orient[:,0])
    orient[:,1] = key_points[:,2] - key_points[:,1]
    orient[:,1] = orient[:,1]/jnp.linalg.norm(orient[:,1])
    orient[:, 2] = jnp.cross(orient[:,0], orient[:,1]) 

    return pos, orient

def generate_random_sphere_point(subkey):
    random_point = jax.random.randint(subkey, shape=(3,), minval=-1, maxval=1)

    norm_val = jnp.linalg.norm(random_point)

    if norm_val == 0:
        return jnp.array([0.0, 0.0, 0.0])

    unit_vector = random_point / norm_val
    return unit_vector

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

def euler_to_rotmat(euler_angles: jnp.ndarray) -> jnp.ndarray:
    #Converts 3D Euler angles (Roll, Pitch, Yaw) in radians to a 3x3 rotation matrix.
    # Unpack angles
    roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]
    
    # Calculate trigonometric values
    c_r, s_r = jnp.cos(roll), jnp.sin(roll)    # cos(alpha), sin(alpha)
    c_p, s_p = jnp.cos(pitch), jnp.sin(pitch)  # cos(beta), sin(beta)
    c_y, s_y = jnp.cos(yaw), jnp.sin(yaw)      # cos(gamma), sin(gamma)
    
    R00 = c_y * c_p
    R01 = c_y * s_p * s_r - s_y * c_r
    R02 = c_y * s_p * c_r + s_y * s_r
    
    R10 = s_y * c_p
    R11 = s_y * s_p * s_r + c_y * c_r
    R12 = s_y * s_p * c_r - c_y * s_r
    
    R20 = -s_p
    R21 = c_p * s_r
    R22 = c_p * c_r
    

    R = jnp.array([
        [R00, R01, R02],
        [R10, R11, R12],
        [R20, R21, R22]
    ])
    
    return R