import jax.numpy as jnp
import jax
import jax.scipy.spatial
import jax.scipy.spatial.transform
from jax.scipy.spatial.transform import Rotation as R
import flax
import optax
import msgpack
from flax import serialization

#This will be the folder with all the utils for jnp

def extract_pos_orient_keypoints(key_points: jnp.ndarray):
    # Position = mean of the 4 keypoints
    pos = jnp.mean(key_points, axis=2)  # (NUM_ENVS, 3)

    # First column: p0 - p1
    col0 = key_points[:, :, 0] - key_points[:, :, 1]          # (NUM_ENVS, 3)
    col0 = col0 / jnp.linalg.norm(col0, axis=1, keepdims=True)

    # Second column: p2 - p1
    col1 = key_points[:, :, 2] - key_points[:, :, 1]          # (NUM_ENVS, 3)
    col1 = col1 / jnp.linalg.norm(col1, axis=1, keepdims=True)

    # Third column: cross(col0, col1)
    col2 = jnp.cross(col0, col1, axis=1)                      # (NUM_ENVS, 3)

    # Stack columns into orientation matrix
    orient = jnp.stack([col0, col1, col2], axis=2)            # (NUM_ENVS, 3, 3)

    return pos, orient

def create_random_rotations(key, num_envs, orient_noise):

    orient_euler_noise = jax.random.normal(key, (num_envs, 3))

    orient_euler_noise_norm = orient_euler_noise / jnp.linalg.norm(orient_euler_noise, axis=1, keepdims=True)
    orient_euler_noise_norm = orient_noise * orient_euler_noise_norm

    matrix_noise = R.from_euler("xyz", orient_euler_noise_norm).as_matrix()

    return matrix_noise

def generate_random_sphere_point_jax(subkey):
    random_point = jax.random.normal(subkey, shape=(3,)) 
    norm_val = jnp.linalg.norm(random_point)
    
    # is_zero should be a scalar boolean here
    is_zero = norm_val == 0 
    
    normalized_vec = random_point / norm_val
    
    # JAX handles broadcasting the scalar condition to the (3,) vector shapes
    result = jnp.where(
        is_zero,
        jnp.array([0.0, 0.0, 0.0]),
        normalized_vec
    )
    
    # Explicitly return the final result as a (3, 1) column vector
    return result.reshape(3,)

def dist(tool_key_points: jnp.array , task_key_points: jnp.array, extracted = False, debug = False):

    if not extracted:
        tool_key_points, _ = extract_pos_orient_keypoints(tool_key_points)
        task_key_points, _ = extract_pos_orient_keypoints(task_key_points)

    if debug:
        print(f"tool key points : {tool_key_points}  task key points: {task_key_points}")

    dist = jnp.linalg.norm(tool_key_points - task_key_points, axis = 1)

    return dist # NUM_ENVS

def realize_points_in_frame(points, pos, frame):
    shifted = points - pos[:, :, None]      
    frame_T = jnp.transpose(frame, (0,2,1))
    return jnp.matmul(frame_T, shifted)   

def compute_geodesic_distance_batch(orient1, orient2):
    arg = (jnp.einsum('bii->b', orient1 @ jnp.transpose(orient2, (0, 2, 1))) - 1) / 2

    arg = jnp.clip(arg, -1.0, 1.0)

    return jnp.arccos(arg)

def continuous_reward_func_batch(a: float, b: float, x: jnp.array):
    exp_ax = jnp.exp(a * x)
    exp_neg_ax = jnp.exp(-a * x)

    denominator = b + exp_neg_ax + exp_ax

    r = 2 * (b + 2) * (denominator ** -1) - 1
    return r

def compute_sigma_masked(fdim: jnp.ndarray,
                         axes: jnp.ndarray):
    #fdim (NUM_ENVS, ) axes (NUM_ENVS, 3)
    B = axes.shape[0]

    # Projection matrix P = a a^T  per env → (B, 3, 3)
    P = jnp.einsum("bi,bj->bij", axes, axes)

    # Identity per env → (B, 3, 3)
    I = jnp.tile(jnp.eye(3), (B, 1, 1))

    # Broadcastable masks (B,1,1)
    m0 = (fdim == 0)[..., None, None]
    m1 = (fdim == 1)[..., None, None]
    m2 = (fdim == 2)[..., None, None]
    m3 = (fdim == 3)[..., None, None]

    # Start from zeros
    sigmaF = jnp.zeros_like(P)
    sigmaM = jnp.zeros_like(P)

    # Case 1: fdim == 1 → σ_F = P, σ_M = I - P
    sigmaF = sigmaF + m1 * P
    sigmaM = sigmaM + m1 * (I - P)

    # Case 2: fdim == 2 → σ_F = I - P, σ_M = P
    sigmaF = sigmaF + m2 * (I - P)
    sigmaM = sigmaM + m2 * P

    # Case 3: fdim == 3 → σ_F = I, σ_M = 0
    sigmaF = sigmaF + m3 * I
    # sigmaM unchanged for case 3 (stays 0 there)

    # Case 0: fdim == 0 → σ_F = 0, σ_M = I
    sigmaM = sigmaM + m0 * I

    return sigmaM, sigmaF


def check_out_of_bounds_batch(male_points: jnp.ndarray, zone: dict) -> jnp.ndarray:
    
    center = jnp.array(zone["center"])  # (3,)

    if zone["shape"] == "cyl":
        in_z_bound = jnp.abs(male_points[:, 2] - center[2]) < (zone["height"]/2)

        delta_xy = male_points[:, :2] - center[:2]
        in_base = jnp.linalg.norm(delta_xy, axis=1) < zone["radius"]
        out_of_bounds = ~(in_z_bound & in_base)

    elif zone["shape"] == "sphere":
        delta = male_points - center
        dist = jnp.linalg.norm(delta, axis=1)
        out_of_bounds = dist > zone["radius"]

    else:
        raise ValueError(f"Unknown zone shape {zone['shape']}")

    return out_of_bounds

class RewardModuleJax():

    def __init__(self, dist_weight, orient_weight,alpha, beta, dist_scale, orient_scale, success_thresh, task_points): 

        self.dist_weight = dist_weight
        self.orient_weight = orient_weight
        self.alpha = alpha 
        self.beta = beta
        self.dist_scale = dist_scale 
        self.orient_scale = orient_scale
        self.success_thresh = success_thresh
        self.task_points_noise = task_points
        self.task_pos , self.task_ori = extract_pos_orient_keypoints(self.task_points_noise)

    def batched_reward(self, tool_points, out_of_bounds):
        pos_tool , ori_tool = extract_pos_orient_keypoints(tool_points)

        angle_rad = compute_geodesic_distance_batch(ori_tool, self.task_ori)
        dist_batch = dist(pos_tool, self.task_pos, extracted = True)

        dist_norm = dist_batch/self.dist_scale
        angle_norm = angle_rad/self.orient_scale

        total_error = dist_norm * self.dist_weight + angle_norm * self.orient_weight

        cont_reward = continuous_reward_func_batch(self.alpha, self.beta, total_error)

        success_reward = jnp.zeros(angle_rad.shape[0])
        success_reward = jnp.where(dist_batch < self.success_thresh, 100 , success_reward)

        oob_penalty = jnp.zeros(success_reward.shape[0])
        oob_penalty = jnp.where(out_of_bounds, -100, oob_penalty)

        return cont_reward + success_reward + oob_penalty

    
    def dist(self, tool_key_points, extracted = False):

        if not extracted:
            tool_key_points, _ = extract_pos_orient_keypoints(tool_key_points)
            

        dist = jnp.linalg.norm(tool_key_points - self.task_pos, axis = 1)

        return dist
    
@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict

from flax.training import train_state

@flax.struct.dataclass
class AgentState:
    params: AgentParams
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    opt_state: optax.OptState
    step: int = 0

#This is the object which we will use to save the agent state
@flax.struct.dataclass
class InferenceState:
    params: AgentParams
    step: int = 0   # optional

def create_agent_state(agent_params, tx):
    opt_state = tx.init(agent_params)
    return AgentState(
        params=agent_params,
        tx=tx,  # now static
        opt_state=opt_state,
        step=0,
    )

def apply_agent_gradients(agent_state: AgentState, grads: AgentParams) -> AgentState:
    updates, new_opt_state = agent_state.tx.update(
        grads,
        agent_state.opt_state,
        agent_state.params,
    )
    new_params = optax.apply_updates(agent_state.params, updates)
    return agent_state.replace(
        params=new_params,
        opt_state=new_opt_state,
        step=agent_state.step + 1,
    )

def save_inference_state(path: str, agent_state: AgentState):
    ckpt = InferenceState(
        params=agent_state.params,
        step=agent_state.step,
    )
    bytes_out = flax.serialization.to_bytes(ckpt)  # ✅ handles jax.Array
    with open(path, "wb") as f:
        f.write(bytes_out)

def load_inference_state(path: str, template_params: AgentParams) -> InferenceState:
    # IMPORTANT: template must match what was saved (InferenceState)
    template = InferenceState(params=template_params, step=0)

    with open(path, "rb") as f:
        bytes_in = f.read()

    return flax.serialization.from_bytes(template, bytes_in)







        











    
    
    