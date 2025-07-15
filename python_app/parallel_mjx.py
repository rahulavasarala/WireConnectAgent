# ----------------------------------- All Imports --------------------------------------- #
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os
import random
import time
from dataclasses import dataclass
from typing import Sequence
import struct

import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter
from flax.core import FrozenDict

#@title Import MuJoCo, MJX, and Brax
from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict


import jax
from jax import numpy as jp
import jax.tree_util as jtu
import numpy as np
from flax.training import orbax_utils
from matplotlib import pyplot as plt
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model
import zmq
from distrax import Normal

#---------------------------------------- All imports --------------------------------------------------#

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"

#Define mjx models
mj_model_path = "/Users/rahulavasarala/Desktop/OpenSai/WireConnectAgent/models/scenes/rizon4s.xml"
mj_model = mujoco.MjModel.from_xml_path(mj_model_path)
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

#Define sockets for zmq
SERVER_ENDPOINT = "ipc:///tmp/zmq_torque_server"
ctx = zmq.Context()
socket = ctx.socket(zmq.REQ)
socket.connect(SERVER_ENDPOINT)

#Define environment for fast stepping
def _reset_env(state: mjx.Data, init_orientation: jp.ndarray):
    qpos_init = mjx_model.qpos0
    qvel_init = jp.zeros(qpos_init.shape)

    return state.replace(qpos = qpos_init, qvel = qvel_init)

_jit_reset_env = jax.jit(jax.vmap(_reset_env))

def _apply_joint_torques_and_step_env(state: mjx.Data, joint_torque: jp.ndarray):
    ctrls = state.ctrl
    ctrls = ctrls.at[:7].set(joint_torque)
    state = state.replace(ctrl = ctrls)

    new_state = mjx.step(mjx_model, state)

    return new_state

_jit_apply_joint_torques_and_step_env = jax.jit(jax.vmap(_apply_joint_torques_and_step_env))

class CustomZMQEnvironment:
    def __init__(self, zmq_socket, n_physics_steps, init_des_cart_pos = jp.zeros((3,)), num_envs = 5, action_space_dim = 7, observation_space_dim = 10):
        self.num_envs = num_envs
        self.no_force_count = jp.zeros(num_envs)
        self.env_ids = jp.arange(num_envs)

        self.zmq_socket = zmq_socket

        self.n_frames = n_physics_steps
        self.init_orientations = jp.zeros((self.num_envs, 4))

        self.des_cart_pos = jp.tile(init_des_cart_pos, (self.num_envs, 1))
        self.des_cart_orient = jp.tile(jp.array([1,0,0,0]), (self.num_envs, 1))

        self.action_space_dim = action_space_dim
        self.observation_space_dim = observation_space_dim

    def reset(self, states):
        states = _jit_reset_env(states, self.init_orientations) #v_map'd

        obs = jp.zeros((self.num_envs, self.observation_space_dim))

        return states, obs
        
    
    def get_joint_torques(self, states) -> jp.ndarray:

        qpos_batch = jax.vmap(lambda d: d.qpos)(states) #we can jit these
        qvel_batch = jax.vmap(lambda d: d.qvel)(states)

        des_targets = jp.concatenate([self.des_cart_pos, self.des_cart_orient], axis = 1)

        stacked_data = jp.concatenate([des_targets.flatten(), qpos_batch.flatten(), qvel_batch.flatten()])
        # packed_data = struct.pack('f' * 21 * self.num_envs, *stacked_data)

        # self.zmq_socket.send(packed_data)
        # reply = socket.recv()
        # joint_torques = jp.frombuffer(reply, dtype = jp.float32)

        # joint_torques = joint_torques.reshape((self.num_envs, 7))

        joint_torques = jp.zeros((self.num_envs, 7))

        return joint_torques

    def apply_joint_torques_and_step(self, states, joint_torques: jp.ndarray):
        states = _jit_apply_joint_torques_and_step_env(states, joint_torques) #vmap'd
        return states

    def step(self, states, actions: jp.ndarray):

        for i in range(self.n_frames -1):
            joint_torques = self.get_joint_torques(states)
            states = self.apply_joint_torques_and_step(states, joint_torques)

        ## Apply the action logic:

        joint_torques = self.get_joint_torques(states)
        states = self.apply_joint_torques_and_step(states, joint_torques)

        ## Return the observation data

        return states, self._get_obs()

    def _get_obs(self):

        rewards = jp.zeros(self.num_envs)
        dones = jp.zeros(self.num_envs)
        obs = jp.zeros((self.num_envs, self.observation_space_dim))
        info = {
            "reward": rewards,
            "terminated": dones,
            "TimeLimit.truncated": jp.zeros_like(dones)
        }

        return (obs, rewards, dones, info)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class Network(nn.Module):
    hidden_sizes: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_sizes:
            x = nn.Dense(size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: int
    log_std_init: float = 0.0

    @nn.compact
    def __call__(self, x):

        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        log_std = self.param("log_std", lambda key, shape: jnp.full(shape, self.log_std_init), (self.action_dim,))
        std = jnp.exp(log_std)
        return mean, std


@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    # env setup
    init_states = [mjx_data.replace() for _ in range(args.num_envs)]
    states = jtu.tree_map(lambda *xs: jp.stack(xs), *init_states)
    mjEnvs = CustomZMQEnvironment(socket, n_physics_steps=5,
                                 init_des_cart_pos=jp.array([0.1,0.1,0.1]), num_envs=args.num_envs)
    
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )

    handle = states

    def step_env_wrappeed(episode_stats, handle, action):
        handle, (next_obs, reward, next_done, info) = mjEnvs.step(handle, action)
        new_episode_return = episode_stats.episode_returns + info["reward"]
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            episode_returns=(new_episode_return) * (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"]),
            episode_lengths=(new_episode_length) * (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"]),
            # only update the `returned_episode_returns` if the episode is done
            returned_episode_returns=jnp.where(
                info["terminated"] + info["TimeLimit.truncated"], new_episode_return, episode_stats.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                info["terminated"] + info["TimeLimit.truncated"], new_episode_length, episode_stats.returned_episode_lengths
            ),
        )
        return episode_stats, handle, (next_obs, reward, next_done, info)

    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_iterations
        return args.learning_rate * frac

    network = Network()
    actor = Actor(action_dim=mjEnvs.action_space_dim)
    critic = Critic()
    network_params = network.init(network_key, np.zeros((1, mjEnvs.observation_space_dim)))

    agent_params = FrozenDict({
        "network": network_params,
        "actor": actor.init(actor_key, network.apply(network_params, np.zeros((1, mjEnvs.observation_space_dim)))),
        "critic": critic.init(critic_key, network.apply(network_params, np.zeros((1, mjEnvs.observation_space_dim)))),
    })

    agent_state = TrainState.create(
        apply_fn=None,
        params=agent_params,
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    # ALGO Logic: Storage setup
    storage = Storage(
        obs=jnp.zeros((args.num_steps, args.num_envs, mjEnvs.observation_space_dim)),
        actions=jnp.zeros((args.num_steps, args.num_envs, mjEnvs.action_space_dim)),
        logprobs=jnp.zeros((args.num_steps, args.num_envs)),
        dones=jnp.zeros((args.num_steps, args.num_envs)),
        values=jnp.zeros((args.num_steps, args.num_envs)),
        advantages=jnp.zeros((args.num_steps, args.num_envs)),
        returns=jnp.zeros((args.num_steps, args.num_envs)),
        rewards=jnp.zeros((args.num_steps, args.num_envs)),
    )

    #need to make this compatible with continuous distributions
    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        step: int,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
        hidden = network.apply(agent_state.params["network"], next_obs)
        mean, std = actor.apply(agent_state.params["actor"], hidden)

        #sample action from distribution(continuous action space)
        dist = Normal(loc=mean, scale=std)
        
        key, subkey = jax.random.split(key)
        action = dist.sample(seed=subkey)
        logprob = dist.log_prob(action).sum(-1) 

        value = critic.apply(agent_state.params["critic"], hidden)
        storage = storage.replace(
            obs=storage.obs.at[step].set(next_obs),
            dones=storage.dones.at[step].set(next_done),
            actions=storage.actions.at[step].set(action),
            logprobs=storage.logprobs.at[step].set(logprob),
            values=storage.values.at[step].set(value.squeeze()),
        )
        return storage, action, key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        hidden = network.apply(params["network"], x)
        mean, std = actor.apply(params["actor"], hidden)

        dist = Normal(loc=mean, scale=std)
        logprob = dist.log_prob(action).sum(-1) 

        entropy = dist.entropy().sum(-1)
        
        value = critic.apply(params["critic"], hidden).squeeze()
        return logprob, entropy, value

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
        next_value = critic.apply(
            agent_state.params["critic"], network.apply(agent_state.params["network"], next_obs)
        ).squeeze()
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - storage.dones[t + 1]
                nextvalues = storage.values[t + 1]
            delta = storage.rewards[t] + args.gamma * nextvalues * nextnonterminal - storage.values[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            storage = storage.replace(advantages=storage.advantages.at[t].set(lastgaelam))
        storage = storage.replace(returns=storage.advantages + storage.values)
        return storage

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        b_obs = storage.obs.reshape((-1, mjEnvs.observation_space_dim)) 
        b_logprobs = storage.logprobs.reshape(-1)
        b_actions = storage.actions.reshape(-1, mjEnvs.action_space_dim)
        b_advantages = storage.advantages.reshape(-1)
        b_returns = storage.returns.reshape(-1)

        def ppo_loss(params, x, a, logp, mb_advantages, mb_returns):
            newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
            logratio = newlogprob - logp
            ratio = jnp.exp(logratio)
            approx_kl = ((ratio - 1) - logratio).mean()

            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # Value loss
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

        ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
        for _ in range(args.update_epochs):
            key, subkey = jax.random.split(key)
            b_inds = jax.random.permutation(subkey, args.batch_size, independent=True)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    b_logprobs[mb_inds],
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                )
                agent_state = agent_state.apply_gradients(grads=grads)
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    handle, next_obs = mjEnvs.reset(handle)
    next_done = np.zeros(args.num_envs)

    def rollout(agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step):
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            storage, action, key = get_action_and_value(agent_state, next_obs, next_done, storage, step, key)

            # TRY NOT TO MODIFY: execute the game and log data.
            episode_stats, handle, (next_obs, reward, next_done, _) = step_env_wrappeed(episode_stats, handle, action)
            storage = storage.replace(rewards=storage.rewards.at[step].set(reward))
        return agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step

    for iteration in range(1, args.num_iterations + 1):
        iteration_time_start = time.time()
        agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step = rollout(
            agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step
        )
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state,
            storage,
            key,
        )
        avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar(
            "charts/avg_episodic_length", np.mean(jax.device_get(episode_stats.returned_episode_lengths)), global_step
        )
        writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar(
            "charts/SPS_update", int(args.num_envs * args.num_steps / (time.time() - iteration_time_start)), global_step
        )

    del mjEnvs
    writer.close()