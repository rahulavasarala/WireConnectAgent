# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os
import random
import time
from dataclasses import dataclass
from typing import Sequence

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
from parallel_env import ParallelEnv, NUM_ENVS, DummyParallelEnv
import mujoco, zmq, mujoco.mjx as mjx
from utils import params_from_yaml
from policy_network import JaxTransformerNetwork, JaxActor, JaxCritic
import msgpack
import shutil
from jax_utils import AgentParams, AgentState, InferenceState, save_inference_state, load_inference_state, create_agent_state, apply_agent_gradients

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"


#-------------ENVS---------------------------------------------------------------
mj_model = mujoco.MjModel.from_xml_path("../models/scenes/fr3squaresquare.xml")
mj_data = mujoco.MjData(mj_model)
home_key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
home_qpos = mj_model.key_qpos[home_key_id]
HOME_QPOS = jnp.array(home_qpos)

ctx = zmq.Context()
mft_socket = ctx.socket(zmq.REQ)
mft_socket.connect("ipc:///tmp/zmq_motion_force_server")

fspf_socket = ctx.socket(zmq.REQ)
fspf_socket.connect("ipc:///tmp/zmq_fspf_server")
#--------------------------------------------------------------------------------

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
    total_timesteps: int = 1000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 2
    """the number of parallel game environments"""
    num_steps: int = 60
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
    entropies: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


if __name__ == "__main__":
    args = tyro.cli(Args)
    params = params_from_yaml("./experiment.yaml")
    args.total_timesteps = params["train_steps"]
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, )

    batched_data = jax.vmap(lambda _: mjx_data.replace(qpos=HOME_QPOS))(rng)

    experiment_run_dir = f"./runs_jax/run_{params["name"]}"
    os.makedirs(experiment_run_dir)
    shutil.copyfile("./experiment.yaml", f"{experiment_run_dir}/experiment_{params["name"]}.yaml")

    # env setup
    envs = ParallelEnv([mft_socket, fspf_socket], mjx_model, mj_model, params)
    batched_data = envs.dry_start(batched_data)
    envs.init_env_params(batched_data) #This initializes all the properties of the envs
    # envs = DummyParallelEnv(3, 25)

    observation_space_dim = envs.obs_stack * envs.obs_size
    action_space_dim = 7 
    
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )
    #handle is the equivalent of the batched data, step_env is the equivalent of step

    def step_env_wrappeed(episode_stats, handle, action):
        handle, (next_obs, reward, next_done, info) = envs.step(handle, action)
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

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_iterations
        return args.learning_rate * frac

    network = JaxTransformerNetwork(
        input_dim=envs.obs_size,
        output_dim=envs.obs_size,
        observation_horizon=envs.obs_stack,
        action_dim=action_space_dim,
    )

    actor = JaxActor(obs_dim=envs.obs_size, action_dim=action_space_dim)
    critic = JaxCritic(obs_dim=envs.obs_size)

    # --- dummy inputs ---
    dummy_x = jnp.zeros((1, observation_space_dim))
    dummy_encoded = jnp.zeros((1, envs.obs_size))

    # --- parameter initialization ---
    network_params = network.init(network_key, dummy_x)
    actor_params   = actor.init(actor_key, dummy_encoded)
    critic_params  = critic.init(critic_key, dummy_encoded)

    agent_params = AgentParams(
        network_params,
        actor_params,
        critic_params,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(
            learning_rate=linear_schedule if args.anneal_lr else args.learning_rate,
            eps=1e-5,
        ),
    )


    agent_state = create_agent_state(agent_params, tx)

    @jax.jit
    def sample_fn(actor_params, obs, rng):
        return actor.sample(actor_params, obs, rng)
    
    @jax.jit
    def evaluate_fn(actor_params, obs, action):
        return actor.evaluate(actor_params, obs, action)


    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    # ALGO Logic: Storage setup
    storage = Storage(
        obs=jnp.zeros((args.num_steps, args.num_envs,observation_space_dim)),
        actions=jnp.zeros((args.num_steps, args.num_envs,action_space_dim) , dtype=jnp.float32),
        logprobs=jnp.zeros((args.num_steps, args.num_envs)),
        dones=jnp.zeros((args.num_steps, args.num_envs)),
        values=jnp.zeros((args.num_steps, args.num_envs)),
        advantages=jnp.zeros((args.num_steps, args.num_envs)),
        returns=jnp.zeros((args.num_steps, args.num_envs)),
        rewards=jnp.zeros((args.num_steps, args.num_envs)),
        entropies=jnp.zeros((args.num_steps, args.num_envs)),
    )

    @jax.jit
    def get_action_and_value(
        agent_state: AgentState,
        next_obs: jnp.ndarray,
        next_done: jnp.ndarray,
        storage: Storage,
        step: int,
        key: jax.random.PRNGKey,
    ):
        """Sample continuous action, compute logprob + entropy + value, and write into storage."""

        key, k_network, k_actor = jax.random.split(key, 3)
        
        hidden = network.apply(
            agent_state.params.network_params,
            next_obs,
            rngs={"dropout": k_network},
        )

        action, logprob, entropy, mean, log_std = sample_fn(
            agent_state.params.actor_params,
            hidden,
            rng=k_actor,
        )

        # ------- Critic -------
        value = critic.apply(
            agent_state.params.critic_params,
            hidden,
        )
        
        # ------- Store results -------
        storage = storage.replace(
            obs=storage.obs.at[step].set(next_obs),
            dones=storage.dones.at[step].set(next_done),
            actions=storage.actions.at[step].set(action),        # (B, action_dim)
            logprobs=storage.logprobs.at[step].set(logprob),     # (B,)
            entropies=storage.entropies.at[step].set(entropy),   # (B,)
            values=storage.values.at[step].set(value),           # (B,)
        )

        return storage, action, key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: jnp.ndarray,
        action: jnp.ndarray,
    ):
        """Compute logprob, entropy, and value for a supplied continuous action."""

        hidden = network.apply(
            params.network_params,
            x,
        )

        logprob, entropy, mean, log_std = evaluate_fn(
            params.actor_params,
            hidden,
            action=action,
        )

        # Critic
        value = critic.apply(
            params.critic_params,
            hidden,
        )

        return logprob, entropy, value


    @jax.jit
    def compute_gae(
        agent_state: AgentState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
        next_value = critic.apply(
            agent_state.params.critic_params, network.apply(agent_state.params.network_params, next_obs)
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
        agent_state: AgentState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        b_obs = storage.obs.reshape((-1, observation_space_dim))
        b_logprobs = storage.logprobs.reshape(-1)
        b_actions = storage.actions.reshape((-1,action_space_dim))
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
                agent_state = apply_agent_gradients(agent_state, grads)
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    batched_data , next_obs = envs.reset(batched_data)
    next_done = np.zeros(args.num_envs)

    def rollout(agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step):
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            storage, action, key = get_action_and_value(agent_state, next_obs, next_done, storage, step, key)

            # TRY NOT TO MODIFY: execute the game and log data.
            episode_stats, handle, (next_obs, reward, next_done, _) = step_env_wrappeed(episode_stats, handle, action)
            storage = storage.replace(rewards=storage.rewards.at[step].set(reward))
        return agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step

    for iteration in range(0, args.num_iterations):
        print(f"On Iteration : {iteration}")
        iteration_time_start = time.time()
        agent_state, episode_stats, next_obs, next_done, storage, key, batched_data, global_step = rollout(
            agent_state, episode_stats, next_obs, next_done, storage, key, batched_data, global_step
        )

        print("Running compute gae!!!!")
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state,
            storage,
            key,
        )
        avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")

        if iteration % 100 == 0:
            save_path = f"{experiment_run_dir}/model{iteration}.cleanrl_model"
            save_inference_state(save_path, agent_state)
            print(f"Saved checkpoint to {save_path}")
