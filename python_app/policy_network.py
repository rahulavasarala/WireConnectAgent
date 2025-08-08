# import flax.linen as nn 

# from flax.linen.initializers import constant, orthogonal
# from typing import Any, Dict, Sequence, Tuple, Union
# import jax.numpy as jnp

import torch
from torch.distributions.normal import Normal
import torch.nn as nn
import numpy as np


# class Network(nn.Module):
#     hidden_sizes: Sequence[int] = (256, 256)

#     @nn.compact
#     def __call__(self, x):
#         for size in self.hidden_sizes:
#             x = nn.Dense(size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
#             x = nn.relu(x)
#         return x


# class Critic(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


# class Actor(nn.Module):
#     action_dim: int
#     log_std_init: float = 0.0

#     @nn.compact
#     def __call__(self, x):

#         mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
#         log_std = self.param("log_std", lambda key, shape: jnp.full(shape, self.log_std_init), (self.action_dim,))
#         std = jnp.exp(log_std)
#         return mean, std
    

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space_dim).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space_dim).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.action_space_dim)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space_dim)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)