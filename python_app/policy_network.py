import torch
from torch.distributions.normal import Normal
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import jax.numpy as jnp
import flax

import jax
from flax import linen as jnn
from typing import Sequence
from functools import partial

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

#Transformer gets (BATCH, fullobslen) ->>>>>> Spits out (BATCH, obs_len) to Agent
class Agent(nn.Module):
    def __init__(self, observation_space_dim, action_space_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space_dim).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space_dim).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_space_dim)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_space_dim)))

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
    

def freeze_model_layers(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = False
        print(f"Frozen parameter: {name}")


class CompactTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, ff_dim, num_layers=1, output_dim = 5):
        super().__init__()
        # Linear layer to project input to d_model
        self.embedding = nn.Linear(input_dim, d_model)
        # Transformer Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final linear layer to output a vector for the RL agent
        self.output_head = nn.Linear(d_model, output_dim) # output_size depends on your task (e.g., num actions, 1 for value)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = self.embedding(x)
        # Add positional encoding here (not shown for simplicity)
        x = self.transformer_encoder(x)
        final_embedding = x[:, -1, :] 
        # Pass the final embedding to the output head
        output = self.output_head(final_embedding)
        return output
    
class TransformerAgent(nn.Module):

    def __init__(self, input_dim, output_dim, observation_horizon, action_dim):
        super().__init__()
        self.encoding_layer = CompactTransformerEncoder(input_dim=input_dim, d_model = 128, n_heads = 4, ff_dim = 128, num_layers = 1, output_dim=output_dim)
        self.observation_horizon = observation_horizon
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Agent = Agent(output_dim, action_dim)

    def get_value(self, x):#assume that x comes in the size (batch, full_obs_len)
        x = self.pass_through_encoder(x)
        print(x.shape)

        x = self.Agent.get_value(x)

        return x
    
    def get_action_and_value(self, x, action = None):
        x = self.pass_through_encoder(x)

        return self.Agent.get_action_and_value(x, action)
    
    def pass_through_encoder(self, x):
        x = x.reshape(-1, self.observation_horizon, self.input_dim)
        x = self.encoding_layer(x)
        # print(f"x shape is : {x.shape}")

        return x
    
class JaxEncoderBlock(jnn.Module):
    d_model: int
    n_heads: int
    ff_dim: int

    @jnn.compact
    def __call__(self, x):
        # Multi-head attention
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
        )(x, x)

        # Residual + LayerNorm
        x = jnn.LayerNorm()(x + attn_out)

        # Feed-forward network
        ff_out = jnn.Dense(self.ff_dim)(x)
        ff_out = jnn.relu(ff_out)
        ff_out = jnn.Dense(self.d_model)(ff_out)

        # Residual + LayerNorm
        x = jnn.LayerNorm()(x + ff_out)

        return x

    
class JaxTransformerEncoder(jnn.Module):
    input_dim: int
    d_model: int
    n_heads: int
    ff_dim: int
    num_layers: int = 1
    output_dim: int = 5

    @jnn.compact
    def __call__(self, x):
        """
        x: (batch, seq_len, input_dim)
        """

        # Linear embedding
        x = jnn.Dense(self.d_model)(x)

        # Stacked Transformer encoder layers
        for _ in range(self.num_layers):
            x = JaxEncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                ff_dim=self.ff_dim,
            )(x)

        # Extract last token embedding
        final_embedding = x[:, -1, :]  # (batch, d_model)

        # Output head
        out = jnn.Dense(self.output_dim)(final_embedding)

        return out

    
class JaxTransformerNetwork(jnn.Module):
    input_dim: int
    output_dim: int
    observation_horizon: int
    action_dim: int

    def setup(self):
        self.encoding_layer = JaxTransformerEncoder(
            input_dim=self.input_dim,
            d_model=128,
            n_heads=4,
            ff_dim=128,
            num_layers=1,
            output_dim=self.output_dim,
        )

    def __call__(self, x):
        # reshape (B, obs_stack * obs_dim) → (B, obs_stack, obs_dim)
        x = x.reshape(-1, self.observation_horizon, self.input_dim)
        return self.encoding_layer(x)


import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Sequence
from flax.linen.initializers import orthogonal


# ---------------------------------------------------------
#                    JAX Critic
# ---------------------------------------------------------

class JaxCritic(jnn.Module):
    obs_dim: int     # flattened observation dimension

    @jnn.compact
    def __call__(self, x):
        x = jnn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = jnn.tanh(x)

        x = jnn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = jnn.tanh(x)

        value = jnn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return value.squeeze(-1)
    

# ---------------------------------------------------------
#                    JAX Actor (Gaussian)
# ---------------------------------------------------------

class JaxActor(nn.Module):
    obs_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # network → mean, logstd
        h = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        h = nn.tanh(h)
        h = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)))(h)
        h = nn.tanh(h)

        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(h)

        log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (1, self.action_dim),
        )
        log_std = jnp.broadcast_to(log_std, mean.shape)

        return mean, log_std

    # -------------------------
    # SAMPLE MODE
    # -------------------------
    def sample(self, variables, x, rng):
        mean, log_std = self.apply(variables, x)
        std = jnp.exp(log_std)

        eps = jax.random.normal(rng, mean.shape)
        action = mean + eps * std

        # logprob
        log_prob = -0.5 * (
            ((action - mean) / std) ** 2
            + 2 * log_std
            + jnp.log(2 * jnp.pi)
        ).sum(-1)

        entropy = (0.5 * (1.0 + jnp.log(2 * jnp.pi) + 2 * log_std)).sum(-1)

        return action, log_prob, entropy, mean, log_std

    # -------------------------
    # EVALUATION MODE
    # -------------------------
    def evaluate(self, variables, x, action):
        mean, log_std = self.apply(variables, x)
        std = jnp.exp(log_std)

        log_prob = -0.5 * (
            ((action - mean) / std) ** 2
            + 2 * log_std
            + jnp.log(2 * jnp.pi)
        ).sum(-1)

        entropy = (0.5 * (1.0 + jnp.log(2 * jnp.pi) + 2 * log_std)).sum(-1)

        return log_prob, entropy, mean, log_std


    




















