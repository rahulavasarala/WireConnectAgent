import torch
from torch.distributions.normal import Normal
import torch.nn as nn
import numpy as np
from torch_geometric.data import Batch
import torch.nn.functional as F
    

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
    
# class GNNAgent(nn.Module):#In this case the total observation dim, is the dim from the observation, and the dim from the output of the point net
#     def __init__(self, observation_space_dim, action_space_dim):
#         super().__init__()
#         self.critic = nn.Sequential(
#             layer_init(nn.Linear(np.array(observation_space_dim).prod(), 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 1), std=1.0),
#         )
#         self.actor_mean = nn.Sequential(
#             layer_init(nn.Linear(np.array(observation_space_dim).prod(), 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, np.prod(action_space_dim)), std=0.01),
#         )

#         self.lwGNN = LightweightGNN(3,20, 12)
#         self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_space_dim)))

#     def get_value(self, obs_data, point_cloud_data):
#         encoded_points = self.lwGNN(point_cloud_data)
#         total_data = torch.cat((obs_data, encoded_points), axis = 1)
#         return self.critic(total_data)

#     def get_action_and_value(self, obs_data, point_cloud_data, action=None):

#         encoded_points = self.lwGNN(point_cloud_data)
#         # print(f"size of encoded points: {encoded_points.shape}, size of obs_data is: {obs_data.shape}")
        
#         total_data = torch.cat((obs_data, encoded_points), axis = 1)
#         action_mean = self.actor_mean(total_data)
#         action_logstd = self.actor_logstd.expand_as(action_mean)
#         action_std = torch.exp(action_logstd)
#         probs = Normal(action_mean, action_std)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(total_data)
    
def freeze_model_layers(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = False
        print(f"Frozen parameter: {name}")
    
# class PointNetAgent(nn.Module):

#     def __init__(self, observation_space_dim, action_space_dim, weights_path):

#         super().__init__()

#         self.point_net = get_model(num_class = 40)

#         state_dict = torch.load(weights_path)
#         self.point_net.load_state_dict(state_dict)

#         freeze_model_layers(self.point_net)

#         self.linear_layer_pointnet = nn.Linear(256, 12)

#         self.critic = nn.Sequential(
#             layer_init(nn.Linear(np.array(observation_space_dim).prod(), 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 1), std=1.0),
#         )
#         self.actor_mean = nn.Sequential(
#             layer_init(nn.Linear(np.array(observation_space_dim).prod(), 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, np.prod(action_space_dim)), std=0.01),
#         )

#         self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_space_dim)))

#     def get_padded_points(self, points): #assume points are in form (num_features, num_points)

#         if points.shape[1] > 1024:
#             raise Exception("Too many points in point buffer!")
        
#         num_points = points.shape[1]

#         full_reps = 1024//num_points
#         partial_part = 1024%num_points

#         tiled_points = np.tile(points, (1, full_reps))
#         tiled_points = np.hstack([tiled_points, points[:partial_part]])

#         return tiled_points
    
#     def get_point_net_encodings(self, points):

#         xyz = self.get_padded_points(points)
#         xyz.reshape((1, xyz.shape[0], xyz.shape[1]))

#         B, _, _ = xyz.shape
#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
#         l1_xyz, l1_points = self.point_net.sa1(xyz, norm)
#         l2_xyz, l2_points = self.point_net.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.point_net.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 1024)
#         x = self.point_net.drop1(F.relu(self.point_net.bn1(self.fc1(x)), inplace=True))
#         x = self.point_net.drop2(F.relu(self.point_net.bn2(self.fc2(x)), inplace=True))

#         x = self.linear_layer_pointnet(x)

#         return x
    
#     def get_value(self, obs, points):
#         encoded_points = self.get_point_net_encodings(points)
#         total_data = torch.cat((obs, encoded_points), axis = 1)
#         return self.critic(total_data)

#     def get_action_and_value(self, obs, points, action=None):

#         encoded_points = self.get_point_net_encodings(points)
        
#         total_data = torch.cat((obs, encoded_points), axis = 1)
#         action_mean = self.actor_mean(total_data)
#         action_logstd = self.actor_logstd.expand_as(action_mean)
#         action_std = torch.exp(action_logstd)
#         probs = Normal(action_mean, action_std)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(total_data)


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

    

# Test

t = TransformerAgent(25, 25, 3, 21)

x = torch.rand(1, 75)

# x = t.pass_through_encoder(x)

# print(x.shape)

print(t.get_action_and_value(x))



















