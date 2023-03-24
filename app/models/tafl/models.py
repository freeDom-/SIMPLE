import numpy as np
import torch as th
import torch.nn.functional as F

from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from torch import nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

ROWS = 7
COLS = 7
GRID_SIZE = COLS * ROWS
ACTIONS_PER_TOKEN = ROWS + COLS

ACTIONS = ROWS * COLS * ACTIONS_PER_TOKEN
ACTION_LAYERS = ACTIONS_PER_TOKEN
KERNEL_SIZE = 3
VALUE_FILTERS = 128
FILTERS = 128
HALF_FILTERS = FILTERS // 2
RESIDUAL_LAYERS = 12


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]

        self.conv = nn.Sequential(
            nn.Conv2d(n_input_channels, FILTERS, KERNEL_SIZE, padding='same'),
            nn.BatchNorm2d(FILTERS), nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_input_channels, HALF_FILTERS, (ROWS, 1), padding='same'),
            nn.BatchNorm2d(HALF_FILTERS), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_input_channels, HALF_FILTERS, (1, COLS), padding='same'),
            nn.BatchNorm2d(HALF_FILTERS), nn.ReLU()
        )
        # 128, 256, 384, 512, ... Input Channels, because of skip connection
        self.residual = nn.ModuleList(
            list(np.concatenate(
            [[
                nn.Conv2d((i+1)*FILTERS, FILTERS, KERNEL_SIZE, padding='same'),
                nn.BatchNorm2d(FILTERS), nn.ReLU(),
                nn.Conv2d(FILTERS, FILTERS, KERNEL_SIZE, padding='same'),
                nn.BatchNorm2d(FILTERS)
            ] for i in range(RESIDUAL_LAYERS)]).flat)
        )
        self.relu = nn.ReLU()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # TODO: implement correct merging of split input observation
        '''a = self.conv1(observations)
        b = self.conv2(observations)
        print(a.shape, b.shape)
        y = th.cat((a, b), -1)'''

        y = self.conv(observations)
        # Residual Layer with skip connection
        for i in range(RESIDUAL_LAYERS):
            x = y
            y = self.residual[i*5](x)
            y = self.residual[i*5+1](y)
            y = self.residual[i*5+2](y)
            y = self.residual[i*5+3](y)
            y = self.residual[i*5+4](y)
            y = th.cat((y, x), 1)
            y = self.relu(y)
        return y


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(self, feature_dim: int,
                last_layer_dim_pi: int = ACTIONS,
                last_layer_dim_vf: int = 1,):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Conv2d(FILTERS * (RESIDUAL_LAYERS+1), 2, 1, padding='same'),
            nn.Flatten(),
            nn.BatchNorm1d(2 * GRID_SIZE), nn.ReLU(),
            nn.Linear(2 * GRID_SIZE, last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Conv2d(FILTERS * (RESIDUAL_LAYERS+1), 1, 1, padding='same'),
            nn.Flatten(),
            nn.BatchNorm1d(GRID_SIZE), nn.ReLU(),
            nn.Linear(GRID_SIZE, VALUE_FILTERS), nn.ReLU(),
            nn.Linear(VALUE_FILTERS, last_layer_dim_vf), nn.Tanh()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        x = self.policy_net(features)
        return x

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomPolicy(ActorCriticPolicy):
    def __init__(self,
                observation_space: spaces.Space,
                action_space: spaces.Space,
                lr_schedule: Callable[[float], float],
                *args,
                **kwargs,):

        super().__init__(observation_space,
                        action_space,
                        lr_schedule,
                        # Pass remaining arguments to base class
                        *args,
                        **kwargs,)
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)
