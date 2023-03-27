import numpy as np
import torch as th
import torch.nn.functional as F

from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from torch import nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from models.templates.residual import ResidualBlock


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 32):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]

        self.convolutional = nn.Sequential(
            nn.Conv2d(n_input_channels, features_dim, 3, padding='same'),
            nn.BatchNorm2d(features_dim), nn.ReLU()
        )
        self.residual = ResidualBlock(features_dim, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        y = self.convolutional(observations)
        y = self.residual(y)
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
                last_layer_dim_pi: int = 9,
                last_layer_dim_vf: int = 1,):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Conv2d(feature_dim, 4, 1, padding='same'),
            nn.BatchNorm2d(4), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(24, last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Conv2d(feature_dim, 4, 1, padding='same'),
            nn.BatchNorm2d(4), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(24, last_layer_dim_vf), nn.Tanh()
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
