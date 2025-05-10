import sys
import gymnasium as gym
import numpy as np

import torch
from torch import nn

from train import make_env

device = "cpu"


class Actor(nn.Module):
    def __init__(self, envs: gym.vector.SyncVectorEnv) -> None:
        super().__init__()
        n_observation = int(np.prod(envs.single_observation_space.shape))
        n_action = int(np.prod(envs.single_action_space.shape))
        action_high = envs.single_action_space.high
        action_low = envs.single_action_space.low

        self.fc = nn.Sequential(
            nn.Linear(n_observation, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, n_action)
        self.fc_logstd = nn.Linear(256, n_action)  # log std is more stable
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.LOG_STD_RANGE = self.LOG_STD_MAX - self.LOG_STD_MIN

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get mean, std
        x = self.fc(state)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + self.LOG_STD_RANGE * (log_std + 1) * 0.5
        std = log_std.exp()

        # get action, log_prob
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)  # output between [-1, 1]
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # mean is deterministic action
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts randomly."""

    def __init__(self):
        envs = gym.vector.SyncVectorEnv([make_env(1)])
        self.actor = Actor(envs)
        envs.close()
        self.actor.load_state_dict(torch.load("actor.pt"))
        self.actor.eval()

    def act(self, observation):
        observation = torch.tensor(
            np.array([observation]), dtype=torch.float32, device=device
        )
        # print(observation.shape)
        _, _, deterministic_action = self.actor(observation)
        # print(f"{deterministic_action.shape=}")
        # print(f"{deterministic_action=}")
        return deterministic_action.detach().cpu().numpy()[0]
