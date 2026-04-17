#!/usr/bin/env python3
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import cast
from gymnasium.spaces import Discrete
import wandb
from minigrid.wrappers import FullyObsWrapper

from env.factory import make_env
from env.rewardsystem import DoorKeyRewardSystem, RewardConfig
from env import doorkey_events as doorev

SEED = 42


class StateEncoder:

    def __init__(self, env: gym.Env, include_stage_progress: bool = True):
        obs_space = env.observation_space
        self.state_dim = 0
        self.include_stage_progress = include_stage_progress

        if isinstance(obs_space, gym.spaces.Dict):
            img_space = cast(gym.spaces.Box, obs_space.spaces["image"])
            assert (
                img_space.shape is not None
            ), "La shape dell'immagine non può essere None"
            self.state_dim += int(np.prod(img_space.shape))

            if "direction" in obs_space.spaces:
                self.state_dim += 1
        elif isinstance(obs_space, gym.spaces.Box):
            self.state_dim = int(np.prod(obs_space.shape))
        else:
            raise ValueError(
                f"Tipo di observation space non supportato: {type(obs_space)}"
            )

        if self.include_stage_progress:
            self.state_dim += 1

    def encode(self, obs: dict, info) -> np.ndarray:
        if isinstance(obs, dict):
            img_flat = obs["image"].flatten()
            if "direction" in obs:
                direction = np.array([obs["direction"]])
                encoded_obs = np.concatenate((img_flat, direction))
            else:
                encoded_obs = img_flat
        else:
            encoded_obs = np.array(obs).flatten()

        if self.include_stage_progress:
            progress_val = info.get("stage_progress", 0.0) if info else 0.0
            progress_arr = np.array([progress_val])
            encoded_obs = np.concatenate((encoded_obs, progress_arr))

        return encoded_obs.astype(np.float32)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim=7) -> None:
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class DQNAgent:
    def __init__(
        self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, buffer_size
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = 0.05
        self.memory = deque(maxlen=buffer_size)

        # Rete Principale
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Target Network
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        q_values = self.model(torch.tensor(state, dtype=torch.float32))
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:

                with torch.no_grad():
                    next_q_values = self.target_model(
                        torch.tensor(next_state, dtype=torch.float32)
                    )
                    max_next_q = torch.max(next_q_values).item()

                target = reward + self.gamma * max_next_q
            target_f = self.model(torch.tensor(state, dtype=torch.float32))
            target_f[action] = target

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(
                target_f, self.model(torch.tensor(state, dtype=torch.float32))
            )
            loss.backward()
            self.optimizer.step()

            if self.epsilon > 0.01:
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """Copia i pesi dalla rete principale alla rete target."""
        self.target_model.load_state_dict(self.model.state_dict())
