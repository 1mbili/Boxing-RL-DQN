"""
Utility functions for the DQN algorithm
"""


from collections import namedtuple, deque
from typing import NamedTuple

import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """A simple numpy replay buffer"""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ReplayBufferSamples(NamedTuple):
    """ A class to represent the samples from the replay buffer """
    states: th.Tensor
    actions: th.Tensor
    next_states: th.Tensor
    rewards: th.Tensor


class AdvancedReplayMemory:
    """ A more advanced replay buffer that stores the data in numpy arrays and pre-allocates the memory """

    def __init__(self, buffer_size, observation_space, device):
        self.buffer_size = buffer_size
        self.idx = 0
        self.full = False
        self.device = device
        self.states = np.zeros(
            (buffer_size, *observation_space.shape), dtype=observation_space.dtype)
        self.next_states = np.zeros(
            (self.buffer_size, *observation_space.shape), dtype=observation_space.dtype)
        self.actions = np.zeros((buffer_size, 1), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)

    def size(self) -> int:
        return self.buffer_size if self.full else self.idx

    def add(self, state, next_state, action, reward) -> None:
        self.states[self.idx] = np.array(state, dtype=np.float32)
        self.next_states[self.idx] = np.array(next_state, dtype=np.float32)
        self.actions[self.idx] = np.array(action)
        self.rewards[self.idx] = np.array(reward)

        self.idx = (self.idx + 1) % self.buffer_size
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        batch_idx = np.random.randint(0, self.size(), size=batch_size)
        next_obs = self.next_states[batch_idx, :]
        obs = self.states[batch_idx, :]
        actions = self.actions[batch_idx, :]
        rewards = self.rewards[batch_idx].reshape(-1, 1)
        r = ReplayBufferSamples(
            states=th.tensor(obs, device=self.device, dtype=th.float32),
            actions=th.tensor(actions, device=self.device, dtype=th.int64),
            next_states=th.tensor(
                next_obs, device=self.device, dtype=th.float32),
            rewards=th.tensor(rewards, device=self.device)
        )
        return r


class DQN(nn.Module):
    """ Deep Q-Network """

    def __init__(self, n_actions):
        super().__init__()
        self.layer1 = nn.Conv2d(4, 32, 8, stride=4)
        self.layer2 = nn.Conv2d(32, 64, 4, stride=2)
        self.layer3 = nn.Conv2d(64, 64, 3, stride=1)
        self.layer4 = nn.Linear(3136, 512)
        self.layer5 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(-1, 3136)
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        return x
