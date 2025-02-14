import gymnasium as gym
import torch
import random
import numpy as np
from collections import deque
from gymnasium.spaces import Discrete, Box, MultiDiscrete, MultiBinary


def get_spaces(env_name):
    """
    Get the dimension of observation and action spaces for any Gym environment.

    Args:
        env_name (str): Name of the Gym environment

    Returns:
        tuple: (action_dim, observation_dim) where each is an integer representing
               the size of the space (flattened in case of multi-dimensional spaces)
    """
    env = gym.make(env_name)

    # Handle action space
    if isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n
    elif isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]  # Take first dimension for Box spaces
    elif isinstance(env.action_space, MultiDiscrete):
        action_dim = sum(env.action_space.nvec)  # Sum up all possible actions
    elif isinstance(env.action_space, MultiBinary):
        action_dim = env.action_space.n
    else:
        raise NotImplementedError(
            f"Action space {type(env.action_space)} not supported"
        )

    # Handle observation space
    if isinstance(env.observation_space, Discrete):
        observation_dim = env.observation_space.n
    elif isinstance(env.observation_space, Box):
        observation_dim = int(
            np.prod(env.observation_space.shape)
        )  # Flatten the observation space
    elif isinstance(env.observation_space, MultiDiscrete):
        observation_dim = sum(env.observation_space.nvec)
    elif isinstance(env.observation_space, MultiBinary):
        observation_dim = env.observation_space.n
    else:
        raise NotImplementedError(
            f"Observation space {type(env.observation_space)} not supported"
        )

    env.close()
    return action_dim, observation_dim


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, 1 if done else 0))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = list(zip(*batch))
        states = torch.tensor(states)
        actions = torch.tensor(actions).unsqueeze(dim=1)
        rewards = torch.tensor(rewards).unsqueeze(dim=1)
        next_states = torch.tensor(next_states)
        dones = torch.tensor(dones).unsqueeze(dim=1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
