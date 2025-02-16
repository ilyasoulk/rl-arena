import json
import torch
import random
import gymnasium as gym
from collections import deque
import torch.nn.functional as F


# from gymnasium.spaces import Discrete, Box, MultiDiscrete, MultiBinary
#
#
# def get_spaces(env_config, env_name):
#     """
#     Get the dimension of observation and action spaces for any Gym environment.
#
#     Args:
#         env_name (str): Name of the Gym environment
#
#     Returns:
#         tuple: (action_dim, observation_dim) where each is an integer representing
#                the size of the space (flattened in case of multi-dimensional spaces)
#     """
#     env = env_config.create_env(env_name)
#
#     # Handle action space
#     if isinstance(env.action_space, Discrete):
#         action_dim = env.action_space.n
#     elif isinstance(env.action_space, Box):
#         # action_dim = env.action_space.shape[0]  # Take first dimension for Box spaces
#         action_dim = env.action_space.shape
#     elif isinstance(env.action_space, MultiDiscrete):
#         action_dim = sum(env.action_space.nvec)  # Sum up all possible actions
#     elif isinstance(env.action_space, MultiBinary):
#         action_dim = env.action_space.n
#     else:
#         raise NotImplementedError(
#             f"Action space {type(env.action_space)} not supported"
#         )
#
#     # Handle observation space
#     if isinstance(env.observation_space, Discrete):
#         observation_dim = env.observation_space.n
#     elif isinstance(env.observation_space, Box):
#         # observation_dim = int(
#         #     np.prod(env.observation_space.shape)
#         # )  # Flatten the observation space
#         observation_dim = env.observation_space.shape
#     elif isinstance(env.observation_space, MultiDiscrete):
#         observation_dim = sum(env.observation_space.nvec)
#     elif isinstance(env.observation_space, MultiBinary):
#         observation_dim = env.observation_space.n
#     else:
#         raise NotImplementedError(
#             f"Observation space {type(env.observation_space)} not supported"
#         )
#
#     env.close()
#     if isinstance(action_dim, tuple):
#         if len(action_dim) == 1:
#             action_dim = action_dim[0]
#     if isinstance(observation_dim, tuple):
#         if len(observation_dim) == 1:
#             observation_dim = observation_dim[0]
#     return action_dim, observation_dim
#
#
def preprocess(state, mode):
    state = torch.tensor(state)
    if mode == "ConvNet":
        if len(state.shape) == 2:  # If no channel dim
            state = state.unsqueeze(-1)

        # Convert to torch and resize
        state = (
            F.interpolate(
                state.float().permute(2, 0, 1).unsqueeze(0),
                size=(84, 84),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )

    return state


class ReplayBuffer:
    def __init__(self, capacity, mode="MLP"):
        self.buffer = deque(maxlen=capacity)
        self.mode = mode

    def add(self, state, action, reward, next_state, done):
        action = torch.tensor(action)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(1 if done else 0)

        state = preprocess(state, mode=self.mode)
        next_state = preprocess(next_state, mode=self.mode)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = list(zip(*batch))
        states = torch.stack(states)
        actions = torch.stack(actions).unsqueeze(dim=1)
        rewards = torch.stack(rewards).unsqueeze(dim=1)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones).unsqueeze(dim=1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class EnvConfig:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.configs = json.load(f)

    def get_env_config(self, env_name: str):
        if env_name not in self.configs:
            raise ValueError(f"Environment {env_name} not found in config file.")
        return self.configs[env_name]

    def create_env(self, env_name: str):
        config = self.get_env_config(env_name)
        env_args = config.get("env_args", {})
        print(env_args)
        return gym.make(env_name, **env_args)

    def get_model_type(self, env_name: str):
        return self.get_env_config(env_name)["model"]

    def get_spaces(self, env_name: str):
        config = self.get_env_config(env_name)
        return config["action_space"], config["observation_space"]
