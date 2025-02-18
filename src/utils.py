import json
import torch
import random
import gymnasium as gym
from collections import deque
import torch.nn.functional as F


@torch.no_grad()
def eval(policy, num_episodes, env_name, env_config, num_frame_stack=1, device="mps"):
    print("Evaluating")
    eval_env = env_config.create_env(env_name)
    mode = env_config.get_model_type(env_name)
    frame_stack = FrameStack(num_frame_stack, mode=mode)
    rewards = []
    for _ in range(num_episodes):
        current_state, _ = eval_env.reset()
        current_state = frame_stack.reset(current_state)
        done = False
        truncated = False
        episode_reward = 0
        while not (done or truncated):
            inputs = preprocess(current_state, mode=mode).to(device)
            action = policy(inputs).argmax().item()
            current_state, reward, done, truncated, _ = eval_env.step(action)
            current_state = frame_stack.update(current_state)
            episode_reward += float(reward)  # Accumulate episode reward

        rewards.append(episode_reward)

    eval_env.close()

    avg_reward = sum(rewards) / num_episodes
    return avg_reward


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
    def __init__(self, capacity, stack_size=4, mode="MLP"):
        self.buffer = deque(maxlen=capacity)
        self.stack_size = stack_size
        self.mode = mode

    def add(self, frame_stack, action, reward, next_frame_stack, done):
        """Stores the stacked frames instead of a single frame."""
        if not isinstance(frame_stack, torch.Tensor):
            frame_stack = preprocess(frame_stack, self.mode)
        if not isinstance(next_frame_stack, torch.Tensor):
            next_frame_stack = preprocess(next_frame_stack, self.mode)
        action = torch.tensor(action)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(1 if done else 0)

        self.buffer.append((frame_stack, action, reward, next_frame_stack, done))

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
        return gym.make(env_name, **env_args)

    def get_model_type(self, env_name: str):
        return self.get_env_config(env_name)["model"]

    def get_spaces(self, env_name: str):
        config = self.get_env_config(env_name)
        return config["action_space"], config["observation_space"]


class FrameStack:
    def __init__(self, stack_size, mode):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.mode = mode

    def reset(self, state):
        processed_state = preprocess(state, mode=self.mode)
        for _ in range(self.stack_size):
            self.frames.append(processed_state)
        return torch.cat(list(self.frames), dim=-1)

    def update(self, state):
        processed_state = preprocess(state, mode=self.mode)
        self.frames.append(processed_state)
        return torch.cat(list(self.frames), dim=-1)
