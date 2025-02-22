import json
import torch
import random
import numpy as np
import cv2
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
            current_state = current_state.to(device)
            estimated_returns = policy(current_state)
            action = estimated_returns.argmax()
            next_state, reward, done, truncated, _ = eval_env.step(action.item())
            current_state = frame_stack.update(next_state)
            episode_reward += float(reward)

        rewards.append(episode_reward)

    eval_env.close()
    avg_reward = sum(rewards) / num_episodes
    return avg_reward


def preprocess(state, mode):
    if mode == "ConvNet":
        # Convert RGB to grayscale
        gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        normalized = resized / 255.0

        # Convert to tensor and add channel dimension
        state_tensor = torch.FloatTensor(normalized).unsqueeze(-1)

        return state_tensor
    else:
        return torch.FloatTensor(state)


class ReplayBuffer:
    def __init__(self, capacity, stack_size=4, mode="MLP", device="mps"):
        self.buffer = deque(maxlen=capacity)
        self.stack_size = stack_size
        self.device = device
        self.mode = mode

    def add(self, frame_stack, action, reward, next_frame_stack, done):
        done = 1 if done else 0
        self.buffer.append((frame_stack, action, reward, next_frame_stack, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = list(zip(*batch))

        states = torch.stack(
            [
                torch.FloatTensor(s, device=self.device)
                if isinstance(s, np.ndarray)
                else s.to(self.device)
                for s in states
            ]
        ).to(self.device)
        actions = (
            torch.stack([torch.tensor(action) for action in actions])
            .unsqueeze(dim=1)
            .to(self.device)
        )
        rewards = (
            torch.stack([torch.tensor(reward) for reward in rewards])
            .unsqueeze(dim=1)
            .to(self.device)
        )
        next_states = torch.stack(
            [
                torch.FloatTensor(s, device=self.device)
                if isinstance(s, np.ndarray)
                else s.to(self.device)
                for s in next_states
            ]
        ).to(self.device)
        dones = torch.tensor(dones).unsqueeze(dim=1).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class EnvConfig:
    def __init__(self, config_path: str, eval=False):
        self.eval = eval
        with open(config_path, "r") as f:
            self.configs = json.load(f)

    def set_eval(self, eval):
        self.eval = eval

    def get_env_config(self, env_name: str):
        if env_name not in self.configs:
            raise ValueError(f"Environment {env_name} not found in config file.")
        return self.configs[env_name]

    def create_env(self, env_name: str):
        config = self.get_env_config(env_name)
        env_args = config.get("env_args", {})
        if self.eval:
            env_args["render_mode"] = "human"
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
            self.frames.append(processed_state.clone())

        stacked = torch.cat(list(self.frames), dim=-1)
        return stacked

    def update(self, state):
        processed_state = preprocess(state, mode=self.mode)
        self.frames.append(processed_state.clone())

        stacked = torch.cat(list(self.frames), dim=-1)
        return stacked

    def visualize_frames(self, tensor=None):
        import matplotlib.pyplot as plt

        if tensor is None:
            frames = self.frames
        else:
            # Handle tensor from replay buffer
            # Assuming tensor shape is [B, C, H, W] or [B, H, W, C]
            if len(tensor.shape) == 4:
                tensor = tensor[0]  # Take first batch
            # Split the stacked frames back into individual frames
            frames = torch.chunk(tensor, self.stack_size, dim=-1)

        num_frames = len(frames)
        fig, axes = plt.subplots(1, num_frames, figsize=(3 * num_frames, 3))

        if num_frames == 1:
            axes = [axes]

        for i, frame in enumerate(frames):
            frame_np = frame.cpu().squeeze().numpy()
            axes[i].imshow(frame_np, cmap="gray")
            axes[i].axis("off")
            axes[i].set_title(f"Frame {i+1}")

        plt.tight_layout()
        plt.show()


def soft_update(target_network, policy_network, tau=0.005):
    for target_param, policy_param in zip(
        target_network.parameters(), policy_network.parameters()
    ):
        target_param.data.copy_(
            tau * policy_param.data + (1.0 - tau) * target_param.data
        )
