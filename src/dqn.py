import gymnasium as gym
from collections import deque
import random
import torch.nn as nn
import torch.nn.functional as F

# Create environment
env = gym.make("CartPole-v1", render_mode="human")  # Use "human" for visualization

# Reset env
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Take random action
    obs, reward, done, truncated, info = env.step(action)

    if done or truncated:
        obs, info = env.reset()

env.close()


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.ln1 = nn.Linear(state_dim, hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        return self.ln3(x)
