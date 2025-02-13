import gymnasium as gym
from collections import deque
import random
import torch.nn as nn
import torch.nn.functional as F
import torch


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


def train(
    target,
    main,
    optimizer,
    env_name="CartPole-v1",
    steps=100_000,
    capacity=100_000,
    epsilon=1,
    decay=0.0001,
    min_eps=0.01,
    batch_size=32,
    gamma=0.99,
    device="mps",
):
    env = gym.make(env_name, render_mode="human")  # Use "human" for visualization
    replay_buffer = ReplayBuffer(capacity)
    min_experiences = 1000
    update_frequency = 1000
    total_steps = 0

    while total_steps < steps:
        obs, info = env.reset()
        current_state = obs
        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            total_steps += 1
            if total_steps > steps:
                break

            if total_steps % update_frequency == 0:
                print("Target weights are being updated")
                target.load_state_dict(main.state_dict())

            if torch.rand(1).item() > epsilon:
                input = torch.tensor(current_state).to(device)
                action = main(input).argmax().item()

            else:
                action = env.action_space.sample()  # Take random action

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward  # Accumulate episode reward
            replay_buffer.add(current_state, action, reward, obs, done)
            current_state = obs

            if len(replay_buffer) > min_experiences:
                current_states, actions, rewards, next_states, dones = (
                    replay_buffer.sample(batch_size)
                )

                current_states = current_states.to(device)
                actions = actions.to(device)
                current_q_values = main(current_states).gather(1, actions)
                with torch.no_grad():
                    next_states = next_states.to(device)
                    rewards = rewards.to(device)
                    dones = dones.to(device)
                    targets = rewards + gamma * target(next_states).max(dim=1)[0] * (
                        1 - dones
                    )

                loss = ((targets - current_q_values) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(min_eps, epsilon - decay)
        print(f"Epsilon value : {epsilon}, Cumulated reward : {episode_reward}")

    env.close()


if __name__ == "__main__":
    # State dim is 4, action space is 2
    state_dim = 4
    action_dim = 2
    hidden_dim = 20
    device = "mps"

    target = QNet(state_dim, hidden_dim, action_dim).to(device)
    main = QNet(state_dim, hidden_dim, action_dim).to(device)

    optimizer = torch.optim.Adam(main.parameters(), lr=1e-3)

    train(target, main, optimizer)
