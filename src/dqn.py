import torch
import argparse
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from utils import get_spaces, ReplayBuffer


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


def train_dqn(
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
    reward_logs = []

    while total_steps < steps:
        obs, _ = env.reset()
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

            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += float(reward)  # Accumulate episode reward
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

        reward_logs.append(episode_reward)
        epsilon = max(min_eps, epsilon - decay)
        print(f"Epsilon value : {epsilon}, Cumulated reward : {episode_reward}")

    env.close()

    return reward_logs


if __name__ == "__main__":
    # State dim is 4, action space is 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str)
    parser.add_argument(
        "--hidden_dim", type=int
    )  # This might be changed later one when having multiple architectures
    parser.add_argument("--steps", type=int)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--decay", type=float)
    parser.add_argument("--min_eps", type=float)
    parser.add_argument("--batch_size", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--lr", type=float)

    args = parser.parse_args()

    action_space, observation_space = get_spaces(args.env_name)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    target = QNet(observation_space, args.hidden_dim, action_space).to(device)
    main = QNet(observation_space, args.hidden_dim, action_space).to(device)

    optimizer = torch.optim.Adam(main.parameters(), lr=args.lr)

    logs = train_dqn(
        target,
        main,
        optimizer,
        env_name=args.env_name,
        steps=args.steps,
        capacity=args.capacity,
        epsilon=args.epsilon,
        decay=args.decay,
        min_eps=args.min_eps,
        batch_size=args.batch_size,
        gamma=args.gamma,
    )
