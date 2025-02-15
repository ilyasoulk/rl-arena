from gymnasium.spaces import Discrete
import torch
import models
import argparse
import numpy as np
import gymnasium as gym
import torch.nn.functional as F
from utils import get_spaces, ReplayBuffer, EnvConfig
from torch.nn.utils import clip_grad_norm_


@torch.no_grad()
def eval_dqn(model, num_episodes, env_name, env_config):
    print("Evaluating")
    eval_env = env_config.create_env(env_name)
    rewards = []
    for _ in range(num_episodes):
        current_state, _ = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not (done or truncated):
            input = torch.tensor(current_state).to(device)
            action = model(input).argmax().item()
            current_state, reward, done, truncated, _ = eval_env.step(action)
            episode_reward += float(reward)  # Accumulate episode reward

        rewards.append(episode_reward)

    eval_env.close()

    avg_reward = sum(rewards) / num_episodes
    return avg_reward


def train_dqn(
    target,
    main,
    optimizer,
    env_config,
    solved_threshold,
    env_name="CartPole-v1",
    steps=100_000,
    capacity=100_000,
    epsilon=1,
    update_frequency=100,
    decay=0.0001,
    min_eps=0.01,
    batch_size=32,
    gamma=0.99,
    output_dir="models",
    device="mps",
):
    env = env_config.create_env(env_name)
    replay_buffer = ReplayBuffer(capacity)
    min_experiences = 100
    eval_freq = 1000
    total_steps = 0
    train_reward_logs = []
    eval_reward_logs = []
    avg_eval_rewards = 0

    while total_steps < steps:
        current_state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            total_steps += 1
            if total_steps > steps:
                break

            if total_steps % eval_freq == 0:
                avg_eval_rewards = eval_dqn(
                    main, num_episodes=10, env_name=env_name, env_config=env_config
                )
                eval_reward_logs.append(avg_eval_rewards)
                if (
                    avg_eval_rewards > solved_threshold
                ):  # This is the score at which we consider CartPole-v1 solved
                    print(f"{env_name} has been solved, saving the Q-function...")
                    torch.save(main.state_dict(), output_dir + "/" + env_name)
                    return train_reward_logs, eval_reward_logs

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

                    next_values = target(next_states)
                    next_values_max = torch.max(next_values, dim=1).values.unsqueeze(
                        dim=1
                    )
                    targets = rewards + gamma * next_values_max * (1 - dones)

                loss = F.smooth_l1_loss(current_q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(main.parameters(), max_norm=1)
                optimizer.step()
            epsilon = max(min_eps, epsilon - decay)

        train_reward_logs.append(episode_reward)

        print(
            f"[{total_steps} step] Epsilon value : {epsilon}, Cumulated train reward : {episode_reward}, Average eval reward : {avg_eval_rewards}"
        )

    env.close()

    return train_reward_logs, eval_reward_logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str)
    parser.add_argument(
        "--hidden_dim", type=int
    )  # This might be changed later one when having multiple architectures
    parser.add_argument("--steps", type=int)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--decay", type=float)
    parser.add_argument("--update_frequency", type=int)
    parser.add_argument("--min_eps", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--solved_threshold", type=float)

    args = parser.parse_args()

    env_config = EnvConfig("configs/envs.json")

    action_space, observation_space = get_spaces(env_config, args.env_name)

    print(f"Action space : {action_space}\nObservation space : {observation_space}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model_type = env_config.get_model_type(args.env_name)
    model_class = getattr(models, model_type)
    target = model_class(observation_space, args.hidden_dim, action_space).to(device)
    main = model_class(observation_space, args.hidden_dim, action_space).to(device)

    optimizer = torch.optim.Adam(main.parameters(), lr=args.lr)

    logs = train_dqn(
        target,
        main,
        optimizer,
        env_config=env_config,
        env_name=args.env_name,
        steps=args.steps,
        capacity=args.capacity,
        epsilon=args.epsilon,
        update_frequency=args.update_frequency,
        decay=args.decay,
        min_eps=args.min_eps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        gamma=args.gamma,
        solved_threshold=args.solved_threshold,
    )
