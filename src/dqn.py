import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils import ReplayBuffer, FrameStack, preprocess, eval, soft_update


def dqn(
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
    num_frame_stack=1,
    device="mps",
):
    env = env_config.create_env(env_name)
    model_type = env_config.get_model_type(env_name)
    replay_buffer = ReplayBuffer(capacity, mode=model_type)
    frame_stack = FrameStack(stack_size=num_frame_stack, mode=model_type, device=device)
    eval_freq = 20_000
    warm_up = 12_500
    total_steps = 0
    train_reward_logs = []
    eval_reward_logs = []
    avg_eval_rewards = 0

    while total_steps < steps:
        current_state, _ = env.reset()
        current_state = frame_stack.reset(current_state)  # Initialize stacked frames
        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            total_steps += 1
            if total_steps > steps:
                break

            if total_steps % eval_freq == 0:
                avg_eval_rewards = eval(
                    main,
                    num_episodes=10,
                    env_name=env_name,
                    env_config=env_config,
                    num_frame_stack=num_frame_stack,
                    device=device,
                )
                eval_reward_logs.append(avg_eval_rewards)
                if (
                    avg_eval_rewards > solved_threshold
                ):  # This is the score at which we consider CartPole-v1 solved
                    print(f"{env_name} has been solved, saving the Q-function...")
                    torch.save(
                        main.state_dict(), output_dir + "/DQN-" + env_name + ".pth"
                    )
                    return train_reward_logs, eval_reward_logs

            if total_steps % update_frequency == 0:
                print("Target weights are being updated")
                soft_update(target, main)

            if torch.rand(1).item() > epsilon:
                inputs = preprocess(current_state, mode=model_type).to(device)
                action = main(inputs).argmax().item()

            else:
                action = env.action_space.sample()  # Take random action

            obs, reward, done, truncated, _ = env.step(action)

            obs = frame_stack.update(obs)
            episode_reward += float(reward)  # Accumulate episode reward
            replay_buffer.add(current_state, action, reward, obs, done)
            current_state = obs

            if len(replay_buffer) > warm_up:
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

                loss = F.mse_loss(current_q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(main.parameters(), max_norm=10)
                optimizer.step()
                epsilon = max(min_eps, epsilon - decay)

        train_reward_logs.append(episode_reward)

        print(
            f"[{total_steps} step] Epsilon value : {epsilon}, Cumulated train reward : {episode_reward}, Average eval reward : {avg_eval_rewards}"
        )

    env.close()

    return train_reward_logs, eval_reward_logs
