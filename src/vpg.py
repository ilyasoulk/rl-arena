import torch
import torch.nn.functional as F
from utils import preprocess, FrameStack, eval


def compute_returns(rewards, gamma):
    T = len(rewards)
    returns = torch.zeros(T, device=rewards.device)
    future_return = 0

    for i in reversed(range(T)):
        future_return = rewards[i] + gamma * future_return
        returns[i] = future_return

    # TODO : Normalize returns
    return returns


# TODO : Use advantage instead of basic REINFORCE, or implement it on more advanced algorithms
def vpg(
    policy,
    env_name,
    env_config,
    steps,
    gamma,
    optimizer,
    num_frame_stack,
    solved_threshold,
    output_dir,
    device="mps",
):
    env = env_config.create_env(env_name)
    model_type = env_config.get_model_type(env_name)
    frame_stack = FrameStack(num_frame_stack, mode=model_type)
    train_reward_logs = []
    eval_reward_logs = []
    total_steps = 0
    eval_freq = 1000

    while total_steps < steps:
        current_state, _ = env.reset()
        current_state = frame_stack.reset(current_state)
        done = False
        truncated = False
        episode_reward = 0
        episode_batch = []

        while not (done or truncated):
            total_steps += 1
            if total_steps > steps:
                break

            if total_steps % eval_freq == 0:
                avg_eval_rewards = eval(
                    policy,
                    num_episodes=10,
                    env_name=env_name,
                    env_config=env_config,
                    num_frame_stack=num_frame_stack,
                )
                eval_reward_logs.append(avg_eval_rewards)
                if (
                    avg_eval_rewards > solved_threshold
                ):  # This is the score at which we consider CartPole-v1 solved
                    print(f"{env_name} has been solved, saving the Q-function...")
                    torch.save(policy.state_dict(), output_dir + "/VPG-" + env_name)
                    return train_reward_logs, eval_reward_logs

            inputs = preprocess(current_state, mode=model_type).to(device)
            inputs = frame_stack.update(inputs)
            logits = policy(inputs)

            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            logprob = distribution.log_prob(action)

            obs, reward, done, truncated, _ = env.step(action)
            episode_batch.append((logprob, reward))
            episode_reward += float(reward)
            current_state = obs

        logprobs, rewards = list(zip(*episode_batch))

        returns = compute_returns(rewards, gamma=gamma)
        logprobs = torch.stack(logprobs)
        loss = -(logprobs * returns).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            f"[{total_steps} steps] Loss : {loss.item()} | Episode Reward : {episode_reward}"
        )

    return train_reward_logs, eval_reward_logs
