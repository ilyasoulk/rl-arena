import torch
from utils import FrameStack, eval


def compute_returns(rewards, gamma, device="mps"):
    T = len(rewards)
    returns = torch.zeros(T, device=device)
    future_return = 0

    for i in reversed(range(T)):
        future_return = rewards[i] + gamma * future_return
        returns[i] = future_return

    # Normalize
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


# This is the original vanilla who uses REINFORCE returns for the loss instead of advantages
# TODO : Add variations of "weight" : Action-Value function (Q(a,s)) or Advantage (A(s, a) = Q(s, a) - V(s))
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
    eval_freq = 10_000
    avg_eval_rewards = 0

    while total_steps < steps:
        current_state, _ = env.reset()
        current_state = frame_stack.reset(current_state)
        print(current_state.shape)
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
                ):  # This is the score at which we consider env to be solved
                    print(f"{env_name} has been solved, saving the policy...")
                    torch.save(
                        policy.state_dict(), output_dir + "/VPG-" + env_name + ".pth"
                    )
                    return train_reward_logs, eval_reward_logs

            logits = policy(current_state)

            # Sampling from distribution is the equivalent of eps-greedy in DQN it allows for exploration/exploitation
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            logprob = distribution.log_prob(action)

            obs, reward, done, truncated, _ = env.step(action.item())
            episode_batch.append((logprob, reward))
            episode_reward += float(reward)
            current_state = frame_stack.update(obs)

        logprobs, rewards = list(zip(*episode_batch))

        returns = compute_returns(rewards, gamma=gamma, device=device)
        logprobs = torch.stack(logprobs)
        loss = -(logprobs * returns).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            f"[{total_steps} steps] Loss : {loss.item()} | Episode Reward : {episode_reward} | Avg Eval Reward : {avg_eval_rewards}"
        )

    return train_reward_logs, eval_reward_logs
