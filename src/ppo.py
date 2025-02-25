import torch
from vpg import compute_returns
import torch.nn.functional as F
from utils import FrameStack, eval


def ppo(
    policy,
    old_policy,
    critic,
    eps,
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
    torch.autograd.set_detect_anomaly(True)
    env = env_config.create_env(env_name)
    model_type = env_config.get_model_type(env_name)
    frame_stack = FrameStack(num_frame_stack, mode=model_type)
    train_reward_logs = []
    eval_reward_logs = []
    total_steps = 0
    eval_freq = 500
    avg_eval_rewards = 0

    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-3)

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
                ):  # This is the score at which we consider env to be solved
                    print(
                        f"{env_name} has been solved, saving the policy with average reward : {avg_eval_rewards}..."
                    )
                    torch.save(
                        policy.state_dict(), output_dir + "/PPO-" + env_name + ".pth"
                    )
                    return train_reward_logs, eval_reward_logs

            current_state = current_state.to(device)
            logits = policy(current_state)

            value = critic(current_state)

            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()

            logprob = distribution.log_prob(action)

            obs, reward, done, truncated, _ = env.step(action.item())
            episode_batch.append(
                (
                    current_state,
                    logprob,
                    logits,
                    reward,
                    value,
                    action,
                )
            )
            episode_reward += float(reward)
            current_state = frame_stack.update(obs)

        (
            observations,
            logprobs,
            logits,
            rewards,
            values,
            actions,
        ) = list(zip(*episode_batch))

        returns = compute_returns(rewards, gamma=gamma, device=device)
        values = torch.stack(values)
        advantages = (
            returns - values.squeeze().detach()
        )  # Avoids computing gradients for the value function
        advantages = (advantages - advantages.mean()) / (advantages.std())
        logprobs = torch.stack(logprobs)
        logits = torch.stack(logits)

        distribution = torch.distributions.Categorical(logits=logits)
        observations = torch.stack(observations)
        old_logits = old_policy(observations)
        old_distribution = torch.distributions.Categorical(logits=old_logits)
        actions = torch.stack(actions)
        old_logprobs = old_distribution.log_prob(actions)

        ratio = torch.exp(logprobs - old_logprobs.detach())
        left_side = ratio * advantages
        right_side = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
        loss = -torch.min(left_side, right_side).mean()
        old_policy.load_state_dict(policy.state_dict())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        critic_loss = F.mse_loss(values.squeeze(), returns)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        print(
            f"[{total_steps} steps] Loss : {-loss.item()} | Episode Reward : {episode_reward} | Avg Eval Reward : {avg_eval_rewards}"
        )

    return train_reward_logs, eval_reward_logs
