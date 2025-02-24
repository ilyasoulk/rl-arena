import torch
from utils import FrameStack, eval
from vpg import compute_returns
import torch.nn.functional as F


def update_policy(policy, natural_gradient, alpha):
    for param in policy.parameters():
        param.data += alpha * natural_gradient


def compute_fim(policy, new_distributions, old_distributions, v):
    kl = torch.distributions.kl_divergence(new_distributions, old_distributions).mean()

    grads = torch.autograd.grad(
        kl, policy.parameters(), create_graph=True, retain_graph=True
    )  # create graph allows us to compute higher order gradients
    grads = torch.cat([grad.view(-1) for grad in grads])  # Flatten

    grad_v_prod = (grads * v).sum()

    second_grads = torch.autograd.grad(
        grad_v_prod, policy.parameters(), retain_graph=True
    )
    fisher_v_prod = torch.cat([grad.contiguous().view(-1) for grad in second_grads])

    return fisher_v_prod


def conjugate_gradient(
    policy, new_distributions, old_distributions, v, n=10, condition=1e-10
):
    x = torch.zeros_like(v)
    r = v.clone()
    p = v.clone()

    for _ in range(n):
        f = compute_fim(policy, new_distributions, old_distributions, p)
        alpha = torch.dot(r, r) / torch.dot(p, f)

        x += alpha * p
        r_new = r - alpha * f

        beta = torch.dot(r_new, r_new) / torch.dot(r, r)
        r = r_new
        p = r + beta * p

        if torch.norm(r) < condition:
            break

    return x


def set_params(policy, params):
    idx = 0
    for param in policy.parameters():
        param_size = param.numel()
        param.data = params[idx : idx + param_size].reshape(param.shape)
        idx += param_size


def compute_surrogate(policy, observations, actions, advantages, old_logprobs):
    logits = policy(observations)
    dist = torch.distributions.Categorical(logits=logits)

    logprobs = dist.log_prob(actions)
    ratio = torch.exp(logprobs - old_logprobs)
    return torch.mean(ratio * advantages), dist


def line_search(
    policy,
    observations,
    actions,
    advantages,
    old_surrogate,
    old_logprobs,
    old_dist,
    natural_gradient,
    max_kl=0.01,
):
    original_params = torch.cat([param.data.view(-1) for param in policy.parameters()])

    for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625]:
        new_params = original_params + alpha * natural_gradient
        set_params(policy, new_params)

        new_surrogate, dist = compute_surrogate(
            policy, observations, actions, advantages, old_logprobs
        )
        kl = torch.distributions.kl_divergence(dist, old_dist).mean()

        if new_surrogate > old_surrogate and kl <= max_kl:
            return new_surrogate

        # If not acceptable, restore original parameters
        set_params(policy, original_params)

    return old_surrogate


def trpo(
    policy,
    old_policy,
    critic,
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
    eval_freq = 2_000
    avg_eval_rewards = 0

    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

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
                    print(f"{env_name} has been solved, saving the policy...")
                    torch.save(
                        policy.state_dict(), output_dir + "/VPG-" + env_name + ".pth"
                    )
                    return train_reward_logs, eval_reward_logs

            current_state = current_state.to(device)
            new_logits = policy(current_state)
            old_logits = old_policy(current_state)

            value = critic(current_state)

            old_distribution = torch.distributions.Categorical(logits=old_logits)
            action = old_distribution.sample()
            old_logprob = old_distribution.log_prob(action)

            new_distribution = torch.distributions.Categorical(logits=new_logits)
            new_logprob = new_distribution.log_prob(action)

            obs, reward, done, truncated, _ = env.step(action.item())
            episode_batch.append(
                (
                    current_state,
                    new_logprob,
                    old_logprob,
                    new_logits,
                    old_logits,
                    reward,
                    value,
                    action,
                )
            )
            episode_reward += float(reward)
            current_state = frame_stack.update(obs)

        (
            observations,
            new_logprobs,
            old_logprobs,
            new_logits,
            old_logits,
            rewards,
            values,
            actions,
        ) = list(zip(*episode_batch))

        returns = compute_returns(rewards, gamma=gamma, device=device)
        values = torch.stack(values)
        advantage = returns - values
        advantage = (advantage - advantage.mean()) / (advantage.std())
        old_logprobs = torch.stack(old_logprobs)
        new_logprobs = torch.stack(new_logprobs)

        new_logits = torch.stack(new_logits)
        old_logits = torch.stack(old_logits)

        new_distribution = torch.distributions.Categorical(logits=new_logits)
        old_distribution = torch.distributions.Categorical(logits=old_logits)
        observations = torch.stack(observations)

        # In the surrogate, the ratio is pi / pi_old, since we use log_probs this is equivalent to exp(new_logprobs - old_logprobs)
        ratio = torch.exp(new_logprobs - old_logprobs)
        loss = (
            ratio * advantage
        ).mean()  # We don't multiply by -1 because this is only useful when updating the parameters
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        policy_grad = torch.cat([param.grad.view(-1) for param in policy.parameters()])
        optimizer.zero_grad()

        critic_loss = F.mse_loss(values.squeeze(), returns)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        natural_gradient = conjugate_gradient(
            policy, new_distribution, old_distribution, policy_grad
        )

        actions = torch.stack(actions)
        new_surrogate = line_search(
            policy,
            observations,
            actions,
            advantage,
            loss,
            old_logprobs,
            old_distribution,
            natural_gradient,
        )

        update_status = new_surrogate > loss

        print(
            f"[{total_steps} steps] Line search status : {update_status} | Loss : {loss.item()} | Episode Reward : {episode_reward} | Avg Eval Reward : {avg_eval_rewards} | New surrogate : {new_surrogate}"
        )

    return train_reward_logs, eval_reward_logs
