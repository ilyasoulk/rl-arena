import torch
import torch.nn.functional as F
from rl_arena.agents.base_agent import RLAgent


class TRPOAgent(RLAgent):
    def __init__(
        self,
        policy,
        old_policy,
        critic,
        max_kl=0.01,
        cg_iters=10,
        cg_damping=1e-3,
        **kwargs,
    ):
        super().__init__(policy=policy, **kwargs)
        self.old_policy = old_policy
        self.critic = critic
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-3)

    def select_action(self, state):
        logits = self.policy(state)
        value = self.critic(state)

        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        logprob = distribution.log_prob(action)

        return action.item(), (logprob, logits, value, action)

    def compute_returns(self, rewards):
        T = len(rewards)
        returns = torch.zeros(T, device=self.device)
        future_return = 0

        for i in reversed(range(T)):
            future_return = rewards[i] + self.gamma * future_return
            returns[i] = future_return

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def compute_fim(self, distributions, v):
        ref_logits = distributions.logits.detach()
        ref_distribution = torch.distributions.Categorical(logits=ref_logits)
        kl = torch.distributions.kl_divergence(distributions, ref_distribution).mean()

        grads = torch.autograd.grad(
            kl, self.policy.parameters(), create_graph=True, retain_graph=True
        )
        grads = torch.cat([grad.view(-1) for grad in grads])

        grad_v_prod = (grads * v).sum()

        second_grads = torch.autograd.grad(
            grad_v_prod, self.policy.parameters(), retain_graph=True
        )
        fisher_v_prod = torch.cat([grad.contiguous().view(-1) for grad in second_grads])

        return fisher_v_prod + self.cg_damping * v

    def conjugate_gradient(self, distributions, v, n=10, condition=1e-10):
        x = torch.zeros_like(v)
        r = v.clone()
        p = v.clone()

        for _ in range(n):
            f = self.compute_fim(distributions, p)
            alpha = torch.dot(r, r) / (torch.dot(p, f) + 1e-8)

            x += alpha * p
            r_new = r - alpha * f

            beta = torch.dot(r_new, r_new) / (torch.dot(r, r) + 1e-8)
            r = r_new
            p = r + beta * p

            if torch.norm(r) < condition:
                break

        return x

    def set_params(self, params):
        idx = 0
        for param in self.policy.parameters():
            param_size = param.numel()
            param.data = params[idx : idx + param_size].reshape(param.shape)
            idx += param_size

    def compute_surrogate(self, observations, actions, advantages, old_logprobs):
        logits = self.policy(observations)
        dist = torch.distributions.Categorical(logits=logits)

        logprobs = dist.log_prob(actions)
        ratio = torch.exp(logprobs - old_logprobs)
        return torch.mean(ratio * advantages), dist

    def line_search(
        self,
        observations,
        actions,
        advantages,
        old_surrogate,
        old_dist,
        natural_gradient,
    ):
        original_params = torch.cat(
            [param.data.view(-1) for param in self.policy.parameters()]
        )
        old_logprobs = old_dist.log_prob(actions)

        alphas = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        for alpha in alphas:
            new_params = original_params + alpha * natural_gradient
            self.set_params(new_params)

            new_surrogate, dist = self.compute_surrogate(
                observations, actions, advantages, old_logprobs
            )
            kl = torch.distributions.kl_divergence(dist, old_dist).mean()

            if new_surrogate > old_surrogate and kl <= self.max_kl:
                self.old_policy.load_state_dict(self.policy.state_dict())
                return new_surrogate, True
            self.set_params(original_params)

        self.old_policy.load_state_dict(self.policy.state_dict())
        return old_surrogate, False

    def update(self, batch):
        observations, logprobs, logits, values, actions, rewards = zip(
            *[(item[0], item[1], item[2], item[3], item[4], item[5]) for item in batch]
        )

        returns = self.compute_returns(rewards)
        values = torch.stack(values).squeeze()
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        logits = torch.stack(logits)
        logprobs = torch.stack(logprobs)

        distribution = torch.distributions.Categorical(logits=logits)

        loss = (logprobs * advantages).mean()
        grads = torch.autograd.grad(loss, self.policy.parameters(), retain_graph=True)
        policy_grad = torch.cat([grad.view(-1) for grad in grads])

        natural_gradient = self.conjugate_gradient(
            distribution, policy_grad, n=self.cg_iters
        )

        self.line_search(
            observations, actions, advantages, loss, distribution, natural_gradient
        )
        critic_loss = F.mse_loss(values, returns)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        return loss.item(), critic_loss.item()

    def train(self):
        while self.total_steps < self.steps:
            episode_batch, episode_reward, solved = self.collect_episode()
            if solved:
                return self.train_reward_logs, self.eval_reward_logs

            policy_loss, critic_loss = self.update(episode_batch)
            self.train_reward_logs.append(episode_reward)

            print(
                f"[{self.total_steps} steps] Policy Loss: {policy_loss:.5f} | Critic Loss: {critic_loss:.5f} | "
                f"Episode Reward: {self.train_reward_logs[-1]:.2f} | "
                f"Avg Eval Reward: {self.avg_eval_rewards:.2f}"
            )

        return self.train_reward_logs, self.eval_reward_logs

    def save_model(self):
        torch.save(
            {"policy": self.policy.state_dict(), "critic": self.critic.state_dict()},
            f"{self.output_dir}/TRPO-{self.env_name}.pth",
        )
