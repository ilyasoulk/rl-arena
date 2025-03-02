import torch
import torch.nn.functional as F
from rl_arena.agents.base_agent import RLAgent


class PPOAgent(RLAgent):
    def __init__(
        self,
        policy,
        old_policy,
        critic,
        clip_param=0.1,
        ppo_epochs=5,
        mini_batch_size=32,
        entropy_coef=0.01,
        target_steps=512,
        **kwargs,
    ):
        super().__init__(policy=policy, **kwargs)
        self.old_policy = old_policy
        self.critic = critic
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.target_steps = target_steps
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def select_action(self, state):
        logits = self.policy(state)
        value = self.critic(state)

        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        logprob = distribution.log_prob(action)

        return action.item(), (logprob, logits, value, action)

    def compute_returns(self, rewards, dones):
        rewards = torch.tensor(rewards, device=self.device)
        T = len(rewards)
        returns = torch.zeros(T, device=self.device)
        future_return = 0

        for t in reversed(range(T)):
            future_return = rewards[t] + self.gamma * future_return
            returns[t] = future_return
            # Reset future_return if this is the last step of an episode
            if dones[t]:
                future_return = 0

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self, batch):
        observations, _, logits, values, actions, dones, rewards = zip(
            *[
                (item[0], item[1], item[2], item[3], item[4], item[5], item[6])
                for item in batch
            ]
        )

        returns = self.compute_returns(rewards, dones)
        values = torch.stack(values).squeeze()
        advantages = returns - values.detach()  # Detach values to avoid graph issues
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        observations = torch.stack(observations)
        actions = torch.stack(actions)

        policy_loss = torch.tensor(0)
        value_loss = torch.tensor(0)

        self.old_policy.load_state_dict(self.policy.state_dict())

        for _ in range(self.ppo_epochs):
            batch_size = len(observations)
            indices = torch.randperm(batch_size)

            for start_idx in range(0, batch_size, self.mini_batch_size):
                idx = indices[start_idx : start_idx + self.mini_batch_size]

                mb_obs = observations[idx]
                mb_actions = actions[idx]
                mb_advantages = advantages[idx]

                old_logits = self.old_policy(mb_obs)
                old_dist = torch.distributions.Categorical(logits=old_logits)
                old_logprobs = old_dist.log_prob(mb_actions)

                # Policy update
                logits = self.policy(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - old_logprobs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * mb_advantages
                )

                policy_loss = (
                    -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                policy_loss.backward()  # Remove retain_graph=True
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

            # Critic update (separate backward pass)
            new_values = self.critic(observations).squeeze()
            value_loss = F.mse_loss(new_values, returns)

            self.critic_opt.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_opt.step()

        return policy_loss.item(), value_loss.item()

    def train(self):
        while self.total_steps < self.steps:
            dataset = []
            collected_steps = 0
            while collected_steps < self.target_steps:
                episode_batch, episode_reward, solved = self.collect_episode()
                if solved:
                    return self.train_reward_logs, self.eval_reward_logs

                collected_steps += len(episode_batch)
                dataset.extend(episode_batch)
                self.train_reward_logs.append(episode_reward)

            policy_loss, value_loss = self.update(dataset)

            print(
                f"[{self.total_steps} steps] Policy Loss: {-policy_loss:.5f} | "
                f"Value Loss: {value_loss:.5f} | "
                f"Episode Reward: {self.train_reward_logs[-1]:.2f} | "
                f"Avg Eval Reward: {self.avg_eval_rewards:.2f}"
            )

        return self.train_reward_logs, self.eval_reward_logs

    def save_model(self):
        torch.save(
            {"policy": self.policy.state_dict(), "critic": self.critic.state_dict()},
            f"{self.output_dir}/PPO-{self.env_name}.pth",
        )
