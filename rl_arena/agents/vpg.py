import torch
import torch.nn.functional as F
from rl_arena.agents.base_agent import RLAgent


class VPGAgent(RLAgent):
    def __init__(self, policy, critic=None, **kwargs):
        super().__init__(policy=policy, **kwargs)
        self.critic = critic
        if self.critic:
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def select_action(self, state):
        logits = self.policy(state)
        if self.critic:
            value = self.critic(state)
        else:
            value = torch.tensor(0)

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

        # Normalize
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self, batch, episode_lengths=None):
        logprobs, rewards, values = list(
            zip(*[(item[1], item[-1], item[3]) for item in batch])
        )

        returns = self.compute_returns(rewards)
        values = torch.stack(values).squeeze().to(self.device)
        advantages = returns - values.detach()  # If no critic, values = 0
        logprobs = torch.stack(logprobs)

        loss = -(logprobs * advantages).mean()
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        if self.critic and self.critic_opt:
            self.critic_opt.zero_grad()
            critic_loss = F.mse_loss(values.float(), returns)
            critic_loss.backward()
            self.critic_opt.step()
            return loss.item(), critic_loss.item()

        return loss.item(), 0.0

    def train(self):
        while self.total_steps < self.steps:
            episode_batch, episode_reward, solved = self.collect_episode()
            if solved:
                return self.train_reward_logs, self.eval_reward_logs

            loss, critic_loss = self.update(episode_batch)
            self.train_reward_logs.append(episode_reward)

            print(
                f"[{self.total_steps} steps] Loss: {-loss:.5f} | "
                f"Episode Reward: {episode_reward:.2f} | "
                f"Avg Eval Reward: {self.avg_eval_rewards:.2f}"
            )
            if self.critic:
                print(f"State-Value Loss: {critic_loss:.5f}")

        return self.train_reward_logs, self.eval_reward_logs
