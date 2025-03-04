import torch
import torch.nn.functional as F
from rl_arena.agents.base_agent import RLAgent


class GRPOAgent(RLAgent):
    def __init__(
        self,
        policy,
        old_policy,
        clip_param=0.1,
        ppo_epochs=5,
        mini_batch_size=32,
        entropy_coef=0.01,
        target_steps=512,
        group_size=10,
        **kwargs,
    ):
        super().__init__(policy=policy, **kwargs)
        self.old_policy = old_policy
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.target_steps = target_steps
        self.group_size = group_size
        # TODO : Define reward env
        # self.reward_env =

    def select_action(self, state):
        logits = self.policy(state)

        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        logprob = distribution.log_prob(action)

        return action.item(), (logprob, logits, action)

    def reward_function(self, obs, action):
        # TODO : Complete the reward function using the reward env
        pass

    def update(self, batch):
        observations, _, logits, actions, dones, rewards = zip(
            *[(item[0], item[1], item[2], item[3], item[4], item[5]) for item in batch]
        )

        observations = torch.stack(observations)
        actions = torch.stack(actions)

        policy_loss = torch.tensor(0)

        self.old_policy.load_state_dict(self.policy.state_dict())

        for _ in range(self.ppo_epochs):
            batch_size = len(observations)
            indices = torch.randperm(batch_size)

            for start_idx in range(0, batch_size, self.mini_batch_size):
                # TODO : Use distribution to sample multiple actions and compute reward using reward function
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

        return policy_loss.item(), 0.0

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
