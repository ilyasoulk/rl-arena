from math import inf
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from rl_arena.utils import ReplayBuffer, soft_update
from rl_arena.agents.base_agent import RLAgent


class DQNAgent(RLAgent):
    def __init__(
        self,
        policy,
        target,
        capacity=100_000,
        epsilon=1.0,
        update_frequency=100,
        decay=0.0001,
        min_eps=0.01,
        batch_size=32,
        tau=0.001,
        **kwargs,
    ):
        super().__init__(policy=policy, **kwargs)
        self.target = target
        self.capacity = capacity
        self.epsilon = epsilon
        self.update_frequency = update_frequency
        self.decay = decay
        self.min_eps = min_eps
        self.batch_size = batch_size
        self.tau = tau

        self.replay_buffer = ReplayBuffer(
            capacity, mode=self.model_type, device=self.device
        )
        self.warm_up = 1000

    def select_action(self, state):
        if torch.rand(1).item() > self.epsilon:
            with torch.no_grad():
                action = self.policy(state).argmax().item()
        else:
            env = self.create_env()
            action = env.action_space.sample()
            env.close()

        return action, ()

    def update(self, batch=None):
        if len(self.replay_buffer) <= self.warm_up:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        current_q_values = self.policy(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy(next_states).argmax(dim=1, keepdim=True)
            next_values = self.target(next_states).gather(1, next_actions)
            targets = rewards + self.gamma * next_values * (1 - dones)

        loss = F.smooth_l1_loss(current_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy.parameters(), max_norm=10)
        self.optimizer.step()

        if self.total_steps % self.update_frequency == 0:
            self.target.load_state_dict(self.policy.state_dict())

        self.epsilon = max(self.min_eps, self.epsilon - self.decay)

        return loss.item(), 0.0

    def train(self):
        env = self.create_env()
        loss = float(inf)

        while self.total_steps < self.steps:
            current_state, _ = env.reset()
            current_state = self.frame_stack.reset(current_state)
            done = False
            truncated = False
            episode_reward = 0

            while not (done or truncated):
                self.total_steps += 1
                if self.total_steps > self.steps:
                    break

                if self.total_steps % self.eval_freq == 0:
                    if self.evaluate():
                        env.close()
                        return self.train_reward_logs, self.eval_reward_logs

                current_state = current_state.to(self.device)
                action, _ = self.select_action(current_state)

                obs, reward, done, truncated, _ = env.step(action)
                next_state = self.frame_stack.update(obs)
                episode_reward += float(reward)

                self.replay_buffer.add(current_state, action, reward, next_state, done)
                current_state = next_state

                if len(self.replay_buffer) > self.warm_up:
                    loss = self.update()

            self.train_reward_logs.append(episode_reward)
            print(
                f"[{self.total_steps} steps] Epsilon: {self.epsilon:.4f} | "
                f"Episode Reward: {episode_reward:.2f} | "
                f"Avg Eval Reward: {self.avg_eval_rewards:.2f} | "
                f"Loss : {loss}"
            )

        env.close()
        return self.train_reward_logs, self.eval_reward_logs

    def save_model(self):
        torch.save(
            {"policy": self.policy.state_dict(), "target": self.target.state_dict()},
            f"{self.output_dir}/DQN-{self.env_name}.pth",
        )
