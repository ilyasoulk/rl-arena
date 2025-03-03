from typing import Tuple, List
import torch
from abc import ABC, abstractmethod
from rl_arena.utils import FrameStack, eval


class RLAgent(ABC):
    def __init__(
        self,
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
        self.policy = policy
        self.env_name = env_name
        self.env_config = env_config
        self.steps = steps
        self.gamma = gamma
        self.optimizer = optimizer
        self.num_frame_stack = num_frame_stack
        self.solved_threshold = solved_threshold
        self.output_dir = output_dir
        self.device = device

        # Common attributes
        self.model_type = env_config.get_model_type(env_name)
        self.frame_stack = FrameStack(num_frame_stack, mode=self.model_type)
        self.train_reward_logs = []
        self.eval_reward_logs = []
        self.total_steps = 0
        self.eval_freq = 1000
        self.avg_eval_rewards = 0

    def create_env(self):
        return self.env_config.create_env(self.env_name)

    def evaluate(self):
        self.avg_eval_rewards = eval(
            self.policy,
            num_episodes=10,
            env_name=self.env_name,
            env_config=self.env_config,
            num_frame_stack=self.num_frame_stack,
            device=self.device,
        )
        self.eval_reward_logs.append(self.avg_eval_rewards)

        if self.avg_eval_rewards > self.solved_threshold:
            print(f"{self.env_name} has been solved, saving the policy...")
            self.save_model()
            return True
        return False

    def save_model(self):
        torch.save(
            self.policy.state_dict(),
            f"{self.output_dir}/{self.__class__.__name__}-{self.env_name}.pth",
        )

    @abstractmethod
    def train(self) -> Tuple[List, List]:
        pass

    @abstractmethod
    def update(self, batch) -> Tuple[float, float]:
        pass

    def compute_returns(self, rewards, dones):
        rewards = torch.tensor(rewards, device=self.device)
        T = len(rewards)
        returns = torch.zeros(T, device=self.device)
        future_return = 0

        for t in reversed(range(T)):
            if dones[t]:
                future_return = 0

            future_return = rewards[t] + self.gamma * future_return
            returns[t] = future_return
            # Reset future_return if this is the last step of an episode

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def collect_episode(self):
        env = self.create_env()
        current_state, _ = env.reset()
        current_state = self.frame_stack.reset(current_state)
        done = False
        truncated = False
        episode_reward = 0
        episode_batch = []

        while not (done or truncated):
            self.total_steps += 1
            if self.total_steps > self.steps:
                break

            if self.total_steps % self.eval_freq == 0:
                if self.evaluate():
                    env.close()
                    return episode_batch, episode_reward, True  # Environment solved

            current_state = current_state.to(self.device)
            action, log_data = self.select_action(current_state)

            obs, reward, done, truncated, _ = env.step(action)
            episode_batch.append((current_state, *log_data, done, reward))
            episode_reward += float(reward)
            current_state = self.frame_stack.update(obs)

        env.close()
        return episode_batch, episode_reward, False

    @abstractmethod
    def select_action(self, state) -> Tuple[float, Tuple]:
        pass
