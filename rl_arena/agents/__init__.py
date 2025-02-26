# Import the agents to make them available at the agents module level
from rl_arena.agents.dqn import DQNAgent
from rl_arena.agents.vpg import VPGAgent
from rl_arena.agents.trpo import TRPOAgent
from rl_arena.agents.ppo import PPOAgent
from rl_arena.agents.base_agent import RLAgent

__all__ = ["RLAgent", "DQNAgent", "VPGAgent", "TRPOAgent", "PPOAgent"]
