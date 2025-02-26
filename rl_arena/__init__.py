# Import the base agent
from rl_arena.agents.base_agent import RLAgent

# Import the agents to make them available at the package level
from rl_arena.agents.dqn import DQNAgent
from rl_arena.agents.vpg import VPGAgent
from rl_arena.agents.trpo import TRPOAgent
from rl_arena.agents.ppo import PPOAgent

# You can also import other useful components
from rl_arena.utils import EnvConfig, eval, ReplayBuffer, FrameStack

# Define what gets imported with "from rl_arena import *"
__all__ = [
    "RLAgent",
    "DQNAgent",
    "VPGAgent",
    "TRPOAgent",
    "PPOAgent",
    "EnvConfig",
    "eval",
    "ReplayBuffer",
    "FrameStack",
]
