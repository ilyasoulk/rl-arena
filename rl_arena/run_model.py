import torch
from models import MLP
from utils import EnvConfig

device = "mps"
policy = MLP(4, 128, 2).to(device)
env_config = EnvConfig("configs/envs.json")
# env_name = "LunarLander-v3"
env_name = "CartPole-v1"
model_path = "models/DQN-CartPole-v1.pth"
env = env_config.create_env(env_name)

# Reset env
obs, _ = env.reset()
policy.load_state_dict(torch.load(model_path, weights_only=True))

for _ in range(1000000):
    current_state = torch.tensor(obs).to(device)
    action = policy(current_state).argmax().item()
    obs, reward, done, truncated, info = env.step(action)

    if done or truncated:
        break

env.close()
