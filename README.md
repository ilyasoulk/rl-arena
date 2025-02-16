# 🎮 RL Arena

A comprehensive implementation of reinforcement learning algorithms following OpenAI's Spinning Up guide. This project aims to provide clear, educational implementations of various RL algorithms while maintaining good software engineering practices.

## 🚀 Currently Implemented
- [x] Deep Q-Network (DQN)
  - Successfully tested on CartPole-v1 and LunarLander-v3
  - Includes experience replay
  - Target network for stability
  - Epsilon-greedy exploration
  - L1 smooth loss

## 🎯 Roadmap
Planning to implement the following algorithms from OpenAI's Spinning Up (and more):
- [ ] Vanilla Policy Gradient (VPG)
- [ ] Trust Region Policy Optimization (TRPO)
- [ ] Proximal Policy Optimization (PPO)
- [ ] Deep Deterministic Policy Gradient (DDPG)
- [ ] Twin Delayed DDPG (TD3)
- [ ] Soft Actor-Critic (SAC)
- [ ] Group Relative Policy Optimization (GRPO)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/ilyasoulkd/rl-arena.git
cd rl-arena

# Install dependencies
uv sync
source .venv/bin/activate

```

## 🏃‍♂️ Running Experiments

### DQN on CartPole

```bash
./launch_cartpole_dqn.sh
```

Or manually with custom parameters:

```bash
python src/dqn.py \
    --env_name CartPole-v1 \
    --hidden_dim 128 \
    --steps 50000 \
    --capacity 10000 \
    --epsilon 1 \
    --update_frequency 1000 \
    --decay 0.001 \
    --min_eps 0.01 \
    --batch_size 32 \
    --gamma 0.99 \
    --lr 0.001 \
    --output_dir models
```

### DQN on LunarLander

```bash
./launch_lunarlander_dqn.sh
```

## 📁 Project Structure
```
Directory structure:
└── ilyasoulk-rl-arena/
    ├── README.md
    ├── LICENSE
    ├── launch_carracing_dqn.sh # Launch DQN CarRacing experiment
    ├── launch_cartpole_dqn.sh # CartPole-v1
    ├── launch_lunarlander_dqn.sh # LunarLander-v3
    ├── pyproject.toml # Project dependencies
    ├── uv.lock # Dependency lock
    ├── configs/
    │   └── envs.json # Env configs, parameters...
    ├── models/ # Models per env, currently only DQN models...
    │   ├── CartPole-v1.pth
    │   └── LunarLander-v3.pth
    └── src/
        ├── dqn.py # DQN implementation
        ├── models.py # Model architecture
        └── utils.py # Utils function for env configs, experience replay, action obs space inference.

```

## 🔧 Technical Details

### DQN Implementation Features
- Experience replay buffer with configurable capacity
- Target network with periodic updates
- Epsilon-greedy exploration with decay
- Gradient clipping for stability
- Support for various gym environments through space detection

## 📚 References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Gymnasium (formerly Gym)](https://gymnasium.farama.org/)

## 📝 License

MIT License
