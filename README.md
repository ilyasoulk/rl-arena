# 🎮 RL Arena

A comprehensive implementation of reinforcement learning algorithms following OpenAI's Spinning Up guide. This project aims to provide clear, educational implementations of various RL algorithms while maintaining good software engineering practices.

## 🚀 Currently Implemented
- [x] Deep Q-Network (DQN)
  - Successfully tested on CartPole-v1
  - Includes experience replay
  - Target network for stability
  - Epsilon-greedy exploration

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

## 📁 Project Structure
```
ilyasoulk-rl-arena/
├── README.md
├── launch_cartpole_dqn.sh       # Launch script for DQN experiment
├── pyproject.toml               # Project dependencies and metadata
├── uv.lock                      # Dependency lock file
└── src/
    ├── dqn.py                   # DQN implementation
    └── utils.py                 # Shared utilities
```

## 🔧 Technical Details

### DQN Implementation Features
- Experience replay buffer with configurable capacity
- Target network with periodic updates
- Epsilon-greedy exploration with decay
- Gradient clipping for stability
- Support for various gym environments through space detection

## 📚 References

- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Gymnasium (formerly Gym)](https://gymnasium.farama.org/)

## 📝 License

MIT License
