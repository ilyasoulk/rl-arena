# 🎮 RL Arena

A comprehensive implementation of reinforcement learning algorithms following OpenAI's Spinning Up guide. This project aims to provide clear, educational implementations of various RL algorithms while maintaining good software engineering practices.

## 🚀 Currently Implemented
- [x] Deep Q-Network (DQN)
  - Successfully tested on CartPole-v1 (solved in ~18k steps) and LunarLander-v3
  - Includes experience replay
  - Target network for stability
  - Epsilon-greedy exploration
  - L1 smooth loss
  - Frame stacking support

- [x] Vanilla Policy Gradient (VPG)
  - Successfully tested on CartPole-v1 (solved in ~11k steps)
  - Direct policy optimization using REINFORCE
  - Return normalization for stability
  - Stochastic action sampling for exploration
  - Frame stacking support
  - Periodic evaluation during training

## 🎯 Roadmap
Planning to implement the following algorithms from OpenAI's Spinning Up (and more):
- [x] Deep-Q-Network (DQN)
- [x] Vanilla Policy Gradient (VPG)
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
python src/main.py \
    --method DQN \
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
    --output_dir models \
    --solved_threshold 475
```

### VPG on CartPole

```bash
./launch_cartpole_vpg.sh
```

Or manually with custom parameters:

```bash
python src/main.py \
    --method VPG \
    --env_name CartPole-v1 \
    --hidden_dim 128 \
    --steps 50000 \
    --gamma 0.99 \
    --lr 0.001 \
    --num_frame_stack 1 \
    --solved_threshold 475 \
    --output_dir models
```

## 📁 Project Structure

```
Directory structure:
└── ilyasoulk-rl-arena/
    ├── README.md
    ├── LICENSE
    ├── launch_carracing_dqn.sh
    ├── launch_cartpole_dqn.sh
    ├── launch_cartpole_vpg.sh
    ├── launch_lunarlander_dqn.sh
    ├── launch_pong_dqn.sh
    ├── pyproject.toml
    ├── uv.lock
    ├── configs/
    │   └── envs.json
    ├── models/
    │   ├── DQN-CartPole-v1.pth
    │   ├── DQN-LunarLander-v3.pth
    │   └── VPG-CartPole-v1.pth
    └── src/
        ├── dqn.py
        ├── main.py
        ├── models.py
        ├── utils.py
        └── vpg.py
```

## 🔧 Technical Details

### DQN Implementation Features
- Experience replay buffer with configurable capacity
- Target network with periodic updates
- Epsilon-greedy exploration with decay
- Gradient clipping for stability
- Support for various gym environments through space detection
- Frame stacking for temporal information
- Configurable network architectures (MLP/CNN)

### VPG Implementation Features
- Direct policy optimization using REINFORCE algorithm
- Return normalization for training stability
- Stochastic action sampling using Categorical distribution
- Frame stacking support for temporal information
- Periodic evaluation during training
- Model saving when environment is solved
- Support for both discrete and continuous action spaces

## 📈 Performance Comparisons

| Algorithm | Environment  | Steps to Solve |
|-----------|-------------|----------------|
| DQN       | CartPole-v1 | ~18,000       |
| VPG       | CartPole-v1 | ~11,000       |

## 📚 References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)
- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Gymnasium (formerly Gym)](https://gymnasium.farama.org/)

## 📝 License

MIT License
