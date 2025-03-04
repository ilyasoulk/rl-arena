# 🎮 RL Arena

A comprehensive implementation of reinforcement learning algorithms following OpenAI's Spinning Up guide. This project aims to provide clear, educational implementations of various RL algorithms while maintaining good software engineering practices.


## 🎯 Roadmap
Planning to implement the following algorithms from OpenAI's Spinning Up (and more):
- [x] Deep-Q-Network (DQN)
- [x] Vanilla Policy Gradient (VPG)
- [x] Trust Region Policy Optimization (TRPO)
- [x] Proximal Policy Optimization (PPO)
- [ ] Group Relative Policy Optimization (GRPO)
- [ ] Deep Deterministic Policy Gradient (DDPG)
- [ ] Twin Delayed DDPG (TD3)
- [ ] Soft Actor-Critic (SAC)

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


### TRPO on CartPole

```bash
./launch_cartpole_trpo.sh
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
- Actor-Critic method implementation as well
- Advantages instead of rewards-to-go implementation
- Return normalization for training stability
- Stochastic action sampling using Categorical distribution
- Frame stacking support for temporal information

### TRPO Implementation Features
- Natural gradient updates that respect the geometry of the policy space
- Trust region constraint to prevent destructive policy updates
- Conjugate gradient method for efficiently computing Fisher-vector products
- Line search to find the optimal step size that satisfies the KL constraint
- Advantage normalization for training stability
- Separate networks for policy and value function

### PPO Implementation Features
- Clipped surrogate objective to constrain policy updates
- Multiple epochs of minibatch updates for sample efficiency
- Entropy bonus for exploration
- Advantage normalization for training stability

## 📈 Performance Comparisons

| Algorithm | Environment  | Steps to Solve |
|-----------|-------------|----------------|
| DQN       | CartPole-v1 | ~18,000       |
| VPG       | CartPole-v1 | ~11,000       |
| TRPO      | CartPole-v1 | ~5,000        |
| PPO       | CartPole-v1 | ~4,000        |
## 📚 References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)
- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Gymnasium (formerly Gym)](https://gymnasium.farama.org/)

## 📝 License

MIT License
