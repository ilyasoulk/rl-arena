import torch
import ale_py
import argparse
import gymnasium as gym
from rl_arena.utils import EnvConfig
from rl_arena import DQNAgent, VPGAgent, TRPOAgent, PPOAgent
import rl_arena.models as models


if __name__ == "__main__":
    gym.register_envs(ale_py)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        choices=["DQN", "VPG", "TRPO", "PPO"],
        help="RL algorithm to use",
    )
    parser.add_argument("--env_name", type=str, help="Environment name")
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Hidden dimension size"
    )
    parser.add_argument(
        "--steps", type=int, default=100000, help="Total number of training steps"
    )
    parser.add_argument(
        "--capacity", type=int, default=100000, help="Replay buffer capacity (for DQN)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Initial epsilon for epsilon-greedy (DQN) or clip parameter (PPO)",
    )
    parser.add_argument(
        "--decay", type=float, default=0.0001, help="Epsilon decay rate (DQN)"
    )
    parser.add_argument(
        "--update_frequency",
        type=int,
        default=100,
        help="Target network update frequency (DQN)",
    )
    parser.add_argument(
        "--min_eps", type=float, default=0.01, help="Minimum epsilon value (DQN)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for updates"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument(
        "--output_dir", type=str, default="./models", help="Directory to save models"
    )
    parser.add_argument(
        "--solved_threshold",
        type=float,
        default=float("inf"),
        help="Reward threshold to consider environment solved",
    )
    parser.add_argument(
        "--num_frame_stack", type=int, default=1, help="Number of frames to stack"
    )
    parser.add_argument(
        "--use_critic", action="store_true", help="Use critic network (for VPG)"
    )
    parser.add_argument(
        "--max_kl", type=float, default=0.01, help="Maximum KL divergence (for TRPO)"
    )
    parser.add_argument(
        "--cg_iters",
        type=int,
        default=10,
        help="Conjugate gradient iterations (for TRPO)",
    )
    parser.add_argument(
        "--ppo_epochs", type=int, default=10, help="Number of PPO epochs per update"
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=64,
        help="Mini-batch size for PPO updates",
    )

    args = parser.parse_args()

    # Load environment configuration
    env_config = EnvConfig("configs/envs.json")
    action_space, observation_space = env_config.get_spaces(args.env_name)
    print(f"Action space: {action_space}\nObservation space: {observation_space}")

    # Determine device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Create model based on environment type
    model_type = env_config.get_model_type(args.env_name)
    model_class = getattr(models, model_type)

    # Determine input dimension based on model type and frame stacking
    if model_type == "ConvNet":
        in_dim = observation_space[-1] * args.num_frame_stack
    else:
        in_dim = observation_space

    # Create policy network
    policy = model_class(in_dim, args.hidden_dim, action_space).to(device)

    # Create optimizer
    if args.method == "DQN":
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # Create and train agent based on method
    if args.method == "DQN":
        # Create target network for DQN
        target = model_class(in_dim, args.hidden_dim, action_space).to(device)
        target.load_state_dict(policy.state_dict())
        target.eval()

        # Create DQN agent
        agent = DQNAgent(
            policy=policy,
            target=target,
            env_name=args.env_name,
            env_config=env_config,
            steps=args.steps,
            capacity=args.capacity,
            epsilon=args.epsilon,
            update_frequency=args.update_frequency,
            decay=args.decay,
            min_eps=args.min_eps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            optimizer=optimizer,
            num_frame_stack=args.num_frame_stack,
            solved_threshold=args.solved_threshold,
            output_dir=args.output_dir,
            device=device,
        )

    elif args.method == "VPG":
        # Create critic network if needed
        critic = None
        if args.use_critic:
            critic = model_class(in_dim, args.hidden_dim, 1).to(device)

        # Create VPG agent
        agent = VPGAgent(
            policy=policy,
            critic=critic,
            env_name=args.env_name,
            env_config=env_config,
            steps=args.steps,
            gamma=args.gamma,
            optimizer=optimizer,
            num_frame_stack=args.num_frame_stack,
            solved_threshold=args.solved_threshold,
            output_dir=args.output_dir,
            device=device,
        )

    elif args.method == "TRPO":
        # Create old policy and critic networks
        old_policy = model_class(in_dim, args.hidden_dim, action_space).to(device)
        old_policy.load_state_dict(policy.state_dict())
        critic = model_class(in_dim, args.hidden_dim, 1).to(device)

        # Create TRPO agent
        agent = TRPOAgent(
            policy=policy,
            old_policy=old_policy,
            critic=critic,
            env_name=args.env_name,
            env_config=env_config,
            steps=args.steps,
            gamma=args.gamma,
            optimizer=optimizer,
            num_frame_stack=args.num_frame_stack,
            solved_threshold=args.solved_threshold,
            output_dir=args.output_dir,
            device=device,
            max_kl=args.max_kl,
            cg_iters=args.cg_iters,
        )

    elif args.method == "PPO":
        # Create old policy and critic networks
        old_policy = model_class(in_dim, args.hidden_dim, action_space).to(device)
        old_policy.load_state_dict(policy.state_dict())
        critic = model_class(in_dim, args.hidden_dim, 1).to(device)

        # Create PPO agent
        agent = PPOAgent(
            policy=policy,
            old_policy=old_policy,
            critic=critic,
            env_name=args.env_name,
            env_config=env_config,
            steps=args.steps,
            gamma=args.gamma,
            optimizer=optimizer,
            num_frame_stack=args.num_frame_stack,
            solved_threshold=args.solved_threshold,
            output_dir=args.output_dir,
            device=device,
            clip_param=args.epsilon,
            ppo_epochs=args.ppo_epochs,
            mini_batch_size=args.mini_batch_size,
        )

    # Train the agent
    print(f"Starting training with {args.method} on {args.env_name}...")
    train_logs, eval_logs = agent.train()
    print(f"Training completed. Final evaluation reward: {eval_logs[-1]:.2f}")

    # Save final model
    agent.save_model()
    print(f"Model saved to {args.output_dir}/{args.method}-{args.env_name}.pth")
