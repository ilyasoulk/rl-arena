import torch
import ale_py
import models
import argparse
import gymnasium as gym
from dqn import dqn
from vpg import vpg
from utils import EnvConfig

if __name__ == "__main__":
    gym.register_envs(ale_py)
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str)
    parser.add_argument("--env_name", type=str)
    parser.add_argument(
        "--hidden_dim", type=int
    )  # This might be changed later one when having multiple architectures
    parser.add_argument("--steps", type=int)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--decay", type=float)
    parser.add_argument("--update_frequency", type=int)
    parser.add_argument("--min_eps", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--solved_threshold", type=float)
    parser.add_argument("--num_frame_stack", type=int, default=1)

    args = parser.parse_args()

    env_config = EnvConfig("configs/envs.json")

    action_space, observation_space = env_config.get_spaces(args.env_name)

    print(f"Action space : {action_space}\nObservation space : {observation_space}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model_type = env_config.get_model_type(args.env_name)
    model_class = getattr(models, model_type)
    if model_type == "ConvNet":
        in_dim = observation_space[-1] * args.num_frame_stack
    else:
        in_dim = observation_space

    main = model_class(in_dim, args.hidden_dim, action_space).to(device)
    optimizer = torch.optim.Adam(main.parameters(), lr=args.lr)

    print(args.method)
    if args.method == "DQN":
        target = model_class(in_dim, args.hidden_dim, action_space).to(device)

        logs = dqn(
            target,
            main,
            optimizer,
            env_config=env_config,
            env_name=args.env_name,
            steps=args.steps,
            capacity=args.capacity,
            epsilon=args.epsilon,
            update_frequency=args.update_frequency,
            decay=args.decay,
            min_eps=args.min_eps,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            gamma=args.gamma,
            solved_threshold=args.solved_threshold,
            num_frame_stack=args.num_frame_stack,
        )

    elif args.method == "VPG":
        vpg(
            main,
            args.env_name,
            env_config,
            args.steps,
            args.gamma,
            optimizer,
            args.num_frame_stack,
            args.solved_threshold,
            args.output_dir,
            device,
        )
