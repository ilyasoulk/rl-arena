#!/bin/sh

python rl_arena/main.py \
	--env_name CartPole-v1 \
	--method PPO \
	--hidden_dim 128 \
	--steps 50_000 \
	--epsilon 0.2 \
	--gamma 0.99 \
	--lr 0.001 \
	--output_dir models \
	--solved_threshold 475 \
	--use_critic
