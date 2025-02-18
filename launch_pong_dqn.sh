#!/bin/sh

python src/dqn.py \
	--method DQN \
	--env_name ALE/Pong-v5 \
	--hidden_dim 512 \
	--steps 5_000_000 \
	--capacity 100_000 \
	--epsilon 1.0 \
	--update_frequency 1000 \
	--decay 0.0000001 \
	--min_eps 0.1 \
	--batch_size 32 \
	--gamma 0.99 \
	--lr 0.00025 \
	--output_dir models \
	--solved_threshold 18 \
	--num_frame_stack 4
