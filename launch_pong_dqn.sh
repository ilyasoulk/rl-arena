#!/bin/sh

python src/main.py \
	--method DQN \
	--env_name ALE/Pong-v5 \
	--hidden_dim 512 \
	--steps 5_000_000 \
	--capacity 100_000 \
	--epsilon 1.0 \
	--update_frequency 10_000 \
	--decay 0.999999 \
	--min_eps 0.1 \
	--batch_size 64 \
	--gamma 0.99 \
	--lr 0.001 \
	--output_dir models \
	--solved_threshold 18 \
	--num_frame_stack 4
