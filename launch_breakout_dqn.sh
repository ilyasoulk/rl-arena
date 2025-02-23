#!/bin/sh

python src/main.py \
	--method DQN \
	--env_name ALE/Breakout-v5 \
	--hidden_dim 512 \
	--steps 12_500_000 \
	--capacity 1_000_000 \
	--epsilon 1.0 \
	--update_frequency 5_000 \
	--decay 1e-5 \
	--min_eps 0.1 \
	--batch_size 32 \
	--gamma 0.99 \
	--lr 0.00025 \
	--output_dir models \
	--solved_threshold 18 \
	--num_frame_stack 4
