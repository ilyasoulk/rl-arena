#!/bin/sh

python src/dqn.py \
	--env_name ALE/Pong-v5 \
	--hidden_dim 128 \
	--steps 500_000 \
	--capacity 100_000 \
	--epsilon 1 \
	--update_frequency 1000 \
	--decay 0.00001 \
	--min_eps 0.01 \
	--batch_size 64 \
	--gamma 0.99 \
	--lr 0.0005 \
	--output_dir models \
	--solved_threshold 21 \
	--num_frame_stack 4
