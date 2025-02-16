#!/bin/sh

python src/dqn.py \
	--env_name LunarLander-v3 \
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
	--solved_threshold 200
