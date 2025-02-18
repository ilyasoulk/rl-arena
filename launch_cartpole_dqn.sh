#!/bin/sh

python src/main.py \
	--env_name CartPole-v1 \
	--method DQN \
	--hidden_dim 128 \
	--steps 50_000 \
	--capacity 10_000 \
	--epsilon 1 \
	--update_frequency 1000 \
	--decay 0.001 \
	--min_eps 0.01 \
	--batch_size 32 \
	--gamma 0.99 \
	--lr 0.001 \
	--output_dir models \
	--solved_threshold 475
