#!/bin/sh

python src/main.py \
	--env_name CartPole-v1 \
	--method VPG \
	--hidden_dim 128 \
	--steps 50_000 \
	--update_frequency 1000 \
	--batch_size 32 \
	--gamma 0.99 \
	--lr 0.001 \
	--output_dir models \
	--solved_threshold 475
