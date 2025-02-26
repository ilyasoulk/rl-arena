#!/bin/sh

python src/main.py \
	--method VPG \
	--env_name ALE/Pong-v5 \
	--hidden_dim 512 \
	--steps 5_000_000 \
	--update_frequency 10_000 \
	--batch_size 32 \
	--gamma 0.99 \
	--lr 0.001 \
	--output_dir models \
	--solved_threshold 18 \
	--num_frame_stack 4 \
	--use_critic
