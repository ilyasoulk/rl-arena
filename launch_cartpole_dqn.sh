#!/bin/sh

python src/dqn.py \
	--env_name CartPole-v1 \
	--hidden_dim 128 \
	--steps 100000 \
	--capacity 100000 \
	--epsilon 1 \
	--decay 0.0001 \
	--min_eps 0.01 \
	--batch_size 32 \
	--gamma 0.99 \
	--lr 0.001
