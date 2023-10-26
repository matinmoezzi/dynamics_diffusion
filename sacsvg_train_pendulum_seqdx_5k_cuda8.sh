#! /bin/bash
set -x

torchrun --nproc_per_node=8 --nnodes=$1 --node_rank=$2 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:22491 scripts/train_sacsvg.py env=pendulum dx=seqdx num_train_steps=5000