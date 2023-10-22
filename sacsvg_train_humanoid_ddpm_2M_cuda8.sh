#! /bin/bash

set -x

torchrun --nproc_per_node=8 --nnodes=$1 --node_rank=$2 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:22411 scripts/train_sacsvg.py env=mbpo_humanoid dx=ddpm_dx num_train_steps=250000