#! /bin/bash

set -x

torchrun --nproc_per_node=$1 --nnodes=1 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:22411 scripts/train_sacsvg.py env=$2 dx=$3 num_train_steps=$4 agent.horizon=$5