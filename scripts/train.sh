#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
export HF_HOME=/dev/shm/
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500
NPROC_PER_NODE=8
MODEL_PATH=/dev/shm/models/BAGEL-7B-MoT

# replace the variables with your own
torchrun \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --model_path $MODEL_PATH \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $MODEL_PATH \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --lr_scheduler cosine \
  --min_lr 1e-6 \
  --num_worker 1 \
  --expected_num_tokens 60000 \
  --max_num_tokens 60000 \
  --max_num_tokens_per_sample 60000 \
  --prefer_buffer_before 30000 \
  --num_shard=$NPROC_PER_NODE \
  --sharding_strategy="HYBRID_SHARD" \
  --wandb_project "zebra-cot" \
  --wandb_name "h200-zebra-cot-$(date +%Y%m%d_%H%M%S)" \
  --save_every 50 \
  --warmup_steps 50 \
  --total_steps 5000 \
  --results_dir results/ \
  --checkpoint_dir results/checkpoints/ > run.out 2> run.err

# --cpu_offload True \