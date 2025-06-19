# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
export HF_HOME=/dev/shm/
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500
NPROC_PER_NODE=4
MODEL_PATH=/home/jovyan/workspace/Bagel/models/BAGEL-7B-MoT

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
  --num_worker 1 \
  --expected_num_tokens 32768 \
  --max_num_tokens 36864 \
  --max_num_tokens_per_sample 16384 \
  --num_shard=$NPROC_PER_NODE \
  --sharding_strategy="FULL_SHARD" \
  --cpu_offload True \
  --wandb_name "zebra-cot-$(date +%Y%m%d_%H%M%S)" \