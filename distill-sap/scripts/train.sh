#!/bin/bash
set -x

# 安装依赖
pip3 install -e /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/gongji/xushuai21/ms-swift -i https://pypi.sankuai.com/simple/
pip3 install swanlab mpi4py -i https://pypi.sankuai.com/simple/

# 训练参数
MODEL_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/common/model/Qwen2.5-3B-Instruct"  # 3/7/14/32/72
DATASET_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/gongji/xushuai21/code-repo/Hotel-Data-Engine/data/sap-distill/train-v1.jsonl"
OUTPUT_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/gongji/xushuai21/code-repo/Hotel-Data-Engine/model/sap-distill/SAP-Distill-V1-Qwen2.5-3B"

# 环境变量设置
source "utils/export_distribute_env.sh"
export PATH="~/.local/bin:$PATH"
export SWANLAB_LOG_DIR=$OUTPUT_DIR

# 32 x A100-80G

# 执行训练命令
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
OMP_NUM_THREADS=8 \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    \
    --train_type full \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --max_length 4500 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    \
    --deepspeed zero2 \
    --ddp_backend nccl \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --use_liger_kernel true \
    \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_only_model true \
    \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    \
    --logging_steps 1 \
    --eval_strategy no \
    --add_version false \
    --enable_channel_loss true \
    \
    --report_to swanlab \
    --swanlab_project SAP-Distill \
    --swanlab_mode offline

MODEL_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/common/model/Qwen2.5-7B-Instruct"  # 3/7/14/32/72
OUTPUT_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/gongji/xushuai21/code-repo/Hotel-Data-Engine/model/sap-distill/SAP-Distill-V1-Qwen2.5-7B"
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
OMP_NUM_THREADS=8 \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    \
    --train_type full \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --max_length 4500 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    \
    --deepspeed zero2 \
    --ddp_backend nccl \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --gradient_checkpointing \
    --use_liger_kernel true \
    \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_only_model true \
    \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    \
    --logging_steps 1 \
    --eval_strategy no \
    --add_version false \
    --enable_channel_loss true \
    \
    --report_to swanlab \
    --swanlab_project SAP-Distill \
    --swanlab_mode offline

MODEL_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/common/model/Qwen2.5-14B-Instruct"  # 3/7/14/32/72
OUTPUT_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/gongji/xushuai21/code-repo/Hotel-Data-Engine/model/sap-distill/SAP-Distill-V1-Qwen2.5-14B"
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
OMP_NUM_THREADS=8 \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    \
    --train_type full \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --max_length 4500 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    \
    --deepspeed zero3 \
    --ddp_backend nccl \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --gradient_checkpointing \
    --use_liger_kernel true \
    \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_only_model true \
    \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    \
    --logging_steps 1 \
    --eval_strategy no \
    --add_version false \
    --enable_channel_loss true \
    \
    --report_to swanlab \
    --swanlab_project SAP-Distill \
    --swanlab_mode offline

MODEL_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/common/model/Qwen2.5-32B-Instruct"  # 3/7/14/32/72
OUTPUT_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/gongji/xushuai21/code-repo/Hotel-Data-Engine/model/sap-distill/SAP-Distill-V1-Qwen2.5-32B"
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
OMP_NUM_THREADS=8 \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    \
    --train_type full \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --max_length 4500 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    \
    --deepspeed zero3 \
    --ddp_backend nccl \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --gradient_checkpointing \
    --use_liger_kernel true \
    \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_only_model true \
    \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    \
    --logging_steps 1 \
    --eval_strategy no \
    --add_version false \
    --enable_channel_loss true \
    \
    --report_to swanlab \
    --swanlab_project SAP-Distill \
    --swanlab_mode offline

MODEL_PATH="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/common/model/Qwen2.5-72B-Instruct"  # 3/7/14/32/72
OUTPUT_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/gongji/xushuai21/code-repo/Hotel-Data-Engine/model/sap-distill/SAP-Distill-V1-Qwen2.5-72B"
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
OMP_NUM_THREADS=8 \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    \
    --train_type full \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --max_length 4500 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    \
    --deepspeed zero3_offload \
    --ddp_backend nccl \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --gradient_checkpointing \
    --use_liger_kernel true \
    \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_only_model true \
    \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    \
    --logging_steps 1 \
    --eval_strategy no \
    --add_version false \
    --enable_channel_loss true \
    \
    --report_to swanlab \
    --swanlab_project SAP-Distill \
    --swanlab_mode offline
