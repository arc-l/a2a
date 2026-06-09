#!/bin/bash
# ============================================================================
# A2A-AffordGen — Train the SegAgent annotator (FULL fine-tune)
# ----------------------------------------------------------------------------
# Fine-tunes a Qwen-VL backbone to predict the next click point that refines a
# part mask, from the human-like click-trajectory data produced by
# data_process/gen_segagent_train_data.py. Uses ms-swift (DeepSpeed ZeRO-3).
# See ../README.md for the full pipeline.
#
# All paths are overridable via environment variables, e.g.:
#   BASE_MODEL=/path/to/Qwen3.5-9B OUTPUT_DIR=runs/full bash train_segagent_full.sh
# ============================================================================
set -e

CONDA_ENV=${CONDA_ENV:-affordance}
BASE_MODEL=${BASE_MODEL:-models/Qwen3.5-9B}                          # HF dir of the VL backbone
TRAIN_JSONL=${TRAIN_JSONL:-data/affordance/segagent_train/train_step0_3x.jsonl}
VAL_JSONL=${VAL_JSONL:-data/affordance/segagent_train/val.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-runs/segagent_full}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1}
NPROC=${NPROC:-2}
MASTER_PORT=${MASTER_PORT:-29500}

CUDA_VISIBLE_DEVICES=$GPUS \
conda run -n "$CONDA_ENV" \
torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    -m swift.cli.sft \
    --model "$BASE_MODEL" \
    --tuner_type full \
    --dataset "$TRAIN_JSONL" \
    --val_dataset "$VAL_JSONL" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --max_pixels 401408 \
    --save_steps 100 \
    --save_total_limit 15 \
    --save_only_model true \
    --eval_steps 100 \
    --logging_steps 10 \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed zero3
