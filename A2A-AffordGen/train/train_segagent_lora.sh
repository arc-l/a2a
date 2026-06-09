#!/bin/bash
# ============================================================================
# A2A-AffordGen — Train the SegAgent annotator (LoRA fine-tune)
# ----------------------------------------------------------------------------
# LoRA variant of train_segagent_full.sh: the base VL weights are frozen, so
# the backbone keeps its original grounding ability and only the adapters are
# learned. Cheaper and usually more stable than full fine-tuning.
# Uses ms-swift (DeepSpeed ZeRO-2). See ../README.md.
#
# All paths are overridable via environment variables, e.g.:
#   BASE_MODEL=/path/to/Qwen3.5-9B OUTPUT_DIR=runs/lora bash train_segagent_lora.sh
# ============================================================================
set -e

CONDA_ENV=${CONDA_ENV:-affordance}
BASE_MODEL=${BASE_MODEL:-models/Qwen3.5-9B}
TRAIN_JSONL=${TRAIN_JSONL:-data/affordance/segagent_train/train.jsonl}
VAL_JSONL=${VAL_JSONL:-data/affordance/segagent_train/val.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-runs/segagent_lora}
GPUS=${CUDA_VISIBLE_DEVICES:-0,1}
NPROC=${NPROC:-2}
MASTER_PORT=${MASTER_PORT:-29501}

CUDA_VISIBLE_DEVICES=$GPUS \
conda run -n "$CONDA_ENV" \
torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    -m swift.cli.sft \
    --model "$BASE_MODEL" \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --dataset "$TRAIN_JSONL" \
    --val_dataset "$VAL_JSONL" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --max_pixels 401408 \
    --save_steps 50 \
    --save_total_limit 50 \
    --eval_steps 50 \
    --logging_steps 10 \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed zero2
