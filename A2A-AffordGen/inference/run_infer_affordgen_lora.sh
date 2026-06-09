#!/bin/bash
# ============================================================================
# A2A-AffordGen — Evaluate a LoRA-finetuned AffordGen annotator
# ----------------------------------------------------------------------------
# Same as run_infer_affordgen_full.sh but point CKPT at a LoRA checkpoint dir
# (ms-swift saves LoRA adapters that infer_affordgen.py loads on top of the
# base model recorded in the checkpoint args). See ../README.md.
#
# Usage:
#   CKPT=/path/to/lora/checkpoint-2332 bash run_infer_affordgen_lora.sh
# ============================================================================
set -e
cd "$(dirname "$0")"

CONDA_ENV=${CONDA_ENV:-affordance}
CKPT=${CKPT:?set CKPT=/path/to/your/lora/checkpoint-XXXX}
SAM3_CKPT=${SAM3_CKPT:-models/sam3/sam3.pt}
MANIFEST=${MANIFEST:-data/affordance/train_and_val/val_set/val_manifest.jsonl}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
conda run -n "$CONDA_ENV" --no-capture-output python infer_affordgen.py \
    --ckpt "$CKPT" \
    --sam3 "$SAM3_CKPT" \
    --manifest "$MANIFEST" \
    --n_images ${N_IMAGES:-100} \
    --max_steps ${MAX_STEPS:-8} \
    --stop_iou ${STOP_IOU:-0.97} \
    --gpu 0
