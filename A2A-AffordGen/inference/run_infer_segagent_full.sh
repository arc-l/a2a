#!/bin/bash
# ============================================================================
# A2A-AffordGen — Evaluate a FULL-finetuned SegAgent annotator
# ----------------------------------------------------------------------------
# Runs the SegAgent VLM + SAM3 interactive-click loop over a validation
# manifest and reports IoU vs. ground-truth masks. See ../README.md.
#
# Usage:
#   CKPT=/path/to/checkpoint-1200 bash run_infer_segagent_full.sh
#   CKPT=... N_IMAGES=1167 CUDA_VISIBLE_DEVICES=0 bash run_infer_segagent_full.sh
# ============================================================================
set -e
cd "$(dirname "$0")"

CONDA_ENV=${CONDA_ENV:-affordance}
CKPT=${CKPT:?set CKPT=/path/to/your/full-finetune/checkpoint-XXXX}
SAM3_CKPT=${SAM3_CKPT:-models/sam3/sam3.pt}
MANIFEST=${MANIFEST:-data/affordance/train_and_val/val_set/val_manifest.jsonl}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
conda run -n "$CONDA_ENV" --no-capture-output python infer_segagent.py \
    --ckpt "$CKPT" \
    --sam3 "$SAM3_CKPT" \
    --manifest "$MANIFEST" \
    --n_images ${N_IMAGES:-100} \
    --max_steps ${MAX_STEPS:-8} \
    --stop_iou ${STOP_IOU:-0.97} \
    --gpu 0
