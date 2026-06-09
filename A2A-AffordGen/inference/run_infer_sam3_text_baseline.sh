#!/bin/bash
# ============================================================================
# A2A-AffordGen — Zero-shot SAM3 + text baseline (no agent, no clicks)
# ----------------------------------------------------------------------------
# Prompts SAM3 directly with the part description, with no AffordGen VLM and no
# iterative clicking. Useful as the lower bound to compare the trained agent
# against on the same validation manifest. See ../README.md.
#
# Usage:
#   bash run_infer_sam3_text_baseline.sh
#   N_IMAGES=1167 CUDA_VISIBLE_DEVICES=0 bash run_infer_sam3_text_baseline.sh
# ============================================================================
set -e
cd "$(dirname "$0")"

CONDA_ENV=${CONDA_ENV:-affordance}
SAM3_CKPT=${SAM3_CKPT:-models/sam3/sam3.pt}
MANIFEST=${MANIFEST:-data/affordance/train_and_val/val_set/val_manifest.jsonl}
OUT_DIR=${OUT_DIR:-runs/baseline_sam3_text}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
conda run -n "$CONDA_ENV" --no-capture-output python infer_sam3_text_baseline.py \
    --sam3 "$SAM3_CKPT" \
    --manifest "$MANIFEST" \
    --out_dir "$OUT_DIR" \
    --n_images ${N_IMAGES:-100} \
    --gpu 0
