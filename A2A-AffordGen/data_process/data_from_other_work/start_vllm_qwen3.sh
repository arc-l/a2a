#!/bin/bash
# ============================================================================
# Start a local Qwen3-VL vLLM server (OpenAI-compatible API).
# This is the "labeler" VLM used by the agent-assisted annotation scripts in
# this folder (filter_and_unify_other_data.py, gen_part_captions.py, the
# cleaning_chain / orps_trps_chain steps). See ../../README.md.
#
# Usage:
#   VLM_MODEL=/path/to/Qwen3-VL-32B-Instruct bash start_vllm_qwen3.sh
# ============================================================================
VLM_MODEL=${VLM_MODEL:-models/Qwen3-VL-32B-Instruct}
SERVED_NAME=${SERVED_NAME:-qwen3-vl-32b}
PORT=${PORT:-8000}
TP=${TP:-1}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python -m vllm.entrypoints.openai.api_server \
    --model "$VLM_MODEL" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --limit-mm-per-prompt '{"image": 1}' \
    --port "$PORT" \
    > /tmp/vllm_qwen3.log 2>&1 &

echo "vLLM server starting on port $PORT, PID: $!"
echo "Log:   tail -f /tmp/vllm_qwen3.log"
echo "Check: curl http://localhost:$PORT/health   (ready after ~1 min)"
