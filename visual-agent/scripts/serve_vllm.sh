#!/usr/bin/env bash
# Serve the vision-language model with vLLM (OpenAI-compatible API on :8000).
#
# Install vLLM first (in its own venv/conda env, needs CUDA):
#   pip install vllm
#
# Pick ONE of the launch commands below.

set -euo pipefail

MODEL="${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct-FP8}"

# --- Single 48 GB GPU (RTX 6000 Ada / L40S / A6000 etc.) -------------------
vllm serve "$MODEL" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image": 4}'

# --- 2 x 24 GB GPUs (2x RTX 3090/4090): uncomment instead ------------------
# vllm serve "$MODEL" \
#   --tensor-parallel-size 2 \
#   --max-model-len 32768 \
#   --gpu-memory-utilization 0.92 \
#   --limit-mm-per-prompt '{"image": 4}'

# --- Lower-VRAM fallbacks ---------------------------------------------------
# Faster MoE variant (~same VRAM, much higher tokens/s):
#   MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 ./scripts/serve_vllm.sh
# Biggest headroom (~19 GB weights, AWQ 4-bit):
#   MODEL_NAME=Qwen/Qwen2.5-VL-32B-Instruct-AWQ ./scripts/serve_vllm.sh
# Specialized GUI-agent model (excellent click grounding, only ~16 GB):
#   MODEL_NAME=ByteDance-Seed/UI-TARS-1.5-7B ./scripts/serve_vllm.sh
