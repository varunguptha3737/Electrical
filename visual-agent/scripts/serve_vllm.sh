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
# 0.90 (not higher) leaves ~2 GB for the voice models (Whisper STT) if you
# run them on the same GPU.
vllm serve "$MODEL" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 4}'

# --- 2 x 24 GB GPUs (2x RTX 3090/4090): uncomment instead ------------------
# vllm serve "$MODEL" \
#   --tensor-parallel-size 2 \
#   --max-model-len 32768 \
#   --gpu-memory-utilization 0.90 \
#   --limit-mm-per-prompt '{"image": 4}'

# --- Rented GPU (RunPod / Vast.ai / Lambda) ---------------------------------
# Run this script on the rented box, then from your laptop:
#   ssh -N -L 8000:localhost:8000 user@rented-box
# and keep VLLM_BASE_URL=http://localhost:8000/v1 in your local .env.
# The agent, browser, UI and voice all run on the laptop; only the model
# lives on the rented GPU.

# --- Lower-VRAM fallbacks ---------------------------------------------------
# Faster MoE variant (~same VRAM, much higher tokens/s):
#   MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 ./scripts/serve_vllm.sh
# Biggest headroom (~19 GB weights, AWQ 4-bit):
#   MODEL_NAME=Qwen/Qwen2.5-VL-32B-Instruct-AWQ ./scripts/serve_vllm.sh
# Specialized GUI-agent model (excellent click grounding, only ~16 GB):
#   MODEL_NAME=ByteDance-Seed/UI-TARS-1.5-7B ./scripts/serve_vllm.sh
