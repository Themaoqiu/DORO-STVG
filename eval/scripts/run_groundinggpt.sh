#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="groundinggpt"
MODEL_PATH="/path/to/groundinggpt"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/home/wangxingjian/DORO-STVG/graph_generator/modules/autoresearch/round_21/query_eval.jsonl"
VIDEO_DIR="/home/wangxingjian/data/vidstg/video"
OUTPUT_DIR="/home/wangxingjian/DORO-STVG/graph_generator/modules/autoresearch/round_21/eval_groundinggpt"
BATCH_SIZE=1
MAX_TOKENS=4096
MAX_MODEL_LEN=8192
TEMPERATURE=0.1
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
PYTHON_BIN="python"

"$PYTHON_BIN" main.py run \
  --model_name="$MODEL_NAME" \
  --model_path="$MODEL_PATH" \
  --data_name="$DATA_NAME" \
  --annotation_path="$ANNOTATION_PATH" \
  --video_dir="$VIDEO_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size="$BATCH_SIZE" \
  --max_tokens="$MAX_TOKENS" \
  --max_model_len="$MAX_MODEL_LEN" \
  --temperature="$TEMPERATURE" \
  --tensor_parallel_size="$TENSOR_PARALLEL_SIZE" \
  --gpu_memory_utilization="$GPU_MEMORY_UTILIZATION"
