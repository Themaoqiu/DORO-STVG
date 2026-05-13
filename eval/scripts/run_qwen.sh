#!/bin/bash
set -euo pipefail

export FORCE_QWENVL_VIDEO_READER=decord
export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="qwen3.5"
MODEL_PATH="/home/wangxingjian/model/qwen3.5-9b"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/home/wangxingjian/DORO-STVG/graph_generator/modules/autoresearch/round_24/query_raw.jsonl"
VIDEO_DIR="/home/wangxingjian/data/vidstg/video"
OUTPUT_DIR="/home/wangxingjian/DORO-STVG/graph_generator/modules/autoresearch/round_24/eval"
BATCH_SIZE=64
MAX_TOKENS=4096
MAX_MODEL_LEN=64000
TEMPERATURE=0.1
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
PYTHON_BIN="python"


# Print configuration
echo "=========================================="
echo "STVG Evaluation Configuration"
echo "=========================================="
echo "Model Name:              $MODEL_NAME"
echo "Model Path:              $MODEL_PATH"
echo "Annotation Path:         $ANNOTATION_PATH"
echo "Video Directory:         $VIDEO_DIR"
echo "Output Directory:        $OUTPUT_DIR"
echo "CUDA Visible Devices:    $CUDA_VISIBLE_DEVICES"
echo "Batch Size:              $BATCH_SIZE"
echo "Max Tokens:              $MAX_TOKENS"
echo "Max Model Length:        $MAX_MODEL_LEN"
echo "Temperature:             $TEMPERATURE"
echo "Tensor Parallel Size:    $TENSOR_PARALLEL_SIZE"
echo "GPU Memory Utilization:  $GPU_MEMORY_UTILIZATION"
echo "Python:                  $PYTHON_BIN"
echo "=========================================="
echo ""

# Run evaluation
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
