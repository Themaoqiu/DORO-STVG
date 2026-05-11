#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"

VISIBLE_GPUS=3
MODEL_NAME="groundinggpt"
GROUNDINGGPT_SOURCE_DIR="${GROUNDINGGPT_SOURCE_DIR:-/mnt/sdc/xingjianwang/yibowang/DORO-STVG-groundingGPT}"
GROUNDINGGPT_PYTHON="${GROUNDINGGPT_PYTHON:-$REPO_ROOT/envs/eval/groundinggpt/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/sdc/xingjianwang/yibowang/model_zoo/GroundingGPT}"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke"
OUTPUT_DIR="./res_groundinggpt"
BATCH_SIZE=1
MAX_TOKENS=512
MAX_MODEL_LEN=8192
TEMPERATURE="${TEMPERATURE:-0.01}"
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
GROUNDINGGPT_MAX_NEW_TOKENS="${GROUNDINGGPT_MAX_NEW_TOKENS:-1024}"
GROUNDINGGPT_KEEP_LOGS="${GROUNDINGGPT_KEEP_LOGS:-0}"
GROUNDINGGPT_PERSISTENT_CLI="${GROUNDINGGPT_PERSISTENT_CLI:-1}"

echo "=========================================="
echo "GroundingGPT Evaluation Configuration"
echo "=========================================="
echo "Model Name:              $MODEL_NAME"
echo "Model Path:              $MODEL_PATH"
echo "GroundingGPT Source Dir: $GROUNDINGGPT_SOURCE_DIR"
echo "GroundingGPT Python:     $GROUNDINGGPT_PYTHON"
echo "Annotation Path:         $ANNOTATION_PATH"
echo "Video Directory:         $VIDEO_DIR"
echo "Output Directory:        $OUTPUT_DIR"
echo "Batch Size:              $BATCH_SIZE"
echo "Max Tokens:              $MAX_TOKENS"
echo "Temperature:             $TEMPERATURE"
echo "Max New Tokens:          $GROUNDINGGPT_MAX_NEW_TOKENS"
echo "Keep Logs:               $GROUNDINGGPT_KEEP_LOGS"
echo "Persistent CLI:          $GROUNDINGGPT_PERSISTENT_CLI"
echo "Visible GPUs:            $VISIBLE_GPUS"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
GROUNDINGGPT_SOURCE_DIR="$GROUNDINGGPT_SOURCE_DIR" \
GROUNDINGGPT_PYTHON="$GROUNDINGGPT_PYTHON" \
GROUNDINGGPT_CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
GROUNDINGGPT_MAX_NEW_TOKENS="$GROUNDINGGPT_MAX_NEW_TOKENS" \
GROUNDINGGPT_TEMPERATURE="$TEMPERATURE" \
GROUNDINGGPT_KEEP_LOGS="$GROUNDINGGPT_KEEP_LOGS" \
GROUNDINGGPT_PERSISTENT_CLI="$GROUNDINGGPT_PERSISTENT_CLI" \
"$REPO_ROOT/envs/eval/groundinggpt/.venv/bin/python" main.py run \
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
