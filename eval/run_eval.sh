#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

MODEL_NAME="${MODEL_NAME:-llava-st-qwen2}"
MODEL_PATH="${MODEL_PATH:-/mnt/sdc/xingjianwang/yibowang/models/LLaVA-ST-Qwen2-7B}"
DATA_NAME="${DATA_NAME:-doro-stvg}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_smoke1.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval/res_llava_st_smoke}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

echo "=========================================="
echo "STVG Evaluation Configuration"
echo "=========================================="
echo "Model Name:              $MODEL_NAME"
echo "Model Path:              $MODEL_PATH"
echo "Data Name:               $DATA_NAME"
echo "Annotation Path:         $ANNOTATION_PATH"
echo "Video Directory:         $VIDEO_DIR"
echo "Output Directory:        $OUTPUT_DIR"
echo "Batch Size:              $BATCH_SIZE"
echo "Max Tokens:              $MAX_TOKENS"
echo "Max Model Length:        $MAX_MODEL_LEN"
echo "Temperature:             $TEMPERATURE"
echo "Tensor Parallel Size:    $TENSOR_PARALLEL_SIZE"
echo "GPU Memory Utilization:  $GPU_MEMORY_UTILIZATION"
echo "Visible GPUs:            $CUDA_VISIBLE_DEVICES"
echo "Video Reader:            $FORCE_QWENVL_VIDEO_READER"
echo "=========================================="
echo ""

cd "$REPO_ROOT"

uv run --project envs/eval python eval/main.py run \
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
