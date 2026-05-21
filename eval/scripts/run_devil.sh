#!/bin/bash
source ../envs/eval/devil/.venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DECORD_EOF_RETRY_MAX=20480

# DeViL is bundled under eval/dependence/devil; no SOURCE_DIR needed.
export DEVIL_PYTHON="${DEVIL_PYTHON:-$VIRTUAL_ENV/bin/python}"
export DEVIL_DEVICE="${DEVIL_DEVICE:-cuda:0}"

MODEL_NAME="devil"
MODEL_PATH="${MODEL_PATH:-/mnt/sdc/xingjianwang/DeViL/weights/DeViL-7B}"
DATA_NAME="${DATA_NAME:-doro-stvg}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/mnt/sdc/xingjianwang/data/vidstg/query_polished.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-/mnt/sdc/xingjianwang/data/vidstg/video}"
OUTPUT_DIR="${OUTPUT_DIR:-./res_devil}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

echo "=========================================="
echo "DeViL Evaluation"
echo "Source: $DEVIL_SOURCE_DIR | Model: $MODEL_PATH | Device: $DEVIL_DEVICE"
echo "=========================================="

python main.py run \
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
