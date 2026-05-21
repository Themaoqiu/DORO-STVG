#!/bin/bash
source ../envs/eval/qwen/.venv/bin/activate

export DECORD_EOF_RETRY_MAX=20480
export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="${MODEL_NAME:-llava-next-video}"
MODEL_PATH="${MODEL_PATH:-llava-hf/LLaVA-NeXT-Video-7B-hf}"
DATA_NAME="${DATA_NAME:-doro-stvg}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/home/wangxingjian/DORO-STVG/graph_generator/modules/autoresearch/round_24/query_raw.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-/home/wangxingjian/data/vidstg/video}"
OUTPUT_DIR="${OUTPUT_DIR:-./res}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
export LLAVA_FPS="${LLAVA_FPS:-2.0}"

echo "=========================================="
echo "LLaVA-NeXT-Video Evaluation: $MODEL_PATH"
echo "Sampling fps: $LLAVA_FPS | TP=$TENSOR_PARALLEL_SIZE | BS=$BATCH_SIZE"
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
