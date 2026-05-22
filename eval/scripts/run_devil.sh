#!/bin/bash
source ../envs/eval/devil/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=20480

# DeViL is bundled under eval/dependence/devil; no SOURCE_DIR needed.
export CUDA_VISIBLE_DEVICES=4
export DEVIL_DEVICE="cuda:0"

MODEL_NAME="devil"
MODEL_PATH="/home/wangxingjian/model/DeViL-7B"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/home/wangxingjian/data/vidstg/query.jsonl"
VIDEO_DIR="/home/wangxingjian/data/vidstg/video"
OUTPUT_DIR="./res"
BATCH_SIZE=16
MAX_TOKENS=4096
TEMPERATURE=0.1

echo "=========================================="
echo "DeViL Evaluation"
echo "Model: $MODEL_PATH | Device: $DEVIL_DEVICE"
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
  --temperature="$TEMPERATURE"
