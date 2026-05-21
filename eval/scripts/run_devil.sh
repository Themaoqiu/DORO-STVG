#!/bin/bash
source ../envs/eval/devil/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=20480

# DeViL is bundled under eval/dependence/devil; no SOURCE_DIR needed.
export DEVIL_PYTHON="$VIRTUAL_ENV/bin/python"
export DEVIL_DEVICE="cuda:0"

MODEL_NAME="devil"
MODEL_PATH="/mnt/sdc/xingjianwang/DeViL/weights/DeViL-7B"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/data/vidstg/query_polished.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/data/vidstg/video"
OUTPUT_DIR="./res_devil"
BATCH_SIZE=1
MAX_TOKENS=1024
TEMPERATURE=0.0

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
