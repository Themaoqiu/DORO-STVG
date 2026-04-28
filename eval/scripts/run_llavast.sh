#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5

MODEL_NAME="llava-st-qwen2"
MODEL_PATH="/home/wangxingjian/model/LLaVA-ST-Qwen2-7B"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/home/wangxingjian/DORO-STVG/graph_generator/modules/autoresearch/round_17/query_eval.jsonl"
VIDEO_DIR="/home/wangxingjian/data/vidstg/video"
OUTPUT_DIR="/home/wangxingjian/DORO-STVG/graph_generator/modules/autoresearch/round_17/eval_llavast"
BATCH_SIZE=8
MAX_TOKENS=4096
MAX_MODEL_LEN=8192
TEMPERATURE=0.1

EVAL_DIR="/home/wangxingjian/DORO-STVG/eval"
GPU_LIST="$CUDA_VISIBLE_DEVICES"

GPU_LIST_SPACE="$(echo "$GPU_LIST" | tr ',' ' ')"
CHUNK_NUM="$(echo "$GPU_LIST" | awk -F, '{print NF}')"
if [[ "$CHUNK_NUM" -lt 1 ]]; then
  echo "No GPUs selected. Set LLAVAST_GPUS or CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
SHARD_DIR="$OUTPUT_DIR/shards_$RUN_ID"
mkdir -p "$SHARD_DIR"

# Print configuration
echo "=========================================="
echo "LLaVA-ST Multiprocess Evaluation Configuration"
echo "=========================================="
echo "Model Name:              $MODEL_NAME"
echo "Model Path:              $MODEL_PATH"
echo "Annotation Path:         $ANNOTATION_PATH"
echo "Video Directory:         $VIDEO_DIR"
echo "Output Directory:        $OUTPUT_DIR"
echo "CUDA Visible Devices:    $GPU_LIST"
echo "Chunk Num:               $CHUNK_NUM"
echo "Batch Size:              $BATCH_SIZE"
echo "Max Tokens:              $MAX_TOKENS"
echo "Max Model Length:        $MAX_MODEL_LEN"
echo "Temperature:             $TEMPERATURE"
echo "Shard Directory:         $SHARD_DIR"
echo "=========================================="
echo ""

# Split annotations
python "$EVAL_DIR/utils/llavast_split_chunks.py" \
  --input "$ANNOTATION_PATH" \
  --output-dir "$SHARD_DIR" \
  --chunks "$CHUNK_NUM"

# Run evaluation
chunk_id=0
for gpu in $GPU_LIST_SPACE; do
  shard="$SHARD_DIR/chunk_$(printf '%03d' "$chunk_id").jsonl"
  chunk_output="$OUTPUT_DIR/chunk_$(printf '%03d' "$chunk_id")"

  (
    cd "$EVAL_DIR"
    CUDA_VISIBLE_DEVICES="$gpu" \
    python main.py run \
      --model_name="$MODEL_NAME" \
      --model_path="$MODEL_PATH" \
      --data_name="$DATA_NAME" \
      --annotation_path="$shard" \
      --video_dir="$VIDEO_DIR" \
      --output_dir="$chunk_output" \
      --batch_size="$BATCH_SIZE" \
      --max_tokens="$MAX_TOKENS" \
      --max_model_len="$MAX_MODEL_LEN" \
      --temperature="$TEMPERATURE"
  ) &

  echo "Started chunk $chunk_id on GPU $gpu"
  chunk_id=$((chunk_id + 1))
done

failed=0
for job in $(jobs -p); do
  if ! wait "$job"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "Some LLaVA-ST chunks failed." >&2
  exit 1
fi

echo "All LLaVA-ST chunks finished."
