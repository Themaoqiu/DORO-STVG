#!/bin/bash

source ../envs/eval/llavast/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

export LLAVA_ST_VISION_TOWER="/mnt/sdc/xingjianwang/models/siglip-so400m-patch14-384"

MODEL_NAME="llava-st"
MODEL_PATH="/mnt/sdc/xingjianwang/models/LLaVA-ST-Qwen2-7B"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/data/vidstg/query_polished.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/data/vidstg/video"
OUTPUT_DIR="./res_llavast"
BATCH_SIZE=16
MAX_TOKENS=4096
TEMPERATURE=0.1
RUN_ID="$(date +%Y%m%d%H%M%S)"

echo "=========================================="
echo "LLaVA-ST Evaluation Configuration"
echo "=========================================="
echo "Model Name:              $MODEL_NAME"
echo "Model Path:              $MODEL_PATH"
echo "Annotation Path:         $ANNOTATION_PATH"
echo "Video Directory:         $VIDEO_DIR"
echo "Output Directory:        $OUTPUT_DIR"
echo "Batch Size:              $BATCH_SIZE"
echo "Max Tokens:              $MAX_TOKENS"
echo "Max Model Length:        $MAX_MODEL_LEN"
echo "Temperature:             $TEMPERATURE"
echo "Tensor Parallel Size:    $TENSOR_PARALLEL_SIZE"
echo "GPU Memory Utilization:  $GPU_MEMORY_UTILIZATION"
echo "Vision Tower Path:       $LLAVA_ST_VISION_TOWER"
echo "Run ID:                  $RUN_ID"
echo "=========================================="
echo ""

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
CHUNK_NUM="${#GPUS[@]}"
PIDS=()

if [ "$CHUNK_NUM" -le 1 ]; then
  python main.py run \
    --model_name="$MODEL_NAME" \
    --model_path="$MODEL_PATH" \
    --data_name="$DATA_NAME" \
    --annotation_path="$ANNOTATION_PATH" \
    --video_dir="$VIDEO_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --batch_size=1 \
    --max_tokens="$MAX_TOKENS" \
    --max_model_len="$MAX_MODEL_LEN" \
    --temperature="$TEMPERATURE" \
    --tensor_parallel_size=1 \
    --gpu_memory_utilization="$GPU_MEMORY_UTILIZATION"
  exit $?
fi

for CHUNK_ID in "${!GPUS[@]}"; do
  GPU_ID="${GPUS[$CHUNK_ID]}"
  EVAL_CHUNK_NUM="$CHUNK_NUM" \
  EVAL_CHUNK_ID="$CHUNK_ID" \
  EVAL_RUN_ID="$RUN_ID" \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python main.py run \
    --model_name="$MODEL_NAME" \
    --model_path="$MODEL_PATH" \
    --data_name="$DATA_NAME" \
    --annotation_path="$ANNOTATION_PATH" \
    --video_dir="$VIDEO_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --batch_size=1 \
    --max_tokens="$MAX_TOKENS" \
    --max_model_len="$MAX_MODEL_LEN" \
    --temperature="$TEMPERATURE" \
    --tensor_parallel_size=1 \
    --gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAILED=1
  fi
done

if [ "$FAILED" -ne 0 ]; then
  echo "One or more LLaVA-ST chunk workers failed; skip aggregation." >&2
  exit 1
fi

python main.py run \
  --model_name="$MODEL_NAME" \
  --model_path="$MODEL_PATH" \
  --data_name="$DATA_NAME" \
  --annotation_path="$ANNOTATION_PATH" \
  --video_dir="$VIDEO_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size=1 \
  --max_tokens="$MAX_TOKENS" \
  --max_model_len="$MAX_MODEL_LEN" \
  --temperature="$TEMPERATURE" \
  --tensor_parallel_size=1 \
  --gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  --aggregate_only=True \
  --run_id="$RUN_ID" \
  --chunk_num="$CHUNK_NUM"
