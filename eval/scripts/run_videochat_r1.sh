#!/bin/bash
source ../envs/eval/qwen/.venv/bin/activate

export FORCE_QWENVL_VIDEO_READER=decord
export CUDA_VISIBLE_DEVICES=0,1

MODEL_NAME="videochat-r1"
MODEL_PATH="/mnt/sdc/xingjianwang/models/VideoChat-R1_7B"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/data/vidstg/query_polished.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/data/vidstg/video"
OUTPUT_DIR="./res_videochat_r1"
BATCH_SIZE=64
MAX_TOKENS=8192
MAX_MODEL_LEN=64000
TEMPERATURE=0.0
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.9

echo "=========================================="
echo "VideoChat-R1 Evaluation Configuration"
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
echo "Visible GPUs:            $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

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
