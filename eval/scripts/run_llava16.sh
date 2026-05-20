#!/bin/bash
source ../envs/eval/llava16/.venv/bin/activate

export FORCE_QWENVL_VIDEO_READER=decord
export CUDA_VISIBLE_DEVICES=3

MODEL_NAME="llava16"
MODEL_PATH="/mnt/sdc/xingjianwang/yibowang/model_zoo/llava-v1.6-mistral-7b-hf"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke"
OUTPUT_DIR="./res_llava16"
BATCH_SIZE=1
MAX_TOKENS=512
MAX_MODEL_LEN=4096
TEMPERATURE=0.0
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.8
LLAVA16_MAX_FRAMES=6
LLAVA16_GRID_COLUMNS=3
LLAVA16_ENFORCE_EAGER=0

echo "=========================================="
echo "LLaVA-1.6 Evaluation Configuration"
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
echo "Max Frames:              $LLAVA16_MAX_FRAMES"
echo "Grid Columns:            $LLAVA16_GRID_COLUMNS"
echo "Enforce Eager:           $LLAVA16_ENFORCE_EAGER"
echo "Visible GPUs:            $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

LLAVA16_MAX_FRAMES="$LLAVA16_MAX_FRAMES" \
LLAVA16_GRID_COLUMNS="$LLAVA16_GRID_COLUMNS" \
LLAVA16_ENFORCE_EAGER="$LLAVA16_ENFORCE_EAGER" \
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
