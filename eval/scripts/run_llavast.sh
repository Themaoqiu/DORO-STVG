#!/bin/bash
source ../envs/eval/llavast/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=3
export LLAVA_ST_SOURCE_DIR="/mnt/sdc/xingjianwang/yibowang/LLaVA-ST"

# Default parameters
MODEL_NAME="llava-st"
MODEL_PATH="/mnt/sdc/xingjianwang/yibowang/models/LLaVA-ST-Qwen2-7B"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke"
OUTPUT_DIR="./res_llavast"
BATCH_SIZE=64
MAX_TOKENS=512
MAX_MODEL_LEN=8192
TEMPERATURE=0.1
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9


# Print configuration
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
echo "LLaVA-ST Source Dir:     $LLAVA_ST_SOURCE_DIR"
echo "=========================================="
echo ""

# Run evaluation
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
