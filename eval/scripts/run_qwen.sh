#!/bin/bash
source ../envs/eval/qwen_internvl/.venv/bin/activate

export FORCE_QWENVL_VIDEO_READER=decord
export CUDA_VISIBLE_DEVICES=5

# Default parameters
MODEL_NAME="qwen3VL"
MODEL_PATH="/home/wangxingjian/model/stvg-grpo-curriculum-px200k-r8192-rb8-step66/stvg_grpo_curriculum_fast_px200k_r8192_rb8_half_save33_merge/global_step_66_huggingface"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/home/wangxingjian/data/compstvg/test.jsonl"
VIDEO_DIR=""
OUTPUT_DIR="./res"
BATCH_SIZE=1
MAX_TOKENS=4096
MAX_MODEL_LEN=64000
TEMPERATURE=0.1
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9


# Print configuration
echo "=========================================="
echo "STVG Evaluation Configuration"
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
