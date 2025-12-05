#!/bin/bash
export FORCE_QWENVL_VIDEO_READER=decord
export CUDA_VISIBLE_DEVICES=2

# Default parameters
MODEL_NAME="qwen3vl"
MODEL_PATH="/home/wangxingjian/model/Qwen3-VL-8B-Instruct"
DATA_NAME="hcstvg2"
ANNOTATION_PATH="/home/wangxingjian/TA-STVG/data/hc-stvg2/annos/test.json"
VIDEO_DIR="/home/wangxingjian/TA-STVG/data/hc-stvg2/v2_video"
OUTPUT_DIR="./res"
ANNOTATED_VIDEO_DIR="./anno_videos"
NUM_FRAMES=100
BATCH_SIZE=1
MAX_TOKENS=2048
MAX_MODEL_LEN=64000
TEMPERATURE=0.1
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
CLEANUP_AFTER=False
CLEANUP_ONLY=false

# If cleanup only, run cleanup and exit
if [ "$CLEANUP_ONLY" = true ]; then
  echo "=========================================="
  echo "Cleanup annotated videos"
  echo "=========================================="
  python main.py cleanup --annotated_video_dir="$ANNOTATED_VIDEO_DIR"
  exit 0
fi


# Print configuration
echo "=========================================="
echo "HC-STVG Evaluation Configuration"
echo "=========================================="
echo "Model Name:              $MODEL_NAME"
echo "Model Path:              $MODEL_PATH"
echo "Annotation Path:         $ANNOTATION_PATH"
echo "Video Directory:         $VIDEO_DIR"
echo "Output Directory:        $OUTPUT_DIR"
echo "Annotated Video Dir:     $ANNOTATED_VIDEO_DIR"
echo "Num Frames:              $NUM_FRAMES"
echo "Batch Size:              $BATCH_SIZE"
echo "Max Tokens:              $MAX_TOKENS"
echo "Max Model Length:        $MAX_MODEL_LEN"
echo "Temperature:             $TEMPERATURE"
echo "Tensor Parallel Size:    $TENSOR_PARALLEL_SIZE"
echo "GPU Memory Utilization:  $GPU_MEMORY_UTILIZATION"
echo "Cleanup After:           $CLEANUP_AFTER"
echo "=========================================="
echo ""

# Run evaluation
python eval/main.py run \
  --model_name="$MODEL_NAME" \
  --model_path="$MODEL_PATH" \
  --data_name="$DATA_NAME" \
  --annotation_path="$ANNOTATION_PATH" \
  --video_dir="$VIDEO_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --annotated_video_dir="$ANNOTATED_VIDEO_DIR" \
  --num_frames="$NUM_FRAMES" \
  --batch_size="$BATCH_SIZE" \
  --max_tokens="$MAX_TOKENS" \
  --max_model_len="$MAX_MODEL_LEN" \
  --temperature="$TEMPERATURE" \
  --tensor_parallel_size="$TENSOR_PARALLEL_SIZE" \
  --gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  --cleanup_after="$CLEANUP_AFTER"
