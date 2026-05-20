#!/bin/bash
source ../envs/eval/videochat_r1/.venv/bin/activate

export FORCE_QWENVL_VIDEO_READER=decord
export CUDA_VISIBLE_DEVICES=3

MODEL_NAME="videochat-r1"
MODEL_PATH="/mnt/sdc/xingjianwang/yibowang/model_zoo/VideoChat-R1_7B"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke"
OUTPUT_DIR="./res_videochat_r1"
BATCH_SIZE=1
MAX_TOKENS=512
MAX_MODEL_LEN=8192
TEMPERATURE=0.0
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.8
VIDEOCHAT_R1_MAX_FRAMES=32
VIDEOCHAT_R1_CLIP_FPS=2.0
VIDEOCHAT_R1_KEEP_TMP=0

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
echo "Max Frames:              $VIDEOCHAT_R1_MAX_FRAMES"
echo "Clip FPS:                $VIDEOCHAT_R1_CLIP_FPS"
echo "Keep Temp Clips:         $VIDEOCHAT_R1_KEEP_TMP"
echo "Visible GPUs:            $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

VIDEOCHAT_R1_MAX_FRAMES="$VIDEOCHAT_R1_MAX_FRAMES" \
VIDEOCHAT_R1_CLIP_FPS="$VIDEOCHAT_R1_CLIP_FPS" \
VIDEOCHAT_R1_KEEP_TMP="$VIDEOCHAT_R1_KEEP_TMP" \
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
