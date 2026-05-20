#!/bin/bash
source ../envs/eval/stvg_r1/.venv/bin/activate

export FORCE_QWENVL_VIDEO_READER=decord
export CUDA_VISIBLE_DEVICES=3

MODEL_NAME="stvg-r1"
MODEL_PATH="/mnt/sdc/xingjianwang/yibowang/model_zoo/stvg-r1-model-7b"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke"
OUTPUT_DIR="./res_stvg_r1"
BATCH_SIZE=1
MAX_TOKENS=512
MAX_MODEL_LEN=8192
TEMPERATURE=0.0
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.8
STVG_R1_MAX_FRAMES=32
STVG_R1_CLIP_FPS=2.0
STVG_R1_MAX_OUTPUT_FRAMES=8
STVG_R1_KEEP_TMP=0

echo "=========================================="
echo "STVG-R1 Evaluation Configuration"
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
echo "Max Frames:              $STVG_R1_MAX_FRAMES"
echo "Clip FPS:                $STVG_R1_CLIP_FPS"
echo "Max Output Frames:       $STVG_R1_MAX_OUTPUT_FRAMES"
echo "Keep Temp Clips:         $STVG_R1_KEEP_TMP"
echo "Visible GPUs:            $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

STVG_R1_MAX_FRAMES="$STVG_R1_MAX_FRAMES" \
STVG_R1_CLIP_FPS="$STVG_R1_CLIP_FPS" \
STVG_R1_MAX_OUTPUT_FRAMES="$STVG_R1_MAX_OUTPUT_FRAMES" \
STVG_R1_KEEP_TMP="$STVG_R1_KEEP_TMP" \
../envs/eval/stvg_r1/.venv/bin/python main.py run \
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
