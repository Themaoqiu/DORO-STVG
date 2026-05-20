#!/bin/bash
source ../envs/eval/videomolmo/.venv/bin/activate

# Default parameters
VISIBLE_GPUS=3
MODEL_NAME="videomolmo"
VIDEOMOLMO_REPO="/mnt/sdc/xingjianwang/yibowang/VideoMolmo"
VIDEOMOLMO_SOURCE_DIR="$VIDEOMOLMO_REPO/VideoMolmo"
MOLMO_SOURCE_DIR="/mnt/sdc/xingjianwang/yibowang/molmo"
SAM2_SOURCE_DIR="$VIDEOMOLMO_SOURCE_DIR/sam2"
VIDEOMOLMO_PYTHON="/mnt/sdc/xingjianwang/yibowang/DORO-STVG-pr-verify/envs/eval/.venv/bin/python"
MODEL_PATH="$VIDEOMOLMO_REPO"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke"
OUTPUT_DIR="./res_videomolmo"
BATCH_SIZE=1
MAX_TOKENS=512
MAX_MODEL_LEN=8192
TEMPERATURE=0.1
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
VIDEOMOLMO_MAX_FRAMES=100
VIDEOMOLMO_SAMPLE_FPS=2.0
VIDEOMOLMO_POINT_BOX_HALF=0.04
VIDEOMOLMO_ALLOW_LOG_FALLBACK=0
VIDEOMOLMO_USE_POINT_PROMPT=1


# Print configuration
echo "=========================================="
echo "VideoMolmo Evaluation Configuration"
echo "=========================================="
echo "Model Name:              $MODEL_NAME"
echo "Model Path:              $MODEL_PATH"
echo "VideoMolmo Repo:         $VIDEOMOLMO_REPO"
echo "VideoMolmo Source Dir:   $VIDEOMOLMO_SOURCE_DIR"
echo "Molmo Source Dir:        $MOLMO_SOURCE_DIR"
echo "SAM2 Source Dir:         $SAM2_SOURCE_DIR"
echo "VideoMolmo Python:       $VIDEOMOLMO_PYTHON"
echo "Annotation Path:         $ANNOTATION_PATH"
echo "Video Directory:         $VIDEO_DIR"
echo "Output Directory:        $OUTPUT_DIR"
echo "Batch Size:              $BATCH_SIZE"
echo "Max Tokens:              $MAX_TOKENS"
echo "Max Model Length:        $MAX_MODEL_LEN"
echo "Temperature:             $TEMPERATURE"
echo "Tensor Parallel Size:    $TENSOR_PARALLEL_SIZE"
echo "GPU Memory Utilization:  $GPU_MEMORY_UTILIZATION"
echo "VideoMolmo Max Frames:   $VIDEOMOLMO_MAX_FRAMES"
echo "VideoMolmo Sample FPS:   $VIDEOMOLMO_SAMPLE_FPS"
echo "Point Box Half:          $VIDEOMOLMO_POINT_BOX_HALF"
echo "Log Fallback:            $VIDEOMOLMO_ALLOW_LOG_FALLBACK"
echo "Point Prompt:            $VIDEOMOLMO_USE_POINT_PROMPT"
echo "Visible GPUs:            $VISIBLE_GPUS"
echo "=========================================="
echo ""

# Run evaluation
CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
VIDEOMOLMO_REPO="$VIDEOMOLMO_REPO" \
VIDEOMOLMO_PYTHON="$VIDEOMOLMO_PYTHON" \
VIDEOMOLMO_MAX_FRAMES="$VIDEOMOLMO_MAX_FRAMES" \
VIDEOMOLMO_SAMPLE_FPS="$VIDEOMOLMO_SAMPLE_FPS" \
VIDEOMOLMO_POINT_BOX_HALF="$VIDEOMOLMO_POINT_BOX_HALF" \
VIDEOMOLMO_ALLOW_LOG_FALLBACK="$VIDEOMOLMO_ALLOW_LOG_FALLBACK" \
VIDEOMOLMO_USE_POINT_PROMPT="$VIDEOMOLMO_USE_POINT_PROMPT" \
VIDEOMOLMO_USE_POINT_PROMPT=1
PYTHONPATH="$SAM2_SOURCE_DIR:$VIDEOMOLMO_SOURCE_DIR:$MOLMO_SOURCE_DIR:${PYTHONPATH:-}" \
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
