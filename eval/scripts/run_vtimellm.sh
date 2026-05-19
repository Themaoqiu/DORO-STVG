#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export VTIMELLM_SOURCE_DIR="${VTIMELLM_SOURCE_DIR:-/mnt/sdc/xingjianwang/yibowang/VTimeLLM}"
export VTIMELLM_MODEL_BASE="${VTIMELLM_MODEL_BASE:-/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/vicuna-7b-v1.5}"
export VTIMELLM_PRETRAIN_MM_MLP_ADAPTER="${VTIMELLM_PRETRAIN_MM_MLP_ADAPTER:-/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin}"
export VTIMELLM_STAGE2="${VTIMELLM_STAGE2:-/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage2}"
export VTIMELLM_STAGE3="${VTIMELLM_STAGE3:-/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3}"
export VTIMELLM_CLIP_PATH="${VTIMELLM_CLIP_PATH:-/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/clip/ViT-L-14.pt}"

MODEL_NAME="vtimellm"
MODEL_PATH="$VTIMELLM_STAGE3"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke"
OUTPUT_DIR="./res_vtimellm"
BATCH_SIZE=1
MAX_TOKENS=512
MAX_MODEL_LEN=8192
TEMPERATURE=0.0
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9

echo "=========================================="
echo "VTimeLLM Evaluation Configuration"
echo "=========================================="
echo "Model Name:              $MODEL_NAME"
echo "Model Path:              $MODEL_PATH"
echo "Annotation Path:         $ANNOTATION_PATH"
echo "Video Directory:         $VIDEO_DIR"
echo "Output Directory:        $OUTPUT_DIR"
echo "VTimeLLM Source Dir:     $VTIMELLM_SOURCE_DIR"
echo "VTimeLLM Model Base:     $VTIMELLM_MODEL_BASE"
echo "VTimeLLM Stage1 Adapter: $VTIMELLM_PRETRAIN_MM_MLP_ADAPTER"
echo "VTimeLLM Stage2:         $VTIMELLM_STAGE2"
echo "VTimeLLM Stage3:         $VTIMELLM_STAGE3"
echo "VTimeLLM CLIP Path:      $VTIMELLM_CLIP_PATH"
echo "=========================================="

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
