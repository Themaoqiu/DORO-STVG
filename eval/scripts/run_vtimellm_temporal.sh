#!/bin/bash
source ../envs/eval/vtimellm/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=3
export VTIMELLM_SOURCE_DIR="/mnt/sdc/xingjianwang/yibowang/VTimeLLM"
export VTIMELLM_MODEL_BASE="/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/vicuna-7b-v1.5"
export VTIMELLM_PRETRAIN_MM_MLP_ADAPTER="/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin"
export VTIMELLM_STAGE2="/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage2"
export VTIMELLM_STAGE3="${VTIMELLM_STAGE3:-/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3}"
export VTIMELLM_CLIP_PATH="${VTIMELLM_CLIP_PATH:-/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/clip/ViT-L-14.pt}"

MODEL_NAME="vtimellm"
MODEL_PATH="$VTIMELLM_STAGE3"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke"
OUTPUT_DIR="./res_vtimellm_temporal"
BATCH_SIZE=1
MAX_TOKENS=512
TEMPERATURE=0.0

echo "=========================================="
echo "VTimeLLM Temporal Evaluation Configuration"
echo "=========================================="
echo "Model Name:              $MODEL_NAME"
echo "Model Path:              $MODEL_PATH"
echo "Annotation Path:         $ANNOTATION_PATH"
echo "Video Directory:         $VIDEO_DIR"
echo "Output Directory:        $OUTPUT_DIR"
echo "Evaluation Type:         temporal_only"
echo "Coordinate System:       sampled_2fps_frame_index"
echo "=========================================="

python temporal_main.py run \
  --model_name="$MODEL_NAME" \
  --model_path="$MODEL_PATH" \
  --data_name="$DATA_NAME" \
  --annotation_path="$ANNOTATION_PATH" \
  --video_dir="$VIDEO_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size="$BATCH_SIZE" \
  --max_tokens="$MAX_TOKENS" \
  --temperature="$TEMPERATURE"
