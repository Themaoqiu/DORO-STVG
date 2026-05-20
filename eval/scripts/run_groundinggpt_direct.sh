#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EVAL_DIR"

export CUDA_VISIBLE_DEVICES=3

MODEL_PATH="/mnt/sdc/xingjianwang/models/GroundingGPT/GroundingGPT-7B"
GROUNDINGGPT_CLIP_VISION_TOWER="/mnt/sdc/xingjianwang/models/GroundingGPT/clip-vit-large-patch14-336"
GROUNDINGGPT_BERT_PATH="/mnt/sdc/xingjianwang/models/GroundingGPT/bert-base-uncased"
GROUNDINGGPT_IMAGEBIND_PATH="/mnt/sdc/xingjianwang/models/GroundingGPT/imagebind/imagebind_huge.pth"
GROUNDINGGPT_EVA_VIT_G_PATH="/mnt/sdc/xingjianwang/models/GroundingGPT/eva_vit_g.pth"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/data/vidstg/query_polished.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/data/vidstg/video"
OUTPUT_DIR="./res_groundinggpt_direct"
BATCH_SIZE=16
TEMPERATURE=0.1
MAX_NEW_TOKENS=2048
MAX_SAMPLES=0
RUN_TMP_DIR="$(mktemp -d "$EVAL_DIR/.groundinggpt_direct_tmp.XXXXXX")"
trap 'rm -rf "$RUN_TMP_DIR"' EXIT

echo "=========================================="
echo "GroundingGPT Direct Import Evaluation"
echo "=========================================="
echo "Model Path:              $MODEL_PATH"
echo "CLIP Vision Tower:       $GROUNDINGGPT_CLIP_VISION_TOWER"
echo "BERT Path:               $GROUNDINGGPT_BERT_PATH"
echo "ImageBind Path:          $GROUNDINGGPT_IMAGEBIND_PATH"
echo "EVA ViT-G Path:          $GROUNDINGGPT_EVA_VIT_G_PATH"
echo "Annotation Path:         $ANNOTATION_PATH"
echo "Video Directory:         $VIDEO_DIR"
echo "Output Directory:        $OUTPUT_DIR"
echo "Batch Size:              $BATCH_SIZE"
echo "Temperature:             $TEMPERATURE"
echo "Max New Tokens:          $MAX_NEW_TOKENS"
echo "Max Samples:             $MAX_SAMPLES"
echo "Visible GPUs:            $CUDA_VISIBLE_DEVICES"
echo "Run Temp Dir:            $RUN_TMP_DIR"
echo "=========================================="
echo ""

TMPDIR="$RUN_TMP_DIR" \
TEMP="$RUN_TMP_DIR" \
TMP="$RUN_TMP_DIR" \
TORCH_EXTENSIONS_DIR="$RUN_TMP_DIR/torch_extensions" \
GROUNDINGGPT_CLIP_VISION_TOWER="$GROUNDINGGPT_CLIP_VISION_TOWER" \
GROUNDINGGPT_BERT_PATH="$GROUNDINGGPT_BERT_PATH" \
GROUNDINGGPT_IMAGEBIND_PATH="$GROUNDINGGPT_IMAGEBIND_PATH" \
GROUNDINGGPT_EVA_VIT_G_PATH="$GROUNDINGGPT_EVA_VIT_G_PATH" \
python utils/groundinggpt_direct_eval.py \
  --model_path="$MODEL_PATH" \
  --annotation_path="$ANNOTATION_PATH" \
  --video_dir="$VIDEO_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size="$BATCH_SIZE" \
  --temperature="$TEMPERATURE" \
  --max_new_tokens="$MAX_NEW_TOKENS" \
  --max_samples="$MAX_SAMPLES"
