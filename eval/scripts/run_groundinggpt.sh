#!/bin/bash
source ../envs/eval/groundinggpt/.venv/bin/activate
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"

VISIBLE_GPUS=3
MODEL_NAME="groundinggpt"
MODEL_PATH="/mnt/sdc/xingjianwang/yibowang/model_zoo/GroundingGPT"
GROUNDINGGPT_CLIP_VISION_TOWER="/mnt/sdc/xingjianwang/models/GroundingGPT/clip-vit-large-patch14-336"
GROUNDINGGPT_BERT_PATH="/mnt/sdc/xingjianwang/models/GroundingGPT/bert-base-uncased"
GROUNDINGGPT_IMAGEBIND_PATH="/mnt/sdc/xingjianwang/models/GroundingGPT/imagebind/imagebind_huge.pth"
GROUNDINGGPT_EVA_VIT_G_PATH="/mnt/sdc/xingjianwang/models/GroundingGPT/eva_vit_g.pth"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke"
OUTPUT_DIR="./res_groundinggpt"
BATCH_SIZE=1
TEMPERATURE="0.1"
GROUNDINGGPT_MAX_NEW_TOKENS="1024"
RUN_TMP_DIR="$(mktemp -d "$EVAL_DIR/.groundinggpt_tmp.XXXXXX")"
trap 'rm -rf "$RUN_TMP_DIR"' EXIT

echo "=========================================="
echo "GroundingGPT Evaluation Configuration"
echo "=========================================="
echo "Model Name:              $MODEL_NAME"
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
echo "Max New Tokens:          $GROUNDINGGPT_MAX_NEW_TOKENS"
echo "Visible GPUs:            $VISIBLE_GPUS"
echo "Run Temp Dir:            $RUN_TMP_DIR"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
TMPDIR="$RUN_TMP_DIR" \
TEMP="$RUN_TMP_DIR" \
TMP="$RUN_TMP_DIR" \
TORCH_EXTENSIONS_DIR="$RUN_TMP_DIR/torch_extensions" \
GROUNDINGGPT_CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
GROUNDINGGPT_CLIP_VISION_TOWER="$GROUNDINGGPT_CLIP_VISION_TOWER" \
GROUNDINGGPT_BERT_PATH="$GROUNDINGGPT_BERT_PATH" \
GROUNDINGGPT_IMAGEBIND_PATH="$GROUNDINGGPT_IMAGEBIND_PATH" \
GROUNDINGGPT_EVA_VIT_G_PATH="$GROUNDINGGPT_EVA_VIT_G_PATH" \
GROUNDINGGPT_MAX_NEW_TOKENS="$GROUNDINGGPT_MAX_NEW_TOKENS" \
GROUNDINGGPT_TEMPERATURE="$TEMPERATURE" \
python main.py run \
  --model_name="$MODEL_NAME" \
  --model_path="$MODEL_PATH" \
  --data_name="$DATA_NAME" \
  --annotation_path="$ANNOTATION_PATH" \
  --video_dir="$VIDEO_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size="$BATCH_SIZE" \
  --temperature="$TEMPERATURE"
