#!/bin/bash
source ../envs/eval/grounded_video_llm/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export GROUNDED_VIDEO_LLM_SOURCE_DIR="/mnt/sdc/xingjianwang/yibowang/Grounded-Video-LLM"
export GROUNDED_VIDEO_LLM_PYTHON="/mnt/sdc/xingjianwang/yibowang/DORO-STVG/envs/eval/grounded_video_llm/.venv/bin/python"
export GROUNDED_VIDEO_LLM_DEVICE="cuda:0"
export GROUNDED_VIDEO_LLM_LLM="llama3"
export GROUNDED_VIDEO_LLM_ATTN_IMPLEMENTATION="eager"
export GROUNDED_VIDEO_LLM_CONFIG_PATH="/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/llama3-llava-next-8b"
export GROUNDED_VIDEO_LLM_TOKENIZER_PATH="/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/Meta-Llama-3-8B-Instruct"
export GROUNDED_VIDEO_LLM_PRETRAINED_VIDEO_PATH="/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt"
export GROUNDED_VIDEO_LLM_PRETRAINED_VISION_PROJ_LLM_PATH="/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/llama3-llava-next-8b-seperated"
export GROUNDED_VIDEO_LLM_CKPT_PATH="/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/ckpt/sft_llava_next_video_llama3_mix_sft_multi_modal_projector_video_projecter_language_model.pth"
export GROUNDED_VIDEO_LLM_PERSISTENT="1"

MODEL_NAME="grounded-video-llm"
MODEL_PATH="$GROUNDED_VIDEO_LLM_CKPT_PATH"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke"
OUTPUT_DIR="./res_grounded_video_llm_llama3"
BATCH_SIZE=1
MAX_TOKENS=1024
TEMPERATURE=0.0

echo "=========================================="
echo "Grounded-VideoLLM (LLaMA3 / LLaVA-Next) Evaluation"
echo "Backbone: $GROUNDED_VIDEO_LLM_LLM | Ckpt: $GROUNDED_VIDEO_LLM_CKPT_PATH"
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
