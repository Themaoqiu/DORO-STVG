#!/bin/bash
source ../envs/eval/grounded_video_llm/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=3
export GROUNDED_VIDEO_LLM_SOURCE_DIR="/mnt/sdc/xingjianwang/yibowang/Grounded-Video-LLM"
export GROUNDED_VIDEO_LLM_PYTHON="/mnt/sdc/xingjianwang/yibowang/DORO-STVG/envs/eval/grounded_video_llm/.venv/bin/python"
export GROUNDED_VIDEO_LLM_DEVICE="cuda:0"
export GROUNDED_VIDEO_LLM_LLM="phi3.5"
export GROUNDED_VIDEO_LLM_ATTN_IMPLEMENTATION="eager"
export GROUNDED_VIDEO_LLM_CONFIG_PATH="/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/Phi-3.5-vision-instruct"
export GROUNDED_VIDEO_LLM_TOKENIZER_PATH="/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/Phi-3.5-mini-instruct"
export GROUNDED_VIDEO_LLM_PRETRAINED_VIDEO_PATH="/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt"
export GROUNDED_VIDEO_LLM_PRETRAINED_VISION_PROJ_LLM_PATH="/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/Phi-3.5-vision-instruct-seperated"
export GROUNDED_VIDEO_LLM_CKPT_PATH="/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/ckpt/sft_llava_next_video_phi3.5_mix_sft_multi_modal_projector_video_projecter_language_model.pth"

MODEL_NAME="grounded-video-llm"
MODEL_PATH="$GROUNDED_VIDEO_LLM_CKPT_PATH"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke"
OUTPUT_DIR="./res_grounded_video_llm_temporal"
BATCH_SIZE=1
MAX_TOKENS=1024
MAX_MODEL_LEN=8192
TEMPERATURE=0.0
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9

echo "=========================================="
echo "Grounded-VideoLLM Temporal Evaluation Configuration"
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
  --max_model_len="$MAX_MODEL_LEN" \
  --temperature="$TEMPERATURE" \
  --tensor_parallel_size="$TENSOR_PARALLEL_SIZE" \
  --gpu_memory_utilization="$GPU_MEMORY_UTILIZATION"
