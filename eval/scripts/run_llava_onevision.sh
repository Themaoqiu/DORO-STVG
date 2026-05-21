#!/bin/bash
# LLaVA-OneVision-1.5 / 2 use transformers (no vLLM). Multi-GPU via device_map="auto".

# source ../envs/eval/llava_onevision1.5/.venv/bin/activate


export DECORD_EOF_RETRY_MAX=409600
export CUDA_VISIBLE_DEVICES=7

# MODEL_NAME options: llava-onevision-1.5 | llava-onevision-2
MODEL_NAME="llava-onevision-1.5"
MODEL_PATH="/home/wangxingjian/model/LLaVA-OneVision-1.5-8B-Instruct"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/home/wangxingjian/DORO-STVG/graph_generator/modules/autoresearch/round_22/query_polished.jsonl"
VIDEO_DIR="/home/wangxingjian/data/vidstg/video"
OUTPUT_DIR="./res"
BATCH_SIZE=1
MAX_TOKENS=4096
MAX_MODEL_LEN=16384
TEMPERATURE=0.1
export LLAVA_FPS=2.0
export LLAVA_MAX_FRAMES=32
export OV1_5_MAX_PIXELS=1003520
export OV2_MAX_PIXELS=200704

echo "=========================================="
echo "LLaVA-OneVision Evaluation: $MODEL_NAME ($MODEL_PATH)"
echo "Sampling fps: $LLAVA_FPS | BS=$BATCH_SIZE"
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
