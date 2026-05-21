#!/bin/bash
source ../envs/eval/qwen/.venv/bin/activate

export DECORD_EOF_RETRY_MAX=20480
export CUDA_VISIBLE_DEVICES=0

# MODEL_NAME options: internvl3 | internvl3.5
MODEL_NAME="internvl3.5"
MODEL_PATH="/home/wangxingjian/model/InternVL3_5-8B"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/home/wangxingjian/DORO-STVG/graph_generator/modules/autoresearch/round_24/query_raw.jsonl"
VIDEO_DIR="/home/wangxingjian/data/vidstg/video"
OUTPUT_DIR="./res"
BATCH_SIZE=16
MAX_TOKENS=4096
MAX_MODEL_LEN=24800
TEMPERATURE=0.0
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
export INTERNVL_FPS=2.0
# Hard cap; if fps sampling exceeds this, frames are uniformly thinned.
export INTERNVL_MAX_FRAMES=64
# Pre-resize each sampled frame to TILE_SIZE x TILE_SIZE so InternVL's
# dynamic_preprocess produces exactly one 256-token tile per frame.
export INTERNVL_TILE_SIZE=448
# GT annotation fps. Defaults to INTERNVL_FPS in the model code.
export EVAL_GT_FPS=2.0

echo "=========================================="
echo "InternVL Evaluation: $MODEL_NAME ($MODEL_PATH)"
echo "Sampling fps: $INTERNVL_FPS | TP=$TENSOR_PARALLEL_SIZE | BS=$BATCH_SIZE"
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
