#!/bin/bash
source ../envs/eval/tubedetr/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=3
export TUBEDETR_DIR=/mnt/sdc/xingjianwang/yibowang/TubeDETR
export TUBEDETR_PYTHON=/mnt/sdc/xingjianwang/yibowang/TubeDETR/.venv/bin/python
export TUBEDETR_INFER_PY=/mnt/sdc/xingjianwang/yibowang/DORO-STVG-3e9aedc/eval/utils/tubedetr_infer_helper.py
export TUBEDETR_CHECKPOINT=/mnt/sdc/xingjianwang/yibowang/TubeDETR/checkpoints/tubedetr_vidstg.pth
export TUBEDETR_CUDA_VISIBLE_DEVICES=3
export TUBEDETR_DATASET_CONFIG=config/vidstg.json
export TUBEDETR_COMBINE_DATASETS=vidstg
export TUBEDETR_COMBINE_DATASETS_VAL=vidstg
export TUBEDETR_RESOLUTION=224
export TUBEDETR_FPS=2
export TUBEDETR_DEVICE=cuda
export ST_ALIGN_BENCHMARK_PATH=/mnt/sdc/xingjianwang/yibowang/LLaVA-ST/data/benchmarks/st-align/stvg.json

MODEL_NAME="tubedetr"
MODEL_PATH="/mnt/sdc/xingjianwang/yibowang/TubeDETR/checkpoints/tubedetr_vidstg.pth"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke"
OUTPUT_DIR="./res_tubedetr"
BATCH_SIZE=1
MAX_TOKENS=512
MAX_MODEL_LEN=8192
TEMPERATURE=0.0
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9

echo "=========================================="
echo "TubeDETR Evaluation Configuration"
echo "=========================================="
echo "Model Name:              $MODEL_NAME"
echo "Model Path:              $MODEL_PATH"
echo "TubeDETR Dir:            /mnt/sdc/xingjianwang/yibowang/TubeDETR"
echo "TubeDETR Python:         /mnt/sdc/xingjianwang/yibowang/TubeDETR/.venv/bin/python"
echo "TubeDETR Infer Helper:   /mnt/sdc/xingjianwang/yibowang/DORO-STVG-3e9aedc/eval/utils/tubedetr_infer_helper.py"
echo "ST-Align Metadata:       /mnt/sdc/xingjianwang/yibowang/LLaVA-ST/data/benchmarks/st-align/stvg.json"
echo "Annotation Path:         $ANNOTATION_PATH"
echo "Video Directory:         $VIDEO_DIR"
echo "Output Directory:        $OUTPUT_DIR"
echo "Batch Size:              $BATCH_SIZE"
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
