#!/bin/bash
source ../envs/eval/cgstvg/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=3
export CGSTVG_DIR=/mnt/sdc/xingjianwang/yibowang/CGSTVG
export CGSTVG_PYTHON=/mnt/sdc/xingjianwang/yibowang/CGSTVG/.venv/bin/python
export CGSTVG_INFER_PY=/mnt/sdc/xingjianwang/yibowang/DORO-STVG-3e9aedc/eval/utils/cgstvg_infer_helper.py
export CGSTVG_MODEL_WEIGHT=/mnt/sdc/xingjianwang/yibowang/CGSTVG/model_zoo/vidstg.pth
export CGSTVG_CUDA_VISIBLE_DEVICES=3
export CGSTVG_INPUT_RESOLUTION=224
export CGSTVG_NUM_CLIP_FRAMES=16
export CGSTVG_NUM_WORKERS=0

# Default parameters
MODEL_NAME="cgstvg"
MODEL_PATH="/mnt/sdc/xingjianwang/yibowang/CGSTVG/model_zoo/vidstg.pth"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke"
OUTPUT_DIR="./res_cgstvg"


# Print configuration
echo "=========================================="
echo "CG-STVG Evaluation Configuration"
echo "=========================================="
echo "Model Name:              $MODEL_NAME"
echo "Model Path:              $MODEL_PATH"
echo "CGSTVG Dir:              /mnt/sdc/xingjianwang/yibowang/CGSTVG"
echo "CGSTVG Python:           /mnt/sdc/xingjianwang/yibowang/CGSTVG/.venv/bin/python"
echo "CGSTVG Infer Helper:     /mnt/sdc/xingjianwang/yibowang/DORO-STVG-3e9aedc/eval/utils/cgstvg_infer_helper.py"
echo "Annotation Path:         $ANNOTATION_PATH"
echo "Video Directory:         $VIDEO_DIR"
echo "Output Directory:        $OUTPUT_DIR"
echo "Input Resolution:        224"
echo "Num Clip Frames:         16"
echo "Num Workers:             0"
echo "Visible GPUs:            3"
echo "=========================================="
echo ""

# Run evaluation
python main.py run \
  --model_name="$MODEL_NAME" \
  --model_path="$MODEL_PATH" \
  --data_name="$DATA_NAME" \
  --annotation_path="$ANNOTATION_PATH" \
  --video_dir="$VIDEO_DIR" \
  --output_dir="$OUTPUT_DIR"
