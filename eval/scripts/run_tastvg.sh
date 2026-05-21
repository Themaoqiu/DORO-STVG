#!/bin/bash
source ../envs/eval/tastvg/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=3
export TASTVG_DIR=/mnt/sdc/xingjianwang/yibowang/TA-STVG
export TASTVG_PYTHON=/mnt/sdc/xingjianwang/yibowang/TA-STVG/.venv/bin/python
export TASTVG_INFER_PY=/mnt/sdc/xingjianwang/yibowang/DORO-STVG-3e9aedc/eval/utils/tastvg_infer_helper.py
export TASTVG_MODEL_WEIGHT=/mnt/sdc/xingjianwang/yibowang/TA-STVG/model_zoo/TASTVG_VidSTG.pth
export TASTVG_CUDA_VISIBLE_DEVICES=3
export TASTVG_INPUT_RESOLUTION=224
export TASTVG_NUM_CLIP_FRAMES=16
export TASTVG_NUM_WORKERS=0
export TASTVG_PREPARE_CACHE=1
export TASTVG_CONFIG_FILE=experiments/vidstg.yaml
export TASTVG_TEST_SCRIPT=scripts/test_net.py
export TASTVG_RESULT_FILE=test_results.json
export TASTVG_OFFICIAL_EXTRA_OPTS="MODEL.TASTVG.USE_ACTION False"

# Default parameters
MODEL_NAME="tastvg"
MODEL_PATH="/mnt/sdc/xingjianwang/yibowang/TA-STVG/model_zoo/TASTVG_VidSTG.pth"
DATA_NAME="doro-stvg"
ANNOTATION_PATH="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl"
VIDEO_DIR="/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke"
OUTPUT_DIR="./res_tastvg"


# Print configuration
echo "=========================================="
echo "TA-STVG Evaluation Configuration"
echo "=========================================="
echo "Model Name:              $MODEL_NAME"
echo "Model Path:              $MODEL_PATH"
echo "TA-STVG Dir:             /mnt/sdc/xingjianwang/yibowang/TA-STVG"
echo "TA-STVG Python:          /mnt/sdc/xingjianwang/yibowang/TA-STVG/.venv/bin/python"
echo "TA-STVG Infer Helper:    /mnt/sdc/xingjianwang/yibowang/DORO-STVG-3e9aedc/eval/utils/tastvg_infer_helper.py"
echo "Annotation Path:         $ANNOTATION_PATH"
echo "Video Directory:         $VIDEO_DIR"
echo "Output Directory:        $OUTPUT_DIR"
echo "Input Resolution:        224"
echo "Num Clip Frames:         16"
echo "Num Workers:             0"
echo "Prepare Cache:           1"
echo "Official Extra Opts:     MODEL.TASTVG.USE_ACTION False"
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
