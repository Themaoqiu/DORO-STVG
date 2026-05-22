#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/envs/eval/cgstvg/.venv/bin/python}"
MODEL_NAME="${MODEL_NAME:-cgstvg}"
MODEL_PATH="${MODEL_PATH:-/home/wangxingjian/model/CGSTVG/vidstg.pth}"
DATA_NAME="${DATA_NAME:-doro-stvg}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/home/wangxingjian/DORO-STVG/graph_generator/modules/autoresearch/round_22/query_polished.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-/home/wangxingjian/data/vidstg/video}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_DIR/res_cgstvg}"
BATCH_SIZE="${BATCH_SIZE:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CGSTVG_PYTHON="${CGSTVG_PYTHON:-$REPO_ROOT/envs/eval/cgstvg/.venv/bin/python}"
export CGSTVG_INFER_PY="${CGSTVG_INFER_PY:-$EVAL_DIR/utils/cgstvg_infer_helper.py}"
export CGSTVG_MODEL_WEIGHT="${CGSTVG_MODEL_WEIGHT:-$MODEL_PATH}"
export CGSTVG_CUDA_VISIBLE_DEVICES="${CGSTVG_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
export CGSTVG_CONFIG_FILE="${CGSTVG_CONFIG_FILE:-dependence/cgstvg/experiments/vidstg.yaml}"
export CGSTVG_TEST_MODULE="${CGSTVG_TEST_MODULE:-dependence.cgstvg.scripts.test_net}"
export CGSTVG_RESULT_FILE="${CGSTVG_RESULT_FILE:-test_results.json}"
export CGSTVG_INPUT_RESOLUTION="${CGSTVG_INPUT_RESOLUTION:-420}"
export CGSTVG_NUM_CLIP_FRAMES="${CGSTVG_NUM_CLIP_FRAMES:-32}"
export CGSTVG_NUM_WORKERS="${CGSTVG_NUM_WORKERS:-8}"
export CGSTVG_OFFICIAL_EXTRA_OPTS="${CGSTVG_OFFICIAL_EXTRA_OPTS:-}"
export CGSTVG_EXTRA_ARGS="${CGSTVG_EXTRA_ARGS:-}"
export CGSTVG_KEEP_TMP="${CGSTVG_KEEP_TMP:-0}"

echo "=========================================="
echo "CG-STVG Evaluation Configuration"
echo "=========================================="
echo "Python Bin:             $PYTHON_BIN"
echo "Model Name:             $MODEL_NAME"
echo "Model Path:             $MODEL_PATH"
echo "CGSTVG Python:          $CGSTVG_PYTHON"
echo "CGSTVG Infer Helper:    $CGSTVG_INFER_PY"
echo "CGSTVG Config:          $CGSTVG_CONFIG_FILE"
echo "CGSTVG Test Module:     $CGSTVG_TEST_MODULE"
echo "CGSTVG Result File:     $CGSTVG_RESULT_FILE"
echo "Annotation Path:        $ANNOTATION_PATH"
echo "Video Directory:        $VIDEO_DIR"
echo "Output Directory:       $OUTPUT_DIR"
echo "Batch Size:             $BATCH_SIZE"
echo "Input Resolution:       $CGSTVG_INPUT_RESOLUTION"
echo "Num Clip Frames:        $CGSTVG_NUM_CLIP_FRAMES"
echo "Num Workers:            $CGSTVG_NUM_WORKERS"
echo "Visible GPUs:           $CGSTVG_CUDA_VISIBLE_DEVICES"
echo "Official Extra Opts:    ${CGSTVG_OFFICIAL_EXTRA_OPTS:-<none>}"
echo "Extra Args:             ${CGSTVG_EXTRA_ARGS:-<none>}"
echo "Keep Tmp:               $CGSTVG_KEEP_TMP"
echo "=========================================="
echo

cd "$EVAL_DIR"
"$PYTHON_BIN" main.py run \
  --model_name="$MODEL_NAME" \
  --model_path="$MODEL_PATH" \
  --data_name="$DATA_NAME" \
  --annotation_path="$ANNOTATION_PATH" \
  --video_dir="$VIDEO_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size="$BATCH_SIZE"
