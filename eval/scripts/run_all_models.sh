#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"

ANNOTATION_PATH="${ANNOTATION_PATH:-/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_first10videos_valid.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke}"
DATA_NAME="${DATA_NAME:-doro-stvg}"
VISIBLE_GPUS="${VISIBLE_GPUS:-${CUDA_VISIBLE_DEVICES:-3}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EVAL_DIR/res_all_models/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"
FIRST_N="${FIRST_N:-0}"
HF_OFFLINE="${HF_OFFLINE:-1}"
CLEAR_PROXY="${CLEAR_PROXY:-1}"
RUN_QWEN="${RUN_QWEN:-0}"
UV_BIN="${UV_BIN:-uv}"
AUTO_INSTALL_EXTERNAL="${AUTO_INSTALL_EXTERNAL:-1}"

DEFAULT_MODELS=(
  llava-st-qwen2
  vtimellm
  grounded-video-llm
  llava16
  videochat-r1
  stvg-r1
  groundinggpt
  cgstvg
  tastvg
  tubedetr
  videomolmo
)

if [[ "$RUN_QWEN" == "1" ]]; then
  DEFAULT_MODELS=(qwen3.5 "${DEFAULT_MODELS[@]}")
fi

if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  SELECTED_MODELS=($MODELS)
else
  SELECTED_MODELS=("${DEFAULT_MODELS[@]}")
fi

if [[ "$FIRST_N" =~ ^[0-9]+$ ]] && (( FIRST_N > 0 )) && (( FIRST_N < ${#SELECTED_MODELS[@]} )); then
  SELECTED_MODELS=("${SELECTED_MODELS[@]:0:FIRST_N}")
fi

mkdir -p "$LOG_DIR" || {
  echo "[ERROR] Failed to create log directory: $LOG_DIR" >&2
  exit 1
}

if [[ "$CLEAR_PROXY" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
fi

if [[ "$HF_OFFLINE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

SUMMARY_FILE="$OUTPUT_ROOT/summary.tsv"
printf "model\teval_type\tentrypoint\tstatus\texit_code\tlog\n" > "$SUMMARY_FILE" || {
  echo "[ERROR] Failed to create summary file: $SUMMARY_FILE" >&2
  exit 1
}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "[WARN] $label does not exist: $path" >&2
  fi
}

configure_model() {
  local model="$1"

  ENV_NAME=""
  PYTHON_BIN=""
  UV_PROJECT=""
  USE_UV=0
  EXTERNAL_INSTALL_PATH=""
  SKIP_MODEL=0
  SKIP_REASON=""
  MODEL_NAME="$model"
  MODEL_PATH=""
  BATCH_SIZE=1
  MAX_TOKENS=512
  MAX_MODEL_LEN=8192
  TEMPERATURE=0.0
  TENSOR_PARALLEL_SIZE=1
  GPU_MEMORY_UTILIZATION=0.9
  EVAL_TYPE="spatiotemporal"
  ENTRYPOINT="main.py"

  case "$model" in
    qwen3.5|qwen3vl|qwen3-vl)
      ENV_NAME="qwen"
      UV_PROJECT="$REPO_ROOT/envs/eval/qwen"
      PYTHON_BIN="${QWEN_PYTHON:-$REPO_ROOT/envs/eval/qwen/.venv/bin/python}"
      MODEL_NAME="${QWEN_MODEL_NAME:-qwen3.5}"
      MODEL_PATH="${QWEN_MODEL_PATH:-/home/wangxingjian/model/qwen3.5-9b}"
      BATCH_SIZE="${QWEN_BATCH_SIZE:-8}"
      MAX_TOKENS="${QWEN_MAX_TOKENS:-4096}"
      MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-64000}"
      TEMPERATURE="${QWEN_TEMPERATURE:-0.1}"
      GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.9}"
      if [[ "$RUN_QWEN" != "1" ]]; then
        SKIP_MODEL=1
        SKIP_REASON="qwen skipped; set RUN_QWEN=1 and QWEN_MODEL_PATH to enable"
      fi
      ;;

    llava-st-qwen2|llavast|llava-st)
      ENV_NAME="llavast"
      UV_PROJECT="$REPO_ROOT/envs/eval/llavast"
      PYTHON_BIN="${LLAVAST_PYTHON:-$REPO_ROOT/envs/eval/llavast/.venv/bin/python}"
      MODEL_NAME="llava-st-qwen2"
      MODEL_PATH="${LLAVAST_MODEL_PATH:-/mnt/sdc/xingjianwang/yibowang/models/LLaVA-ST-Qwen2-7B}"
      TEMPERATURE="${LLAVAST_TEMPERATURE:-0.1}"
      export LLAVA_ST_SOURCE_DIR="${LLAVA_ST_SOURCE_DIR:-/mnt/sdc/xingjianwang/yibowang/LLaVA-ST}"
      EXTERNAL_INSTALL_PATH="$LLAVA_ST_SOURCE_DIR"
      export LLAVA_ST_MAX_FRAMES="${LLAVA_ST_MAX_FRAMES:-100}"
      export LLAVA_ST_VT_CHUNK="${LLAVA_ST_VT_CHUNK:-1}"
      ;;

    vtimellm|vtime-llm|vtime_llm)
      ENV_NAME="vtimellm"
      UV_PROJECT="$REPO_ROOT/envs/eval/vtimellm"
      PYTHON_BIN="${VTIMELLM_PYTHON_BIN:-$REPO_ROOT/envs/eval/vtimellm/.venv/bin/python}"
      MODEL_NAME="vtimellm"
      export VTIMELLM_SOURCE_DIR="${VTIMELLM_SOURCE_DIR:-/mnt/sdc/xingjianwang/yibowang/VTimeLLM}"
      EXTERNAL_INSTALL_PATH="$VTIMELLM_SOURCE_DIR"
      export VTIMELLM_MODEL_BASE="${VTIMELLM_MODEL_BASE:-/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/vicuna-7b-v1.5}"
      export VTIMELLM_PRETRAIN_MM_MLP_ADAPTER="${VTIMELLM_PRETRAIN_MM_MLP_ADAPTER:-/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin}"
      export VTIMELLM_STAGE2="${VTIMELLM_STAGE2:-/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage2}"
      export VTIMELLM_STAGE3="${VTIMELLM_STAGE3:-/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3}"
      export VTIMELLM_CLIP_PATH="${VTIMELLM_CLIP_PATH:-/mnt/sdc/xingjianwang/yibowang/VTimeLLM/checkpoints/clip/ViT-L-14.pt}"
      MODEL_PATH="${VTIMELLM_MODEL_PATH:-$VTIMELLM_STAGE3}"
      EVAL_TYPE="${VTIMELLM_EVAL_TYPE:-temporal_only}"
      if [[ "$EVAL_TYPE" == "temporal_only" ]]; then
        ENTRYPOINT="temporal_main.py"
      fi
      ;;

    grounded-video-llm|grounded_video_llm|groundedvideollm)
      ENV_NAME="grounded_video_llm"
      UV_PROJECT="$REPO_ROOT/envs/eval/grounded_video_llm"
      PYTHON_BIN="${GROUNDED_VIDEO_LLM_EVAL_PYTHON:-$REPO_ROOT/envs/eval/grounded_video_llm/.venv/bin/python}"
      MODEL_NAME="grounded-video-llm"
      export GROUNDED_VIDEO_LLM_SOURCE_DIR="${GROUNDED_VIDEO_LLM_SOURCE_DIR:-/mnt/sdc/xingjianwang/yibowang/Grounded-Video-LLM}"
      EXTERNAL_INSTALL_PATH="$GROUNDED_VIDEO_LLM_SOURCE_DIR"
      export GROUNDED_VIDEO_LLM_PYTHON="${GROUNDED_VIDEO_LLM_PYTHON:-$REPO_ROOT/envs/eval/grounded_video_llm/.venv/bin/python}"
      export GROUNDED_VIDEO_LLM_DEVICE="${GROUNDED_VIDEO_LLM_DEVICE:-cuda:0}"
      export GROUNDED_VIDEO_LLM_LLM="${GROUNDED_VIDEO_LLM_LLM:-phi3.5}"
      export GROUNDED_VIDEO_LLM_CONFIG_PATH="${GROUNDED_VIDEO_LLM_CONFIG_PATH:-/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/Phi-3.5-vision-instruct}"
      export GROUNDED_VIDEO_LLM_TOKENIZER_PATH="${GROUNDED_VIDEO_LLM_TOKENIZER_PATH:-/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/Phi-3.5-mini-instruct}"
      export GROUNDED_VIDEO_LLM_PRETRAINED_VIDEO_PATH="${GROUNDED_VIDEO_LLM_PRETRAINED_VIDEO_PATH:-/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt}"
      export GROUNDED_VIDEO_LLM_PRETRAINED_VISION_PROJ_LLM_PATH="${GROUNDED_VIDEO_LLM_PRETRAINED_VISION_PROJ_LLM_PATH:-/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/Phi-3.5-vision-instruct-seperated}"
      export GROUNDED_VIDEO_LLM_CKPT_PATH="${GROUNDED_VIDEO_LLM_CKPT_PATH:-/mnt/sdc/xingjianwang/yibowang/models/Grounded-Video-LLM/ckpt/sft_llava_next_video_phi3.5_mix_sft_multi_modal_projector_video_projecter_language_model.pth}"
      MODEL_PATH="${GROUNDED_VIDEO_LLM_MODEL_PATH:-$GROUNDED_VIDEO_LLM_CKPT_PATH}"
      MAX_TOKENS="${GROUNDED_VIDEO_LLM_MAX_TOKENS:-1024}"
      EVAL_TYPE="${GROUNDED_VIDEO_LLM_EVAL_TYPE:-temporal_only}"
      if [[ "$EVAL_TYPE" == "temporal_only" ]]; then
        ENTRYPOINT="temporal_main.py"
      fi
      ;;

    llava16|llava-1.6|llava-v1.6)
      ENV_NAME="llava16"
      UV_PROJECT="$REPO_ROOT/envs/eval/llava16"
      PYTHON_BIN="${LLAVA16_PYTHON:-$REPO_ROOT/envs/eval/llava16/.venv/bin/python}"
      MODEL_NAME="llava16"
      MODEL_PATH="${LLAVA16_MODEL_PATH:-/mnt/sdc/xingjianwang/yibowang/model_zoo/llava-v1.6-mistral-7b-hf}"
      MAX_MODEL_LEN="${LLAVA16_MAX_MODEL_LEN:-4096}"
      GPU_MEMORY_UTILIZATION="${LLAVA16_GPU_MEMORY_UTILIZATION:-0.8}"
      export LLAVA16_MAX_FRAMES="${LLAVA16_MAX_FRAMES:-6}"
      export LLAVA16_GRID_COLUMNS="${LLAVA16_GRID_COLUMNS:-3}"
      export LLAVA16_ENFORCE_EAGER="${LLAVA16_ENFORCE_EAGER:-0}"
      ;;

    videochat-r1|videochat_r1|videochatr1)
      ENV_NAME="videochat_r1"
      UV_PROJECT="$REPO_ROOT/envs/eval/videochat_r1"
      PYTHON_BIN="${VIDEOCHAT_R1_PYTHON:-$REPO_ROOT/envs/eval/videochat_r1/.venv/bin/python}"
      MODEL_NAME="videochat-r1"
      MODEL_PATH="${VIDEOCHAT_R1_MODEL_PATH:-/mnt/sdc/xingjianwang/yibowang/model_zoo/VideoChat-R1_7B}"
      GPU_MEMORY_UTILIZATION="${VIDEOCHAT_R1_GPU_MEMORY_UTILIZATION:-0.8}"
      ;;

    stvg-r1|stvg_r1|stvgr1)
      ENV_NAME="stvg_r1"
      UV_PROJECT="$REPO_ROOT/envs/eval/stvg_r1"
      PYTHON_BIN="${STVG_R1_PYTHON:-$REPO_ROOT/envs/eval/stvg_r1/.venv/bin/python}"
      MODEL_NAME="stvg-r1"
      MODEL_PATH="${STVG_R1_MODEL_PATH:-/mnt/sdc/xingjianwang/yibowang/model_zoo/stvg-r1-model-7b}"
      GPU_MEMORY_UTILIZATION="${STVG_R1_GPU_MEMORY_UTILIZATION:-0.8}"
      export STVG_R1_MAX_FRAMES="${STVG_R1_MAX_FRAMES:-32}"
      export STVG_R1_CLIP_FPS="${STVG_R1_CLIP_FPS:-2.0}"
      export STVG_R1_MAX_OUTPUT_FRAMES="${STVG_R1_MAX_OUTPUT_FRAMES:-8}"
      export STVG_R1_KEEP_TMP="${STVG_R1_KEEP_TMP:-0}"
      ;;

    groundinggpt|grounding-gpt|grounding_gpt)
      ENV_NAME="groundinggpt"
      UV_PROJECT="$REPO_ROOT/envs/eval/groundinggpt"
      PYTHON_BIN="${GROUNDINGGPT_EVAL_PYTHON:-$REPO_ROOT/envs/eval/groundinggpt/.venv/bin/python}"
      MODEL_NAME="groundinggpt"
      MODEL_PATH="${GROUNDINGGPT_MODEL_PATH:-/mnt/sdc/xingjianwang/yibowang/model_zoo/GroundingGPT}"
      TEMPERATURE="${GROUNDINGGPT_TEMPERATURE:-0.01}"
      export GROUNDINGGPT_CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
      export GROUNDINGGPT_CLIP_VISION_TOWER="${GROUNDINGGPT_CLIP_VISION_TOWER:-/mnt/sdc/xingjianwang/models/GroundingGPT/clip-vit-large-patch14-336}"
      export GROUNDINGGPT_BERT_PATH="${GROUNDINGGPT_BERT_PATH:-/mnt/sdc/xingjianwang/models/GroundingGPT/bert-base-uncased}"
      export GROUNDINGGPT_IMAGEBIND_PATH="${GROUNDINGGPT_IMAGEBIND_PATH:-/mnt/sdc/xingjianwang/models/GroundingGPT/imagebind/imagebind_huge.pth}"
      export GROUNDINGGPT_EVA_VIT_G_PATH="${GROUNDINGGPT_EVA_VIT_G_PATH:-/mnt/sdc/xingjianwang/models/GroundingGPT/eva_vit_g.pth}"
      export GROUNDINGGPT_MAX_NEW_TOKENS="${GROUNDINGGPT_MAX_NEW_TOKENS:-1024}"
      ;;

    videomolmo|video-molmo)
      ENV_NAME="videomolmo"
      UV_PROJECT="$REPO_ROOT/envs/eval/videomolmo"
      PYTHON_BIN="${VIDEOMOLMO_EVAL_PYTHON:-$REPO_ROOT/envs/eval/videomolmo/.venv/bin/python}"
      MODEL_NAME="videomolmo"
      export VIDEOMOLMO_REPO="${VIDEOMOLMO_REPO:-/mnt/sdc/xingjianwang/yibowang/VideoMolmo}"
      export VIDEOMOLMO_SOURCE_DIR="${VIDEOMOLMO_SOURCE_DIR:-$VIDEOMOLMO_REPO/VideoMolmo}"
      export MOLMO_SOURCE_DIR="${MOLMO_SOURCE_DIR:-/mnt/sdc/xingjianwang/yibowang/molmo}"
      export SAM2_SOURCE_DIR="${SAM2_SOURCE_DIR:-$VIDEOMOLMO_SOURCE_DIR/sam2}"
      export VIDEOMOLMO_PYTHON="${VIDEOMOLMO_PYTHON:-/mnt/sdc/xingjianwang/yibowang/DORO-STVG-pr-verify/envs/eval/.venv/bin/python}"
      export VIDEOMOLMO_MAX_FRAMES="${VIDEOMOLMO_MAX_FRAMES:-100}"
      export VIDEOMOLMO_SAMPLE_FPS="${VIDEOMOLMO_SAMPLE_FPS:-2.0}"
      export VIDEOMOLMO_POINT_BOX_HALF="${VIDEOMOLMO_POINT_BOX_HALF:-0.04}"
      export VIDEOMOLMO_ALLOW_LOG_FALLBACK="${VIDEOMOLMO_ALLOW_LOG_FALLBACK:-0}"
      export VIDEOMOLMO_USE_POINT_PROMPT="${VIDEOMOLMO_USE_POINT_PROMPT:-1}"
      export PYTHONPATH="$SAM2_SOURCE_DIR:$VIDEOMOLMO_SOURCE_DIR:$MOLMO_SOURCE_DIR:${PYTHONPATH:-}"
      MODEL_PATH="${VIDEOMOLMO_MODEL_PATH:-$VIDEOMOLMO_REPO}"
      TEMPERATURE="${VIDEOMOLMO_TEMPERATURE:-0.1}"
      ;;

    cgstvg|cg-stvg|cg_stvg)
      ENV_NAME="cgstvg"
      UV_PROJECT="$REPO_ROOT/envs/eval/cgstvg"
      PYTHON_BIN="${CGSTVG_EVAL_PYTHON:-$REPO_ROOT/envs/eval/cgstvg/.venv/bin/python}"
      MODEL_NAME="cgstvg"
      export CGSTVG_DIR="${CGSTVG_DIR:-/mnt/sdc/xingjianwang/yibowang/CGSTVG}"
      export CGSTVG_PYTHON="${CGSTVG_PYTHON:-$CGSTVG_DIR/.venv/bin/python}"
      export CGSTVG_INFER_PY="${CGSTVG_INFER_PY:-$EVAL_DIR/utils/cgstvg_infer_helper.py}"
      export CGSTVG_MODEL_WEIGHT="${CGSTVG_MODEL_WEIGHT:-$CGSTVG_DIR/model_zoo/vidstg.pth}"
      export CGSTVG_CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
      export CGSTVG_INPUT_RESOLUTION="${CGSTVG_INPUT_RESOLUTION:-224}"
      export CGSTVG_NUM_CLIP_FRAMES="${CGSTVG_NUM_CLIP_FRAMES:-16}"
      export CGSTVG_NUM_WORKERS="${CGSTVG_NUM_WORKERS:-0}"
      MODEL_PATH="${CGSTVG_MODEL_PATH:-$CGSTVG_MODEL_WEIGHT}"
      ;;

    tastvg|ta-stvg|ta_stvg)
      ENV_NAME="tastvg"
      UV_PROJECT="$REPO_ROOT/envs/eval/tastvg"
      PYTHON_BIN="${TASTVG_EVAL_PYTHON:-$REPO_ROOT/envs/eval/tastvg/.venv/bin/python}"
      MODEL_NAME="tastvg"
      export TASTVG_DIR="${TASTVG_DIR:-/mnt/sdc/xingjianwang/yibowang/TA-STVG}"
      export TASTVG_PYTHON="${TASTVG_PYTHON:-$TASTVG_DIR/.venv/bin/python}"
      export TASTVG_INFER_PY="${TASTVG_INFER_PY:-$EVAL_DIR/utils/tastvg_infer_helper.py}"
      export TASTVG_MODEL_WEIGHT="${TASTVG_MODEL_WEIGHT:-$TASTVG_DIR/model_zoo/TASTVG_VidSTG.pth}"
      export TASTVG_CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
      export TASTVG_INPUT_RESOLUTION="${TASTVG_INPUT_RESOLUTION:-224}"
      export TASTVG_NUM_CLIP_FRAMES="${TASTVG_NUM_CLIP_FRAMES:-16}"
      export TASTVG_NUM_WORKERS="${TASTVG_NUM_WORKERS:-0}"
      export TASTVG_PREPARE_CACHE="${TASTVG_PREPARE_CACHE:-1}"
      export TASTVG_CONFIG_FILE="${TASTVG_CONFIG_FILE:-experiments/vidstg.yaml}"
      export TASTVG_TEST_SCRIPT="${TASTVG_TEST_SCRIPT:-scripts/test_net.py}"
      export TASTVG_RESULT_FILE="${TASTVG_RESULT_FILE:-test_results.json}"
      export TASTVG_OFFICIAL_EXTRA_OPTS="${TASTVG_OFFICIAL_EXTRA_OPTS:-MODEL.TASTVG.USE_ACTION False}"
      MODEL_PATH="${TASTVG_MODEL_PATH:-$TASTVG_MODEL_WEIGHT}"
      ;;

    tubedetr|tube-detr|tube_detr)
      ENV_NAME="tubedetr"
      UV_PROJECT="$REPO_ROOT/envs/eval/tubedetr"
      PYTHON_BIN="${TUBEDETR_EVAL_PYTHON:-$REPO_ROOT/envs/eval/tubedetr/.venv/bin/python}"
      MODEL_NAME="tubedetr"
      export TUBEDETR_DIR="${TUBEDETR_DIR:-/mnt/sdc/xingjianwang/yibowang/TubeDETR}"
      export TUBEDETR_PYTHON="${TUBEDETR_PYTHON:-$TUBEDETR_DIR/.venv/bin/python}"
      export TUBEDETR_INFER_PY="${TUBEDETR_INFER_PY:-$EVAL_DIR/utils/tubedetr_infer_helper.py}"
      export TUBEDETR_CHECKPOINT="${TUBEDETR_CHECKPOINT:-$TUBEDETR_DIR/checkpoints/tubedetr_vidstg.pth}"
      export TUBEDETR_CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
      export TUBEDETR_DATASET_CONFIG="${TUBEDETR_DATASET_CONFIG:-config/vidstg.json}"
      export TUBEDETR_COMBINE_DATASETS="${TUBEDETR_COMBINE_DATASETS:-vidstg}"
      export TUBEDETR_COMBINE_DATASETS_VAL="${TUBEDETR_COMBINE_DATASETS_VAL:-vidstg}"
      export TUBEDETR_RESOLUTION="${TUBEDETR_RESOLUTION:-224}"
      export TUBEDETR_FPS="${TUBEDETR_FPS:-2}"
      export TUBEDETR_DEVICE="${TUBEDETR_DEVICE:-cuda}"
      export ST_ALIGN_BENCHMARK_PATH="${ST_ALIGN_BENCHMARK_PATH:-/mnt/sdc/xingjianwang/yibowang/LLaVA-ST/data/benchmarks/st-align/stvg.json}"
      MODEL_PATH="${TUBEDETR_MODEL_PATH:-$TUBEDETR_CHECKPOINT}"
      ;;

    *)
      echo "[ERROR] Unknown model in MODELS: $model" >&2
      return 1
      ;;
  esac

  if [[ "$SKIP_MODEL" != "1" && ! -x "$PYTHON_BIN" ]]; then
    USE_UV=1
  fi

  return 0
}

run_model() {
  local model="$1"
  local safe_model="${model//[^A-Za-z0-9_.-]/_}"
  local output_dir="$OUTPUT_ROOT/$safe_model"
  local log_file="$LOG_DIR/$safe_model.log"

  configure_model "$model" || {
    printf "%s\tunknown\tunknown\tconfig_error\t1\t%s\n" "$model" "$log_file" >> "$SUMMARY_FILE"
    return 1
  }

  if [[ "$SKIP_MODEL" == "1" ]]; then
    echo "[SKIP] $model: $SKIP_REASON"
    printf "%s\t%s\t%s\tskipped\t0\t%s\n" "$model" "$EVAL_TYPE" "$ENTRYPOINT" "$SKIP_REASON" >> "$SUMMARY_FILE"
    return 0
  fi

  echo "=========================================="
  echo "Running model:        $MODEL_NAME"
  echo "Eval env:             $ENV_NAME"
  echo "Python:               $PYTHON_BIN"
  echo "UV project:           $UV_PROJECT"
  echo "Use uv:               $USE_UV"
  echo "Eval type:            $EVAL_TYPE"
  echo "Entrypoint:           $ENTRYPOINT"
  echo "External install:     ${EXTERNAL_INSTALL_PATH:-none}"
  echo "Model path:           $MODEL_PATH"
  echo "Annotation path:      $ANNOTATION_PATH"
  echo "Video dir:            $VIDEO_DIR"
  echo "Output dir:           $output_dir"
  echo "Log file:             $log_file"
  echo "Visible GPUs:         $VISIBLE_GPUS"
  echo "=========================================="

  require_path "annotation" "$ANNOTATION_PATH"
  require_path "video dir" "$VIDEO_DIR"
  require_path "entrypoint" "$EVAL_DIR/$ENTRYPOINT"
  if [[ "$USE_UV" == "1" ]]; then
    require_path "uv project" "$UV_PROJECT"
  else
    require_path "python" "$PYTHON_BIN"
  fi
  require_path "model path" "$MODEL_PATH"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "%s\t%s\t%s\tdry_run\t0\t%s\n" "$model" "$EVAL_TYPE" "$ENTRYPOINT" "$log_file" >> "$SUMMARY_FILE"
    return 0
  fi

  mkdir -p "$output_dir"

  (
    cd "$EVAL_DIR"
    if [[ "$AUTO_INSTALL_EXTERNAL" == "1" && -n "$EXTERNAL_INSTALL_PATH" && -d "$EXTERNAL_INSTALL_PATH" ]]; then
      echo "[INFO] Installing external dependency into $ENV_NAME env: $EXTERNAL_INSTALL_PATH"
      if [[ "$USE_UV" == "1" ]]; then
        UV_LINK_MODE="${UV_LINK_MODE:-copy}" "$UV_BIN" run --project "$UV_PROJECT" uv pip install -e "$EXTERNAL_INSTALL_PATH"
      else
        UV_LINK_MODE="${UV_LINK_MODE:-copy}" "$UV_BIN" pip install --python "$PYTHON_BIN" -e "$EXTERNAL_INSTALL_PATH"
      fi
    fi

    if [[ "$USE_UV" == "1" ]]; then
      UV_LINK_MODE="${UV_LINK_MODE:-copy}" "$UV_BIN" run --project "$UV_PROJECT" python "$ENTRYPOINT" run \
        --model_name="$MODEL_NAME" \
        --model_path="$MODEL_PATH" \
        --data_name="$DATA_NAME" \
        --annotation_path="$ANNOTATION_PATH" \
        --video_dir="$VIDEO_DIR" \
        --output_dir="$output_dir" \
        --batch_size="$BATCH_SIZE" \
        --max_tokens="$MAX_TOKENS" \
        --max_model_len="$MAX_MODEL_LEN" \
        --temperature="$TEMPERATURE" \
        --tensor_parallel_size="$TENSOR_PARALLEL_SIZE" \
        --gpu_memory_utilization="$GPU_MEMORY_UTILIZATION"
    else
      "$PYTHON_BIN" "$ENTRYPOINT" run \
        --model_name="$MODEL_NAME" \
        --model_path="$MODEL_PATH" \
        --data_name="$DATA_NAME" \
        --annotation_path="$ANNOTATION_PATH" \
        --video_dir="$VIDEO_DIR" \
        --output_dir="$output_dir" \
        --batch_size="$BATCH_SIZE" \
        --max_tokens="$MAX_TOKENS" \
        --max_model_len="$MAX_MODEL_LEN" \
        --temperature="$TEMPERATURE" \
        --tensor_parallel_size="$TENSOR_PARALLEL_SIZE" \
        --gpu_memory_utilization="$GPU_MEMORY_UTILIZATION"
    fi
  ) 2>&1 | tee "$log_file"

  local exit_code=${PIPESTATUS[0]}
  if [[ "$exit_code" -eq 0 ]]; then
    printf "%s\t%s\t%s\tok\t0\t%s\n" "$model" "$EVAL_TYPE" "$ENTRYPOINT" "$log_file" >> "$SUMMARY_FILE"
  else
    printf "%s\t%s\t%s\tfailed\t%s\t%s\n" "$model" "$EVAL_TYPE" "$ENTRYPOINT" "$exit_code" "$log_file" >> "$SUMMARY_FILE"
  fi

  return "$exit_code"
}

echo "=========================================="
echo "STVG all-model evaluation"
echo "=========================================="
echo "Repo root:        $REPO_ROOT"
echo "Eval dir:         $EVAL_DIR"
echo "Models:           ${SELECTED_MODELS[*]}"
echo "Annotation path:  $ANNOTATION_PATH"
echo "Video dir:        $VIDEO_DIR"
echo "Output root:      $OUTPUT_ROOT"
echo "Logs:             $LOG_DIR"
echo "Visible GPUs:     $VISIBLE_GPUS"
echo "HF offline:       $HF_OFFLINE"
echo "Clear proxy:      $CLEAR_PROXY"
echo "Continue errors:  $CONTINUE_ON_ERROR"
echo "Dry run:          $DRY_RUN"
echo "Run Qwen:         $RUN_QWEN"
echo "UV bin:           $UV_BIN"
echo "Auto external:    $AUTO_INSTALL_EXTERNAL"
echo "=========================================="

overall_status=0
for model in "${SELECTED_MODELS[@]}"; do
  if ! run_model "$model"; then
    overall_status=1
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      break
    fi
  fi
done

echo "=========================================="
echo "Summary: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
echo "=========================================="

exit "$overall_status"
