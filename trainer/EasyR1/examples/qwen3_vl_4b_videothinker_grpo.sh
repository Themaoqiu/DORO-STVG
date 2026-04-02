#!/bin/bash

set -euo pipefail
set -x

MODEL_PATH="/home/wangxingjian/model/qwen3-vl-2b-instruct"
export CUDA_VISIBLE_DEVICES=0
export FORCE_QWENVL_VIDEO_READER=${FORCE_QWENVL_VIDEO_READER:-decord}
export RAY_local_fs_capacity_threshold=${RAY_local_fs_capacity_threshold:-0.999}
export BAD_SAMPLES_LOG=${BAD_SAMPLES_LOG:-./checkpoints/easy_r1/qwen3_vl_4b_videothinker_grpo/bad_samples.txt}

python3 -m verl.trainer.main \
  config=examples/videothinker_config.yaml \
  worker.actor.model.model_path=${MODEL_PATH}
