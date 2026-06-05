#!/usr/bin/env bash
# CompSTVG (mose + perception) SFT for Qwen3-VL-8B-Instruct on 8x A800 80GB.
# Hyperparameters aligned with Video-Thinker SFT.
# Run: bash examples/vlm_finetune/qwen3/run_compstvg_sft_8b.sh
set -euo pipefail

cd /home/wangxingjian/DORO-STVG/trainer/Automodel

CONFIG=examples/vlm_finetune/qwen3/qwen3_vl_8b_compstvg.yaml
NPROC=2

export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com
export DECORD_EOF_RETRY_MAX=16384
export TOKENIZERS_PARALLELISM=false

# Prepare splits if missing
SPLIT_DIR=/home/wangxingjian/data/compstvg/splits
if [ ! -f $SPLIT_DIR/sft.jsonl ]; then
  echo "[run_compstvg_sft_8b] sft.jsonl missing, regenerating splits..."
  uv run python scripts/split_compstvg_sft_rl.py \
    --inputs /home/wangxingjian/data/compstvg/mose.queries.jsonl \
             /home/wangxingjian/data/compstvg/preception.queries.jsonl \
    --out-dir $SPLIT_DIR \
    --seed 42
fi

echo "============================================================"
echo "Config:        $CONFIG"
echo "NPROC:         $NPROC"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "HF_ENDPOINT:   $HF_ENDPOINT"
echo "============================================================"

uv run automodel $CONFIG --nproc-per-node $NPROC "$@"
