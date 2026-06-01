#!/usr/bin/env bash
# Compare CP-SAT (ours) against ILP / Greedy / Random baselines on graph .jsonl files.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
VENV="$REPO_ROOT/.venv-baseline"

# if [[ ! -x "$VENV/bin/python" ]]; then
#     echo "[bench] creating venv at $VENV"
#     (cd "$REPO_ROOT" && uv venv .venv-baseline --python 3.11)
#     VIRTUAL_ENV="$VENV" uv pip install ortools pulp fire openai
# fi

# ---- I/O ----
INPUTS=/home/wangxingjian/data/compstvg/preception.jsonl,/home/wangxingjian/data/compstvg/mose.jsonl
OUT=/home/wangxingjian/DORO-STVG/graph_generator/outputs/baselines/mose_preception_full_run.json

# ---- benchmark scope ----
NUM_GRAPHS=-1            # -1 = every graph in each file
MAX_CASES=-1             # -1 = no cap

# ---- solver knobs ----
TIME_LIMIT=2.0
RANDOM_TRIALS=500
RANDOM_SEED=0
NUM_WORKERS=8

mkdir -p "$(dirname "$OUT")"

echo "[bench] inputs      : $INPUTS"
echo "[bench] num_graphs  : $NUM_GRAPHS  (per file)"
echo "[bench] max_cases   : $MAX_CASES"
echo "[bench] num_workers : $NUM_WORKERS"
echo "[bench] out         : $OUT"

cd "$PROJECT_ROOT"
python -m modules.query_generator.baseline.benchmark \
    --inputs "$INPUTS" \
    --num_graphs "$NUM_GRAPHS" \
    --max_cases "$MAX_CASES" \
    --time_limit_sec "$TIME_LIMIT" \
    --random_max_trials "$RANDOM_TRIALS" \
    --random_seed "$RANDOM_SEED" \
    --num_workers "$NUM_WORKERS" \
    --out "$OUT"
