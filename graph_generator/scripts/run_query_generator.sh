#!/usr/bin/env bash
# Synthesize STVG queries from one or more scene-graph .jsonl files.
# Each input file is processed independently and written to its own output.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

# ---- I/O ----
# Comma-separated list of input scene-graph .jsonl files.
INPUTS=/tmp/preception_50.jsonl,/tmp/mose_50.jsonl
# Output directory; per-input file is written as <stem>.queries.jsonl here.
OUTPUT_DIR=/tmp/queries_q100

# ---- sampler / solver ----
MIN_INTERVAL_LEN=3
MAX_INTERVALS_PER_OBJECT=15
MAX_TARGET_ARITY=4
MAX_MULTI_INTERVALS_PER_GROUP=15
MAX_MULTI_CANDIDATES_TOTAL=240
STRICT_TIME_UNIQUENESS_MULTI=False
MAX_CHAIN_LEN=6
MAX_QUERIES_PER_CANDIDATE=1
TIME_LIMIT_SEC=2.0
SEED=7
EPS=0.01
LAMBDA_WEIGHT=0.5

# ---- post-processing ----
MAX_QUERIES_PER_VIDEO=5
MAX_QUERIES_PER_BUCKET=2
EXPAND_NON_TEMPORAL_QUERY_TRACKS=True

# ---- optional LLM polishing ----
USE_LLM_POLISH=False
POLISH_MODEL_NAME=gemini-3-flash
MAX_CONCURRENT_PER_KEY=30
MAX_RETRIES=5

# ---- python ----
# Prefer the project main env when fully installed; fall back to the baseline
# env (which has fire+ortools+pulp but not cv2/decord — fine for query gen).
PY=python3
_main_py="$PROJECT_ROOT/../envs/graph_generator/main/.venv/bin/python"
_base_py="$PROJECT_ROOT/../.venv-baseline/bin/python"
if [[ -x "$_main_py" ]] && "$_main_py" -c "import fire, ortools" >/dev/null 2>&1; then
    PY="$_main_py"
elif [[ -x "$_base_py" ]]; then
    PY="$_base_py"
fi

mkdir -p "$OUTPUT_DIR"

echo "[query] python  : $PY"
echo "[query] inputs  : $INPUTS"
echo "[query] out_dir : $OUTPUT_DIR"
echo "[query] llm     : $USE_LLM_POLISH ($POLISH_MODEL_NAME)"

REPO_ROOT="$(dirname "$PROJECT_ROOT")"

IFS=',' read -r -a INPUT_LIST <<< "$INPUTS"
for INPUT in "${INPUT_LIST[@]}"; do
    INPUT="$(echo "$INPUT" | xargs)"   # trim spaces
    [[ -z "$INPUT" ]] && continue
    if [[ ! -f "$INPUT" ]]; then
        echo "[query] skip missing input: $INPUT" >&2
        continue
    fi
    STEM="$(basename "$INPUT")"
    STEM="${STEM%.jsonl}"
    OUTPUT="$OUTPUT_DIR/${STEM}.queries.jsonl"
    echo
    echo "[query] === processing $INPUT -> $OUTPUT ==="

    CMD=(
        "$PY" "$PROJECT_ROOT/query_generator.py"
        --input_path "$INPUT"
        --output_path "$OUTPUT"
        --min_interval_len "$MIN_INTERVAL_LEN"
        --max_intervals_per_object "$MAX_INTERVALS_PER_OBJECT"
        --max_target_arity "$MAX_TARGET_ARITY"
        --max_multi_intervals_per_group "$MAX_MULTI_INTERVALS_PER_GROUP"
        --max_multi_candidates_total "$MAX_MULTI_CANDIDATES_TOTAL"
        --strict_time_uniqueness_multi_target "$STRICT_TIME_UNIQUENESS_MULTI"
        --max_chain_len "$MAX_CHAIN_LEN"
        --max_queries_per_candidate "$MAX_QUERIES_PER_CANDIDATE"
        --time_limit_sec "$TIME_LIMIT_SEC"
        --seed "$SEED"
        --eps "$EPS"
        --lambda_weight "$LAMBDA_WEIGHT"
        --expand_non_temporal_query_tracks "$EXPAND_NON_TEMPORAL_QUERY_TRACKS"
        --use_llm_polish "$USE_LLM_POLISH"
        --polish_model_name "$POLISH_MODEL_NAME"
        --max_concurrent_per_key "$MAX_CONCURRENT_PER_KEY"
        --max_retries "$MAX_RETRIES"
    )

    if [[ -n "${API_KEYS:-}" ]]; then
        CMD+=(--api_keys "$API_KEYS")
    fi
    if [[ -n "$MAX_QUERIES_PER_VIDEO" ]]; then
        CMD+=(--max_queries_per_video "$MAX_QUERIES_PER_VIDEO")
    fi
    if [[ -n "$MAX_QUERIES_PER_BUCKET" ]]; then
        CMD+=(--max_queries_per_difficulty_bucket "$MAX_QUERIES_PER_BUCKET")
    fi

    (
        cd "$PROJECT_ROOT"
        PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT" "${CMD[@]}"
    )
done

echo
echo "[query] all inputs done"
