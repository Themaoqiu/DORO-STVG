#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"

UV_BIN="${UV_BIN:-uv}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
LLAVA_ST_SOURCE_DIR="${LLAVA_ST_SOURCE_DIR:-/mnt/sdc/xingjianwang/yibowang/LLaVA-ST}"
LLAVAST_PROJECT="$REPO_ROOT/envs/eval/llavast"
LLAVAST_PYTHON="$LLAVAST_PROJECT/.venv/bin/python"

echo "=========================================="
echo "LLaVA-ST environment setup"
echo "=========================================="
echo "Repo root:           $REPO_ROOT"
echo "uv project:          $LLAVAST_PROJECT"
echo "env python:          $LLAVAST_PYTHON"
echo "LLaVA-ST source dir: $LLAVA_ST_SOURCE_DIR"
echo "=========================================="

if [[ ! -d "$LLAVA_ST_SOURCE_DIR" ]]; then
  echo "[ERROR] LLaVA-ST source dir does not exist: $LLAVA_ST_SOURCE_DIR" >&2
  exit 1
fi

if [[ ! -f "$LLAVAST_PROJECT/pyproject.toml" ]]; then
  echo "[ERROR] Missing pyproject.toml: $LLAVAST_PROJECT/pyproject.toml" >&2
  exit 1
fi

UV_LINK_MODE="$UV_LINK_MODE" "$UV_BIN" lock --project "$LLAVAST_PROJECT"
UV_LINK_MODE="$UV_LINK_MODE" "$UV_BIN" sync --project "$LLAVAST_PROJECT" --locked
UV_LINK_MODE="$UV_LINK_MODE" "$UV_BIN" pip install --python "$LLAVAST_PYTHON" -e "$LLAVA_ST_SOURCE_DIR"

"$LLAVAST_PYTHON" - <<'PY'
import llava
from llava.model.builder import load_lora_model

print("llava:", llava.__file__)
print("load_lora_model:", load_lora_model)
PY

echo "=========================================="
echo "LLaVA-ST environment is ready."
echo "=========================================="
