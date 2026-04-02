#!/bin/bash
set -euo pipefail

python3 utils/format_query_jsonl.py \
  --input=/home/wangxingjian/DORO-STVG/graph_generator/output/query.jsonl \
  --output=/home/wangxingjian/DORO-STVG/graph_generator/output/query_formatted.jsonl \
  --fps=2.0