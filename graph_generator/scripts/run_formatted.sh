#!/bin/bash
set -euo pipefail

python3 utils/format_train.py \
  --input=/home/wangxingjian/DORO-STVG/graph_generator/output/query.jsonl \
  --output=/home/wangxingjian/DORO-STVG/graph_generator/output/query_train.jsonl \
  --fps=2.0