#!/bin/bash
set -euo pipefail

python3 -m utils.video_reencode.py \
  --input /home/wangxingjian/data/vidstg/video \
  --output /home/wangxingjian/data/vidstg/video_2fps \
  --fps 2 \
  --codec libx264
