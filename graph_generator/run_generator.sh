#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

python -m main \
    --video /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is.mkv \
    --output scene_graphs.jsonl \
    --yolo_model /home/wangxingjian/DORO-STVG/graph_generator/models/yolo26x/yolo26x.pt \
    --tracker_config /home/wangxingjian/DORO-STVG/graph_generator/models/yolo11x/botsort.yaml \
    --conf 0.5