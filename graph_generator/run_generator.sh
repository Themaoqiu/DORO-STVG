#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

python -m main \
    --video /home/wangxingjian/DORO-STVG/anno_videos/50_TM5MPJIq1Is_annotated_100frames.mp4 \
    --output scene_graphs.jsonl \
    --yolo_model /home/wangxingjian/DORO-STVG/graph_generator/models/yolo11x/yolo11x.pt \
    --tracker_config /home/wangxingjian/DORO-STVG/graph_generator/models/yolo11x/botsort.yaml \
    --conf 0.3