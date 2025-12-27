#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -m graph_generator.main \
    --video anno_videos/50_TM5MPJIq1Is_annotated_100frames.mp4 \
    --output output/scene_graphs.jsonl \
    --yolo_model yolo11n.pt \
    --tracker_config botsort.yaml \
    --conf 0.3