#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7

USE_SAM3=True
SAM3_MODEL="/home/wangxingjian/model/sam3/sam3.pt"

python -m main \
    --video /home/wangxingjian/data/vidstg/2400492172.mp4 \
    --output scene_graphs.jsonl \
    --yolo_model /home/wangxingjian/DORO-STVG/graph_generator/models/yolo26x/yolo26x.pt \
    --skip_filter=True \
    --scene_threshold 3.0 \
    --min_scene_duration 1.0 \
    --conf 0.3 \
    --iou 0.5 \
    --use_sam3=${USE_SAM3} \
    --sam3_model="${SAM3_MODEL}" \
    --sam3_redetection_interval=15 \
    --sam3_iou_threshold=0.01
    # --filter_min_frames=1 \
    # --filter_max_gap_ratio=1 \
    # --filter_min_temporal_coverage=0.3 \
    # --filter_max_flicker_segments=3 \
    # --filter_min_stable_segment_length=1
