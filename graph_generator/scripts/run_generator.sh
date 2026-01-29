#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

USE_SAM3=false
SAM3_MODEL="/home/wangxingjian/model/sam3.pt"

python -m main \
    --video_dir /home/wangxingjian/data/vidstg \
    --output scene_graphs.jsonl \
    --yolo_model /home/wangxingjian/DORO-STVG/graph_generator/models/yolo26x/yolo26x.pt \
    --tracker_config /home/wangxingjian/DORO-STVG/graph_generator/models/yolo26x/botsort.yaml \
    --scene_threshold 3.0 \
    --min_scene_duration 1.0 \
    --conf 0.3 \
    --iou 0.5 \
    --gap_threshold 5 \
    --min_track_length 10 \
    --use_sam3 ${USE_SAM3} \
    --sam3_model ${SAM3_MODEL} \
    --sam3_redetection_interval 15 \
    --filter_min_frames 30 \
    --filter_max_gap_ratio 0.5 \
    --filter_min_temporal_coverage 0.3 \
    --filter_max_flicker_segments 3 \
    --filter_min_stable_segment_length 15
