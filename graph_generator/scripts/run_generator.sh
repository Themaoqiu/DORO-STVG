#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
USE_SAM3=True
SAM3_MODEL="/home/wangxingjian/model/sam3/sam3.pt"
# python -m main \
#     --video /home/wangxingjian/data/vidstg/2425461420.mp4 \
#     --output scene_graphs.jsonl \
#     --yolo_model /home/wangxingjian/DORO-STVG/graph_generator/models/yolo26x/yolo26x.pt \
#     --skip_filter=True \
#     --scene_threshold 3.0 \
#     --min_scene_duration 1.0 \
#     --conf 0.5 \
#     --iou 0.5 \
#     --use_sam3=${USE_SAM3} \
#     --sam3_model="${SAM3_MODEL}" \
#     --sam3_redetection_interval=15 \
#     --sam3_iou_threshold=0.4 \
#     --sam3_match_output_dir output/match_debug \

python -m modules.action_detector \
  --config /home/wangxingjian/DORO-STVG/graph_generator/mmaction2/configs/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py \
  --checkpoint /home/wangxingjian/model/vit-large-p16_videomae-k400-pre.pth \
  --label-map /home/wangxingjian/DORO-STVG/graph_generator/mmaction2/tools/data/ava/label_map.txt \
  --jsonl /home/wangxingjian/DORO-STVG/graph_generator/scene_graphs.jsonl \
  --video /home/wangxingjian/data/vidstg/2425461420.mp4 \
  --output /home/wangxingjian/DORO-STVG/graph_generator/output/scene_graphs_with_actions.jsonl
