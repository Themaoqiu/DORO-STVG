#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=5

set -a
source /home/wangxingjian/DORO-STVG/graph_generator/.env
set +a

SAM2_MODEL_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT="/home/wangxingjian/DORO-STVG/graph_generator/dependence/GroundedSAM2/checkpoints/sam2.1_hiera_large.pt"

# source /home/wangxingjian/DORO-STVG/graph_generator/.venv/main/bin/activate
# python -m main \
#     --video /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is_2fps.mp4 \
#     --output scene_graphs.jsonl \
#     --yolo_model /home/wangxingjian/model/yolo26x/yolo26x.pt \
#     --tracker_backend "groundedsam2" \
#     --skip_filter=True \
#     --scene_threshold 3.0 \
#     --min_scene_duration 1.0 \
#     --conf 0.5 \
#     --iou 0.5 \
#     --sam2_model_cfg "${SAM2_MODEL_CFG}" \
#     --sam2_checkpoint "${SAM2_CHECKPOINT}" \
#     --sam3_redetection_interval 15 \
#     --filter_min_frames 5

# source /home/wangxingjian/DORO-STVG/graph_generator/.venv/mmaction/bin/activate
# export PYTHONPATH="/home/wangxingjian/DORO-STVG/graph_generator/dependence/mmaction2:${PYTHONPATH}"
# python -m modules.action_detector \
#   --config /home/wangxingjian/DORO-STVG/graph_generator/dependence/mmaction2/configs/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py \
#   --checkpoint /home/wangxingjian/model/vit-large-p16_videomae-k400-pre.pth \
#   --label-map /home/wangxingjian/DORO-STVG/graph_generator/dependence/mmaction2/tools/data/ava/label_map.txt \
#   --jsonl /home/wangxingjian/DORO-STVG/graph_generator/scene_graphs.jsonl \
#   --video /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is_2fps.mp4

# python -m modules.attribute_generator \
#   --jsonl scene_graphs.jsonl \
#   --video /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is_2fps.mp4 \
#   --model_name gemini-3-flash-preview

# python -m modules.relation_generator \
#   --jsonl scene_graphs.jsonl \
#   --video /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is_2fps.mp4 \
#   --model_name gemini-3-flash-preview \

python -m modules.reference_edge_generator \
  --jsonl scene_graphs.jsonl \
  --video /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is_2fps.mp4 \
  --model_name gemini-3-flash-preview \
  --max_pairs_per_object 3 \
  --similarity_threshold 0.35 \
  --verbose True
