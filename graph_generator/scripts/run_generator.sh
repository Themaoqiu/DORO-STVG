#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=5

set -a
source /home/wangxingjian/DORO-STVG/graph_generator/.env
set +a

SAM3_MODEL="/home/wangxingjian/model/sam3/sam3.pt"
SAM2_MODEL_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT="/home/wangxingjian/DORO-STVG/graph_generator/GroundedSAM2/checkpoints/sam2.1_hiera_large.pt"

# source /home/wangxingjian/DORO-STVG/graph_generator/.venv/main/bin/activate
python -m main \
    --video /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is_2fps.mp4 \
    --output scene_graphs.jsonl \
    --yolo_model /home/wangxingjian/model/yolo26x/yolo26x.pt \
    --tracker_backend "groundedsam2" \
    --skip_filter=True \
    --scene_threshold 3.0 \
    --min_scene_duration 1.0 \
    --conf 0.5 \
    --iou 0.5 \
    --sam3_model="${SAM3_MODEL}" \
    --sam2_model_cfg "${SAM2_MODEL_CFG}" \
    --sam2_checkpoint "${SAM2_CHECKPOINT}" \
    --sam3_redetection_interval=15 \
    --sam3_iou_threshold=0.4 \

# source /home/wangxingjian/DORO-STVG/graph_generator/.venv/mmaction/bin/activate
# python -m modules.action_detector \
#   --config /home/wangxingjian/DORO-STVG/graph_generator/mmaction2/configs/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py \
#   --checkpoint /home/wangxingjian/model/vit-large-p16_videomae-k400-pre.pth \
#   --label-map /home/wangxingjian/DORO-STVG/graph_generator/mmaction2/tools/data/ava/label_map.txt \
#   --jsonl /home/wangxingjian/DORO-STVG/graph_generator/scene_graphs.jsonl \
#   --video /home/wangxingjian/data/vidstg/2451862173_2fps.mp4 \

# python -m modules.attribute_generator \
#   --jsonl scene_graphs.jsonl \
#   --video /home/wangxingjian/data/vidstg/2451862173_2fps.mp4 \
#   --model_name gemini-3-flash-preview

# python -m modules.relation_generator \
#   --jsonl scene_graphs.jsonl \
#   --video /home/wangxingjian/data/vidstg/2451862173_2fps.mp4 \
#   --model_name gemini-3-flash-preview \
#   --verbose True

# python -m modules.sam3_text_only_test \
#   --video /home/wangxingjian/data/vidstg/2451862173_2fps.mp4 \
#   --output output/scene_graphs_sam3_text_only.jsonl \
#   --sam3_model="${SAM3_MODEL}" \
#   --text_prompt cat
