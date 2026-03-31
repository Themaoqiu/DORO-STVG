#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=7
export HF_ENDPOINT=https://hf-mirror.com

SAM2_MODEL_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT="/home/wangxingjian/DORO-STVG/graph_generator/dependence/GroundedSAM2/checkpoints/sam2.1_hiera_large.pt"

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
#     --groundedsam2_mask_output_dir /home/wangxingjian/DORO-STVG/graph_generator/output/sam2_masks \
#     --sam3_redetection_interval 15 \
#     --filter_min_frames 5

# python -m modules.attribute_generator \
#   --jsonl /home/wangxingjian/DORO-STVG/graph_generator/scene_graphs.jsonl \
#   --video /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is_2fps.mp4 \
#   --model_name gemini-3-flash-preview \
#   --masks_json /home/wangxingjian/DORO-STVG/graph_generator/output/sam2_masks/50_TM5MPJIq1Is_2fps_sam2_masks_indexed.json \
#   --model_path /home/wangxingjian/model/DAM-3B-Video

# source /home/wangxingjian/DORO-STVG/graph_generator/.venv/mmaction/bin/activate
# export PYTHONPATH="/home/wangxingjian/DORO-STVG/graph_generator/dependence/mmaction2:${PYTHONPATH}"
# python -m modules.action_detector \
#   --config /home/wangxingjian/DORO-STVG/graph_generator/dependence/mmaction2/configs/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py \
#   --checkpoint /home/wangxingjian/model/vit-large-p16_videomae-k400-pre.pth \
#   --label-map /home/wangxingjian/DORO-STVG/graph_generator/dependence/mmaction2/tools/data/ava/label_map.txt \
#   --frame_interval 1 \
#   --jsonl /home/wangxingjian/DORO-STVG/graph_generator/scene_graphs.jsonl \
#   --video /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is_2fps.mp4 \

# python -m modules.relation_generator \
#   --jsonl /home/wangxingjian/DORO-STVG/graph_generator/scene_graphs.jsonl \
#   --video /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is_2fps.mp4 \
#   --model_name gemini-3-flash-preview \
#   --crop_output_dir /home/wangxingjian/DORO-STVG/graph_generator/output/relation_crops \
#   --min_shared_frames 3 \
#   --save_intermediate_frames=True \
#   --verbose=False


# python -m modules.reference_edge_generator \
#   --jsonl scene_graphs.jsonl \
#   --video /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is_2fps.mp4 \
#   --model_name gemini-3-flash-preview \
#   --max_pairs_per_object 3 \

# python3 -m scripts.test_reference_id_matching \
#   --jsonl /home/wangxingjian/DORO-STVG/graph_generator/scene_graphs.jsonl \
#   --video /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is_2fps.mp4 \
#   --model_name gemini-3-flash-preview \
#   --output_dir /home/wangxingjian/DORO-STVG/graph_generator/output/reference_id_match_test \
#   --shot_a 0 \
#   --shot_b 1 \
#   --frames_per_shot 3

# python -m modules.query_generator \
#   --input_path=scene_graphs.jsonl \
#   --output_path=output/query_minimal.jsonl \
#   --d_star=0.6 \
#   --model_name=gpt-5.4-nano-2026-03-17




# full command (graph generation only)
cd /home/wangxingjian/DORO-STVG/graph_generator && \
python -m main \
  --full_pipeline True \
  --video /home/wangxingjian/data/hc-stvg2/v2_video/50_TM5MPJIq1Is_2fps.mp4 \
  --max_videos 20 \
  --output scene_graphs.jsonl \
  --yolo_model /home/wangxingjian/model/yolo26x/yolo26x.pt \
  --tracker_backend groundedsam2 \
  --skip_filter True \
  --scene_threshold 3.0 \
  --min_scene_duration 2.0 \
  --conf 0.5 \
  --iou 0.5 \
  --sam2_model_cfg ${SAM2_MODEL_CFG} \
  --sam2_checkpoint ${SAM2_CHECKPOINT} \
  --groundedsam2_mask_output_dir /home/wangxingjian/DORO-STVG/graph_generator/output/sam2_masks \
  --sam3_redetection_interval 15 \
  --filter_min_frames 5 \
  --attribute_model_name gemini-3-flash-preview \
  --attribute_model_path /home/wangxingjian/model/DAM-3B-Video \
  --action_config /home/wangxingjian/DORO-STVG/graph_generator/dependence/mmaction2/configs/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py \
  --action_checkpoint /home/wangxingjian/model/vit-large-p16_videomae-k400-pre.pth \
  --action_label_map /home/wangxingjian/DORO-STVG/graph_generator/dependence/mmaction2/tools/data/ava/label_map.txt \
  --action_python /home/wangxingjian/DORO-STVG/venv/graph_generator/action_detector/.venv/bin/python \
  --relation_model_name gemini-3-flash-preview \
  --relation_crop_output_dir /home/wangxingjian/DORO-STVG/graph_generator/output/relation_crops \
  --relation_min_shared_frames 3 \
  --relation_save_intermediate_frames False \
  --with_reference False

# cd /Users/themaoqiu/CodeRepo/DORO-STVG
# python -m modules.query_generator_cpsat \
#   --input_path /Users/themaoqiu/CodeRepo/DORO-STVG/graph_generator/scene_graphs.jsonl \
#   --output_path /Users/themaoqiu/CodeRepo/DORO-STVG/graph_generator/output/query.jsonl \
#   --queries_per_graph 12 \
#   --time_limit_sec 2.0 \
#   --seed 7