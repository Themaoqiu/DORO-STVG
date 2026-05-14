#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3

PROJECT_ROOT=/home/wangxingjian/DORO-STVG/graph_generator
set -a
source $PROJECT_ROOT/.env
set +a

SAM2_MODEL_CFG=configs/sam2.1/sam2.1_hiera_l.yaml
SAM2_CHECKPOINT=sam2.1_hiera_large.pt

VIDEO_INPUT_DIR=/home/wangxingjian/data/vidstg/video
VIDEO_OUTPUT_DIR=/home/wangxingjian/data/vidstg/video_2fps
TARGET_FPS=2
NUM_SHARDS=4

OUTPUT_DIR=$PROJECT_ROOT/output
FINAL_OUTPUT=$OUTPUT_DIR/scene_graphs.jsonl
mkdir -p $OUTPUT_DIR

echo "[run_generator] re-encoding $VIDEO_INPUT_DIR -> $VIDEO_OUTPUT_DIR @ ${TARGET_FPS}fps"
python3 $PROJECT_ROOT/utils/video_reencode.py \
  --input $VIDEO_INPUT_DIR \
  --output $VIDEO_OUTPUT_DIR \
  --fps $TARGET_FPS \
  --codec libx264

cd graph_generator

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  echo "[run_generator] launching shard $i on GPU $i"
  CUDA_VISIBLE_DEVICES=$i python -m graph_generator \
    --full_pipeline True \
    --video_dir $VIDEO_OUTPUT_DIR \
    --shard_idx $i \
    --num_shards $NUM_SHARDS \
    --output output/scene_graphs.shard$i.jsonl \
    --yolo_model /home/wangxingjian/model/yolo26x/yolo26x.pt \
    --tracker_backend groundedsam2 \
    --skip_filter True \
    --scene_threshold 3.0 \
    --min_scene_duration 2.0 \
    --conf 0.5 \
    --iou 0.5 \
    --sam2_model_cfg $SAM2_MODEL_CFG \
    --sam2_checkpoint $SAM2_CHECKPOINT \
    --groundedsam2_mask_output_dir $OUTPUT_DIR/sam2_masks \
    --sam3_redetection_interval 15 \
    --filter_min_frames 5 \
    --attribute_model_name gemini-3-flash-preview \
    --attribute_model_path /model/DAM-3B-Video \
    --action_config $PROJECT_ROOT/dependence/mmaction2/configs/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py \
    --action_checkpoint /model/vit-large-p16_videomae-k400-pre.pth \
    --action_label_map $PROJECT_ROOT/dependence/mmaction2/tools/data/ava/label_map.txt \
    --action_python envs/graph_generator/action_detector/.venv/bin/python \
    --action_device cuda:0 \
    --relation_model_name gemini-3-flash-preview \
    --relation_crop_output_dir $OUTPUT_DIR/relation_crops \
    --relation_min_shared_frames 3 \
    --relation_save_intermediate_frames False \
    --with_reference True \
    > $OUTPUT_DIR/run_shard$i.log 2>&1 &
done

wait

echo "[run_generator] merging shards -> $FINAL_OUTPUT"
cat $OUTPUT_DIR/scene_graphs.shard*.jsonl > $FINAL_OUTPUT
echo "[run_generator] done"
