# Shot Scene Graph Generator

This module generates shot-level scene graphs and STVG queries from a video.

[中文版](README.zh.md)

## 1. Pipeline Overview

The full pipeline (as organized in `scripts/run_generator.sh`) is:

1. Scene split + YOLO keyframe detection + tracker (SAM3 / Grounded-SAM2(main))
2. Object attribute extraction
3. Human action detection
4. Inter-object relation generation
5. Cross-shot same-entity reference edge generation
6. STVG query generation

## 2. Environment Setup

### 2.1 Prerequisites

- Linux + NVIDIA GPU
- CUDA available in PyTorch runtime
- Python 3.11
- `uv` installed

### 2.2 Main environment (scene/object/relation/query modules)

```bash
cd /home/wangxingjian/DORO-STVG/venv/graph_generator/main

uv sync
```

This environment is used for:

- `main.py`
- `modules.attribute_generator`
- `modules.relation_generator`
- `modules.reference_edge_generator`
- `modules.query_generator`

### 2.3 Action-detector environment (MMAction2/VideoMAE)

```bash
cd /home/wangxingjian/DORO-STVG/graph_generator/venv/graph_generator/action_detector

uv sync
```

### 2.4 Weights and checkpoints

1) YOLO weight (`yolo26x.pt`)

```bash
hf download Ultralytics/YOLO26 yolo26x.pt --local-dir /model/yolo26x
```

2) SAM2 checkpoint (for Grounded-SAM2 backend)

```bash
#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Use either wget or curl to download the checkpoints
if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

# Define the URLs for SAM 2 checkpoints
# SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824"
# sam2_hiera_t_url="${SAM2_BASE_URL}/sam2_hiera_tiny.pt"
# sam2_hiera_s_url="${SAM2_BASE_URL}/sam2_hiera_small.pt"
# sam2_hiera_b_plus_url="${SAM2_BASE_URL}/sam2_hiera_base_plus.pt"
# sam2_hiera_l_url="${SAM2_BASE_URL}/sam2_hiera_large.pt"

# Download each of the four checkpoints using wget
# echo "Downloading sam2_hiera_tiny.pt checkpoint..."
# $CMD $sam2_hiera_t_url || { echo "Failed to download checkpoint from $sam2_hiera_t_url"; exit 1; }

# echo "Downloading sam2_hiera_small.pt checkpoint..."
# $CMD $sam2_hiera_s_url || { echo "Failed to download checkpoint from $sam2_hiera_s_url"; exit 1; }

# echo "Downloading sam2_hiera_base_plus.pt checkpoint..."
# $CMD $sam2_hiera_b_plus_url || { echo "Failed to download checkpoint from $sam2_hiera_b_plus_url"; exit 1; }

# echo "Downloading sam2_hiera_large.pt checkpoint..."
# $CMD $sam2_hiera_l_url || { echo "Failed to download checkpoint from $sam2_hiera_l_url"; exit 1; }

# Define the URLs for SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

# SAM 2.1 checkpoints


echo "Downloading sam2.1_hiera_large.pt checkpoint..."
$CMD $sam2p1_hiera_l_url || { echo "Failed to download checkpoint from $sam2p1_hiera_l_url"; exit 1; }

echo "All checkpoints are downloaded successfully."
```

3) VideoMAE action checkpoint

```bash
wget https://download.openmmlab.com/mmaction/v1.0/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb_20230314-bf93c9ea.pth \
  -O /model/vit-large-p16_videomae-k400-pre.pth
```

4) Describe anything model

```bash
hf download nvidia/DAM-3B-Video --local-dir /model/DAM-3B-Video
```

### 2.5 API environment variables

Create `/DORO-STVG/graph_generator/.env`:

```bash
API_KEYS=your_key_1,your_key_2
MM_API_BASE_URL=https://xxx
```

Notes:

- `API_KEYS` is required by attribute/relation/reference/query modules.
- Multiple keys can be comma-separated for concurrent requests.
- If `MM_API_BASE_URL` is not set, code defaults to DashScope compatible endpoint.

## 3. Running

### 3.1 Full command

Run this one complete command to generate scene graphs (including attribute/action/relation stages in the main pipeline):

You can run this command from [run_generator.sh](scripts/run_generator.sh)

```bash
cd /DORO-STVG/graph_generator && \
python -m main \
  --full_pipeline True \
  --video_dir /data/vidstg/video \
  --max_videos 20 \
  --output scene_graphs.jsonl \
  --yolo_model /model/yolo26x/yolo26x.pt \
  --tracker_backend groundedsam2 \
  --skip_filter True \
  --scene_threshold 3.0 \
  --min_scene_duration 2.0 \
  --conf 0.5 \
  --iou 0.5 \
  --sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2_checkpoint /DORO-STVG/graph_generator/dependence/GroundedSAM2/checkpoints/sam2.1_hiera_large.pt \
  --groundedsam2_mask_output_dir /DORO-STVG/graph_generator/output/sam2_masks \
  --sam3_redetection_interval 15 \
  --filter_min_frames 5 \
  --attribute_model_name gemini-3-flash-preview \
  --attribute_model_path /model/DAM-3B-Video \
  --action_config /DORO-STVG/graph_generator/dependence/mmaction2/configs/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py \
  --action_checkpoint /model/vit-large-p16_videomae-k400-pre.pth \
  --action_label_map /DORO-STVG/graph_generator/dependence/mmaction2/tools/data/ava/label_map.txt \
  --action_python /DORO-STVG/graph_generator/.venv/mmaction/bin/python \
  --relation_model_name gemini-3-flash-preview \
  --relation_crop_output_dir /DORO-STVG/graph_generator/output/relation_crops \
  --relation_min_shared_frames 3 \
  --relation_save_intermediate_frames False \
  --with_reference False
```
### 3.2 Parameter explanation

| Parameter | Description |
| --- | --- |
| `--full_pipeline` | Enables the integrated multi-stage graph pipeline in `main.py`. |
| `--video_dir` | Input directory containing videos (`.mp4`/`.avi`). |
| `--max_videos` | Maximum number of videos to process from `video_dir` (0 means all). |
| `--output` | Output JSONL path for generated scene graphs. |
| `--yolo_model` | YOLO checkpoint path for keyframe object detection. |
| `--tracker_backend` | Tracking backend (`groundedsam2`, `sam3`, or `yolo`). |
| `--scene_threshold` | Scene split threshold. Higher values usually produce fewer cuts. |
| `--min_scene_duration` | Minimum shot duration (seconds) to keep. |
| `--conf` | YOLO confidence threshold. |
| `--iou` | YOLO NMS IoU threshold. |
| `--sam2_model_cfg` | Grounded-SAM2 model config path. |
| `--sam2_checkpoint` | Grounded-SAM2 checkpoint path. |
| `--groundedsam2_mask_output_dir` | Directory to save SAM2 indexed masks. |
| `--sam3_redetection_interval` | Keyframe interval / redetection interval used by tracker. |
| `--skip_filter` | If `True`, skip object filtering and only normalize graph format. |
| `--filter_min_frames` | Minimum tracked-frame count when filtering is enabled. |
| `--attribute_model_name` | LLM model name used by attribute generation stage. |
| `--attribute_model_path` | Local DAM model path used by attribute generation. |
| `--action_config` | MMAction2 config for VideoMAE action detection. |
| `--action_checkpoint` | VideoMAE action checkpoint path. |
| `--action_label_map` | Action label map file path. |
| `--action_python` | Python executable for the action-detector environment. |
| `--relation_model_name` | LLM model name used by relation generation stage. |
| `--relation_crop_output_dir` | Directory for relation crop images/intermediate files. |
| `--relation_min_shared_frames` | Minimum shared frames for relation candidate pairs. |
| `--relation_save_intermediate_frames` | Whether to save relation intermediate frames. |
| `--with_reference` | Whether to run cross-shot same-entity reference edge generation. |

### 3.3 Query generation (run separately)

After `scene_graphs.jsonl` is generated, run query generation with a separate command:

```bash
cd /DORO-STVG/graph_generator && \
python -m modules.query_generator \
  --input_path scene_graphs.jsonl \
  --output_path output/query_minimal.jsonl \
  --d_star 0.9 \
  --model_name gpt-5.4-nano-2026-03-17
```

## 4. Visualization

```bash
bash scripts/run_visualizer.sh
```

## 5. Main Outputs

- `scene_graphs.jsonl`: generated scene graph
- `output/sam2_masks/*.json`: indexed masks (if using Grounded-SAM2)
- `output/relation_crops/`: pair crops for relation module (optional)
- `output/query_minimal.jsonl`: generated STVG queries
