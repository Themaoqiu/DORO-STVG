# Shot Scene Graph Generator

This module generates shot-level scene graphs and STVG queries from a video.

## 1. Pipeline Overview

The full pipeline (as organized in `scripts/run_generator.sh`) is:

1. Scene split + YOLO keyframe detection + tracker (SAM3 / Grounded-SAM2)
2. Object attribute extraction
3. Human action detection (optional)
4. Inter-object relation generation
5. Cross-shot same-entity reference edge generation
6. STVG query generation

## 2. Environment Setup

### 2.1 Prerequisites

- Linux + NVIDIA GPU
- CUDA available in PyTorch runtime
- Python 3.11
- `uv` installed

Recommended (for data/tooling):

- `ffmpeg`
- `wget`
- `huggingface-cli` (`hf` command)

### 2.2 Main environment (scene/object/relation/query modules)

```bash
cd /home/wangxingjian/DORO-STVG/graph_generator

uv venv .venv/main --python 3.11
source .venv/main/bin/activate
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
cd /home/wangxingjian/DORO-STVG/graph_generator

uv venv .venv/mmaction --python 3.11
source .venv/mmaction/bin/activate
uv pip install -r requirements_action.txt
```

Action module depends on local MMAction2 code under:

- `dependence/mmaction2`

`action_detector.py` will automatically append this path to `PYTHONPATH` at runtime.

### 2.4 Weights and checkpoints

1) YOLO weight (`yolo26x.pt`)

```bash
hf download Ultralytics/YOLO26 yolo26x.pt --local-dir /home/wangxingjian/model/yolo26x
```

2) SAM2 checkpoint (for Grounded-SAM2 backend)

```bash
cd /home/wangxingjian/DORO-STVG/graph_generator/dependence/GroundedSAM2/checkpoints
bash download_ckpts.sh
```

3) VideoMAE action checkpoint

```bash
wget https://download.openmmlab.com/mmaction/v1.0/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb_20230314-bf93c9ea.pth \
  -O /home/wangxingjian/model/vit-large-p16_videomae-k400-pre.pth
```

4) DAM model for attribute extraction (local path expected by script example)

- Example path: `/home/wangxingjian/model/DAM-3B-Video`

### 2.5 API environment variables

Create `/DORO-STVG/graph_generator/.env`:

```bash
API_KEYS=your_key_1,your_key_2
MM_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QUERY_MODEL_NAME=gpt-5.4-nano-2026-03-17
```

Notes:

- `API_KEYS` is required by attribute/relation/reference/query modules.
- Multiple keys can be comma-separated for concurrent requests.
- If `MM_API_BASE_URL` is not set, code defaults to DashScope compatible endpoint.

## 3. Running

### 3.1 One-click script

```bash
cd /home/wangxingjian/DORO-STVG/graph_generator
bash scripts/run_generator.sh
```

Important:

- In `run_generator.sh`, most stages are currently commented out; uncomment the stages you want.
- The script exports:
  - `CUDA_VISIBLE_DEVICES=5`
  - `HF_ENDPOINT=https://hf-mirror.com`
- Final command in script currently calls `modules.query_generator_minimal`, but this module file does not exist in current codebase.
  - Use `modules.query_generator` instead.

### 3.2 Recommended stage-by-stage commands

1) Build scene graph

```bash
python -m main \
  --video /path/to/video.mp4 \
  --output scene_graphs.jsonl \
  --yolo_model /home/wangxingjian/model/yolo26x/yolo26x.pt \
  --tracker_backend groundedsam2 \
  --scene_threshold 3.0 \
  --min_scene_duration 1.0 \
  --conf 0.5 \
  --iou 0.5 \
  --sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2_checkpoint /home/wangxingjian/DORO-STVG/graph_generator/dependence/GroundedSAM2/checkpoints/sam2.1_hiera_large.pt \
  --groundedsam2_mask_output_dir output/sam2_masks \
  --sam3_redetection_interval 15 \
  --skip_filter True \
  --filter_min_frames 5
```

2) Attributes

```bash
python -m modules.attribute_generator \
  --jsonl scene_graphs.jsonl \
  --video /path/to/video.mp4 \
  --model_name gemini-3-flash-preview \
  --masks_json output/sam2_masks/<video_stem>_sam2_masks_indexed.json \
  --model_path /home/wangxingjian/model/DAM-3B-Video
```

3) Actions (optional, use mmaction env)

```bash
source .venv/mmaction/bin/activate
python -m modules.action_detector \
  --config dependence/mmaction2/configs/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py \
  --checkpoint /home/wangxingjian/model/vit-large-p16_videomae-k400-pre.pth \
  --label_map dependence/mmaction2/tools/data/ava/label_map.txt \
  --frame_interval 1 \
  --jsonl scene_graphs.jsonl \
  --video /path/to/video.mp4
```

4) Relations

```bash
source .venv/main/bin/activate
python -m modules.relation_generator \
  --jsonl scene_graphs.jsonl \
  --video /path/to/video.mp4 \
  --model_name gemini-3-flash-preview \
  --crop_output_dir output/relation_crops \
  --min_shared_frames 3 \
  --save_intermediate_frames True
```

5) Cross-shot reference edges

```bash
python -m modules.reference_edge_generator \
  --jsonl scene_graphs.jsonl \
  --video /path/to/video.mp4 \
  --model_name gemini-3-flash-preview \
  --shot_a 0 \
  --shot_b 1 \
  --frames_per_shot 3
```

6) Query generation

```bash
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
