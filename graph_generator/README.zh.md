# Shot Scene Graph Generator（中文版）

该模块用于从视频生成镜头级场景图（scene graph）与 STVG 查询。

English version: [README.md](README.md)

## 1. 流水线概览

完整流程（对应 `scripts/run_generator.sh`）如下：

1. 镜头切分 + YOLO 关键帧检测 + 跟踪（SAM3 / Grounded-SAM2）
2. 目标属性抽取
3. 人体动作检测
4. 目标间关系生成
5. 跨镜头同一实体引用边生成
6. STVG 查询生成

## 2. 环境准备

### 2.1 前置条件

- Linux + NVIDIA GPU
- PyTorch 运行时可用 CUDA
- Python 3.11
- 已安装 `uv`

### 2.2 主环境（scene/object/relation/query 模块）

```bash
cd /envs/graph_generator/main

uv sync
```

该环境用于：

- `main.py`
- `modules.attribute_generator`
- `modules.relation_generator`
- `modules.reference_edge_generator`
- `modules.query_generator`

### 2.3 动作检测环境（MMAction2/VideoMAE）

```bash
cd /envs/graph_generator/action_detector

uv sync
```

### 2.4 权重与检查点

1) YOLO 权重（`yolo26x.pt`）

```bash
hf download Ultralytics/YOLO26 yolo26x.pt --local-dir /model/yolo26x
```

2) SAM2 权重（Grounded-SAM2 后端）

在 `dependence/GroundedSAM2/checkpoints` 目录执行下载脚本即可（仓库内脚本为 `download.sh`）。

3) VideoMAE 动作检测权重

```bash
wget https://download.openmmlab.com/mmaction/v1.0/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb_20230314-bf93c9ea.pth \
  -O /model/vit-large-p16_videomae-k400-pre.pth
```

4) Describe Anything Model

```bash
hf download nvidia/DAM-3B-Video --local-dir /model/DAM-3B-Video
```

### 2.5 API 环境变量

在 `/DORO-STVG/graph_generator/.env` 中配置：

```bash
API_KEYS=your_key_1,your_key_2
MM_API_BASE_URL=https://xxx
```

说明：

- `API_KEYS` 是 attribute/relation/reference/query 模块必需项。
- 多个 key 可用逗号分隔，以支持并发请求。
- 未设置 `MM_API_BASE_URL` 时，代码默认走 DashScope 兼容端点。

## 3. 运行方式

### 3.1 一条完整命令（只做图生成）

下面这条命令会完整执行图生成相关阶段（属性/动作/关系等）：

你也可以直接参考 [scripts/run_generator.sh](scripts/run_generator.sh)。

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

### 3.2 参数说明

| 参数 | 说明 |
| --- | --- |
| `--full_pipeline` | 启用 `main.py` 中的一体化多阶段流水线。 |
| `--video_dir` | 输入视频目录（支持 `.mp4`/`.avi`）。 |
| `--max_videos` | 从 `video_dir` 最多处理多少个视频（`0` 表示全部）。 |
| `--output` | 生成的场景图 JSONL 输出路径。 |
| `--yolo_model` | YOLO 关键帧检测模型路径。 |
| `--tracker_backend` | 跟踪后端（`groundedsam2`、`sam3` 或 `yolo`）。 |
| `--scene_threshold` | 镜头切分阈值，越高通常切分越少。 |
| `--min_scene_duration` | 最短镜头时长（秒）。 |
| `--conf` | YOLO 置信度阈值。 |
| `--iou` | YOLO NMS 的 IoU 阈值。 |
| `--sam2_model_cfg` | Grounded-SAM2 配置文件路径。 |
| `--sam2_checkpoint` | Grounded-SAM2 权重路径。 |
| `--groundedsam2_mask_output_dir` | SAM2 索引掩码输出目录。 |
| `--sam3_redetection_interval` | 跟踪器重检测间隔（也影响关键帧间隔）。 |
| `--skip_filter` | `True` 时跳过对象过滤，仅做图格式规范化。 |
| `--filter_min_frames` | 开启过滤时对象最少跟踪帧数。 |
| `--attribute_model_name` | 属性生成阶段使用的模型名。 |
| `--attribute_model_path` | 属性生成阶段本地 DAM 模型路径。 |
| `--action_config` | VideoMAE/MMAction2 配置文件路径。 |
| `--action_checkpoint` | VideoMAE 动作权重路径。 |
| `--action_label_map` | 动作标签映射文件路径。 |
| `--action_python` | 动作检测环境对应的 Python 可执行文件。 |
| `--relation_model_name` | 关系生成阶段使用的模型名。 |
| `--relation_crop_output_dir` | 关系阶段中间裁剪图输出目录。 |
| `--relation_min_shared_frames` | 候选关系对所需的最小共享帧数。 |
| `--relation_save_intermediate_frames` | 是否保存关系阶段中间帧。 |
| `--with_reference` | 是否启用跨镜头同一实体引用边生成。 |

### 3.3 Query 生成（单独执行）

`scene_graphs.jsonl` 生成后，再单独执行 Query 生成：

```bash
cd /DORO-STVG/graph_generator && \
python -m modules.query_generator \
  --input_path scene_graphs.jsonl \
  --output_path output/query_minimal.jsonl \
  --d_star 0.9 \
  --model_name gpt-5.4-nano-2026-03-17
```

## 4. 可视化

```bash
bash scripts/run_visualizer.sh
```

## 5. 主要输出

- `scene_graphs.jsonl`：生成的场景图
- `output/sam2_masks/*.json`：索引掩码（Grounded-SAM2 模式）
- `output/relation_crops/`：关系模块中间裁剪图（可选）
- `output/query_minimal.jsonl`：生成的 STVG 查询
