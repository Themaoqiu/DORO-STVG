# DORO-STVG

## 1. Evaluation Framework

`eval/main.py` is the unified entry point. The current code supports:

- Models: `qwen2.5vl` / `qwen3vl`
- Datasets: `hcstvg`, `vidstg`, `doro-stvg`


The default script is [`eval/run_eval.sh`](/home/wangxingjian/DORO-STVG/eval/run_eval.sh). You can edit it directly to change model paths, annotation paths, video paths, and output paths.

Typical outputs:

- `results.json`: per-sample predictions, parsed outputs, GT, and metrics
- `status.json`: overall summary and averaged metrics

## 2. Data Engine

`graph_generator/` is to generate structured data from raw videos. Based on the current code, the main pipeline includes:

1. Scene splitting
2. Object detection and tracking
3. Attribute generation
4. Action detection
5. Relation generation
6. Cross-shot reference edge generation (optional)
7. STVG query generation from scene graphs
8. Formatting query outputs into training-friendly JSONL

Relevant entry points:

- [`graph_generator/main.py`](/home/wangxingjian/DORO-STVG/graph_generator/main.py): main scene graph generation entry
- [`graph_generator/modules/query_generator_cpsat.py`](/home/wangxingjian/DORO-STVG/graph_generator/modules/query_generator_cpsat.py): generate queries from scene graphs
- [`graph_generator/utils/format_train.py`](/home/wangxingjian/DORO-STVG/graph_generator/utils/format_train.py): convert query outputs into training format
- [`graph_generator/scripts/run_generator.sh`](/home/wangxingjian/DORO-STVG/graph_generator/scripts/run_generator.sh): current command collection used in practice

## 3. Environment Setup

This repository does not currently use a single root-level setup script. The actual setup should follow the module-specific `pyproject.toml` files under `envs/`.

### 3.1 Requirements

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 3.2 Virtual Environment

```bash
cd /home/wangxingjian/DORO-STVG/envs/eval
uv sync
```


```bash
cd /home/wangxingjian/DORO-STVG/envs/graph_generator/main
uv sync
```

This environment is used for:

- `graph_generator/main.py`
- the main pipeline modules for attributes, relations, reference edges, and query generation


```bash
cd /home/wangxingjian/DORO-STVG/envs/graph_generator/action_detector
uv sync
```

This separate environment is mainly used by the action detection module to avoid dependency conflicts with the main environment.

### 3.5 Video Reader Backend

The evaluation script currently defaults to `decord`:

```bash
export FORCE_QWENVL_VIDEO_READER=decord
```

You can switch to `torchvision` or `torchcodec` if needed.

### 3.6 Extra Configuration for `graph_generator`

`graph_generator` depends on both model checkpoints and API-related environment variables. The repository already contains [`graph_generator/.env`](/home/wangxingjian/DORO-STVG/graph_generator/.env), and the scripts load it automatically.

The most important variables are:

```bash
API_KEYS=your_key_1,your_key_2
MM_API_BASE_URL=https://your-compatible-endpoint
```

You also need to prepare:

- YOLO weights
- SAM2 / Grounded-SAM2 checkpoints
- VideoMAE action detection checkpoints
- DAM or other attribute-description models

For those details, refer to [`graph_generator/README.md`](/home/wangxingjian/DORO-STVG/graph_generator/README.md).

## 4. Usage

### 4.1 Run Evaluation

```bash
cd /home/wangxingjian/DORO-STVG/eval
bash run_eval.sh
```

If you prefer not to use the shell script, you can call the entry point directly:

```bash
cd /home/wangxingjian/DORO-STVG/eval
python main.py run \
  --model_name=qwen3vl \
  --model_path=/path/to/model \
  --data_name=hcstvg2 \
  --annotation_path=/path/to/test.json \
  --video_dir=/path/to/videos \
  --output_dir=./res
```

### 4.2 Run the Data Engine

The current `run_generator.sh` contains the full pipeline command examples, and the bottom part of the script keeps the active query-generation example.

A typical workflow is:

1. Generate `scene_graphs.jsonl`
2. Generate `query.jsonl`
3. Convert it into `query_train.jsonl`

## 5. Output Data Formats

### 5.3 Training Data Format

This is the training-friendly formatted output generated from `query.jsonl` by `utils/format_train.py`. The main fields include:

- `videopath`
- `queryid`
- `query`
- `Difficulty`
- `Width` / `Height`
- `box`

`box` is a trajectory string in the following format:

```text
target description: <frame_idx, time_sec, x1, y1, x2, y2; ... />
```

Here the coordinates are already normalized to `[0, 1]` using the video width and height, which makes this format easier to use for training and annotation consumption.
