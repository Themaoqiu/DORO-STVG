# DORO-STVG

## 1. Evaluation Framework

`eval/main.py` is the unified entry point. The current code supports:

- Models: `qwen2.5vl` / `qwen3vl` / `llava-st-qwen2` / `llava16` / `videochat-r1` / `videomolmo`(waiting for processing) / `cgstvg` / `tastvg` / `tubedetr` / `groundinggpt`
- Datasets: `hcstvg`, `vidstg`, `doro-stvg`


Evaluation scripts live under `eval/scripts/`. Each model family has its own
environment and launch script:

- Qwen-VL: `envs/eval/qwen`, `eval/scripts/run_qwen.sh`
- LLaVA-ST-Qwen2: `envs/eval/llavast`, `eval/scripts/run_llavast.sh`
- LLaVA-1.6: `envs/eval/llava16`, `eval/scripts/run_llava16.sh`
- VideoChat-R1: `envs/eval/videochat_r1`, `eval/scripts/run_videochat_r1.sh`
- VideoMolmo: `envs/eval/videomolmo`, `eval/scripts/run_videomolmo.sh`
- CG-STVG: `envs/eval/cgstvg`, `eval/scripts/run_cgstvg.sh`
- TA-STVG: `envs/eval/tastvg`, `eval/scripts/run_tastvg.sh`
- TubeDETR: `envs/eval/tubedetr`, `eval/scripts/run_tubedetr.sh`
- GroundingGPT: `envs/eval/groundinggpt`, `eval/scripts/run_groundinggpt.sh`

Edit the corresponding script directly to change model paths, annotation paths,
video paths, GPU id, and output paths.

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

The evaluation environments are split by model family to avoid dependency
conflicts:

```bash
cd /home/wangxingjian/DORO-STVG
UV_LINK_MODE=copy uv sync --project envs/eval/qwen
UV_LINK_MODE=copy uv sync --project envs/eval/llavast
UV_LINK_MODE=copy uv sync --project envs/eval/llava16
UV_LINK_MODE=copy uv sync --project envs/eval/videochat_r1
UV_LINK_MODE=copy uv sync --project envs/eval/videomolmo
UV_LINK_MODE=copy uv sync --project envs/eval/cgstvg
UV_LINK_MODE=copy uv sync --project envs/eval/tastvg
UV_LINK_MODE=copy uv sync --project envs/eval/tubedetr
UV_LINK_MODE=copy uv sync --project envs/eval/groundinggpt
```

Use `envs/eval/qwen` for Qwen-VL models and `envs/eval/llavast` for
LLaVA-ST-Qwen2. Use `envs/eval/llava16` for LLaVA-1.6 and
`envs/eval/videochat_r1` for VideoChat-R1. Use `envs/eval/videomolmo` for the
VideoMolmo wrapper.
LLaVA-ST and VideoMolmo are not vendored in this repository.
Use `envs/eval/cgstvg` and `envs/eval/tastvg` for the lightweight framework
wrappers around external CG-STVG and TA-STVG checkouts. These environments only
run the DORO-STVG evaluation framework; the official model runtimes stay in
their own external repositories.
Use `envs/eval/tubedetr` for TubeDETR and `envs/eval/groundinggpt` for the
GroundingGPT wrapper.

After syncing `envs/eval/llavast`, install an external LLaVA-ST checkout into
that environment:

```bash
cd /home/wangxingjian/DORO-STVG
UV_LINK_MODE=copy uv sync --project envs/eval/llavast
UV_LINK_MODE=copy uv run --project envs/eval/llavast uv pip install -e /path/to/LLaVA-ST
```

If the external checkout is not importable as `inference`, set
`LLAVA_ST_SOURCE_DIR=/path/to/LLaVA-ST` so the wrapper can load
`inference/src/utils.py`.

For VideoMolmo, the repository only provides the evaluation wrapper. The actual
VideoMolmo runtime is external because its Molmo, SAM2, torch, CUDA,
bitsandbytes, and accelerate versions are server-specific. Configure these
paths inside `eval/scripts/run_videomolmo.sh`:

- `VIDEOMOLMO_REPO`: external VideoMolmo repository root.
- `VIDEOMOLMO_SOURCE_DIR`: directory containing `infer.py`.
- `MOLMO_SOURCE_DIR`: external Molmo source root that provides the `olmo`
  package.
- `SAM2_SOURCE_DIR`: SAM2 source path used by VideoMolmo.
- `VIDEOMOLMO_PYTHON`: Python executable that can run VideoMolmo `infer.py`.

`envs/eval/videomolmo` is the lightweight wrapper environment for this
evaluation framework. It does not try to lock the full VideoMolmo runtime.

For CG-STVG and TA-STVG, keep the official repositories outside this project:

- CG-STVG: https://github.com/HengLan/CGSTVG
- TA-STVG: https://github.com/HengLan/TA-STVG

The DORO-STVG framework reads the JSONL annotations, sends each raw query and
video path to a server-side helper, parses the returned tube, and computes all
metrics with `eval/utils/metrics.py`. The shared prompt in `eval/prompts.py` is
not modified. Set these helper paths in the scripts:

- `CGSTVG_DIR`, `CGSTVG_PYTHON`, `CGSTVG_INFER_PY`, `CGSTVG_MODEL_WEIGHT`
- `TASTVG_DIR`, `TASTVG_PYTHON`, `TASTVG_INFER_PY`, `TASTVG_MODEL_WEIGHT`

Each helper receives a JSONL manifest with `query` and `video_path`, runs the
official model inference, and writes JSONL predictions such as:

```json
{"target": {"12": [0.1, 0.2, 0.3, 0.4], "13": [0.1, 0.2, 0.3, 0.4]}}
```

The placeholder files `eval/utils/cgstvg_infer_helper.py` and
`eval/utils/tastvg_infer_helper.py` document this contract. On the server, point
`CGSTVG_INFER_PY` or `TASTVG_INFER_PY` to the concrete helper that already runs
the official model.


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

For Qwen-VL:

```bash
cd /home/wangxingjian/DORO-STVG/eval
source ../envs/eval/qwen/.venv/bin/activate
bash scripts/run_qwen.sh
```

For LLaVA-ST-Qwen2:

```bash
cd /home/wangxingjian/DORO-STVG/eval
source ../envs/eval/llavast/.venv/bin/activate
export LLAVA_ST_SOURCE_DIR=/path/to/LLaVA-ST
bash scripts/run_llavast.sh
```

For LLaVA-1.6:

```bash
cd /home/wangxingjian/DORO-STVG/eval
source ../envs/eval/llava16/.venv/bin/activate
bash scripts/run_llava16.sh
```

For VideoChat-R1:

```bash
cd /home/wangxingjian/DORO-STVG/eval
source ../envs/eval/videochat_r1/.venv/bin/activate
bash scripts/run_videochat_r1.sh
```

For VideoMolmo:

```bash
cd /home/wangxingjian/DORO-STVG/eval
source ../envs/eval/videomolmo/.venv/bin/activate
bash scripts/run_videomolmo.sh
```

For CG-STVG:

```bash
cd /home/wangxingjian/DORO-STVG/eval
source ../envs/eval/cgstvg/.venv/bin/activate
bash scripts/run_cgstvg.sh
```

For TA-STVG:

```bash
cd /home/wangxingjian/DORO-STVG/eval
source ../envs/eval/tastvg/.venv/bin/activate
bash scripts/run_tastvg.sh
```

For TubeDETR:

```bash
cd /home/wangxingjian/DORO-STVG/eval
source ../envs/eval/tubedetr/.venv/bin/activate
bash scripts/run_tubedetr.sh
```

For GroundingGPT:

```bash
cd /home/wangxingjian/DORO-STVG/eval
source ../envs/eval/groundinggpt/.venv/bin/activate
bash scripts/run_groundinggpt.sh
```

If you prefer not to use the shell script, you can call the entry point directly:

```bash
cd /home/wangxingjian/DORO-STVG/eval
UV_LINK_MODE=copy uv run --project ../envs/eval/qwen python main.py run \
  --model_name=qwen3vl \
  --model_path=/path/to/model \
  --data_name=hcstvg2 \
  --annotation_path=/path/to/test.json \
  --video_dir=/path/to/videos \
  --output_dir=./res
```

LLaVA-ST-Qwen2 direct command:

```bash
cd /home/wangxingjian/DORO-STVG/eval
CUDA_VISIBLE_DEVICES=3 \
LLAVA_ST_SOURCE_DIR=/path/to/LLaVA-ST \
UV_LINK_MODE=copy \
uv run --project ../envs/eval/llavast python main.py run \
  --model_name=llava-st-qwen2 \
  --model_path=/path/to/LLaVA-ST-Qwen2-7B \
  --data_name=doro-stvg \
  --annotation_path=/path/to/query_eval.jsonl \
  --video_dir=/path/to/videos \
  --output_dir=./res_llavast \
  --batch_size=1 \
  --max_tokens=512 \
  --temperature=0.1
```

LLaVA-ST outputs spatial tokens such as
`12:[<WIDTH-000><HEIGHT-000><WIDTH-042><HEIGHT-099>]`. The evaluation wrapper
converts these tokens into the JSON-style frame-to-box format used by the
existing metrics. Outputs without an explicit frame id are kept as raw responses
and cannot be scored as trajectories.

Smoke test used during integration:

```bash
CUDA_VISIBLE_DEVICES=3 \
LLAVA_ST_SOURCE_DIR=/mnt/sdc/xingjianwang/yibowang/LLaVA-ST \
uv run --project ../envs/eval/llavast python main.py run \
  --model_name=llava-st-qwen2 \
  --model_path=/mnt/sdc/xingjianwang/yibowang/models/LLaVA-ST-Qwen2-7B \
  --data_name=doro-stvg \
  --annotation_path=/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/query_train_for_eval_3uniq.jsonl \
  --video_dir=/mnt/sdc/xingjianwang/yibowang/datasets/ST-Align-Benchmark/video_test1_smoke \
  --output_dir=./res_llavast \
  --batch_size=1 \
  --max_tokens=512 \
  --temperature=0.1
```

In that smoke set, `3783730077.mp4` is missing from `video_test1_smoke`, so the
run loads two samples and skips one missing video.

VideoMolmo predicts points and uses SAM2 to produce masks through its external
`infer.py`. The evaluation wrapper extracts frames with `ffmpeg`, runs
VideoMolmo, first tries to read final mask outputs and convert each mask to its
minimum enclosing normalized bbox, and then returns the shared STVG JSON format.

There are two prompt layers in this integration:

1. The framework has one public STVG prompt type in `eval/prompts.py`: a shared
   bbox JSON prompt used by all JSON-capable models.
2. The VideoMolmo adapter has one model-specific internal prompt type: a
   `point to ...` prompt used only before calling external VideoMolmo `infer.py`.

The public evaluation prompt in `eval/prompts.py` is not modified. Most
generative VLM adapters, such as LLaVA-ST-Qwen2 and VideoChat-R1, should keep the
shared framework prompt and only parse their generated text back into the shared
frame-to-bbox format. VideoMolmo is different because its official interface is
not a bbox JSON generator: it expects a pointing instruction, emits Molmo point
tokens, and then uses SAM2 to propagate masks. Passing the long STVG JSON prompt
directly to VideoMolmo can produce no `<point ...>` output, leaving
`points.jsonl` empty and returning `{}`.

For this reason, `eval/scripts/run_videomolmo.sh` enables
`VIDEOMOLMO_USE_POINT_PROMPT=1`, which lets the VideoMolmo adapter convert the
wrapped STVG query into a model-native prompt such as `point to the person
holding the black camera` before calling external `infer.py`. This is an adapter
compatibility step, not a change to the common evaluation prompt. The adapter
must still convert the final masks back to normalized STVG boxes and score them
with the same metrics. Set `VIDEOMOLMO_USE_POINT_PROMPT=0` to pass the unchanged
STVG bbox prompt through for ablations.

If no mask output is found, the wrapper falls back to reading `points.jsonl` and
expanding points into small normalized boxes for debugging only.

`VIDEOMOLMO_ALLOW_LOG_FALLBACK` is disabled by default. It can parse JSON from
VideoMolmo logs for debugging, but it may also capture JSON examples from the
prompt, so it should not be used for reported metrics unless the logs are
manually verified.

Current VideoMolmo status and caveats:

- The framework integration has been smoke-tested through the external
  VideoMolmo `infer.py` subprocess. The wrapper can launch inference, extract
  frames, locate mask outputs when the external runtime saves them, convert
  masks to the shared STVG bbox prediction format, and fall back to
  `points.jsonl` point boxes when masks are unavailable.
- VideoMolmo's paper reports point and mask metrics, not STVG bbox metrics. This
  integration adapts VideoMolmo to STVG by converting SAM2 final masks to
  minimum enclosing boxes before computing `tIoU`, `vIoU`, `vIoU@0.3`, and
  `vIoU@0.5`.
- Prompt handling is intentionally model-specific: the framework still has one
  shared bbox JSON prompt, while the VideoMolmo adapter may translate that prompt
  into a `point to ...` instruction because VideoMolmo's expected output is a
  point that seeds SAM2, not JSON boxes.
- Server debugging showed that the VideoMolmo checkpoint can generate point
  predictions on the official `examples/video_sample1`, and SAM2 can generate a
  non-empty mask when the same point is passed directly to
  `build_sam2_video_predictor`. The all-black PNG symptom was confirmed to be
  in the external `infer.py` mask saving/post-processing path. After resizing
  squeezed SAM2 masks back to the original frame size before saving, the official
  example produced non-empty mask PNGs and the DORO-STVG wrapper converted them
  into STVG boxes.
- The external VideoMolmo `infer.py` fix is local to that external checkout, not
  vendored in this repository. The relevant saving logic should squeeze SAM2 mask
  tensors, convert them to a 2D boolean mask, resize them to the original
  `(height, width)` with nearest-neighbor interpolation when needed, OR multiple
  object masks together, and save the final PNG. Without this fix, VideoMolmo can
  produce valid points while all saved PNG masks remain black, which makes the
  DORO-STVG wrapper return no mask boxes.
- The smoke test is considered healthy when the log shows `Point Prompt: 1`,
  `prompt=point to ...`, and `Converted ... VideoMolmo mask frames to STVG
  boxes`. In the server smoke run, the two available videos converted 10 and 37
  mask frames respectively and produced non-empty STVG predictions.
- VideoMolmo's runtime environment is sensitive to server-specific torch, CUDA,
  bitsandbytes, accelerate, Molmo, and SAM2 versions. On a new server, use a
  `VIDEOMOLMO_PYTHON` where CUDA is available and the VideoMolmo/Molmo/SAM2
  imports work.
- With `VIDEOMOLMO_USE_POINT_PROMPT=0`, the unmodified STVG JSON-style prompt
  may make VideoMolmo produce no point output, no mask output, and an empty
  `points.jsonl`, resulting in `{}` predictions and zero metrics.
- Enabling `VIDEOMOLMO_ALLOW_LOG_FALLBACK=1` can recover JSON-looking text from
  logs, but this may be the JSON example embedded in the prompt rather than a
  real model prediction. Do not report metrics from fallback outputs without
  manual inspection.
- For meaningful VideoMolmo scores, use the documented
  `VIDEOMOLMO_USE_POINT_PROMPT=1` path and report that VideoMolmo was evaluated
  through its point-to-SAM2-mask pipeline before mask-to-bbox STVG scoring.

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


### CG-STVG / TA-STVG evaluation note

The DORO-STVG framework reads the JSONL annotations, calls the official-model helper, parses the returned tube, and computes final metrics with `eval/utils/metrics.py`. The official evaluator may run inside the helper because the public code exposes inference through its test script, but those official metrics are not used as final DORO-STVG scores.

`eval/utils/cgstvg_infer_helper.py` probes the real video frame count with `ffprobe` and creates full-video dummy boxes so official prediction frame ids map back to DORO-STVG frame ids. `eval/utils/tastvg_infer_helper.py` reuses the same bridge for TA-STVG when the TA-STVG checkout keeps the same VidSTG test interface.

The converter writes a dummy `fps=1.0` only to satisfy the official VidSTG annotation schema. CG-STVG still uses its own config internally, for example `INPUT.SAMPLE_FPS: 3.2`; DORO-STVG metrics compare integer frame ids from dataset annotations and returned tubes.


### VideoChat-R1

- Model name: `videochat-r1`
- Environment: `envs/eval/videochat_r1`
- Script: `eval/scripts/run_videochat_r1.sh`
- Default model path: `/mnt/sdc/xingjianwang/yibowang/model_zoo/VideoChat-R1_7B`
- The wrapper follows the Qwen2.5-VL-style `vLLM` video path with `qwen_vl_utils`.
- `eval/prompts.py` is not modified. The adapter samples the input video into a temporary clip, parses frame-to-box JSON, and remaps sampled clip frame ids back to original video frame ids.

### GroundingGPT

- Model name: `groundinggpt`
- Environment: `envs/eval/groundinggpt`
- Script: `eval/scripts/run_groundinggpt.sh`
- Model weights stay outside this evaluation framework and are passed through
  `MODEL_PATH`, matching the Qwen-style layout.
- The official GroundingGPT source/runtime is provided through
  `GROUNDINGGPT_SOURCE_DIR` and `GROUNDINGGPT_PYTHON`; the DORO-STVG repository
  only keeps the adapter, script, and lightweight wrapper environment.
- `eval/prompts.py` is not modified. The adapter normalizes JSON frame-to-box
  output and computes metrics in the shared DORO-STVG framework.


### STVG-R1

- Model name: `stvg-r1`
- Environment: `envs/eval/stvg_r1`
- Script: `eval/scripts/run_stvg_r1.sh`
- Default model path: `/mnt/sdc/xingjianwang/yibowang/model_zoo/stvg-r1-model-7b`
- The wrapper follows the Qwen2.5-VL-style `vLLM` video path with `qwen_vl_utils`.
- `eval/prompts.py` is not modified. The adapter samples the input video into a temporary clip, asks STVG-R1 to return STVG-friendly output, parses JSON/frame/time fallbacks, and remaps sampled clip frame ids back to original video frame ids.
- Fairness note: STVG-R1 is evaluated with the same DORO-STVG annotations, videos, shared prompt template, and metrics as Qwen, and the adapter does not use ground-truth boxes. The main adapter-specific difference is that it samples each video into a temporary 32-frame clip and appends output-format constraints so STVG-R1 responses can be normalized into the framework's frame-to-box JSON format. For strict paper-level comparisons, report these sampling/output-normalization details and consider matching Qwen's video sampling settings.
