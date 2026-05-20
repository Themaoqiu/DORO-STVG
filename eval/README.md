# Eval Framework README

This directory contains the STVG evaluation framework. It supports two evaluation types:

- **Spatiotemporal grounding**: models output temporal spans plus frame-indexed boxes; metrics include `m_tIoU`, `m_vIoU`, `vIoU@0.3`, and `vIoU@0.5`.
- **Temporal-only grounding**: models output only temporal spans; metrics include `m_tIoU`, `tIoU@0.3`, and `tIoU@0.5`.

The temporal coordinate convention used by prompts and temporal-only evaluation is:

```text
coordinate_system = sampled_2fps_frame_index
```

## Directory Layout

| Path | Responsibility |
| --- | --- |
| `main.py` | Main spatiotemporal evaluation CLI. Builds the model and dataset pipeline. |
| `temporal_main.py` | Separate temporal-only evaluation CLI. |
| `prompts.py` | Spatiotemporal prompt and response parser. |
| `temporal_prompts.py` | Temporal-only prompt and response parser. |
| `models/` | Model adapters. Each adapter exposes `predict_batch(...)`; temporal-only adapters can also expose `predict_temporal_batch(...)`. |
| `pipelines/` | Dataset pipelines and evaluation loops. |
| `utils/metrics.py` | Spatiotemporal metrics. |
| `utils/temporal_metrics.py` | Temporal-only metrics. |
| `utils/*_infer_helper.py` | Helper scripts for official/external repositories. |
| `scripts/` | Model launch scripts. |

## Spatiotemporal Evaluation Flow

```text
scripts/run_*.sh
  -> python main.py run ...
  -> STVGEvaluator._build_model(...)
  -> dataset pipeline
  -> prompts.format_prompt(...)
  -> model.predict_batch(...)
  -> prompts.parse_response(...)
  -> metrics.compute_metrics(...) / compute_multi_target_metrics(...)
```

Expected model output:

```json
{"target": {"12": [0.1, 0.2, 0.3, 0.4]}}
```

`prompts.py` tells generative models that videos are sampled at 2 fps and asks them to output frame indices instead of timestamps.

## Temporal-Only Evaluation Flow

Temporal-only evaluation is separated from the main STVG path. It is intended for models such as VTimeLLM and Grounded-Video-LLM when only temporal localization is needed.

```text
scripts/run_*_temporal.sh
  -> python temporal_main.py run ...
  -> TemporalEvaluator._build_model(...)
  -> DOROTemporalPipeline
  -> temporal_prompts.format_temporal_prompt(...)
  -> model.predict_temporal_batch(...) if available
  -> temporal_prompts.parse_temporal_response(...)
  -> temporal_metrics.compute_temporal_metrics(...)
```

Expected temporal-only output:

```json
{"temporal_span": [12, 38]}
```

Supported temporal launchers:

- `scripts/run_vtimellm_temporal.sh`
- `scripts/run_grounded_video_llm_temporal.sh`

Temporal-only results are saved as:

```text
temporal_results.jsonl
temporal_status.json
```

Temporal-only results use:

```text
evaluation_type = temporal_only
coordinate_system = sampled_2fps_frame_index
```

They do not report spatial metrics such as `m_vIoU`, `vIoU@0.3`, or `vIoU@0.5`.

## Common Launch Parameters

Most spatiotemporal model scripts call:

```bash
python main.py run \
  --model_name="$MODEL_NAME" \
  --model_path="$MODEL_PATH" \
  --data_name="$DATA_NAME" \
  --annotation_path="$ANNOTATION_PATH" \
  --video_dir="$VIDEO_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size="$BATCH_SIZE" \
  --max_tokens="$MAX_TOKENS" \
  --max_model_len="$MAX_MODEL_LEN" \
  --temperature="$TEMPERATURE" \
  --tensor_parallel_size="$TENSOR_PARALLEL_SIZE" \
  --gpu_memory_utilization="$GPU_MEMORY_UTILIZATION"
```

| Parameter | Meaning |
| --- | --- |
| `MODEL_NAME` | Selects the model adapter in `main.py`. |
| `MODEL_PATH` | Model directory, checkpoint, or external model path. |
| `DATA_NAME` | Dataset pipeline. Current main values include `doro-stvg`, `hcstvg`, and `vidstg`. |
| `ANNOTATION_PATH` | Dataset annotation file. |
| `VIDEO_DIR` | Video root directory. |
| `OUTPUT_DIR` | Evaluation output root. |
| `BATCH_SIZE` | Dataset batch size. |
| `MAX_TOKENS` | Generation token budget for generative models. Ignored by most official detector helpers. |
| `MAX_MODEL_LEN` | vLLM context length for vLLM-backed models. |
| `TEMPERATURE` | Generation temperature. |
| `TENSOR_PARALLEL_SIZE` | vLLM tensor parallel size. |
| `GPU_MEMORY_UTILIZATION` | vLLM GPU memory utilization target. |
| `CUDA_VISIBLE_DEVICES` | GPU visibility control set by launch scripts. |

## Model Integration Types

| Type | Models | Integration pattern |
| --- | --- | --- |
| Native vLLM multimodal models | Qwen2.5-VL, Qwen3-VL/Qwen3.5 | Load directly with `vllm.LLM` and `AutoProcessor`. |
| vLLM-derived models with video preprocessing | VideoChat-R1, STVG-R1, LLaVA-1.6 | Preprocess videos into temporary clips or frame grids before generation. |
| External source loaded in-process | LLaVA-ST-Qwen2, VTimeLLM | Import external repository code and load models inside the eval Python process. |
| External CLI/subprocess generative models | Grounded-Video-LLM, GroundingGPT, VideoMolmo | Call external `inference.py`, CLI, or `infer.py` and normalize output. |
| Official detector/helper pipelines | CG-STVG, TA-STVG, TubeDETR | Write a manifest, call official helper logic, read predictions, and normalize to benchmark JSON. |

### Existing Model Files

| Model | Adapter | Launch script |
| --- | --- | --- |
| Qwen2.5-VL / Qwen3-VL | `models/qwen_family.py` | `scripts/run_qwen.sh` |
| LLaVA-ST-Qwen2 | `models/llava_st.py` | `scripts/run_llavast.sh` |
| VTimeLLM | `models/vtimellm.py` | `scripts/run_vtimellm.sh`, `scripts/run_vtimellm_temporal.sh` |
| Grounded-Video-LLM | `models/grounded_video_llm.py` | `scripts/run_grounded_video_llm.sh`, `scripts/run_grounded_video_llm_temporal.sh` |
| LLaVA-1.6 | `models/llava16.py` | `scripts/run_llava16.sh` |
| VideoChat-R1 | `models/qwen_family.py` | `scripts/run_videochat_r1.sh` |
| STVG-R1 | `models/stvg_r1.py` | `scripts/run_stvg_r1.sh` |
| GroundingGPT | `models/groundinggpt.py` | `scripts/run_groundinggpt.sh` |
| VideoMolmo | `models/videomolmo.py` | `scripts/run_videomolmo.sh` |
| CG-STVG | `models/cgstvg.py` | `scripts/run_cgstvg.sh` |
| TA-STVG | `models/tastvg.py` | `scripts/run_tastvg.sh` |
| TubeDETR | `models/tubedetr.py` | `scripts/run_tubedetr.sh` |

## Script-Specific Notes

### Qwen

`scripts/run_qwen.sh` launches Qwen VL models through `models/qwen_family.py`. Qwen is the most direct generative VLM reference path:

```text
video path + common prompt
  -> Qwen processor / qwen_vl_utils
  -> vLLM generation
  -> common response parsing
  -> metrics
```

### LLaVA-ST

`scripts/run_llavast.sh` requires `LLAVA_ST_SOURCE_DIR`. `scripts/setup_llavast_env.sh` prepares the LLaVA-ST eval environment.

### VTimeLLM

`scripts/run_vtimellm.sh` runs spatiotemporal evaluation. `scripts/run_vtimellm_temporal.sh` runs temporal-only evaluation. Temporal-only VTimeLLM uses `predict_temporal_batch(...)` and returns `{"temporal_span": [...]}` rather than synthetic full-frame boxes.

### Grounded-Video-LLM

`scripts/run_grounded_video_llm.sh` runs spatiotemporal-style output parsing. `scripts/run_grounded_video_llm_temporal.sh` runs temporal-only evaluation through `temporal_main.py`.

### LLaVA-1.6

`scripts/run_llava16.sh` samples video frames, builds a frame-grid image, and runs a vLLM image-model path.

### VideoChat-R1 and STVG-R1

These models create temporary video clips and remap local frame predictions back to sampled/original frame indices. STVG-R1 also has temporal fallback logic.

### Official Detector Helpers

CG-STVG, TA-STVG, and TubeDETR use official helper pipelines. They should normally be reported separately from generative VLMs.

The explicit helper FPS values are aligned to the 2 fps prompt convention:

| Location | FPS |
| --- | --- |
| `scripts/run_tubedetr.sh` | `TUBEDETR_FPS=2` |
| `scripts/run_all_models.sh` TubeDETR branch | `TUBEDETR_FPS="${TUBEDETR_FPS:-2}"` |
| `utils/tubedetr_infer_helper.py` | `--fps` default `2` |
| `utils/cgstvg_infer_helper.py` converter call | `--fps 2.0` |
| `utils/tastvg_infer_helper.py` converter call | `--fps 2.0` |

### VideoMolmo

VideoMolmo extracts frames, calls external `infer.py`, then converts point predictions into boxes. Check `scripts/run_videomolmo.sh` carefully because environment variables must be exported or kept in the same command environment for Python to read them.

## Batch Evaluation

`scripts/run_all_models.sh` runs multiple model adapters and writes:

```text
res_all_models/<RUN_ID>/<model>/
res_all_models/<RUN_ID>/logs/<model>.log
res_all_models/<RUN_ID>/summary.tsv
```

Useful controls:

| Variable | Meaning |
| --- | --- |
| `MODELS` | Space-separated model list. Overrides the default model list. |
| `RUN_QWEN` | Include Qwen in the default list when set to `1`. |
| `FIRST_N` | Run only the first N selected models. |
| `DRY_RUN` | Print/check configuration without running models. |
| `CONTINUE_ON_ERROR` | Continue after a model fails when set to `1`. |
| `OUTPUT_ROOT` | Batch output root. |
| `LOG_DIR` | Log directory. |
| `HF_OFFLINE` | Sets Hugging Face offline mode when `1`. |
| `CLEAR_PROXY` | Clears proxy environment variables when `1`. |
| `AUTO_INSTALL_EXTERNAL` | Installs external editable repos into envs when possible. |

## Fairness Scope

The benchmark is fair in the limited sense that final predictions are evaluated by the same dataset pipeline and metric code after normalization. It is not a pure measurement of model weights alone.

Interpret scores as:

```text
model + input adapter + output adapter + postprocessing
```

### Fair Parts

- Same annotation/video roots for a run.
- Same final metric implementations.
- Same normalized spatiotemporal JSON target for spatiotemporal evaluation.
- Same temporal-only metric implementation for temporal-only evaluation.
- Explicit helper FPS values are now aligned to the 2 fps prompt convention.

### Fairness Caveats

1. **Input adaptation differs.** Qwen receives video through Qwen processor logic, LLaVA-1.6 receives frame grids, VideoMolmo receives frame directories, R1 models receive temporary clips, and official detector models run through helper pipelines.
2. **Temporal coordinates may differ.** Some adapters sample by `max_frames`, clip duration, helper logic, or model-specific processors. Incorrect local-frame-to-benchmark-frame mapping biases temporal metrics.
3. **Prompting is not identical.** Some models require specialized prompts or conversation templates. Official detectors do not follow the same generative JSON prompt path.
4. **Output adapters differ.** Some outputs are parsed directly; others use custom token conversion, point-to-box conversion, temporal fallback, or official prediction normalization.
5. **Official detector pipelines are a different category.** TA-STVG, CG-STVG, and TubeDETR use task-specific code/configs/postprocessing and should be reported separately from generative VLMs.
6. **Runtime parameters differ.** `max_tokens`, `temperature`, `top_p`, `max_frames`, `num_clip_frames`, and `resolution` can affect scores.
7. **Fallback behavior differs.** Retries, log fallback, time-range fallback, and row filling improve robustness but make raw-model comparisons less clean.

## Recommended Reporting

Avoid presenting all models as one undifferentiated pure-model leaderboard. Prefer grouped reporting:

- Qwen-style generative VLMs.
- VLMs with required video preprocessing.
- External CLI/subprocess VLMs.
- Official detector/helper pipelines.
- Temporal-only models.

If a single table is shown, include adapter notes such as input form, FPS/sampling policy, prompt type, postprocessing type, fallback status, and whether the run is `spatiotemporal` or `temporal_only`.

## Adding A New Model

1. Decide the integration type: native vLLM, preprocessed vLLM, in-process external source, external CLI, or official helper.
2. Add a model adapter under `models/`.
3. Implement the common constructor signature used by `main.py`.
4. Implement `predict_batch(...) -> list[str]`.
5. For temporal-only support, optionally implement `predict_temporal_batch(...) -> list[str]`.
6. Register the adapter in `main.py::_build_model(...)`.
7. Add lazy export in `models/__init__.py` if needed.
8. Add a launch script in `scripts/`.
9. Ensure the output parser can parse the model output.
10. Record any non-standard input sampling, prompt, postprocessing, or fallback behavior.
