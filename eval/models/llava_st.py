import copy
import importlib.util
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu

os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
logger = logging.getLogger(__name__)
LLAVA_ST_DEPENDENCIES_LOADED = False


def _bundled_llava_st_root() -> Path:
    env_root = os.getenv("LLAVA_ST_SOURCE_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()

    raise ImportError(
        "LLAVA_ST_SOURCE_DIR is not set. Install the external LLaVA-ST repository "
        "into envs/eval/llavast or set LLAVA_ST_SOURCE_DIR=/path/to/LLaVA-ST."
    )


def _load_bundled_get_variables():
    source_root = _bundled_llava_st_root()
    utils_path = source_root / "inference" / "src" / "utils.py"
    if not utils_path.exists():
        raise ImportError(f"LLaVA-ST inference utilities not found at {utils_path}")

    spec = importlib.util.spec_from_file_location("_llava_st_inference_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec for {utils_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_variables


def _ensure_llava_st_dependencies() -> None:
    global LLAVA_ST_DEPENDENCIES_LOADED
    global conversation_lib
    global get_variables
    global load_lora_model
    global DEFAULT_IM_END_TOKEN
    global DEFAULT_IM_START_TOKEN
    global DEFAULT_IMAGE_PATCH_TOKEN
    global DEFAULT_IMAGE_TOKEN
    global DEFAULT_SLOW_VID_END_TOKEN
    global DEFAULT_SLOW_VID_START_TOKEN
    global DEFAULT_VID_END_TOKEN
    global DEFAULT_VID_START_TOKEN
    global DEFAULT_VIDEO_PATCH_TOKEN
    global DEFAULT_VIDEO_TOKEN
    global IGNORE_INDEX
    global IMAGE_TOKEN_INDEX

    if LLAVA_ST_DEPENDENCIES_LOADED:
        return

    try:
        from llava import conversation as imported_conversation_lib
        from llava.constants import (
            DEFAULT_IM_END_TOKEN as imported_default_im_end_token,
            DEFAULT_IM_START_TOKEN as imported_default_im_start_token,
            DEFAULT_IMAGE_PATCH_TOKEN as imported_default_image_patch_token,
            DEFAULT_IMAGE_TOKEN as imported_default_image_token,
            DEFAULT_SLOW_VID_END_TOKEN as imported_default_slow_vid_end_token,
            DEFAULT_SLOW_VID_START_TOKEN as imported_default_slow_vid_start_token,
            DEFAULT_VID_END_TOKEN as imported_default_vid_end_token,
            DEFAULT_VID_START_TOKEN as imported_default_vid_start_token,
            DEFAULT_VIDEO_PATCH_TOKEN as imported_default_video_patch_token,
            DEFAULT_VIDEO_TOKEN as imported_default_video_token,
            IGNORE_INDEX as imported_ignore_index,
            IMAGE_TOKEN_INDEX as imported_image_token_index,
        )
        from llava.model.builder import load_lora_model as imported_load_lora_model
    except ImportError as exc:
        raise ImportError(
            "Failed to import LLaVA-ST dependencies. Run with envs/eval/llavast "
            "and install the external LLaVA-ST repository, for example: "
            "uv pip install -e /path/to/LLaVA-ST"
        ) from exc

    try:
        from inference.src.utils import get_variables as imported_get_variables
    except ModuleNotFoundError as exc:
        if exc.name != "inference":
            raise
        imported_get_variables = _load_bundled_get_variables()

    conversation_lib = imported_conversation_lib
    get_variables = imported_get_variables
    load_lora_model = imported_load_lora_model
    DEFAULT_IM_END_TOKEN = imported_default_im_end_token
    DEFAULT_IM_START_TOKEN = imported_default_im_start_token
    DEFAULT_IMAGE_PATCH_TOKEN = imported_default_image_patch_token
    DEFAULT_IMAGE_TOKEN = imported_default_image_token
    DEFAULT_SLOW_VID_END_TOKEN = imported_default_slow_vid_end_token
    DEFAULT_SLOW_VID_START_TOKEN = imported_default_slow_vid_start_token
    DEFAULT_VID_END_TOKEN = imported_default_vid_end_token
    DEFAULT_VID_START_TOKEN = imported_default_vid_start_token
    DEFAULT_VIDEO_PATCH_TOKEN = imported_default_video_patch_token
    DEFAULT_VIDEO_TOKEN = imported_default_video_token
    IGNORE_INDEX = imported_ignore_index
    IMAGE_TOKEN_INDEX = imported_image_token_index
    LLAVA_ST_DEPENDENCIES_LOADED = True


def preprocess_qwen(
    sources,
    tokenizer,
    has_image: bool = False,
    system_message: str = "You are a helpful assistant.",
):
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    system_tokens = tokenizer("system").input_ids + nl_tokens

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id: List[int] = []
    target: List[int] = []
    system = [im_start] + system_tokens + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens

    for sentence in source:
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and DEFAULT_IMAGE_TOKEN in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split(DEFAULT_IMAGE_TOKEN)
            current_input = tokenizer(role).input_ids + nl_tokens
            for idx, text in enumerate(texts):
                current_input += tokenizer(text).input_ids
                if idx < len(texts) - 1:
                    current_input += [IMAGE_TOKEN_INDEX] + nl_tokens
            current_input += [im_end] + nl_tokens
            assert sum(token == IMAGE_TOKEN_INDEX for token in current_input) == num_image
        else:
            if sentence["value"] is None:
                current_input = tokenizer(role).input_ids + nl_tokens
            else:
                current_input = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens

        input_id += current_input
        if role == "<|im_start|>user":
            current_target = [im_start] + [IGNORE_INDEX] * (len(current_input) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            role_tokens = tokenizer(role).input_ids
            current_target = [im_start] + [IGNORE_INDEX] * len(role_tokens) + current_input[len(role_tokens) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError(f"Unsupported role: {role}")
        target += current_target

    del target
    return torch.tensor([input_id], dtype=torch.long)


def preprocess_multimodal(sources, vision_config):
    for source in sources:
        for sentence in source:
            value = sentence.get("value")
            if value is None:
                continue

            if DEFAULT_IMAGE_TOKEN in value:
                num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, value))
                if num_im == 1 and not value.startswith(DEFAULT_IMAGE_TOKEN):
                    value = value.replace(DEFAULT_IMAGE_TOKEN, "").strip()
                    value = f"{DEFAULT_IMAGE_TOKEN}\n{value}".strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        value = value.replace(DEFAULT_IMAGE_TOKEN, f"<Image>{DEFAULT_IMAGE_TOKEN}</Image>")
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * (vision_config.image_token_num - 2)
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = value.replace(DEFAULT_IMAGE_TOKEN, replace_token).replace("QA_GT_caption_based_noisy", "")
                continue

            if DEFAULT_VIDEO_TOKEN in value:
                num_vid = len(re.findall(DEFAULT_VIDEO_TOKEN, value))
                if num_vid == 1 and not value.startswith(DEFAULT_VIDEO_TOKEN):
                    value = value.replace(DEFAULT_VIDEO_TOKEN, "").strip()
                    value = f"{DEFAULT_VIDEO_TOKEN}\n{value}".strip()

                if getattr(vision_config, "slow_token", None):
                    replace_token = (
                        DEFAULT_VID_START_TOKEN
                        + DEFAULT_VIDEO_PATCH_TOKEN * (vision_config.fast_token_num * vision_config.fast_frame_num - 2)
                        + DEFAULT_VID_END_TOKEN
                        + DEFAULT_SLOW_VID_START_TOKEN
                        + DEFAULT_VIDEO_PATCH_TOKEN * (vision_config.slow_token_num * vision_config.slow_frame_num - 2)
                        + DEFAULT_SLOW_VID_END_TOKEN
                    )
                else:
                    replace_token = (
                        DEFAULT_VID_START_TOKEN
                        + DEFAULT_VIDEO_PATCH_TOKEN * (vision_config.slow_token_num * vision_config.slow_frame_num)
                        + DEFAULT_VID_END_TOKEN
                    )
                sentence["value"] = value.replace(DEFAULT_VIDEO_TOKEN, replace_token)

    return sources


def _load_video_frames(video_path: str, max_frames: int) -> Tuple[np.ndarray, List[int]]:
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Empty video: {video_path}")

    if max_frames <= 1:
        indices = [0]
    else:
        indices = [round((i / (max_frames - 1)) * (total - 1)) for i in range(max_frames)]
    frames = vr.get_batch(indices).asnumpy()
    return frames, indices


def _llava_st_tokens_to_json(response_text: str, sampled_indices: List[int]) -> str:
    frame_map: Dict[str, List[float]] = {}

    temp_matches = re.findall(
        r"<TEMP-(\d{3})>\s*:\s*\[<WIDTH-(\d{3})><HEIGHT-(\d{3})><WIDTH-(\d{3})><HEIGHT-(\d{3})>\]",
        response_text,
    )
    for temp_idx, x1, y1, x2, y2 in temp_matches:
        sample_pos = int(temp_idx)
        if sample_pos < 0 or sample_pos >= len(sampled_indices):
            continue
        frame_idx = int(sampled_indices[sample_pos])
        frame_map[str(frame_idx)] = [
            max(0.0, min(1.0, int(x1) / 99.0)),
            max(0.0, min(1.0, int(y1) / 99.0)),
            max(0.0, min(1.0, int(x2) / 99.0)),
            max(0.0, min(1.0, int(y2) / 99.0)),
        ]

    frame_matches = re.findall(
        r"(?m)^\s*(?:[-*]\s*)?(\d{1,8})\s*:\s*\[<WIDTH-(\d{3})><HEIGHT-(\d{3})><WIDTH-(\d{3})><HEIGHT-(\d{3})>\]",
        response_text,
    )
    for frame_idx, x1, y1, x2, y2 in frame_matches:
        frame_map[str(int(frame_idx))] = [
            max(0.0, min(1.0, int(x1) / 99.0)),
            max(0.0, min(1.0, int(y1) / 99.0)),
            max(0.0, min(1.0, int(x2) / 99.0)),
            max(0.0, min(1.0, int(y2) / 99.0)),
        ]

    if not frame_map:
        return response_text
    return json.dumps({"target": frame_map}, ensure_ascii=False)


class LlavaSTQwen2:
    """LLaVA-ST-Qwen2 wrapper for the existing STVG evaluation pipeline."""

    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        max_tokens: int = 512,
        max_model_len: int = 8192,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        del max_model_len, tensor_parallel_size, gpu_memory_utilization

        self.model_path = model_path
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_frames = int(os.getenv("LLAVA_ST_MAX_FRAMES", "100"))
        self.vt_chunk = int(os.getenv("LLAVA_ST_VT_CHUNK", "1"))
        self.use_cache = os.getenv("LLAVA_ST_USE_CACHE", "1").lower() in {"1", "true", "yes"}
        self.decode_temperature = temperature if temperature > 0 else float(os.getenv("LLAVA_ST_FALLBACK_TEMPERATURE", "0.01"))

        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []
        self.last_video_frame_indices: List[List[int]] = []

        _ensure_llava_st_dependencies()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for llava-st-qwen2.")

        torch.cuda.set_device(0)
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16

        self.tokenizer, self.model, self.image_processor, _ = load_lora_model(
            [],
            self.model_path,
            "llava_qwen",
            device_map="auto",
            attn_implementation="sdpa",
            overwrite_config={
                "num_spatial_tokens": 100,
                "num_temporal_tokens": 100,
            },
        )
        self.model.eval()

        base = self.model.get_model() if hasattr(self.model, "get_model") else self.model
        self.vision_config = getattr(base, "vision_config", None)
        if self.vision_config is None and hasattr(self.model, "model"):
            self.vision_config = getattr(self.model.model, "vision_config", None)

        vt = base.get_vision_tower() if hasattr(base, "get_vision_tower") else getattr(base, "vision_tower", None)
        if vt is not None and self.vt_chunk > 0 and hasattr(vt, "forward"):
            original_forward = vt.forward

            def chunked_forward(images, *args, **kwargs):
                if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[0] > self.vt_chunk:
                    feat_chunks = []
                    last_outs = None
                    for chunk in torch.split(images, self.vt_chunk, dim=0):
                        feat, outs = original_forward(chunk, *args, **kwargs)
                        feat_chunks.append(feat)
                        last_outs = outs
                    return torch.cat(feat_chunks, dim=0), last_outs
                return original_forward(images, *args, **kwargs)

            vt.forward = chunked_forward

        logger.info(
            "Initialized llava-st-qwen2 | source=%s device=%s dtype=%s max_frames=%s vt_chunk=%s",
            _bundled_llava_st_root(),
            self.device,
            self.dtype,
            self.max_frames,
            self.vt_chunk,
        )

    def _predict_one(self, query: str, video_path: str, system_prompt: str) -> Tuple[str, List[int]]:
        del system_prompt

        frames, sampled_indices = _load_video_frames(video_path, max_frames=self.max_frames)
        video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        video_tensor = video_tensor.to(device=self.device, dtype=self.dtype, non_blocking=True)

        conversations = [
            {
                "from": "human",
                "value": f"{DEFAULT_VIDEO_TOKEN}\n{query.strip()}",
            },
            {
                "from": "gpt",
                "value": None,
            },
        ]
        conversations, variables = get_variables(conversations)
        sources = preprocess_multimodal([copy.deepcopy(conversations)], self.vision_config)
        input_ids = preprocess_qwen(
            [sources[0][0], {"from": "gpt", "value": None}],
            self.tokenizer,
            has_image=True,
        ).to(self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=[video_tensor],
                modalities=["video"],
                variables=[variables],
                do_sample=True,
                temperature=self.decode_temperature,
                top_p=None,
                num_beams=1,
                max_new_tokens=self.max_tokens,
                use_cache=self.use_cache,
            )

        raw_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
        raw_output = raw_output.replace("<|im_end|>", "").strip()
        return raw_output, sampled_indices

    def predict_batch(self, queries: List[str], video_paths: List[str], system_prompt: str) -> List[str]:
        self.last_user_prompts = list(queries)

        raw_outputs: List[str] = []
        converted_outputs: List[str] = []
        frame_indices: List[List[int]] = []
        for query, video_path in zip(queries, video_paths):
            raw_output, sampled_indices = self._predict_one(query, video_path, system_prompt)
            raw_outputs.append(raw_output)
            converted_outputs.append(_llava_st_tokens_to_json(raw_output, sampled_indices))
            frame_indices.append(sampled_indices)

        self.last_raw_responses = raw_outputs
        self.last_video_frame_indices = frame_indices
        return converted_outputs
