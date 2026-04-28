import copy
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu

os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
logger = logging.getLogger(__name__)


def _discover_llava_st_source() -> Path:
    env_path = os.getenv("LLAVA_ST_SOURCE_DIR")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root.parent / "LLaVA-ST",
        repo_root.parent / "models" / "LLaVA-ST",
        repo_root / "models" / "LLaVA-ST",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Unable to locate LLaVA-ST source. Set LLAVA_ST_SOURCE_DIR to the "
        "repository root that contains the `llava/` and `inference/` packages."
    )


LLAVA_ST_SOURCE = _discover_llava_st_source()
if str(LLAVA_ST_SOURCE) not in sys.path:
    sys.path.insert(0, str(LLAVA_ST_SOURCE))

from inference.src.utils import get_variables, replace_and_normalize  # noqa: E402
from llava import conversation as conversation_lib  # noqa: E402
from llava.constants import (  # noqa: E402
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_SLOW_VID_END_TOKEN,
    DEFAULT_SLOW_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VIDEO_PATCH_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.model.builder import load_lora_model  # noqa: E402


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


def _extract_video_spec(video_input: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    marker = "::split="
    if marker not in video_input:
        return video_input, None

    video_path, split_text = video_input.split(marker, 1)
    try:
        start_text, end_text = split_text.split(":", 1)
        split = (int(start_text), int(end_text))
    except ValueError:
        logger.warning("Failed to parse split from video input %s", video_input)
        return video_path, None
    return video_path, split


def _load_video_frames(video_path: str, max_frames: int, split: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, List[int]]:
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Empty video: {video_path}")

    if split is None:
        split = (0, total - 1)
    start, end = split
    start = max(0, min(total - 1, start))
    end = max(start, min(total - 1, end))

    if max_frames <= 1:
        indices = [start]
    else:
        duration = end - start + 1
        indices = [
            start + round((i / (max_frames - 1)) * (duration - 1))
            for i in range(max_frames)
        ]
    frames = vr.get_batch(indices).asnumpy()
    return frames, indices


def _parse_box_values(coords_text: str) -> Optional[List[float]]:
    try:
        coords = [float(part.strip()) for part in coords_text.split(",")]
    except ValueError:
        return None
    if len(coords) != 4:
        return None
    return [max(0.0, min(1.0, value)) for value in coords]


def _frame_from_token(frame_token: str, sampled_indices: List[int]) -> Optional[int]:
    token = frame_token.strip().strip("\"' ")
    try:
        numeric = float(token)
    except ValueError:
        return None

    if sampled_indices and int(round(numeric)) in sampled_indices:
        return int(round(numeric))

    if len(sampled_indices) == 0:
        return None

    if 0.0 <= numeric <= 1.0:
        position = int(round(numeric * (len(sampled_indices) - 1)))
        position = max(0, min(len(sampled_indices) - 1, position))
        return int(sampled_indices[position])

    position = int(round(numeric))
    if 0 <= position < len(sampled_indices):
        return int(sampled_indices[position])

    if 0 <= numeric <= 99:
        position = int(round((numeric / 99.0) * (len(sampled_indices) - 1)))
        position = max(0, min(len(sampled_indices) - 1, position))
        return int(sampled_indices[position])

    return None


def _brace_block_after(text: str, anchor: str) -> Optional[str]:
    start_anchor = text.find(anchor)
    if start_anchor == -1:
        return None
    start = text.find("{", start_anchor)
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _convert_official_output(raw_text: str, sampled_indices: List[int], query: str) -> Optional[str]:
    if not raw_text.strip():
        return None

    frame_map: Dict[str, List[float]] = {}
    raw_matches = re.findall(
        r"<TEMP-(\d{3})>\s*:\s*\[<WIDTH-(\d{3})><HEIGHT-(\d{3})><WIDTH-(\d{3})><HEIGHT-(\d{3})>\]",
        raw_text,
    )
    if raw_matches:
        for temp_idx, x1, y1, x2, y2 in raw_matches:
            frame_position = int(temp_idx)
            if not (0 <= frame_position < len(sampled_indices)):
                continue
            frame_idx = int(sampled_indices[frame_position])
            frame_map[str(frame_idx)] = [
                int(x1) / 99.0,
                int(y1) / 99.0,
                int(x2) / 99.0,
                int(y2) / 99.0,
            ]

    if not frame_map:
        normalized_text = replace_and_normalize(raw_text).replace("<|im_end|>", "").strip()
        bbox_block = _brace_block_after(normalized_text, "Object bounding box")
        if not bbox_block:
            return None

        matches = re.findall(
            r'["\']?([0-9]*\.?[0-9]+),?["\']?\s*:?\s*\[\s*([^\]]+?)\s*\]',
            bbox_block,
        )
        for frame_token, coords_text in matches:
            frame_token = frame_token.rstrip(",")
            frame_idx = _frame_from_token(frame_token, sampled_indices)
            coords = _parse_box_values(coords_text)
            if frame_idx is None or coords is None:
                continue
            frame_map[str(frame_idx)] = coords

    if not frame_map:
        return None

    description = query.strip()
    suffix = " Please find the corresponding time period in the video."
    if suffix in description:
        description = description.split(suffix, 1)[0].strip()
    description = description or "target"
    return json.dumps({description: frame_map}, ensure_ascii=False)


class LlavaSTQwen2:
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
            LLAVA_ST_SOURCE,
            self.device,
            self.dtype,
            self.max_frames,
            self.vt_chunk,
        )

    def _predict_one(self, query: str, video_input: str, system_prompt: str) -> Tuple[str, str]:
        del system_prompt

        video_path, split = _extract_video_spec(video_input)
        frames, sampled_indices = _load_video_frames(video_path, max_frames=self.max_frames, split=split)
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

        converted = _convert_official_output(raw_output, sampled_indices, query)
        if converted is None:
            converted = raw_output
        return raw_output, converted

    def predict_batch(self, queries: List[str], video_paths: List[str], system_prompt: str) -> List[str]:
        self.last_user_prompts = list(queries)

        converted_outputs: List[str] = []
        raw_outputs: List[str] = []
        for query, video_path in zip(queries, video_paths):
            raw_output, converted_output = self._predict_one(query, video_path, system_prompt)
            raw_outputs.append(raw_output)
            converted_outputs.append(converted_output)

        self.last_raw_responses = raw_outputs
        return converted_outputs
