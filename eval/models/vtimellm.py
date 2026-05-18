import json
import logging
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
from decord import VideoReader, cpu
from PIL import Image


logger = logging.getLogger(__name__)
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
VTIMELLM_DEPENDENCIES_LOADED = False


def _vtimellm_source_root() -> Path:
    env_root = os.getenv("VTIMELLM_SOURCE_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()
    raise ImportError(
        "VTIMELLM_SOURCE_DIR is not set. Install the external VTimeLLM repository "
        "into envs/eval/vtimellm or set VTIMELLM_SOURCE_DIR=/path/to/VTimeLLM."
    )


def _ensure_vtimellm_dependencies() -> None:
    global VTIMELLM_DEPENDENCIES_LOADED
    global clip, conv_templates, disable_torch_init, tokenizer_image_token
    global KeywordsStoppingCriteria, load_pretrained_model, IMAGE_TOKEN_INDEX, SeparatorStyle

    if VTIMELLM_DEPENDENCIES_LOADED:
        return

    source_root = _vtimellm_source_root()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    try:
        import clip as imported_clip
        from vtimellm.constants import IMAGE_TOKEN_INDEX as imported_image_token_index
        from vtimellm.conversation import SeparatorStyle as imported_separator_style
        from vtimellm.conversation import conv_templates as imported_conv_templates
        from vtimellm.mm_utils import (
            KeywordsStoppingCriteria as imported_keywords_stopping_criteria,
            tokenizer_image_token as imported_tokenizer_image_token,
        )
        from vtimellm.model.builder import load_pretrained_model as imported_load_pretrained_model
        from vtimellm.utils import disable_torch_init as imported_disable_torch_init
    except ImportError as exc:
        raise ImportError(
            "Failed to import VTimeLLM dependencies. Run with envs/eval/vtimellm "
            "and install the external VTimeLLM repository, for example: "
            "uv pip install -e /path/to/VTimeLLM"
        ) from exc

    clip = imported_clip
    conv_templates = imported_conv_templates
    disable_torch_init = imported_disable_torch_init
    tokenizer_image_token = imported_tokenizer_image_token
    KeywordsStoppingCriteria = imported_keywords_stopping_criteria
    load_pretrained_model = imported_load_pretrained_model
    IMAGE_TOKEN_INDEX = imported_image_token_index
    SeparatorStyle = imported_separator_style
    VTIMELLM_DEPENDENCIES_LOADED = True


def _sample_video_features(video_path, clip_model, image_processor, device, max_frames):
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Empty video: {video_path}")

    indices = [0] if max_frames <= 1 else [round((i / (max_frames - 1)) * (total - 1)) for i in range(max_frames)]
    frames = vr.get_batch(indices).asnumpy()
    fps = float(vr.get_avg_fps() or 0.0)

    images = [image_processor(Image.fromarray(frame)).unsqueeze(0) for frame in frames]
    image_tensor = torch.cat(images, dim=0).to(device=device, dtype=torch.float16)

    with torch.inference_mode():
        features = clip_model.encode_image(image_tensor)

    return features, indices, fps


def _parse_seconds_from_text(text: str) -> Optional[Tuple[float, float]]:
    patterns = [
        r"from\s+(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)?\s+to\s+(\d+(?:\.\d+)?)",
        r"between\s+(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)?\s+and\s+(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)\s*[-~]\s*(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)",
        r"(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            start, end = float(match.group(1)), float(match.group(2))
            return (start, end) if start <= end else (end, start)
    return None


def _parse_frame_span_from_text(text: str) -> Optional[Tuple[int, int]]:
    patterns = [
        r'"temporal_span"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
        r"'temporal_span'\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]",
        r"frames?\s+(\d+)\s*(?:to|through|until|-|~)\s*(\d+)",
        r"frame\s+range\s*[:=]\s*(\d+)\s*(?:to|through|until|-|~)\s*(\d+)",
        r"(\d+)\s*(?:to|through|until|-|~)\s*(\d+)\s*frames?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            return (start, end) if start <= end else (end, start)
    return None


def _span_to_frame_map(span, total_frames: int, max_dense_frames: int) -> Dict[str, List[float]]:
    if span is None or total_frames <= 0:
        return {}
    start, end = span
    start = max(0, min(total_frames - 1, int(start)))
    end = max(start, min(total_frames - 1, int(end)))

    length = end - start + 1
    if length <= max_dense_frames:
        frame_indices = list(range(start, end + 1))
    else:
        step = max(1, round(length / max_dense_frames))
        frame_indices = list(range(start, end + 1, step))
        if frame_indices[-1] != end:
            frame_indices.append(end)

    return {str(frame_idx): [0.0, 0.0, 1.0, 1.0] for frame_idx in frame_indices}


def _vtimellm_response_to_json(response_text: str, video_path: str, max_dense_frames: int) -> str:
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = float(vr.get_avg_fps() or 0.0)

    frame_span = _parse_frame_span_from_text(response_text)
    if frame_span is None and fps > 0:
        second_span = _parse_seconds_from_text(response_text)
        if second_span is not None:
            frame_span = (round(second_span[0] * fps), round(second_span[1] * fps))

    frame_map = _span_to_frame_map(frame_span, total_frames, max_dense_frames)
    if not frame_map:
        return response_text
    return json.dumps({"target": frame_map}, ensure_ascii=False)


def _scale_temporal_span(span: Tuple[int, int], input_fps: float, output_fps: float) -> Tuple[int, int]:
    if input_fps <= 0 or output_fps <= 0 or input_fps == output_fps:
        return span
    scale = output_fps / input_fps
    return (round(span[0] * scale), round(span[1] * scale))


def _vtimellm_response_to_temporal_json(
    response_text: str,
    input_fps: float = 2.0,
    output_fps: float = 2.0,
) -> str:
    frame_span = _parse_frame_span_from_text(response_text)
    if frame_span is None:
        second_span = _parse_seconds_from_text(response_text)
        if second_span is not None:
            frame_span = (round(second_span[0] * output_fps), round(second_span[1] * output_fps))

    if frame_span is None:
        return response_text

    start, end = int(frame_span[0]), int(frame_span[1])
    if end < start:
        start, end = end, start
    start, end = _scale_temporal_span((start, end), input_fps=input_fps, output_fps=output_fps)
    return json.dumps({"temporal_span": [start, end]}, ensure_ascii=False)


class VTimeLLMModel:
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
        self.decode_temperature = temperature if temperature > 0 else float(os.getenv("VTIMELLM_FALLBACK_TEMPERATURE", "0.05"))

        self.model_base = os.getenv("VTIMELLM_MODEL_BASE", "lmsys/vicuna-7b-v1.5")
        self.stage2 = os.getenv("VTIMELLM_STAGE2", "")
        self.stage3 = os.getenv("VTIMELLM_STAGE3", self.model_path)
        self.pretrain_mm_mlp_adapter = os.getenv(
            "VTIMELLM_PRETRAIN_MM_MLP_ADAPTER",
            str(_vtimellm_source_root() / "checkpoints" / "vtimellm-vicuna-v1-5-7b-stage1" / "mm_projector.bin"),
        )
        self.clip_path = os.getenv(
            "VTIMELLM_CLIP_PATH",
            str(_vtimellm_source_root() / "checkpoints" / "clip" / "ViT-L-14.pt"),
        )
        self.conv_mode = os.getenv("VTIMELLM_CONV_MODE", "v1")
        self.max_frames = int(os.getenv("VTIMELLM_MAX_FRAMES", "100"))
        self.max_dense_frames = int(os.getenv("VTIMELLM_MAX_DENSE_FRAMES", "256"))
        self.temporal_input_fps = float(os.getenv("VTIMELLM_TEMPORAL_INPUT_FPS", "2.0"))
        self.temporal_output_fps = float(os.getenv("VTIMELLM_TEMPORAL_OUTPUT_FPS", str(self.temporal_input_fps)))

        self.last_user_prompts = []
        self.last_raw_responses = []

        _ensure_vtimellm_dependencies()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for VTimeLLM.")

        disable_torch_init()
        torch.cuda.set_device(0)
        self.device = torch.device("cuda:0")

        model_args = SimpleNamespace(
            clip_path=self.clip_path,
            model_base=self.model_base,
            pretrain_mm_mlp_adapter=self.pretrain_mm_mlp_adapter,
        )
        self.tokenizer, self.model, _context_len = load_pretrained_model(
            model_args,
            self.stage2 or None,
            self.stage3 or None,
        )
        self.model = self.model.to(device=self.device, dtype=torch.float16)
        self.model.eval()

        self.clip_model, self.image_processor = clip.load(self.clip_path, device=self.device)
        self.clip_model.eval()

        logger.info(
            "Initialized vtimellm | source=%s model_base=%s stage2=%s stage3=%s device=%s max_frames=%s",
            _vtimellm_source_root(),
            self.model_base,
            self.stage2 or None,
            self.stage3 or None,
            self.device,
            self.max_frames,
        )

    def _predict_one(self, query: str, video_path: str, system_prompt: str) -> str:
        del system_prompt

        video_features, _sampled_indices, _fps = _sample_video_features(
            video_path,
            clip_model=self.clip_model,
            image_processor=self.image_processor,
            device=self.device,
            max_frames=self.max_frames,
        )

        question = f"<video>\n {query.strip()}"
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video_features.unsqueeze(0).half().to(self.device),
                do_sample=True,
                temperature=self.decode_temperature,
                num_beams=1,
                max_new_tokens=self.max_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        raw_output = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        if raw_output.endswith(stop_str):
            raw_output = raw_output[: -len(stop_str)]
        return raw_output.strip()

    def predict_batch(self, queries: List[str], video_paths: List[str], system_prompt: str) -> List[str]:
        self.last_user_prompts = list(queries)

        raw_outputs = []
        converted_outputs = []
        for query, video_path in zip(queries, video_paths):
            raw_output = self._predict_one(query, video_path, system_prompt)
            raw_outputs.append(raw_output)
            converted_outputs.append(
                _vtimellm_response_to_json(
                    raw_output,
                    video_path=video_path,
                    max_dense_frames=self.max_dense_frames,
                )
            )

        self.last_raw_responses = raw_outputs
        return converted_outputs

    def predict_temporal_batch(self, queries: List[str], video_paths: List[str], system_prompt: str) -> List[str]:
        self.last_user_prompts = list(queries)

        raw_outputs = []
        converted_outputs = []
        for query, video_path in zip(queries, video_paths):
            raw_output = self._predict_one(query, video_path, system_prompt)
            raw_outputs.append(raw_output)
            converted_outputs.append(
                _vtimellm_response_to_temporal_json(
                    raw_output,
                    input_fps=self.temporal_input_fps,
                    output_fps=self.temporal_output_fps,
                )
            )

        self.last_raw_responses = raw_outputs
        return converted_outputs
