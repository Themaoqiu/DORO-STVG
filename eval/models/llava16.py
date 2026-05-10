import json
import logging
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from decord import VideoReader, cpu
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


logger = logging.getLogger(__name__)
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"


def _extract_json_candidate(response_text: str) -> str | None:
    text = str(response_text or "").strip()
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*([\[{].*[\]}])\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return None
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        return None
    return text[start : end + 1].strip()


def _coerce_box(value: Any) -> List[float] | None:
    if not isinstance(value, list):
        return None

    if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        return [float(v) for v in value]

    for item in value:
        if isinstance(item, list) and len(item) == 4 and all(isinstance(v, (int, float)) for v in item):
            return [float(v) for v in item]
    return None


def _normalize_box(coords: List[float]) -> List[float]:
    if any(abs(v) > 1.0 for v in coords):
        coords = [v / 100.0 if 0.0 <= v <= 100.0 else v for v in coords]
    x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in coords]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _parse_llava16_response(raw_text: str, sampled_indices: List[int]) -> Dict[str, List[float]]:
    candidate = _extract_json_candidate(raw_text)
    if not candidate:
        return {}

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    parsed: Dict[int, List[float]] = {}

    def collect(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                collect(item)
            return

        if not isinstance(node, dict):
            return

        frame_map = node.get("target") if isinstance(node.get("target"), dict) else node
        if not isinstance(frame_map, dict):
            return

        for frame_key, value in frame_map.items():
            try:
                frame_idx = int(str(frame_key).strip())
            except ValueError:
                continue
            box = _coerce_box(value)
            if box is not None and frame_idx not in parsed:
                parsed[frame_idx] = _normalize_box(box)

    collect(payload)

    if not parsed:
        return {}

    sampled_set = set(sampled_indices)
    if not any(frame_idx in sampled_set for frame_idx in parsed) and all(0 <= frame_idx < len(sampled_indices) for frame_idx in parsed):
        return {str(sampled_indices[tile_idx]): box for tile_idx, box in parsed.items()}
    return {str(frame_idx): box for frame_idx, box in parsed.items()}


def _recover_raw_query(query_text: str) -> str:
    text = str(query_text or "").strip()
    match = re.match(r"Where does (.+?) occur in the video\?", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _build_grid_prompt(query_text: str, frame_indices: List[int]) -> str:
    raw_query = _recover_raw_query(query_text)
    allowed = ", ".join(str(idx) for idx in frame_indices)
    return (
        "The image is a grid of sampled frames from one video. "
        "Each tile has its original video frame id printed in the top-left corner.\n"
        f"Target query: {raw_query}\n"
        f"Allowed frame ids: [{allowed}]\n"
        "For each frame where the target is visible, return one normalized box [x1, y1, x2, y2] "
        "inside that tile only.\n"
        "Return only strict JSON in this exact shape:\n"
        "{\"target\": {\"<frame_id>\": [x1, y1, x2, y2]}}\n"
        "Do not repeat examples. Do not explain."
    )


def _sample_video_frames(video_path: str, max_frames: int) -> Tuple[List[np.ndarray], List[int]]:
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Empty video: {video_path}")

    count = max(1, min(max_frames, total))
    if count == 1:
        indices = [0]
    else:
        indices = sorted({round((i / (count - 1)) * (total - 1)) for i in range(count)})
    frames = vr.get_batch(indices).asnumpy()
    return [frames[i] for i in range(len(indices))], [int(i) for i in indices]


def _build_frame_grid(frames: List[np.ndarray], frame_indices: List[int], out_path: Path, columns: int) -> Image.Image:
    if not frames:
        raise ValueError("No frames to build LLaVA-1.6 grid.")

    frame_height, frame_width = frames[0].shape[:2]
    columns = max(1, min(columns, len(frames)))
    rows = int(math.ceil(len(frames) / columns))
    canvas = Image.new("RGB", (columns * frame_width, rows * frame_height), color=(255, 255, 255))
    drawer = ImageDraw.Draw(canvas)

    for idx, frame in enumerate(frames):
        row = idx // columns
        col = idx % columns
        x0 = col * frame_width
        y0 = row * frame_height
        canvas.paste(Image.fromarray(frame).convert("RGB"), (x0, y0))

        label = str(frame_indices[idx])
        label_width = max(42, 10 + 8 * len(label))
        drawer.rectangle([x0 + 6, y0 + 6, x0 + label_width, y0 + 30], fill=(0, 0, 0))
        drawer.text((x0 + 12, y0 + 10), label, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)
    return canvas


class Llava16Model:
    """LLaVA-1.6 image-model wrapper using vLLM for the STVG evaluation pipeline."""

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
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_frames = int(os.getenv("LLAVA16_MAX_FRAMES", "6"))
        self.grid_columns = int(os.getenv("LLAVA16_GRID_COLUMNS", "3"))
        self.keep_tmp = os.getenv("LLAVA16_KEEP_TMP", "0").lower() in {"1", "true", "yes"}
        self.enforce_eager = os.getenv("LLAVA16_ENFORCE_EAGER", "0").lower() in {"1", "true", "yes"}

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=float(os.getenv("LLAVA16_TOP_P", "1.0")),
            max_tokens=self.max_tokens,
            min_tokens=int(os.getenv("LLAVA16_MIN_TOKENS", "1")),
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
            trust_remote_code=True,
            dtype="auto",
            enforce_eager=self.enforce_eager,
        )

        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []
        self.last_video_frame_indices: List[List[int]] = []

    def _apply_chat_template(self, query: str, system_prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{system_prompt}\n\n{query}"},
                ],
            },
        ]
        try:
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"[INST] <image>\n{system_prompt}\n\n{query} [/INST]"

    def _predict_one(self, query: str, video_path: str, system_prompt: str) -> Tuple[str, str, List[int]]:
        tmp_root = Path(tempfile.mkdtemp(prefix="llava16_eval_"))
        try:
            frames, frame_indices = _sample_video_frames(video_path, self.max_frames)
            image = _build_frame_grid(frames, frame_indices, tmp_root / "frame_grid.jpg", self.grid_columns)
            prompt = self._apply_chat_template(_build_grid_prompt(query, frame_indices), system_prompt)
            outputs = self.llm.generate(
                [{"prompt": prompt, "multi_modal_data": {"image": image}}],
                sampling_params=self.sampling_params,
            )
            raw_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            normalized = json.dumps({"target": _parse_llava16_response(raw_text, frame_indices)}, ensure_ascii=False)
            return normalized, raw_text, frame_indices
        finally:
            if not self.keep_tmp:
                shutil.rmtree(tmp_root, ignore_errors=True)

    def predict_batch(self, queries: List[str], video_paths: List[str], system_prompt: str) -> List[str]:
        self.last_user_prompts = list(queries)
        normalized_outputs: List[str] = []
        raw_outputs: List[str] = []
        frame_indices: List[List[int]] = []

        for query, video_path in zip(queries, video_paths):
            normalized, raw_text, indices = self._predict_one(query, video_path, system_prompt)
            normalized_outputs.append(normalized)
            raw_outputs.append(raw_text)
            frame_indices.append(indices)

        self.last_raw_responses = raw_outputs
        self.last_video_frame_indices = frame_indices
        return normalized_outputs
