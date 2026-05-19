import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .qwen_family import Qwen2_5VL
from utils.stvg_adapter_utils import (
    _extract_json_candidate,
    _extract_video_spec,
    _load_sampled_frames,
    _parse_model_frame_map,
    _remap_frame_map,
    _sample_output_positions,
    _write_sampled_clip,
)


logger = logging.getLogger(__name__)


def _extract_answer_text(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_time_range_seconds(text: str) -> Optional[Tuple[float, float]]:
    if not text:
        return None
    patterns = [
        r"Time\s*range\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:to|-)\s*([0-9]+(?:\.[0-9]+)?)",
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:to|-)\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        start_sec = float(match.group(1))
        end_sec = float(match.group(2))
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        return start_sec, end_sec
    return None


def _build_temporal_only_prediction(
    sampled_indices: List[int],
    clip_fps: float,
    time_range_seconds: Tuple[float, float],
) -> Dict[str, List[float]]:
    if not sampled_indices:
        return {}
    start_sec, end_sec = time_range_seconds
    fps = clip_fps if clip_fps > 0 else 2.0
    start_local = int(round(start_sec * fps))
    end_local = int(round(end_sec * fps))
    start_local = max(0, min(len(sampled_indices) - 1, start_local))
    end_local = max(start_local, min(len(sampled_indices) - 1, end_local))
    return {
        str(int(sampled_indices[start_local])): [0.0, 0.0, 0.99, 0.99],
        str(int(sampled_indices[end_local])): [0.0, 0.0, 0.99, 0.99],
    }


def _build_frame_text_prediction(
    sampled_indices: List[int],
    frame_positions: List[int],
) -> Dict[str, List[float]]:
    if not sampled_indices or not frame_positions:
        return {}
    frame_map: Dict[str, List[float]] = {}
    for pos in sorted(set(frame_positions)):
        if 0 <= pos < len(sampled_indices):
            frame_map[str(int(sampled_indices[pos]))] = [0.0, 0.0, 0.99, 0.99]
    return frame_map


def _build_full_clip_prediction(sampled_indices: List[int]) -> Dict[str, List[float]]:
    if not sampled_indices:
        return {}
    frame_map: Dict[str, List[float]] = {}
    for pos in sorted(set(_sample_output_positions(len(sampled_indices), max_points=8))):
        if 0 <= pos < len(sampled_indices):
            frame_map[str(int(sampled_indices[pos]))] = [0.0, 0.0, 0.99, 0.99]
    return frame_map


def _parse_frame_text_positions(raw_text: str) -> List[int]:
    json_candidate = _extract_json_candidate(raw_text)
    if not json_candidate:
        return []
    try:
        payload = json.loads(json_candidate)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    target_payload = payload.get("target", payload)
    if not isinstance(target_payload, dict):
        return []

    positions: List[int] = []
    for key, value in target_payload.items():
        try:
            frame_pos = int(str(key).strip())
        except ValueError:
            continue
        if isinstance(value, str) and value.strip():
            positions.append(frame_pos)
    return sorted(set(positions))


def _parse_fallback_frames(raw_text: str) -> List[int]:
    json_candidate = _extract_json_candidate(raw_text)
    if not json_candidate:
        return []
    try:
        payload = json.loads(json_candidate)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return []

    parsed_frames: List[int] = []
    for item in frames:
        try:
            parsed_frames.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(parsed_frames))


def _parse_fallback_time_range(raw_text: str) -> Optional[Tuple[float, float]]:
    json_candidate = _extract_json_candidate(raw_text)
    if not json_candidate:
        return None
    try:
        payload = json.loads(json_candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    time_range = payload.get("time_range")
    if not isinstance(time_range, list) or len(time_range) != 2:
        return None
    try:
        start_sec = float(time_range[0])
        end_sec = float(time_range[1])
    except (TypeError, ValueError):
        return None
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    return start_sec, end_sec


class STVGR1(Qwen2_5VL):
    prompt_style = "stvg_r1"

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
        super().__init__(
            model_path=model_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.clip_max_frames = int(os.getenv("STVG_R1_MAX_FRAMES", "32"))
        self.clip_fps = float(os.getenv("STVG_R1_CLIP_FPS", "2.0"))
        self.max_output_frames = int(os.getenv("STVG_R1_MAX_OUTPUT_FRAMES", "8"))
        self.keep_tmp = os.getenv("STVG_R1_KEEP_TMP", "0").lower() in {"1", "true", "yes"}
        self.last_video_frame_indices: List[List[int]] = []

    def _augment_prompt(self, query: str, output_positions: List[int]) -> str:
        index_text = ", ".join(str(i) for i in output_positions)
        suffix = (
            "\n\nOutput requirements:\n"
            "- You may reason in <think>...</think> if needed.\n"
            '- Put the final result in <answer>...</answer>.\n'
            '- Inside <answer>, prefer a strict JSON object with one top-level key "target".\n'
            "- The value of target should map sampled frame indices to bounding boxes.\n"
            "- Use only the sampled frame indices listed below.\n"
            "- If exact boxes are difficult, still return a strict JSON object.\n"
            '- The fallback JSON may use "target_description" and "frames".\n'
            '- If only time is clear, use "time_range": [start_sec, end_sec].\n'
            f"- Allowed sampled frame indices: [{index_text}]\n"
            "- Do not copy any fixed example numbers or frame ids from the prompt.\n"
            "- If you return JSON, use only allowed sampled frame indices as keys.\n"
        )
        return query.rstrip() + suffix

    def _prepare_video_clip(self, video_input: str) -> Tuple[Path, List[int], int, int, Path]:
        video_path, split = _extract_video_spec(video_input)
        frames, sampled_indices, width, height = _load_sampled_frames(
            video_path,
            split=split,
            max_frames=self.clip_max_frames,
        )
        tmp_root = Path(tempfile.mkdtemp(prefix="stvg_r1_eval_"))
        clip_path = tmp_root / "sampled_clip.mp4"
        _write_sampled_clip(frames, clip_path, fps=self.clip_fps)
        return clip_path, sampled_indices, width, height, tmp_root

    def predict_batch(
        self,
        queries: List[str],
        video_paths: List[str],
        system_prompt: str,
    ) -> List[str]:
        from qwen_vl_utils import process_vision_info

        batch_messages = []
        batch_tmp_roots: List[Path] = []
        batch_original_indices: List[List[int]] = []
        batch_width_height: List[Tuple[int, int]] = []
        batch_output_positions: List[List[int]] = []

        try:
            for query, video_path in zip(queries, video_paths):
                clip_path, sampled_indices, width, height, tmp_root = self._prepare_video_clip(video_path)
                output_positions = _sample_output_positions(
                    num_frames=len(sampled_indices),
                    max_points=self.max_output_frames,
                )
                prompt = self._augment_prompt(query, output_positions)
                messages = self.prepare_messages(prompt, str(clip_path), system_prompt)

                batch_messages.append(messages)
                batch_tmp_roots.append(tmp_root)
                batch_original_indices.append(sampled_indices)
                batch_width_height.append((width, height))
                batch_output_positions.append(output_positions)

            self.last_user_prompts = list(queries)
            self.last_video_frame_indices = batch_original_indices

            prompts = [
                self.processor.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for msg in batch_messages
            ]
            _image_inputs, video_inputs, video_kwargs = process_vision_info(
                batch_messages,
                return_video_kwargs=True,
            )

            llm_inputs = []
            for idx, prompt in enumerate(prompts):
                sample_mm_data = {"video": video_inputs[idx]}
                sample_video_kw = {}
                for key, value in video_kwargs.items():
                    sample_video_kw[key] = value[idx] if isinstance(value, (list, tuple)) else value
                llm_inputs.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": sample_mm_data,
                        "mm_processor_kwargs": sample_video_kw,
                    }
                )

            outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
            raw_responses = [output.outputs[0].text for output in outputs]
            self.last_raw_responses = raw_responses

            normalized_outputs: List[str] = []
            for raw_text, original_indices, wh, output_positions in zip(
                raw_responses,
                batch_original_indices,
                batch_width_height,
                batch_output_positions,
            ):
                width, height = wh
                parsed_frame_map = _parse_model_frame_map(raw_text)
                allowed_positions = set(output_positions)
                filtered_frame_map = {
                    frame_idx: coords
                    for frame_idx, coords in parsed_frame_map.items()
                    if frame_idx in allowed_positions
                }
                remapped = _remap_frame_map(
                    filtered_frame_map,
                    original_indices=original_indices,
                    width=width,
                    height=height,
                )

                if not remapped:
                    frame_positions = _parse_fallback_frames(raw_text) or _parse_frame_text_positions(raw_text)
                    if frame_positions:
                        remapped = _build_frame_text_prediction(
                            sampled_indices=original_indices,
                            frame_positions=frame_positions,
                        )

                if not remapped:
                    answer_text = _extract_answer_text(raw_text)
                    time_range_seconds = _parse_fallback_time_range(raw_text) or _parse_time_range_seconds(answer_text)
                    if time_range_seconds is not None:
                        remapped = _build_temporal_only_prediction(
                            sampled_indices=original_indices,
                            clip_fps=self.clip_fps,
                            time_range_seconds=time_range_seconds,
                        )

                if not remapped:
                    remapped = _build_full_clip_prediction(original_indices)

                normalized_outputs.append(json.dumps({"target": remapped}, ensure_ascii=False))

            return normalized_outputs
        finally:
            if not self.keep_tmp:
                for tmp_root in batch_tmp_roots:
                    shutil.rmtree(tmp_root, ignore_errors=True)
