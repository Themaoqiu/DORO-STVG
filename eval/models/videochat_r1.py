import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .qwen_family import Qwen2_5VL
from utils.stvg_adapter_utils import (
    _extract_video_spec,
    _load_sampled_frames,
    _parse_model_frame_map,
    _remap_frame_map,
    _write_sampled_clip,
)


logger = logging.getLogger(__name__)


class VideoChatR1(Qwen2_5VL):
    prompt_style = "videochat_r1"

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
        self.clip_max_frames = int(os.getenv("VIDEOCHAT_R1_MAX_FRAMES", "32"))
        self.clip_fps = float(os.getenv("VIDEOCHAT_R1_CLIP_FPS", "2.0"))
        self.keep_tmp = os.getenv("VIDEOCHAT_R1_KEEP_TMP", "0").lower() in {"1", "true", "yes"}
        self.last_video_frame_indices: List[List[int]] = []

    def _prepare_video_clip(self, video_input: str) -> Tuple[Path, List[int], int, int, Path]:
        video_path, split = _extract_video_spec(video_input)
        frames, sampled_indices, width, height = _load_sampled_frames(
            video_path,
            split=split,
            max_frames=self.clip_max_frames,
        )
        tmp_root = Path(tempfile.mkdtemp(prefix="videochat_r1_eval_"))
        clip_path = tmp_root / "sampled_clip.mp4"
        _write_sampled_clip(frames, clip_path, fps=self.clip_fps)
        return clip_path, sampled_indices, width, height, tmp_root

    def predict_batch(self, queries: List[str], video_paths: List[str], system_prompt: str) -> List[str]:
        from qwen_vl_utils import process_vision_info

        batch_messages = []
        batch_tmp_roots: List[Path] = []
        batch_original_indices: List[List[int]] = []
        batch_width_height: List[Tuple[int, int]] = []

        try:
            for query, video_path in zip(queries, video_paths):
                clip_path, sampled_indices, width, height, tmp_root = self._prepare_video_clip(video_path)
                messages = self.prepare_messages(query, str(clip_path), system_prompt)
                batch_messages.append(messages)
                batch_tmp_roots.append(tmp_root)
                batch_original_indices.append(sampled_indices)
                batch_width_height.append((width, height))

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
                sample_video_kw: Dict[str, Any] = {}
                for key, value in video_kwargs.items():
                    sample_video_kw[key] = value[idx] if isinstance(value, (list, tuple)) else value
                llm_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": sample_mm_data,
                    "mm_processor_kwargs": sample_video_kw,
                })

            outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
            raw_responses = [output.outputs[0].text for output in outputs]
            self.last_raw_responses = raw_responses

            normalized_outputs = []
            for raw_text, original_indices, wh in zip(raw_responses, batch_original_indices, batch_width_height):
                width, height = wh
                parsed_frame_map = _parse_model_frame_map(raw_text)
                remapped = _remap_frame_map(
                    parsed_frame_map,
                    original_indices=original_indices,
                    width=width,
                    height=height,
                )
                normalized_outputs.append(json.dumps({"target": remapped}, ensure_ascii=False))
            return normalized_outputs
        finally:
            if not self.keep_tmp:
                for tmp_root in batch_tmp_roots:
                    shutil.rmtree(tmp_root, ignore_errors=True)
