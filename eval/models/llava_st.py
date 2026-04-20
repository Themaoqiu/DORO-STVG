import os
from typing import Any, Dict, List

import numpy as np
from decord import VideoReader, cpu
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


os.environ["DECORD_EOF_RETRY_MAX"] = "20480"


def _load_video_frames(video_path: str, max_frames: int = 48) -> np.ndarray:
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Empty video: {video_path}")

    sample_count = min(max_frames, total)
    indices = np.linspace(0, total - 1, sample_count, dtype=np.int64)
    frames = vr.get_batch(indices.tolist()).asnumpy()
    return frames


class LlavaSTQwen2:
    """LLaVA-ST Qwen2 model wrapper for the STVG evaluation pipeline."""

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

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.001,
            max_tokens=self.max_tokens,
            stop_token_ids=[],
        )

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1, "video": 1},
            trust_remote_code=True,
            dtype="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []

    def _build_messages(self, query: str, video_path: str, system_prompt: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": query},
                ],
            },
        ]

    def _prepare_llm_input(self, query: str, video_path: str, system_prompt: str) -> Dict[str, Any]:
        messages = self._build_messages(query, video_path, system_prompt)
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        try:
            from qwen_vl_utils import process_vision_info

            image_patch_size = None
            if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "patch_size"):
                image_patch_size = self.processor.image_processor.patch_size

            kwargs: Dict[str, Any] = {
                "return_video_kwargs": True,
                "return_video_metadata": True,
            }
            if image_patch_size is not None:
                kwargs["image_patch_size"] = image_patch_size

            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, **kwargs)

            mm_data: Dict[str, Any] = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            if not mm_data:
                raise ValueError("No multimodal inputs parsed from process_vision_info.")

            return {
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            }
        except Exception:
            # Fallback path for models whose processor is not compatible with qwen_vl_utils.
            video_frames = _load_video_frames(video_path)
            return {
                "prompt": prompt,
                "multi_modal_data": {"video": video_frames},
            }

    def predict_batch(
        self,
        queries: List[str],
        video_paths: List[str],
        system_prompt: str,
    ) -> List[str]:
        self.last_user_prompts = list(queries)

        llm_inputs = [
            self._prepare_llm_input(query, video_path, system_prompt)
            for query, video_path in zip(queries, video_paths)
        ]

        outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
        raw_responses = [output.outputs[0].text for output in outputs]
        self.last_raw_responses = raw_responses
        return raw_responses
