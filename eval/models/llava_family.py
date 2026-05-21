import os
from typing import List

import numpy as np
import torch

from utils.video_loader import remap_frame_indices, sample_video_uniform


os.environ["DECORD_EOF_RETRY_MAX"] = "20480"


class _BaseLlava:
    FPS_ENV = "LLAVA_FPS"
    DEFAULT_FPS = 2.0
    DEFAULT_MAX_FRAMES = 32

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
        self.fps = float(os.getenv(self.FPS_ENV, str(self.DEFAULT_FPS)))
        # GT annotation fps; defaults to sampling fps. Set EVAL_GT_FPS when
        # GT and model fps differ (e.g. GT@2fps, you sample @1fps).
        self.gt_fps = float(os.getenv("EVAL_GT_FPS", str(self.fps)))
        self.max_frames = int(os.getenv("LLAVA_MAX_FRAMES", str(self.DEFAULT_MAX_FRAMES)))
        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []
        self.load_model()

    def load_model(self):
        raise NotImplementedError

    def predict_batch(
        self,
        queries: List[str],
        video_paths: List[str],
        system_prompt: str,
    ) -> List[str]:
        raise NotImplementedError


class LlavaNextVideo(_BaseLlava):
    """LLaVA-NeXT-Video via vLLM."""

    def load_model(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.001 if self.temperature == 0.0 else 0.9,
            max_tokens=self.max_tokens,
        )
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt={"video": 1},
            trust_remote_code=True,
            dtype="auto",
        )
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception:
            self.processor = None

    def _build_prompt(self, query: str, system_prompt: str) -> str:
        # LLaVA-NeXT-Video's chat_template silently drops system messages, so
        # fold the system prompt into the user turn.
        user_text = f"{system_prompt}\n\n{query}" if system_prompt else query
        if self.processor is not None and getattr(self.processor, "tokenizer", None) is not None:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},
                            {"type": "text", "text": user_text},
                        ],
                    },
                ]
                return self.processor.apply_chat_template(messages, add_generation_prompt=True)
            except Exception:
                pass
        return f"USER: <video>\n{user_text} ASSISTANT:"

    def predict_batch(self, queries, video_paths, system_prompt):
        self.last_user_prompts = list(queries)
        llm_inputs = []
        per_sample_indices: List[List[int]] = []
        for query, video_path in zip(queries, video_paths):
            frames, sampled_indices, _ = sample_video_uniform(
                video_path, fps=self.fps, max_frames=self.max_frames, gt_fps=self.gt_fps
            )
            prompt = self._build_prompt(query, system_prompt)
            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"video": frames},
            })
            per_sample_indices.append(sampled_indices)

        outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
        raw_responses = [out.outputs[0].text for out in outputs]
        self.last_raw_responses = raw_responses
        self.last_video_frame_indices = per_sample_indices
        return [
            remap_frame_indices(resp, sampled)
            for resp, sampled in zip(raw_responses, per_sample_indices)
        ]


class LlavaOneVision1_5(_BaseLlava):
    """LLaVA-OneVision-1.5 via transformers.

    Mirrors the official lmms-eval ``llava_onevision1_5`` chat model:
    - ``AutoModelForCausalLM`` + ``AutoProcessor`` with ``trust_remote_code``.
    - Build messages with ``{"type": "video", "video": path}`` and pass ``fps``
      / ``max_frames`` through ``qwen_vl_utils.process_vision_info``.
    - Optionally cap to ``LLAVA_MAX_FRAMES`` via uniform ``np.linspace`` resampling.

    Frame-index remap is NOT applied here: ``process_vision_info`` decodes the
    video itself, so we don't have a clean sampled-frame ↔ original-frame map.
    The model's frame-key output is in its own sampled space; STVG metrics
    will compare in that space too.
    """

    DEFAULT_MAX_FRAMES = 32
    UNSUPPORTED_INPUT_KEYS = ("second_per_grid_ts",)

    def load_model(self):
        from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM

        self.max_num_frames = int(os.getenv("LLAVA_MAX_FRAMES", str(self.DEFAULT_MAX_FRAMES)))
        self.min_pixels = int(os.getenv("OV1_5_MIN_PIXELS", str(256 * 28 * 28)))
        self.max_pixels = int(os.getenv("OV1_5_MAX_PIXELS", "1605632"))

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            config=config,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
            trust_remote_code=True,
        )
        if self.batch_size > 1 and getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def _build_messages(self, query: str, video_path: str, system_prompt: str):
        video_item = {
            "type": "video",
            "video": video_path,
            "fps": self.fps,
            "max_frames": self.max_num_frames,
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels,
        }
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    video_item,
                    {"type": "text", "text": query},
                ],
            },
        ]

    @torch.inference_mode()
    def predict_batch(self, queries, video_paths, system_prompt):
        from qwen_vl_utils import process_vision_info

        self.last_user_prompts = list(queries)
        all_messages = [
            self._build_messages(q, v, system_prompt) for q, v in zip(queries, video_paths)
        ]
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in all_messages
        ]
        image_inputs, video_inputs = process_vision_info(all_messages)

        # Per lmms-eval: uniformly resample each video to ~max_num_frames and
        # ensure the last frame is included (may produce max_num_frames+1).
        if video_inputs:
            resampled = []
            for vid in video_inputs:
                if isinstance(vid, torch.Tensor) and vid.shape[0] > 1:
                    total = vid.shape[0]
                    indices = np.linspace(0, total - 1, self.max_num_frames, dtype=int)
                    if total - 1 not in indices:
                        indices = np.append(indices, total - 1)
                    vid = vid[indices]
                resampled.append(vid)
            video_inputs = resampled

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # LOV-1.5's forward does not accept Qwen2.5-VL's second_per_grid_ts.
        for key in self.UNSUPPORTED_INPUT_KEYS:
            inputs.pop(key, None)
        inputs = inputs.to(self.model.device)

        gen_kwargs = {"max_new_tokens": self.max_tokens, "do_sample": self.temperature > 0.0}
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature
        gen = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [g[len(i):] for g, i in zip(gen, inputs.input_ids)]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)
        self.last_raw_responses = list(raw)
        return list(raw)


class LlavaOneVision2(_BaseLlava):
    """LLaVA-OneVision-2 via transformers.

    Mirrors the official lmms-eval ``llava_onevision2`` chat model:
    - ``AutoModelForImageTextToText`` + ``AutoProcessor`` (trust_remote_code).
    - Pre-fetch frames + timestamps with ``qwen_vl_utils.fetch_video``.
    - Interleave per-frame ``<t seconds>`` text and ``{"type":"image"}`` items;
      pass PIL frames via ``images=``, ``videos=None``.

    Frame-index remap is NOT applied here: the model is conditioned on
    timestamps (seconds), not frame indices, so its output frame keys are in
    its own sampled space. STVG metrics will compare in that space too.
    """

    DEFAULT_MAX_FRAMES = 32
    TIMESTAMP_DECIMALS = 1

    def load_model(self):
        from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText

        self.max_num_frames = int(os.getenv("LLAVA_MAX_FRAMES", str(self.DEFAULT_MAX_FRAMES)))
        self.min_pixels = int(os.getenv("OV2_MIN_PIXELS", str(28 * 28 * 4)))
        self.max_pixels = int(os.getenv("OV2_MAX_PIXELS", "200704"))

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            config=config,
            trust_remote_code=True,
        )
        for sub in ("video_processor", "image_processor"):
            proc = getattr(self.processor, sub, None)
            if proc is not None:
                if hasattr(proc, "max_pixels"):
                    proc.max_pixels = self.max_pixels
                if hasattr(proc, "min_pixels"):
                    proc.min_pixels = self.min_pixels
        if self.batch_size > 1 and getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        ).eval()

    def _video_to_frames_and_timestamps(self, video_path: str):
        from PIL import Image
        from qwen_vl_utils import fetch_video

        request = {
            "type": "video",
            "video": video_path,
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels,
            "fps": self.fps,
            "max_frames": self.max_num_frames,
        }
        video_input, video_metadata = fetch_video(request, return_video_metadata=True)
        # video_input: torch.Tensor [T,3,H,W] uint8
        timestamps = getattr(video_metadata, "timestamps", None)
        if timestamps is None:
            duration = float(getattr(video_metadata, "duration", 0.0)) or 0.0
            T = video_input.shape[0]
            timestamps = list(np.linspace(0.0, duration, T)) if duration > 0 else list(range(T))

        pil_frames = [
            Image.fromarray(f.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            for f in video_input
        ]
        return pil_frames, [float(t) for t in timestamps]

    def _build_messages(self, query: str, system_prompt: str, pil_frames, timestamps):
        video_content = []
        for img, t in zip(pil_frames, timestamps):
            video_content.append({"type": "text", "text": f"<{t:.{self.TIMESTAMP_DECIMALS}f} seconds>"})
            video_content.append({"type": "image", "image": img})
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": video_content + [{"type": "text", "text": query}],
            },
        ]

    @torch.inference_mode()
    def predict_batch(self, queries, video_paths, system_prompt):
        self.last_user_prompts = list(queries)

        all_messages = []
        all_pil_frames: List[list] = []
        for query, video_path in zip(queries, video_paths):
            pil_frames, timestamps = self._video_to_frames_and_timestamps(video_path)
            all_messages.append(self._build_messages(query, system_prompt, pil_frames, timestamps))
            all_pil_frames.append(pil_frames)

        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in all_messages
        ]
        # Per lmms-eval: per-frame PIL images go via images=, NOT videos=.
        flat_images = [img for frames in all_pil_frames for img in frames]
        inputs = self.processor(
            text=texts,
            images=flat_images,
            videos=None,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        gen_kwargs = {"max_new_tokens": self.max_tokens, "do_sample": self.temperature > 0.0}
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature
        gen = self.model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        trimmed = gen[:, input_len:]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)
        self.last_raw_responses = list(raw)
        return list(raw)
