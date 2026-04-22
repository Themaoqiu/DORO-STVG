import copy
import logging
import os
from typing import List

import numpy as np
import torch
from decord import VideoReader, cpu

os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
logger = logging.getLogger(__name__)


def _load_video_frames(video_path: str, max_frames: int) -> np.ndarray:
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Empty video: {video_path}")

    sample_count = min(max_frames, total)
    indices = np.linspace(0, total - 1, sample_count, dtype=np.int64)
    return vr.get_batch(indices.tolist()).asnumpy()


class LlavaSTQwen2:
    """LLaVA-ST-Qwen2 wrapper for the evaluation pipeline."""

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
        # Reserved for constructor parity with other backends.
        del max_model_len, tensor_parallel_size, gpu_memory_utilization

        self.model_path = model_path
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_frames = int(os.getenv("LLAVA_ST_MAX_FRAMES", "100"))
        self.vt_chunk = int(os.getenv("LLAVA_ST_VT_CHUNK", "1"))
        self.use_cache = os.getenv("LLAVA_ST_USE_CACHE", "0").lower() in {"1", "true", "yes"}

        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for llava-st-qwen2 inference.")

        torch.cuda.set_device(0)
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16

        from llava.conversation import conv_templates
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        self.conv_templates = conv_templates
        self.conv_template = "qwen_1_5" if "qwen_1_5" in conv_templates else next(iter(conv_templates.keys()))

        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_path,
            None,
            model_name,
            device_map=None,
            dtype=self.dtype,
            attn_implementation="eager",
        )

        tok_size = len(self.tokenizer)
        emb_size = self.model.get_input_embeddings().weight.shape[0]
        if emb_size != tok_size:
            self.model.resize_token_embeddings(tok_size)

        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        base = self.model.get_model() if hasattr(self.model, "get_model") else self.model
        if hasattr(base, "mm_projector") and base.mm_projector is not None:
            base.mm_projector = base.mm_projector.to(self.device, dtype=self.dtype)

        vt = base.get_vision_tower() if hasattr(base, "get_vision_tower") else getattr(base, "vision_tower", None)
        if vt is not None:
            if hasattr(vt, "to"):
                vt.to(self.device, dtype=self.dtype)
            if hasattr(vt, "vision_tower") and vt.vision_tower is not None and hasattr(vt.vision_tower, "to"):
                vt.vision_tower.to(self.device, dtype=self.dtype)

            # Reduce peak memory on consumer GPUs by chunking visual forward.
            if self.vt_chunk > 0 and hasattr(vt, "forward"):
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
            "Initialized llava-st-qwen2 | device=%s dtype=%s max_frames=%s vt_chunk=%s",
            self.device,
            self.dtype,
            self.max_frames,
            self.vt_chunk,
        )

    def _build_prompt(self, query: str, system_prompt: str) -> str:
        from llava.constants import DEFAULT_IMAGE_TOKEN

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{system_prompt}\n{query}")
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _predict_one(self, query: str, video_path: str, system_prompt: str) -> str:
        frames = _load_video_frames(video_path, max_frames=self.max_frames)
        video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        video_tensor = video_tensor.to(device=self.device, dtype=self.dtype, non_blocking=True)

        prompt = self._build_prompt(query, system_prompt)
        tokenized = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device) if "attention_mask" in tokenized else None

        vocab = self.model.get_input_embeddings().weight.shape[0]
        unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
        input_ids = torch.where(
            (input_ids < 0) | (input_ids >= vocab),
            torch.full_like(input_ids, unk_id),
            input_ids,
        )

        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            "use_cache": self.use_cache,
        }
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = max(self.temperature, 1e-5)
        else:
            gen_kwargs["do_sample"] = False

        call_kwargs = dict(gen_kwargs)
        if attention_mask is not None:
            call_kwargs["attention_mask"] = attention_mask

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[video_tensor],
                modalities=["video"],
                **call_kwargs,
            )

        gen_ids = output_ids[:, input_ids.shape[1] :]
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    def predict_batch(self, queries: List[str], video_paths: List[str], system_prompt: str) -> List[str]:
        self.last_user_prompts = list(queries)
        outputs = [self._predict_one(q, vp, system_prompt) for q, vp in zip(queries, video_paths)]
        self.last_raw_responses = outputs
        return outputs

