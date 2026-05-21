import os
from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.video_loader import sample_video_uniform


os.environ["DECORD_EOF_RETRY_MAX"] = "20480"


class InternVLBase:
    """Base class for InternVL3 / InternVL3.5 via vLLM.

    Frames are sampled at the configured fps and fed as a list of PIL.Image
    under the "image" multi-modal key, with one `<image>` placeholder per
    frame. Same code path across InternVL3 and InternVL3.5 regardless of
    text backbone.
    """

    FPS_ENV = "INTERNVL_FPS"
    DEFAULT_FPS = 2.0
    # vLLM scheduling cap; not a sampling cap. Set high enough that fps
    # sampling never exceeds it for typical clips.
    IMAGE_SLOT_LIMIT = 256

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

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.001 if self.temperature == 0.0 else 0.9,
            max_tokens=self.max_tokens,
        )
        self.llm = None
        self.tokenizer = None
        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []
        self.load_model()

    def load_model(self):
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt={"image": self.IMAGE_SLOT_LIMIT},
            trust_remote_code=True,
            dtype="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def _build_prompt(self, query: str, system_prompt: str, num_frames: int) -> str:
        frame_tokens = "\n".join(f"Frame{idx + 1}: <image>" for idx in range(num_frames))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{frame_tokens}\n{query}"},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def predict_batch(
        self,
        queries: List[str],
        video_paths: List[str],
        system_prompt: str,
    ) -> List[str]:
        from PIL import Image

        self.last_user_prompts = list(queries)
        llm_inputs = []
        for query, video_path in zip(queries, video_paths):
            frames, _, _ = sample_video_uniform(video_path, fps=self.fps)
            pil_frames = [Image.fromarray(f).convert("RGB") for f in frames]
            prompt = self._build_prompt(query, system_prompt, len(pil_frames))
            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": pil_frames},
            })

        outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
        raw_responses = [out.outputs[0].text for out in outputs]
        self.last_raw_responses = raw_responses
        return raw_responses


class InternVL3(InternVLBase):
    pass


class InternVL3_5(InternVLBase):
    pass
