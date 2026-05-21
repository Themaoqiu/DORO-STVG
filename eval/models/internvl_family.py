import os
from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.video_loader import remap_frame_indices, sample_video_uniform


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
    # Hard cap on frames per video; if fps sampling produces more, the
    # sequence is uniformly downsampled (keeping first/last).
    DEFAULT_MAX_FRAMES = 64
    # vLLM scheduling cap; not a sampling cap. Set high enough that fps
    # sampling never exceeds it for typical clips.
    IMAGE_SLOT_LIMIT = 256
    # Native tile size InternVL ViT consumes. Resizing each frame to this
    # square forces dynamic_preprocess to emit a single 256-token tile (no
    # extra tiles, since aspect ratio matches 1:1 exactly).
    TILE_SIZE = 448

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
        # GT annotation fps. If the dataset was annotated at a different fps
        # than what we feed the model (e.g. dataset@2fps, model@1fps), set
        # EVAL_GT_FPS to the annotation's fps so frame indices map back to
        # GT space. Defaults to the sampling fps (typical case).
        self.gt_fps = float(os.getenv("EVAL_GT_FPS", str(self.fps)))
        self.max_frames = int(os.getenv("INTERNVL_MAX_FRAMES", str(self.DEFAULT_MAX_FRAMES)))
        self.tile_size = int(os.getenv("INTERNVL_TILE_SIZE", str(self.TILE_SIZE)))

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
        # Matches VLMEvalKit's video prompt: "Frame-i: <image>\n...\n{query}".
        frame_tokens = "\n".join(f"Frame-{idx + 1}: <image>" for idx in range(num_frames))
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
        per_sample_indices: List[List[int]] = []
        for query, video_path in zip(queries, video_paths):
            frames, sampled_indices, _ = sample_video_uniform(
                video_path, fps=self.fps, max_frames=self.max_frames, gt_fps=self.gt_fps
            )
            pil_frames = [
                Image.fromarray(f).convert("RGB").resize(
                    (self.tile_size, self.tile_size), Image.BICUBIC
                )
                for f in frames
            ]
            prompt = self._build_prompt(query, system_prompt, len(pil_frames))
            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": pil_frames},
            })
            per_sample_indices.append(sampled_indices)

        outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
        raw_responses = [out.outputs[0].text for out in outputs]
        self.last_raw_responses = raw_responses
        self.last_video_frame_indices = per_sample_indices

        # Prompt uses "Frame-1..K" over sampled frames; remap the model's
        # 1..K (or 0..K-1) keys back to original-video frame indices so STVG
        # metrics align with GT (mirrors llava_st's _llava_st_tokens_to_json).
        return [
            remap_frame_indices(resp, sampled)
            for resp, sampled in zip(raw_responses, per_sample_indices)
        ]


class InternVL3(InternVLBase):
    pass


class InternVL3_5(InternVLBase):
    pass
