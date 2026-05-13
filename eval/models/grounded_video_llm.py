import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


logger = logging.getLogger(__name__)


def _extract_video_path(video_input: str) -> str:
    return str(video_input or "").split("::split=", 1)[0]


class GroundedVideoLLMModel:
    """Grounded-VideoLLM adapter for forced STVG generation mode."""

    prompt_style = "json"

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

        source_dir = os.getenv("GROUNDED_VIDEO_LLM_SOURCE_DIR")
        if not source_dir:
            raise RuntimeError("GROUNDED_VIDEO_LLM_SOURCE_DIR is required.")
        self.source_dir = Path(source_dir).expanduser().resolve()
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Grounded-Video-LLM source directory not found: {self.source_dir}")

        self.python_bin = os.getenv("GROUNDED_VIDEO_LLM_PYTHON", "python")
        self.inference_py = Path(
            os.getenv("GROUNDED_VIDEO_LLM_INFERENCE_PY", str(self.source_dir / "inference.py"))
        ).expanduser()
        if not self.inference_py.exists():
            raise FileNotFoundError(f"Grounded-Video-LLM inference.py not found: {self.inference_py}")

        self.timeout = float(os.getenv("GROUNDED_VIDEO_LLM_TIMEOUT", "900"))
        self.keep_logs = os.getenv("GROUNDED_VIDEO_LLM_KEEP_LOGS", "0").lower() in {"1", "true", "yes"}

        self.attn_implementation = os.getenv("GROUNDED_VIDEO_LLM_ATTN_IMPLEMENTATION", "eager")
        self.extra_args = os.getenv("GROUNDED_VIDEO_LLM_EXTRA_ARGS", "").strip().split()
        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []

        logger.info(
            "Initialized grounded-video-llm adapter | source=%s inference=%s",
            self.source_dir,
            self.inference_py,
        )

    def _build_cmd(self, prompt: str, video_path: str) -> List[str]:
        cmd = [
            self.python_bin,
            str(self.inference_py),
            "--video_path",
            video_path,
            "--prompt_grounding",
            prompt,
            "--attn_implementation",
            self.attn_implementation,
        ]

        for env_name, arg_name in [
            ("GROUNDED_VIDEO_LLM_DEVICE", "--device"),
            ("GROUNDED_VIDEO_LLM_LLM", "--llm"),
            ("GROUNDED_VIDEO_LLM_CONFIG_PATH", "--config_path"),
            ("GROUNDED_VIDEO_LLM_TOKENIZER_PATH", "--tokenizer_path"),
            ("GROUNDED_VIDEO_LLM_PRETRAINED_VIDEO_PATH", "--pretrained_video_path"),
            ("GROUNDED_VIDEO_LLM_PRETRAINED_VISION_PROJ_LLM_PATH", "--pretrained_vision_proj_llm_path"),
            ("GROUNDED_VIDEO_LLM_CKPT_PATH", "--ckpt_path"),
        ]:
            value = os.getenv(env_name)
            if value:
                cmd.extend([arg_name, value])

        if self.extra_args:
            cmd.extend(self.extra_args)

        return cmd

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(self.source_dir) if not existing else f"{self.source_dir}:{existing}"
        return env

    def _run_one(self, query: str, video_input: str) -> Tuple[str, str]:
        prompt = (query or "").strip()
        if not prompt:
            return "{}", "{}"

        real_video_path = _extract_video_path(video_input)
        cmd = self._build_cmd(prompt=prompt, video_path=real_video_path)
        logger.info("Running Grounded-VideoLLM on %s", Path(real_video_path).name)

        proc = subprocess.run(
            cmd,
            cwd=str(self.source_dir),
            env=self._build_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
        )

        raw_output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if proc.returncode != 0:
            raise RuntimeError(
                "Grounded-VideoLLM inference failed with exit code "
                f"{proc.returncode}:\n{raw_output[-4000:]}"
            )

        if self.keep_logs:
            log_dir = Path(tempfile.mkdtemp(prefix="grounded_video_llm_eval_"))
            (log_dir / "session.log").write_text(raw_output, encoding="utf-8")
            logger.info("Saved Grounded-VideoLLM log to %s", log_dir / "session.log")

        return raw_output, raw_output

    def predict_batch(self, queries: List[str], video_paths: List[str], system_prompt: str) -> List[str]:
        del system_prompt
        self.last_user_prompts = list(queries)
        pairs = [self._run_one(query, video_path) for query, video_path in zip(queries, video_paths)]
        outputs = [output for output, _raw in pairs]
        self.last_raw_responses = [raw for _output, raw in pairs]
        return outputs
