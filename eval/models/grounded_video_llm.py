import logging
import os
import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


def _extract_video_path(video_input: str) -> str:
    return str(video_input or "").split("::split=", 1)[0]


def _normalize_span(span: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return None
    try:
        start = int(round(float(span[0])))
        end = int(round(float(span[1])))
    except (TypeError, ValueError):
        return None
    if end < start:
        start, end = end, start
    return start, end


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
        self.persistent = os.getenv("GROUNDED_VIDEO_LLM_PERSISTENT", "0").lower() in {"1", "true", "yes"}
        self.worker_py = Path(
            os.getenv(
                "GROUNDED_VIDEO_LLM_WORKER_PY",
                str(Path(__file__).resolve().parents[1] / "utils" / "grounded_video_llm_worker.py"),
            )
        ).expanduser()
        self.worker_proc: Optional[subprocess.Popen] = None
        self.temporal_coord_mode = os.getenv("GROUNDED_VIDEO_LLM_TEMPORAL_COORD_MODE", "raw").strip().lower()
        self.temporal_num_bins = int(os.getenv("GROUNDED_VIDEO_LLM_TEMPORAL_NUM_BINS", "12"))
        self.temporal_output_fps = float(os.getenv("GROUNDED_VIDEO_LLM_TEMPORAL_OUTPUT_FPS", "2.0"))

        self.attn_implementation = os.getenv("GROUNDED_VIDEO_LLM_ATTN_IMPLEMENTATION", "eager")
        self.extra_args = os.getenv("GROUNDED_VIDEO_LLM_EXTRA_ARGS", "").strip().split()
        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []

        logger.info(
            "Initialized grounded-video-llm adapter | source=%s inference=%s temporal_coord_mode=%s temporal_num_bins=%s temporal_output_fps=%s",
            self.source_dir,
            self.inference_py,
            self.temporal_coord_mode,
            self.temporal_num_bins,
            self.temporal_output_fps,
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

    def _build_worker_cmd(self) -> List[str]:
        cmd = [
            self.python_bin,
            str(self.worker_py),
            "--source_dir",
            str(self.source_dir),
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

        for env_name, arg_name in [
            ("GROUNDED_VIDEO_LLM_NUM_FRAMES", "--num_frames"),
            ("GROUNDED_VIDEO_LLM_NUM_SEGS", "--num_segs"),
            ("GROUNDED_VIDEO_LLM_NUM_TEMPORAL_TOKENS", "--num_temporal_tokens"),
            ("GROUNDED_VIDEO_LLM_MAX_NEW_TOKENS", "--max_new_tokens"),
            ("GROUNDED_VIDEO_LLM_TEMPERATURE", "--temperature"),
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

    def _read_worker_protocol_line(self) -> Dict[str, Any]:
        assert self.worker_proc is not None and self.worker_proc.stdout is not None
        while True:
            line = self.worker_proc.stdout.readline()
            if not line:
                raise RuntimeError("Grounded-VideoLLM worker exited before returning a protocol message.")
            line = line.strip()
            if not line:
                continue
            if not line.startswith("__DORO_GROUNDED_VIDEO_LLM__ "):
                logger.info("Grounded-VideoLLM worker: %s", line)
                continue
            return json.loads(line.split(" ", 1)[1])

    def _ensure_worker(self) -> None:
        if self.worker_proc is not None and self.worker_proc.poll() is None:
            return
        if not self.worker_py.exists():
            raise FileNotFoundError(f"Grounded-Video-LLM worker not found: {self.worker_py}")

        cmd = self._build_worker_cmd()
        logger.info("Starting persistent Grounded-VideoLLM worker: %s", " ".join(cmd))
        self.worker_proc = subprocess.Popen(
            cmd,
            cwd=str(self.source_dir),
            env=self._build_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        ready = self._read_worker_protocol_line()
        if ready.get("type") != "ready":
            raise RuntimeError(f"Grounded-VideoLLM worker failed to start: {ready}")

    def _run_one_persistent(self, query: str, video_input: str) -> Tuple[str, str]:
        prompt = (query or "").strip()
        if not prompt:
            return "{}", "{}"

        self._ensure_worker()
        assert self.worker_proc is not None and self.worker_proc.stdin is not None
        real_video_path = _extract_video_path(video_input)
        request_id = uuid.uuid4().hex
        payload = {
            "type": "request",
            "id": request_id,
            "video_path": real_video_path,
            "prompt_grounding": prompt,
        }
        logger.info("Running persistent Grounded-VideoLLM on %s", Path(real_video_path).name)
        self.worker_proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.worker_proc.stdin.flush()

        response = self._read_worker_protocol_line()
        if response.get("type") == "error":
            raise RuntimeError(f"Grounded-VideoLLM worker inference failed: {response}")
        if response.get("type") != "response" or response.get("id") != request_id:
            raise RuntimeError(f"Unexpected Grounded-VideoLLM worker response: {response}")
        raw_output = str(response.get("output", ""))
        return raw_output, raw_output

    def _run_one(self, query: str, video_input: str) -> Tuple[str, str]:
        if self.persistent:
            return self._run_one_persistent(query, video_input)

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

    def predict_temporal_batch(self, queries: List[str], video_paths: List[str], system_prompt: str) -> List[str]:
        return self.predict_batch(queries=queries, video_paths=video_paths, system_prompt=system_prompt)

    def close(self) -> None:
        if self.worker_proc is None:
            return
        proc = self.worker_proc
        self.worker_proc = None
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"type": "shutdown"}, ensure_ascii=False) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def map_temporal_span(self, pred_span: Tuple[int, int], sample: Dict[str, Any]) -> Tuple[int, int]:
        span = _normalize_span(pred_span)
        if span is None or self.temporal_coord_mode != "segment":
            return pred_span

        sampled_frames = _sampled_video_frame_count(sample, self.temporal_output_fps)
        if sampled_frames is None or sampled_frames <= 0 or self.temporal_num_bins <= 1:
            return span

        start_bin, end_bin = span
        start_bin = max(0, min(self.temporal_num_bins - 1, start_bin))
        end_bin = max(start_bin, min(self.temporal_num_bins - 1, end_bin))
        sampled_end = max(0, sampled_frames - 1)
        start = int(round(start_bin * sampled_end / (self.temporal_num_bins - 1)))
        end = int(round(end_bin * sampled_end / (self.temporal_num_bins - 1)))
        return start, max(start, end)


def _sampled_video_frame_count(sample: Dict[str, Any], output_fps: float) -> Optional[int]:
    video_path = sample.get("video_path")
    if not video_path or output_fps <= 0:
        return None
    video_path_text = str(video_path).split("::split=", 1)[0]
    try:
        import cv2

        cap = cv2.VideoCapture(video_path_text)
        try:
            if not cap.isOpened():
                raise RuntimeError("OpenCV could not open video")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()
    except Exception:
        probed = _probe_video_frame_count_and_fps(video_path_text)
        if probed is None:
            return None
        frame_count, fps = probed
    if frame_count <= 0 or fps <= 0:
        return None
    return max(1, int(round((frame_count / fps) * output_fps)))


def _probe_video_frame_count_and_fps(video_path: str) -> Optional[Tuple[int, float]]:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,nb_frames,r_frame_rate,avg_frame_rate,duration",
                "-of",
                "json",
                video_path,
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return None
    stream = streams[0]
    frame_count = _first_positive_int(stream.get(key) for key in ("nb_read_frames", "nb_frames"))
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    if (frame_count is None or frame_count <= 0) and fps and fps > 0:
        try:
            duration = float(stream.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0:
            frame_count = int(round(duration * fps))
    if frame_count is None or frame_count <= 0 or fps is None or fps <= 0:
        return None
    return frame_count, fps


def _first_positive_int(values) -> Optional[int]:
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _parse_rate(value: Any) -> Optional[float]:
    if value in (None, "", "0/0"):
        return None
    text = str(value)
    if "/" in text:
        num_text, denom_text = text.split("/", 1)
        try:
            denom = float(denom_text)
            if denom == 0:
                return None
            return float(num_text) / denom
        except (TypeError, ValueError):
            return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None
