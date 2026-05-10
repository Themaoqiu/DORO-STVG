import json
import logging
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


def _extract_video_path(video_input: str) -> str:
    return str(video_input or "").split("::split=", 1)[0]


def _extract_json_candidate(text: str) -> Optional[str]:
    if not text:
        return None

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1].strip()


def _extract_json_prefix(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(text)):
        char = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1].strip()

    return None


def _one_line_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_frame_boxes(frame_map) -> Dict[str, List[float]]:
    normalized: Dict[str, List[float]] = {}
    if not isinstance(frame_map, dict):
        return normalized

    for frame_idx_text, coords in frame_map.items():
        try:
            frame_idx = int(str(frame_idx_text).strip())
            values = [float(v) for v in coords]
        except (TypeError, ValueError):
            continue

        if len(values) != 4:
            continue

        normalized[str(frame_idx)] = [max(0.0, min(1.0, v)) for v in values]

    return normalized


def _normalize_response_text(text: str, fallback_description: str) -> str:
    candidate = _extract_json_prefix(text) or _extract_json_candidate(text)
    if not candidate:
        return "{}"

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return "{}"

    if not isinstance(payload, dict):
        return "{}"

    normalized: Dict[str, Dict[str, List[float]]] = {}
    for description, frame_map in payload.items():
        cleaned = _normalize_frame_boxes(frame_map)
        if cleaned:
            key = str(description).strip() or fallback_description
            normalized[key] = cleaned

    if normalized:
        return json.dumps(normalized, ensure_ascii=False)

    cleaned_single = _normalize_frame_boxes(payload)
    if cleaned_single:
        return json.dumps({fallback_description: cleaned_single}, ensure_ascii=False)

    return "{}"


def _tail_text(text: str, limit: int = 4000) -> str:
    if not text:
        return ""
    return text[-limit:]


def _enabled_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


class GroundingGPTModel:
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
        self.keep_logs = os.getenv("GROUNDINGGPT_KEEP_LOGS", "0").lower() in {"1", "true", "yes"}

        source_dir = os.getenv("GROUNDINGGPT_SOURCE_DIR") or os.getenv("GROUNDINGGPT_REPO")
        if not source_dir:
            raise RuntimeError(
                "GROUNDINGGPT_SOURCE_DIR is required. Set it to the GroundingGPT code checkout "
                "prepared by the evaluation environment, while MODEL_PATH points to the external model weights."
            )
        self.source_dir = Path(source_dir).expanduser().resolve()
        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"GroundingGPT source directory not found at {self.source_dir}. "
                "Clone https://github.com/lzw-lzw/GroundingGPT outside this evaluation framework "
                "or provide a prepared source checkout through GROUNDINGGPT_SOURCE_DIR."
            )

        python_bin = os.getenv("GROUNDINGGPT_PYTHON")
        if not python_bin:
            raise RuntimeError("GROUNDINGGPT_PYTHON is required (python executable with GroundingGPT dependencies).")
        self.python_bin = str(Path(python_bin).expanduser())

        self.cli_py = self.source_dir / "lego" / "serve" / "cli.py"
        if not self.cli_py.exists():
            raise FileNotFoundError(f"GroundingGPT cli.py not found at {self.cli_py}")

        self.default_max_new_tokens = int(os.getenv("GROUNDINGGPT_MAX_NEW_TOKENS", str(max(self.max_tokens, 1024))))
        self.default_temperature = float(os.getenv("GROUNDINGGPT_TEMPERATURE", str(self.temperature)))
        self.cuda_visible_devices = os.getenv("GROUNDINGGPT_CUDA_VISIBLE_DEVICES") or os.getenv("CUDA_VISIBLE_DEVICES")
        self.persistent_cli = _enabled_env("GROUNDINGGPT_PERSISTENT_CLI", "1")
        self.cli_timeout = float(os.getenv("GROUNDINGGPT_CLI_TIMEOUT", "600"))

        self._proc: Optional[subprocess.Popen] = None
        self._stdout_queue: "queue.Queue[str]" = queue.Queue()
        self._stderr_chunks: List[str] = []
        self._session_chunks: List[str] = []

        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []
        logger.info("Initialized groundinggpt adapter | source=%s cli=%s", self.source_dir, self.cli_py)

    def __del__(self):
        try:
            self._stop_cli()
        except Exception:
            pass

    def _build_cmd(self) -> List[str]:
        return [
            self.python_bin,
            str(self.cli_py),
            "--model_path",
            self.model_path,
            "--temperature",
            str(self.default_temperature),
            "--max_new_tokens",
            str(self.default_max_new_tokens),
        ]

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.source_dir)
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
            env.pop(key, None)
        if self.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        return env

    @staticmethod
    def _read_stream(stream, out_queue: Optional["queue.Queue[str]"] = None, chunks: Optional[List[str]] = None) -> None:
        while True:
            char = stream.read(1)
            if not char:
                break
            if out_queue is not None:
                out_queue.put(char)
            if chunks is not None:
                chunks.append(char)
                if len(chunks) > 200000:
                    del chunks[:100000]

    def _read_until(self, patterns: List[str], timeout: float) -> str:
        deadline = time.time() + timeout
        chunks: List[str] = []
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                stderr_tail = _tail_text("".join(self._stderr_chunks))
                raise RuntimeError(
                    f"GroundingGPT CLI exited unexpectedly with code {self._proc.returncode}.\n"
                    f"STDERR tail:\n{stderr_tail}"
                )
            try:
                char = self._stdout_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            chunks.append(char)
            text = "".join(chunks)
            if any(pattern in text for pattern in patterns):
                self._session_chunks.append(text)
                return text
        tail = _tail_text("".join(chunks))
        raise TimeoutError(f"Timed out waiting for GroundingGPT prompt {patterns}. STDOUT tail:\n{tail}")

    def _start_cli(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        self._stdout_queue = queue.Queue()
        self._stderr_chunks = []
        self._session_chunks = []
        self._proc = subprocess.Popen(
            self._build_cmd(),
            cwd=str(self.source_dir),
            env=self._build_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=0,
        )
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        threading.Thread(target=self._read_stream, args=(self._proc.stdout, self._stdout_queue, None), daemon=True).start()
        threading.Thread(target=self._read_stream, args=(self._proc.stderr, None, self._stderr_chunks), daemon=True).start()
        self._read_until(["Human:"], timeout=self.cli_timeout)

    def _stop_cli(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write("exit\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.terminate()
        self._proc = None

    def _write_cli(self, text: str) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.poll() is not None:
            raise RuntimeError("GroundingGPT CLI is not running.")
        self._proc.stdin.write(text + "\n")
        self._proc.stdin.flush()

    def _run_persistent(self, prompt: str, real_video_path: str) -> str:
        self._start_cli()
        self._write_cli("change video")
        out = self._read_until(["Please input new video path:"], timeout=self.cli_timeout)
        self._write_cli(str(real_video_path))
        out += self._read_until(["Human:"], timeout=self.cli_timeout)
        self._write_cli(_one_line_prompt(prompt))
        out += self._read_until(["Human:"], timeout=self.cli_timeout)
        return out

    def _run_subprocess(self, prompt: str, real_video_path: str) -> str:
        session_input = "\n".join(
            [
                "change video",
                str(real_video_path),
                _one_line_prompt(prompt),
                "",
            ]
        )

        logger.info("Running GroundingGPT CLI on %s", Path(real_video_path).name)
        try:
            proc = subprocess.run(
                self._build_cmd(),
                cwd=str(self.source_dir),
                env=self._build_env(),
                input=session_input,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stdout_tail = _tail_text(exc.stdout or "")
            stderr_tail = _tail_text(exc.stderr or "")
            logger.error(
                "GroundingGPT CLI failed on %s | returncode=%s\nSTDOUT tail:\n%s\nSTDERR tail:\n%s",
                Path(real_video_path).name,
                exc.returncode,
                stdout_tail,
                stderr_tail,
            )
            raise

        return ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()

    def _run_one(self, query: str, video_path: str) -> Tuple[str, str]:
        prompt = (query or "").strip()
        if not prompt:
            return "{}", "{}"

        real_video_path = _extract_video_path(video_path)
        if real_video_path != video_path:
            logger.info(
                "GroundingGPT CLI does not support split video markers; falling back to full video %s",
                real_video_path,
            )

        logger.info("Running GroundingGPT CLI on %s", Path(real_video_path).name)
        if self.persistent_cli:
            full_output = self._run_persistent(prompt, real_video_path)
        else:
            full_output = self._run_subprocess(prompt, real_video_path)

        if self.keep_logs:
            log_dir = Path(tempfile.mkdtemp(prefix="groundinggpt_eval_"))
            (log_dir / "session.log").write_text(full_output, encoding="utf-8")
            logger.info("Saved GroundingGPT session log to %s", log_dir / "session.log")

        normalized = _normalize_response_text(full_output, fallback_description="target")
        if normalized != "{}":
            return normalized, full_output

        lines = [line.strip() for line in full_output.splitlines() if line.strip()]
        if not lines:
            return "{}", full_output

        response_text = lines[-1]
        if response_text.lower() == "exit...":
            response_text = lines[-2] if len(lines) >= 2 else "{}"

        return _normalize_response_text(response_text, fallback_description="target"), full_output

    def predict_batch(self, queries: List[str], video_paths: List[str], system_prompt: str) -> List[str]:
        del system_prompt

        self.last_user_prompts = list(queries)
        pairs = [self._run_one(query, video_path) for query, video_path in zip(queries, video_paths)]
        outputs = [normalized for normalized, _raw in pairs]
        self.last_raw_responses = [raw for _normalized, raw in pairs]
        if self.keep_logs and self.persistent_cli and self._session_chunks:
            log_dir = Path(tempfile.mkdtemp(prefix="groundinggpt_eval_session_"))
            (log_dir / "session.log").write_text("".join(self._session_chunks), encoding="utf-8")
            (log_dir / "stderr.log").write_text("".join(self._stderr_chunks), encoding="utf-8")
            logger.info("Saved GroundingGPT persistent session logs to %s", log_dir)
        return outputs
