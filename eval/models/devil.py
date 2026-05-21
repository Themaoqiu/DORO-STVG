import json
import logging
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


class DeViLModel:
    """DeViL adapter using the bundled dependence.devil package.

    Spawns a persistent worker that loads DeViL once and serves JSON requests
    over stdin/stdout. The worker returns the model text plus per-frame
    bounding boxes parsed from the bundled GroundingDINO head.
    """

    DEFAULT_QUERY_TEMPLATE = (
        "Locate the visual content described by the given textual query "
        "<query>{query}</query> in the video. Please output the start and end "
        "timestamps in seconds and the spatial location of the object."
    )

    def __init__(
        self,
        model_path: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.python_bin = os.getenv("DEVIL_PYTHON", "python")
        self.device = os.getenv("DEVIL_DEVICE", "cuda:0")
        self.query_template = os.getenv("DEVIL_QUERY_TEMPLATE", self.DEFAULT_QUERY_TEMPLATE)
        self.eval_root = Path(__file__).resolve().parents[1]
        self.worker_py = self.eval_root / "utils" / "devil_worker.py"

        self.proc: Optional[subprocess.Popen] = None
        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []

        logger.info("Initialized DeViL adapter | eval_root=%s device=%s", self.eval_root, self.device)

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(self.eval_root) if not existing else f"{self.eval_root}{os.pathsep}{existing}"
        return env

    def _read_proto(self) -> Dict[str, Any]:
        assert self.proc is not None and self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("DeViL worker exited unexpectedly.")
            line = line.strip()
            if not line:
                continue
            if not line.startswith("__DORO_DEVIL__ "):
                logger.info("DeViL worker: %s", line)
                continue
            return json.loads(line.split(" ", 1)[1])

    def _ensure_worker(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        cmd = [
            self.python_bin,
            str(self.worker_py),
            "--model_path", self.model_path,
            "--device", self.device,
        ]
        logger.info("Starting DeViL worker: %s", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.eval_root),
            env=self._build_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        msg = self._read_proto()
        if msg.get("type") != "ready":
            raise RuntimeError(f"DeViL worker failed to start: {msg}")

    def _format_query(self, query: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(query or "")).strip()
        return self.query_template.format(query=cleaned)

    def _run_one(self, query: str, video_path: str) -> Tuple[str, str]:
        cleaned = (query or "").strip()
        if not cleaned:
            return "{}", "{}"

        self._ensure_worker()
        assert self.proc is not None and self.proc.stdin is not None
        request_id = uuid.uuid4().hex
        payload = {
            "type": "request",
            "id": request_id,
            "video_path": str(video_path).split("::split=", 1)[0],
            "query": self._format_query(cleaned),
            "do_sample": self.temperature > 0,
            "max_new_tokens": self.max_tokens,
        }
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

        msg = self._read_proto()
        if msg.get("type") == "error":
            raise RuntimeError(f"DeViL worker error: {msg}")
        if msg.get("type") != "response" or msg.get("id") != request_id:
            raise RuntimeError(f"Unexpected DeViL worker response: {msg}")

        raw_text = str(msg.get("text", ""))
        boxes = msg.get("boxes") or {}
        normalized = json.dumps({cleaned: boxes}, ensure_ascii=False) if boxes else "{}"
        return normalized, raw_text

    def predict_batch(
        self,
        queries: List[str],
        video_paths: List[str],
        system_prompt: str,
    ) -> List[str]:
        del system_prompt
        self.last_user_prompts = list(queries)
        outputs: List[str] = []
        raw: List[str] = []
        for query, video_path in zip(queries, video_paths):
            normalized, raw_text = self._run_one(query, video_path)
            outputs.append(normalized)
            raw.append(raw_text)
        self.last_raw_responses = raw
        return outputs
