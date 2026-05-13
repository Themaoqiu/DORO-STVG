import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


def _recover_raw_query(query_text: str) -> str:
    text = str(query_text or "").strip()
    m = re.match(r"Where does (.+?) occur in the video\?", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text


def _extract_video_path(video_input: str) -> str:
    return str(video_input or "").split("::split=", 1)[0]


def _normalize_frame_boxes(frame_map: Any) -> Dict[str, List[float]]:
    out = {}
    if not isinstance(frame_map, dict):
        return out
    for k, coords in frame_map.items():
        try:
            frame = int(str(k).strip())
            vals = [float(v) for v in coords]
        except (TypeError, ValueError):
            continue
        if len(vals) == 4:
            out[str(frame)] = [max(0.0, min(1.0, v)) for v in vals]
    return out


def _normalize_prediction_payload(payload: Any, fallback_description: str = "target") -> str:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return payload
    if not isinstance(payload, dict):
        return "{}"
    raw = payload.get("raw_response")
    if isinstance(raw, str):
        try:
            return _normalize_prediction_payload(json.loads(raw), fallback_description)
        except json.JSONDecodeError:
            return raw
    normalized = {}
    for desc, frame_map in payload.items():
        if desc in {"objects", "prediction", "raw_response", "query", "video_path"}:
            continue
        boxes = _normalize_frame_boxes(frame_map)
        if boxes:
            normalized[str(desc).strip() or fallback_description] = boxes
    return json.dumps(normalized, ensure_ascii=False) if normalized else "{}"


class TASTVGModel:
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
        del batch_size, max_tokens, max_model_len, temperature, tensor_parallel_size, gpu_memory_utilization
        self.model_path = model_path
        self.repo_dir = os.getenv("TASTVG_DIR")
        if not self.repo_dir:
            raise RuntimeError("TASTVG_DIR is required.")

        repo_dir = Path(self.repo_dir).expanduser().resolve()
        if not repo_dir.exists():
            raise FileNotFoundError(f"TASTVG_DIR does not exist: {repo_dir}")
        self.repo_dir = str(repo_dir)

        self.python_bin = os.getenv("TASTVG_PYTHON") or str(repo_dir / ".venv" / "bin" / "python")
        self.util_path = os.getenv("TASTVG_INFER_PY") or str(
            Path(__file__).resolve().parents[1] / "utils" / "tastvg_infer_util.py"
        )
        self.model_weight = os.getenv("TASTVG_MODEL_WEIGHT") or self.model_path
        self.cuda_visible_devices = os.getenv("TASTVG_CUDA_VISIBLE_DEVICES") or os.getenv("CUDA_VISIBLE_DEVICES")
        self.input_resolution = os.getenv("TASTVG_INPUT_RESOLUTION", "224")
        self.num_clip_frames = os.getenv("TASTVG_NUM_CLIP_FRAMES", "16")
        self.num_workers = os.getenv("TASTVG_NUM_WORKERS", "0")
        self.config_file = os.getenv("TASTVG_CONFIG_FILE")
        self.test_script = os.getenv("TASTVG_TEST_SCRIPT")
        self.result_file = os.getenv("TASTVG_RESULT_FILE")
        self.prepare_cache = os.getenv("TASTVG_PREPARE_CACHE", "0").lower() in {"1", "true", "yes", "on"}
        self.official_extra_opts = os.getenv("TASTVG_OFFICIAL_EXTRA_OPTS")
        self.extra_args = os.getenv("TASTVG_EXTRA_ARGS", "").strip()
        self.keep_tmp = os.getenv("TASTVG_KEEP_TMP", "0").lower() in {"1", "true", "yes"}
        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []

    def _run_util(self, manifest_path: Path, output_path: Path) -> List[Dict[str, Any]]:
        cmd = [
            self.python_bin,
            str(Path(self.util_path).expanduser().resolve()),
            "--manifest", str(manifest_path),
            "--output", str(output_path),
            "--tastvg-dir", self.repo_dir,
            "--model-weight", self.model_weight,
            "--input-resolution", self.input_resolution,
            "--num-clip-frames", self.num_clip_frames,
            "--num-workers", self.num_workers,
        ]
        if self.config_file:
            cmd.extend(["--config-file", self.config_file])
        if self.test_script:
            cmd.extend(["--test-script", self.test_script])
        if self.result_file:
            cmd.extend(["--result-file", self.result_file])
        if self.prepare_cache:
            cmd.append("--prepare-cache")
        if self.official_extra_opts:
            cmd.extend(["--official-extra-opts", self.official_extra_opts])
        if self.extra_args:
            cmd.extend(self.extra_args.split())

        env = os.environ.copy()
        if self.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices

        logger.info("Running TA-STVG util: %s", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=self.repo_dir, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.stdout.strip():
            logger.info("TA-STVG util stdout:\n%s", proc.stdout.strip())
        if proc.stderr.strip():
            logger.warning("TA-STVG util stderr:\n%s", proc.stderr.strip())
        if proc.returncode != 0:
            raise RuntimeError(
                f"TA-STVG util failed.\nCommand: {' '.join(cmd)}\nReturn code: {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        rows: List[Dict[str, Any]] = []
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def predict_batch(self, queries: List[str], video_paths: List[str], system_prompt: str, video_metas: List[Dict[str, Any]] | None = None) -> List[str]:
        del system_prompt
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix="tastvg_eval_")
        tmp_dir = Path(tmp_dir_obj.name)
        manifest_path = tmp_dir / "manifest.jsonl"
        output_path = tmp_dir / "predictions.jsonl"

        self.last_user_prompts = [_recover_raw_query(query) for query in queries]
        with manifest_path.open("w", encoding="utf-8") as f:
            metas = video_metas or [{} for _ in video_paths]
            for query, video_path, meta in zip(self.last_user_prompts, video_paths, metas):
                f.write(json.dumps({
                    "query": query,
                    "video_path": _extract_video_path(video_path),
                    "video_input_path": video_path,
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "queryid": meta.get("queryid"),
                }, ensure_ascii=False) + "\n")

        try:
            rows = self._run_util(manifest_path, output_path)
            outputs, raw_responses = [], []
            for idx, row in enumerate(rows):
                fallback = self.last_user_prompts[idx] if idx < len(self.last_user_prompts) else "target"
                normalized = _normalize_prediction_payload(row, fallback_description=fallback)
                outputs.append(normalized)
                raw_responses.append(str(row.get("raw_response") or normalized) if isinstance(row, dict) else normalized)
            while len(outputs) < len(queries):
                outputs.append("{}")
                raw_responses.append("{}")
            self.last_raw_responses = raw_responses
            return outputs[:len(queries)]
        finally:
            tmp_dir_obj.cleanup()
