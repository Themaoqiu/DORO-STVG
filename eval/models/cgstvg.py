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
    match = re.match(r"Where does (.+?) occur in the video\?", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def _extract_video_path(video_input: str) -> str:
    return str(video_input or "").split("::split=", 1)[0]


def _normalize_frame_boxes(frame_map: Any) -> Dict[str, List[float]]:
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


def _normalize_prediction_payload(payload: Any, fallback_description: str = "target") -> str:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return payload
    if not isinstance(payload, dict):
        return "{}"
    raw_response = payload.get("raw_response")
    if isinstance(raw_response, str):
        try:
            return _normalize_prediction_payload(json.loads(raw_response), fallback_description)
        except json.JSONDecodeError:
            return raw_response
    normalized: Dict[str, Dict[str, List[float]]] = {}
    objects = payload.get("objects")
    if isinstance(objects, list):
        for item in objects:
            if not isinstance(item, dict):
                continue
            description = str(item.get("description") or fallback_description).strip() or fallback_description
            boxes = _normalize_frame_boxes(item.get("spatial_bboxes"))
            if boxes:
                normalized[description] = boxes
        if normalized:
            return json.dumps(normalized, ensure_ascii=False)
    prediction = payload.get("prediction")
    if isinstance(prediction, dict):
        for description, frame_map in prediction.items():
            boxes = _normalize_frame_boxes(frame_map)
            if boxes:
                normalized[str(description).strip() or fallback_description] = boxes
        if normalized:
            return json.dumps(normalized, ensure_ascii=False)
    for description, frame_map in payload.items():
        if description in {"objects", "prediction", "raw_response", "query", "video_path"}:
            continue
        boxes = _normalize_frame_boxes(frame_map)
        if boxes:
            normalized[str(description).strip() or fallback_description] = boxes
    if normalized:
        return json.dumps(normalized, ensure_ascii=False)
    return "{}"


class CGSTVGModel:
    prompt_style = "json"
    env_prefix = "CGSTVG"
    model_label = "CGSTVG"
    repo_flag = "--cgstvg-dir"
    default_util_name = "cgstvg_infer_util.py"

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
        prefix = self.env_prefix
        self.repo_dir = os.getenv(f"{prefix}_DIR")
        if not self.repo_dir:
            raise RuntimeError(f"{prefix}_DIR is required.")

        repo_dir = Path(self.repo_dir).expanduser().resolve()
        if not repo_dir.exists():
            raise FileNotFoundError(f"{prefix}_DIR does not exist: {repo_dir}")
        self.repo_dir = str(repo_dir)

        self.python_bin = os.getenv(f"{prefix}_PYTHON") or str(repo_dir / ".venv" / "bin" / "python")
        self.util_path = os.getenv(f"{prefix}_INFER_PY") or str(
            Path(__file__).resolve().parents[1] / "utils" / self.default_util_name
        )
        self.model_weight = os.getenv(f"{prefix}_MODEL_WEIGHT") or self.model_path
        self.cuda_visible_devices = os.getenv(f"{prefix}_CUDA_VISIBLE_DEVICES") or os.getenv("CUDA_VISIBLE_DEVICES")
        self.input_resolution = os.getenv(f"{prefix}_INPUT_RESOLUTION", "224")
        self.num_clip_frames = os.getenv(f"{prefix}_NUM_CLIP_FRAMES", "16")
        self.num_workers = os.getenv(f"{prefix}_NUM_WORKERS", "0")
        self.extra_args = os.getenv(f"{prefix}_EXTRA_ARGS", "").strip()
        self.keep_tmp = os.getenv(f"{prefix}_KEEP_TMP", "0").lower() in {"1", "true", "yes"}
        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []

    def _run_util(self, manifest_path: Path, output_path: Path) -> List[Dict[str, Any]]:
        util = Path(self.util_path).expanduser().resolve()
        if not util.exists():
            raise FileNotFoundError(
                f"{self.model_label} util not found at {util}. "
                f"Set {self.env_prefix}_INFER_PY to the official inference util."
            )

        cmd = [
            self.python_bin,
            str(util),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            self.repo_flag,
            self.repo_dir,
            "--model-weight",
            self.model_weight,
            "--input-resolution",
            self.input_resolution,
            "--num-clip-frames",
            self.num_clip_frames,
            "--num-workers",
            self.num_workers,
        ]
        if self.extra_args:
            cmd.extend(self.extra_args.split())

        env = os.environ.copy()
        if self.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices

        logger.info("Running %s util: %s", self.model_label, " ".join(cmd))
        proc = subprocess.run(
            cmd,
            cwd=self.repo_dir,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.stdout.strip():
            logger.info("%s util stdout:\n%s", self.model_label, proc.stdout.strip())
        if proc.stderr.strip():
            logger.warning("%s util stderr:\n%s", self.model_label, proc.stderr.strip())
        if proc.returncode != 0:
            raise RuntimeError(
                f"{self.model_label} util failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Return code: {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )
        if not output_path.exists():
            raise RuntimeError(f"{self.model_label} util did not create output file: {output_path}")

        rows: List[Dict[str, Any]] = []
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def predict_batch(
        self,
        queries: List[str],
        video_paths: List[str],
        system_prompt: str,
        video_metas: List[Dict[str, Any]] | None = None,
    ) -> List[str]:
        del system_prompt
        if len(queries) != len(video_paths):
            raise ValueError("queries and video_paths must have the same length")

        tmp_dir_obj = None
        if self.keep_tmp:
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"{self.env_prefix.lower()}_eval_"))
        else:
            tmp_dir_obj = tempfile.TemporaryDirectory(prefix=f"{self.env_prefix.lower()}_eval_")
            tmp_dir = Path(tmp_dir_obj.name)
        manifest_path = tmp_dir / "manifest.jsonl"
        output_path = tmp_dir / "predictions.jsonl"

        self.last_user_prompts = [_recover_raw_query(query) for query in queries]
        with manifest_path.open("w", encoding="utf-8") as f:
            metas = video_metas or [{} for _ in video_paths]
            for query, video_path, meta in zip(self.last_user_prompts, video_paths, metas):
                f.write(
                    json.dumps(
                        {
                            "query": query,
                            "video_path": _extract_video_path(video_path),
                            "video_input_path": video_path,
                            "width": meta.get("width"),
                            "height": meta.get("height"),
                            "queryid": meta.get("queryid"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        try:
            rows = self._run_util(manifest_path, output_path)
            outputs: List[str] = []
            raw_responses: List[str] = []
            for idx, row in enumerate(rows):
                fallback = self.last_user_prompts[idx] if idx < len(self.last_user_prompts) else "target"
                normalized = _normalize_prediction_payload(row, fallback_description=fallback)
                outputs.append(normalized)
                raw_responses.append(str(row.get("raw_response") or normalized) if isinstance(row, dict) else normalized)
            while len(outputs) < len(queries):
                outputs.append("{}")
                raw_responses.append("{}")
            self.last_raw_responses = raw_responses
            return outputs[: len(queries)]
        finally:
            if self.keep_tmp:
                logger.info("Keeping %s temp dir at %s", self.model_label, tmp_dir)
            elif tmp_dir_obj is not None:
                tmp_dir_obj.cleanup()
