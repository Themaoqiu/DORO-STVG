import json, logging, os, re, subprocess, tempfile
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

class TubeDETRModel:
    prompt_style = "json"
    use_video_input_path = True

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.repo_dir = os.getenv("TUBEDETR_DIR")
        if not self.repo_dir:
            raise RuntimeError("TUBEDETR_DIR is required.")
        repo_dir = Path(self.repo_dir).expanduser().resolve()
        if not repo_dir.exists():
            raise FileNotFoundError(f"TUBEDETR_DIR does not exist: {repo_dir}")
        self.repo_dir = str(repo_dir)
        self.python_bin = os.getenv("TUBEDETR_PYTHON") or str(repo_dir / ".venv" / "bin" / "python")
        self.helper_path = os.getenv("TUBEDETR_INFER_PY") or str(Path(__file__).resolve().parents[1] / "utils" / "tubedetr_infer_helper.py")
        self.checkpoint = os.getenv("TUBEDETR_CHECKPOINT") or self.model_path
        self.dataset_config = os.getenv("TUBEDETR_DATASET_CONFIG", "config/vidstg.json")
        self.combine_datasets = os.getenv("TUBEDETR_COMBINE_DATASETS", "vidstg").split(",")
        self.combine_datasets_val = os.getenv("TUBEDETR_COMBINE_DATASETS_VAL", "vidstg").split(",")
        self.resolution = os.getenv("TUBEDETR_RESOLUTION", "224")
        self.fps = os.getenv("TUBEDETR_FPS", "5")
        self.device = os.getenv("TUBEDETR_DEVICE", "cuda")
        self.cuda_visible_devices = os.getenv("TUBEDETR_CUDA_VISIBLE_DEVICES") or os.getenv("CUDA_VISIBLE_DEVICES")
        self.keep_tmp = os.getenv("TUBEDETR_KEEP_TMP", "0").lower() in {"1", "true", "yes"}
        self.last_user_prompts = []
        self.last_raw_responses = []

    def _run_helper(self, manifest_path: Path, output_path: Path):
        cmd = [
            self.python_bin, str(Path(self.helper_path).expanduser().resolve()),
            "--manifest", str(manifest_path), "--output", str(output_path),
            "--tubedetr-dir", self.repo_dir, "--checkpoint", self.checkpoint,
            "--dataset-config", self.dataset_config,
            "--combine-datasets", *self.combine_datasets,
            "--combine-datasets-val", *self.combine_datasets_val,
            "--resolution", self.resolution, "--fps", self.fps, "--device", self.device,
        ]
        env = os.environ.copy()
        if self.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        proc = subprocess.run(cmd, cwd=self.repo_dir, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.stdout.strip():
            logger.info("TubeDETR helper stdout:\n%s", proc.stdout.strip())
        if proc.stderr.strip():
            logger.warning("TubeDETR helper stderr:\n%s", proc.stderr.strip())
        if proc.returncode != 0:
            raise RuntimeError(f"TubeDETR helper failed.\nCommand: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        rows = []
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def predict_batch(self, queries: List[str], video_paths: List[str], system_prompt: str, video_metas: List[Dict[str, Any]] | None = None) -> List[str]:
        del system_prompt
        tmp = tempfile.TemporaryDirectory(prefix="tubedetr_eval_")
        tmp_dir = Path(tmp.name)
        manifest_path = tmp_dir / "manifest.jsonl"
        output_path = tmp_dir / "predictions.jsonl"
        self.last_user_prompts = [_recover_raw_query(q) for q in queries]
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
            rows = self._run_helper(manifest_path, output_path)
            outputs, raws = [], []
            for idx, row in enumerate(rows):
                fallback = self.last_user_prompts[idx] if idx < len(self.last_user_prompts) else "target"
                normalized = _normalize_prediction_payload(row, fallback)
                outputs.append(normalized)
                raws.append(str(row.get("raw_response") or normalized) if isinstance(row, dict) else normalized)
            while len(outputs) < len(queries):
                outputs.append("{}")
                raws.append("{}")
            self.last_raw_responses = raws
            return outputs[:len(queries)]
        finally:
            if self.keep_tmp:
                logger.info("Keeping TubeDETR temp dir at %s", tmp_dir)
            else:
                tmp.cleanup()
