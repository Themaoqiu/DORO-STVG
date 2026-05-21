import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


logger = logging.getLogger(__name__)
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"

MASK_FILE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
MASK_PATH_HINTS = {"mask", "masks", "seg", "segs", "segmentation", "segmentations"}
MASK_SKIP_HINTS = {"vis", "visual", "visualization", "overlay", "overlays", "frame", "frames", "point", "points"}


def _extract_frames(video_path: str, out_dir: Path, max_frames: int, sample_fps: float) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = sample_fps if sample_fps and sample_fps > 0 else 2.0
    frame_pattern = out_dir / "frame_%04d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        str(frame_pattern),
        "-y",
    ]
    subprocess.run(cmd, check=True)

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError(f"ffmpeg extracted no frames from: {video_path}")

    if max_frames > 0 and len(frames) > max_frames:
        for f in frames[max_frames:]:
            f.unlink(missing_ok=True)
        frames = frames[:max_frames]

    with Image.open(frames[0]) as im:
        width, height = im.size
    return int(width), int(height)


def _maybe_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _maybe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _extract_xy_from_point_text(text: str) -> Optional[Tuple[float, float]]:
    if not isinstance(text, str):
        return None
    mx = re.search(r'\bx\s*=\s*["\']?(-?\d+(?:\.\d+)?)', text, flags=re.IGNORECASE)
    my = re.search(r'\by\s*=\s*["\']?(-?\d+(?:\.\d+)?)', text, flags=re.IGNORECASE)
    if not mx or not my:
        return None
    px = _maybe_float(mx.group(1))
    py = _maybe_float(my.group(1))
    if px is None or py is None:
        return None
    return (px, py)



def _recover_query_from_stvg_prompt(query_text: str) -> str:
    text = str(query_text or "").strip()
    match = re.search(r"Where does\s+(.+?)\s+occur in the video\?", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return " ".join(match.group(1).split())
    return text


def _strip_question_prefix(query: str) -> str:
    text = " ".join(str(query or "").strip().split())
    if not text:
        return text
    lowered = text.lower()
    for prefix in ("who is ", "who ", "what is ", "what ", "which is ", "which "):
        if lowered.startswith(prefix):
            return text[len(prefix):].strip(" ?.")
    return text.strip(" ?.")


def _verb_to_gerund(phrase: str) -> str:
    words = phrase.split(maxsplit=1)
    if not words:
        return phrase

    verb = words[0]
    rest = words[1] if len(words) > 1 else ""
    irregular = {
        "has": "having",
        "is": "",
        "are": "",
        "does": "doing",
    }
    lowered = verb.lower()
    if lowered in irregular:
        converted = irregular[lowered]
    elif lowered.endswith("ies") and len(lowered) > 3:
        converted = verb[:-3] + "ying"
    elif lowered.endswith("es") and len(lowered) > 3:
        converted = verb[:-2] + "ing"
    elif lowered.endswith("s") and len(lowered) > 2:
        converted = verb[:-1] + "ing"
    elif lowered.endswith("e") and len(lowered) > 2:
        converted = verb[:-1] + "ing"
    elif lowered.endswith("ing"):
        converted = verb
    else:
        converted = verb + "ing"

    return " ".join(part for part in (converted, rest) if part).strip()


def _build_point_prompt(query_text: str) -> str:
    raw_query = _recover_query_from_stvg_prompt(query_text)
    target = _strip_question_prefix(raw_query)
    if not target:
        return raw_query

    lowered = raw_query.lower().strip()
    if lowered.startswith("who "):
        if not lowered.startswith("who is ") and not lowered.startswith("who are "):
            target = _verb_to_gerund(target)
        return f"point to the person {target}"
    return f"point to the {target}"

def _collect_points(payload) -> List[Tuple[int, float, float]]:
    """Collect (frame_idx, x, y) tuples from flexible points.jsonl schemas."""
    points: List[Tuple[int, float, float]] = []

    def add_point(frame_idx: Optional[int], x, y):
        fi = _maybe_int(frame_idx)
        px = _maybe_float(x)
        py = _maybe_float(y)
        if fi is not None and px is not None and py is not None:
            points.append((fi, px, py))

    def add_from_value(frame_idx: Optional[int], value):
        if frame_idx is None or value is None:
            return

        if isinstance(value, dict):
            if "x" in value and "y" in value:
                add_point(frame_idx, value.get("x"), value.get("y"))
            if "point" in value:
                add_from_value(frame_idx, value.get("point"))
            if "points" in value:
                add_from_value(frame_idx, value.get("points"))
            return

        if isinstance(value, (list, tuple)):
            if len(value) >= 2 and all(isinstance(v, (int, float)) for v in value[:2]):
                add_point(frame_idx, value[0], value[1])
            for item in value:
                add_from_value(frame_idx, item)
            return

        if isinstance(value, str):
            xy = _extract_xy_from_point_text(value)
            if xy is not None:
                add_point(frame_idx, xy[0], xy[1])

    def visit(node):
        if isinstance(node, dict):
            for raw_key, raw_value in node.items():
                key_as_frame = _maybe_int(raw_key)
                if key_as_frame is not None:
                    add_from_value(key_as_frame, raw_value)

            frame_idx = None
            for k in ("frame_idx", "frame_index", "frame", "idx", "index", "t"):
                if k in node:
                    frame_idx = _maybe_int(node.get(k))
                    break

            if frame_idx is not None:
                if "x" in node and "y" in node:
                    add_point(frame_idx, node.get("x"), node.get("y"))
                if "point" in node:
                    add_from_value(frame_idx, node.get("point"))
                if "points" in node:
                    add_from_value(frame_idx, node.get("points"))

            for v in node.values():
                visit(v)

        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(payload)
    return points


def _point_to_box(x: float, y: float, width: int, height: int, half_side: float) -> List[float]:
    # Accept normalized / percentage / absolute pixel points.
    if x > 1.0 or y > 1.0:
        # VideoMolmo point tag often uses percentage coordinates (0~100).
        if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0:
            x = x / 100.0
            y = y / 100.0
        else:
            x = x / max(width, 1)
            y = y / max(height, 1)

    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))

    x1 = max(0.0, x - half_side)
    y1 = max(0.0, y - half_side)
    x2 = min(1.0, x + half_side)
    y2 = min(1.0, y + half_side)
    return [x1, y1, x2, y2]



def _batched_mask_to_box(masks):
    import torch

    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    masks = masks.bool()
    in_height = masks.any(dim=-1)
    y_coords = in_height * torch.arange(h, device=masks.device)[None, :]
    bottom_edges = y_coords.max(dim=-1).values
    top_edges = (y_coords + h * (~in_height)).min(dim=-1).values

    in_width = masks.any(dim=-2)
    x_coords = in_width * torch.arange(w, device=masks.device)[None, :]
    right_edges = x_coords.max(dim=-1).values
    left_edges = (x_coords + w * (~in_width)).min(dim=-1).values

    empty = (right_edges < left_edges) | (bottom_edges < top_edges)
    boxes = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    boxes = boxes * (~empty).unsqueeze(-1)

    if len(shape) > 2:
        return boxes.reshape(*shape[:-2], 4)
    return boxes[0]


def _normalize_pixel_box(box, width: int, height: int) -> Optional[List[float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        max(0.0, min(1.0, x1 / max(width, 1))),
        max(0.0, min(1.0, y1 / max(height, 1))),
        max(0.0, min(1.0, (x2 + 1.0) / max(width, 1))),
        max(0.0, min(1.0, (y2 + 1.0) / max(height, 1))),
    ]


def _frame_index_from_path(path: Path) -> Optional[int]:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return None
    frame_idx = _maybe_int(matches[-1])
    if frame_idx is None:
        return None
    if path.stem.lower().startswith("frame_") and frame_idx > 0:
        return frame_idx - 1
    return frame_idx


def _path_has_mask_hint(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    tokens = {part.lower() for part in rel_parts[:-1]}
    stem_tokens = set(re.split(r"[_\-.]+", path.stem.lower()))
    all_tokens = tokens | stem_tokens
    if all_tokens & MASK_SKIP_HINTS and not (all_tokens & MASK_PATH_HINTS):
        return False
    return bool(all_tokens & MASK_PATH_HINTS)


def _iter_mask_files(out_dir: Path, frames_dir_name: str) -> Iterable[Path]:
    search_roots = [out_dir / frames_dir_name, out_dir]
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MASK_FILE_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if path.parent == root and root.name == frames_dir_name:
                yield path
                continue
            if _path_has_mask_hint(path, root):
                yield path


def _mask_image_to_box(path: Path, width: int, height: int) -> Optional[List[float]]:
    with Image.open(path) as im:
        mask = im.convert("L")
        try:
            import torch

            mask_tensor = torch.as_tensor(list(mask.getdata()), dtype=torch.bool).reshape(mask.height, mask.width)
            if not mask_tensor.any().item():
                return None
            box = _batched_mask_to_box(mask_tensor).tolist()
            return _normalize_pixel_box(box, width, height)
        except Exception:
            box = mask.point(lambda px: 255 if px > 0 else 0).getbbox()
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(1.0, x1 / max(width, 1))),
        max(0.0, min(1.0, y1 / max(height, 1))),
        max(0.0, min(1.0, x2 / max(width, 1))),
        max(0.0, min(1.0, y2 / max(height, 1))),
    ]


def _mask_array_to_boxes(path: Path, width: int, height: int) -> Dict[str, List[float]]:
    try:
        import numpy as np
    except ImportError:
        logger.warning("Cannot read %s because numpy is not installed.", path)
        return {}

    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as exc:
        logger.warning("Cannot load mask array %s: %s", path, exc)
        return {}

    if arr.ndim < 2:
        return {}
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.shape[-1] in (1, 3, 4) and arr.ndim == 3:
        arr = arr[..., 0][None, ...]
    else:
        arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))

    frame_start = _frame_index_from_path(path)
    if frame_start is None:
        frame_start = 0

    boxes: Dict[str, List[float]] = {}
    try:
        import torch

        mask_tensor = torch.as_tensor(arr > 0)
        mask_boxes = _batched_mask_to_box(mask_tensor).reshape(-1, 4).tolist()
        non_empty = mask_tensor.reshape(-1, mask_tensor.shape[-2] * mask_tensor.shape[-1]).any(dim=1).tolist()
        for offset, box in enumerate(mask_boxes):
            if not non_empty[offset]:
                continue
            normalized = _normalize_pixel_box(box, width, height)
            if normalized is not None:
                boxes[str(frame_start + offset)] = normalized
        return boxes
    except Exception:
        pass

    for offset, mask in enumerate(arr):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        boxes[str(frame_start + offset)] = [
            max(0.0, min(1.0, float(xs.min()) / max(width, 1))),
            max(0.0, min(1.0, float(ys.min()) / max(height, 1))),
            max(0.0, min(1.0, float(xs.max() + 1) / max(width, 1))),
            max(0.0, min(1.0, float(ys.max() + 1) / max(height, 1))),
        ]
    return boxes


def _read_mask_boxes(out_dir: Path, frames_dir_name: str, width: int, height: int) -> Dict[str, List[float]]:
    frame_to_box: Dict[str, List[float]] = {}
    mask_files = sorted(_iter_mask_files(out_dir, frames_dir_name))
    if not mask_files:
        return frame_to_box

    logger.info("Found %d candidate VideoMolmo mask files under %s", len(mask_files), out_dir)
    for path in mask_files:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            frame_to_box.update(_mask_array_to_boxes(path, width, height))
            continue

        frame_idx = _frame_index_from_path(path)
        if frame_idx is None:
            continue
        box = _mask_image_to_box(path, width, height)
        if box is not None:
            frame_to_box[str(frame_idx)] = box

    return dict(sorted(frame_to_box.items(), key=lambda item: int(item[0])))

def _normalize_frame_boxes(frame_map: Dict) -> Dict[str, List[float]]:
    normalized: Dict[str, List[float]] = {}
    if not isinstance(frame_map, dict):
        return normalized
    for frame_idx_text, coords in frame_map.items():
        try:
            frame_idx = int(str(frame_idx_text).strip())
        except (TypeError, ValueError):
            continue
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in coords]
        except (TypeError, ValueError):
            continue
        normalized[str(frame_idx)] = [
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)),
            max(0.0, min(1.0, y2)),
        ]
    return normalized


def _extract_json_dicts(text: str) -> List[Dict]:
    """Extract JSON dict candidates from arbitrary mixed text."""
    if not text:
        return []

    decoder = json.JSONDecoder()
    results: List[Dict] = []
    i = 0
    n = len(text)
    while i < n:
        start = text.find("{", i)
        if start < 0:
            break
        try:
            obj, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            i = start + 1
            continue
        if isinstance(obj, dict):
            results.append(obj)
        i = start + consumed
    return results


def _extract_response_json_from_logs(text: str) -> str:
    """
    Parse infer stdout/stderr and return the last valid object-track JSON as string.
    Expected shape:
      {
        "target desc": {"12": [x1, y1, x2, y2], ...},
        ...
      }
    """
    if not text:
        return "{}"

    candidates: List[Dict] = []

    # Prefer explicitly labeled model answers.
    output_blocks = re.findall(r"Output:\s*(.*?)(?=\nOutput:|\Z)", text, flags=re.DOTALL)
    for block in output_blocks:
        snippet = block.strip()
        if not snippet or "there are none" in snippet.lower():
            continue
        candidates.extend(_extract_json_dicts(snippet))

    # Fallback to any JSON dict present in logs.
    if not candidates:
        candidates = _extract_json_dicts(text)

    for obj in reversed(candidates):
        normalized_obj: Dict[str, Dict[str, List[float]]] = {}
        for desc, frame_map in obj.items():
            cleaned = _normalize_frame_boxes(frame_map)
            if cleaned:
                key = str(desc).strip() or "target"
                normalized_obj[key] = cleaned
        if normalized_obj:
            return json.dumps(normalized_obj, ensure_ascii=False)
    return "{}"


class VideoMolmoModel:
    """
    Adapter to run VideoMolmo inference and convert point outputs to bbox JSON.

    Required env vars:
    - VIDEOMOLMO_REPO: path to cloned VideoMolmo repo root.

    Optional env vars:
    - VIDEOMOLMO_PYTHON: python executable for VideoMolmo env (default: current python).
    - VIDEOMOLMO_MAX_FRAMES: max sampled frames from each video (default: 100).
    - VIDEOMOLMO_POINT_BOX_HALF: half side for point->box conversion in normalized coords (default: 0.04).
    - VIDEOMOLMO_KEEP_TMP: keep temp directories for debugging, set to 1/true/yes.
    """

    def __init__(
        self,
        model_path: str,
    ):
        del model_path
        self.max_frames = int(os.getenv("VIDEOMOLMO_MAX_FRAMES", "100"))
        self.sample_fps = float(os.getenv("VIDEOMOLMO_SAMPLE_FPS", "2.0"))
        self.infer_retries = max(1, int(os.getenv("VIDEOMOLMO_INFER_RETRIES", "3")))
        self.box_half = float(os.getenv("VIDEOMOLMO_POINT_BOX_HALF", "0.04"))
        self.keep_tmp = os.getenv("VIDEOMOLMO_KEEP_TMP", "0").lower() in {"1", "true", "yes"}
        self.allow_log_fallback = os.getenv("VIDEOMOLMO_ALLOW_LOG_FALLBACK", "0").lower() in {"1", "true", "yes"}
        self.use_point_prompt = os.getenv("VIDEOMOLMO_USE_POINT_PROMPT", "0").lower() in {"1", "true", "yes"}
        self.python_bin = os.getenv("VIDEOMOLMO_PYTHON", "python")

        repo = os.getenv("VIDEOMOLMO_REPO")
        if not repo:
            raise RuntimeError("VIDEOMOLMO_REPO is required (path to VideoMolmo clone).")
        self.repo = Path(repo).expanduser().resolve()

        # Support both repo root and nested VideoMolmo/VideoMolmo layouts.
        infer_candidate_1 = self.repo / "infer.py"
        infer_candidate_2 = self.repo / "VideoMolmo" / "infer.py"
        if infer_candidate_1.exists():
            self.work_dir = self.repo
            self.infer_py = infer_candidate_1
        elif infer_candidate_2.exists():
            self.work_dir = self.repo / "VideoMolmo"
            self.infer_py = infer_candidate_2
        else:
            raise FileNotFoundError(
                f"infer.py not found under {self.repo}. Expected {infer_candidate_1} or {infer_candidate_2}."
            )

        self.last_user_prompts: List[str] = []
        self.last_raw_responses: List[str] = []
        logger.info("Initialized videomolmo adapter | repo=%s infer=%s", self.repo, self.infer_py)

    def _run_one(self, query: str, video_path: str) -> str:
        tmp_root = Path(tempfile.mkdtemp(prefix="videomolmo_eval_"))
        frames_dir = tmp_root / "frames"
        out_dir = tmp_root / "results"

        try:
            width, height = _extract_frames(video_path, frames_dir, self.max_frames, self.sample_fps)
            prompt_for_infer = _build_point_prompt(query) if self.use_point_prompt else (query.strip() if query else query)

            cmd = [
                self.python_bin,
                str(self.infer_py),
                "--video_path",
                str(frames_dir),
                "--prompt",
                prompt_for_infer,
                "--save_path",
                str(out_dir),
            ]
            logger.info("Running VideoMolmo infer on %s | prompt=%s", Path(video_path).name, prompt_for_infer)
            clean_env = os.environ.copy()
            clean_env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

            last_logs = ""
            for attempt in range(1, self.infer_retries + 1):
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)

                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(self.work_dir),
                        env=clean_env,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except subprocess.CalledProcessError as exc:
                    fail_logs = ((exc.stdout or "") + "\n" + (exc.stderr or "")).strip()
                    last_logs = fail_logs
                    if fail_logs:
                        tail = fail_logs[-1200:]
                        logger.error("VideoMolmo infer failed attempt %d/%d (tail):\n%s", attempt, self.infer_retries, tail)
                    if attempt >= self.infer_retries:
                        raise
                    logger.warning("VideoMolmo infer failed attempt %d/%d; retrying", attempt, self.infer_retries)
                    continue

                logs = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
                last_logs = logs

                mask_frame_to_box = _read_mask_boxes(out_dir, frames_dir.name, width, height)
                if mask_frame_to_box:
                    logger.info(
                        "Converted %d VideoMolmo mask frames to STVG boxes for %s",
                        len(mask_frame_to_box),
                        video_path,
                    )
                    return json.dumps({"target": mask_frame_to_box}, ensure_ascii=False)

                points_file = out_dir / frames_dir.name / "points.jsonl"
                if not points_file.exists():
                    alt = out_dir / "points.jsonl"
                    if alt.exists():
                        points_file = alt
                    else:
                        logger.info("points.jsonl not found for %s on attempt %d/%d", video_path, attempt, self.infer_retries)
                        continue

                frame_to_box: Dict[str, List[float]] = {}
                with open(points_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        for frame_idx, x, y in _collect_points(payload):
                            if frame_idx is None:
                                continue
                            frame_to_box[str(int(frame_idx))] = _point_to_box(x, y, width, height, self.box_half)

                if frame_to_box:
                    return json.dumps({"target": frame_to_box}, ensure_ascii=False)

                logger.info("points.jsonl is empty for %s on attempt %d/%d", video_path, attempt, self.infer_retries)

            if self.allow_log_fallback and last_logs:
                logger.info("all retries empty for %s; fallback to infer logs", video_path)
                return _extract_response_json_from_logs(last_logs)

            logger.info("all retries empty for %s; log fallback disabled, return {}", video_path)
            return "{}"
        finally:
            if not self.keep_tmp:
                shutil.rmtree(tmp_root, ignore_errors=True)

    def predict_batch(self, queries: List[str], video_paths: List[str], system_prompt: str) -> List[str]:
        del system_prompt
        self.last_user_prompts = list(queries)
        outputs = [self._run_one(q, vp) for q, vp in zip(queries, video_paths)]
        self.last_raw_responses = outputs
        return outputs
