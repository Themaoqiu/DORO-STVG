import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from decord import VideoReader, cpu
from PIL import Image


logger = logging.getLogger(__name__)
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"


def _extract_video_spec(video_input: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    marker = "::split="
    if marker not in video_input:
        return video_input, None

    video_path, split_text = video_input.split(marker, 1)
    try:
        start_text, end_text = split_text.split(":", 1)
        return video_path, (int(start_text), int(end_text))
    except ValueError:
        logger.warning("Failed to parse clip split from %s", video_input)
        return video_path, None


def _linearly_sample_indices(start: int, end: int, max_frames: int) -> List[int]:
    if end < start:
        end = start
    total = end - start + 1
    if max_frames <= 1 or total <= 1:
        return [start]
    if total <= max_frames:
        return list(range(start, end + 1))

    positions = np.linspace(start, end, num=max_frames)
    sampled = [int(round(v)) for v in positions]
    deduped, seen = [], set()
    for idx in sampled:
        idx = max(start, min(end, idx))
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    if deduped[0] != start:
        deduped[0] = start
    if deduped[-1] != end:
        deduped[-1] = end
    return deduped


def _load_sampled_frames(video_path: str, split: Optional[Tuple[int, int]], max_frames: int):
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Empty video: {video_path}")

    if split is None:
        start, end = 0, total - 1
    else:
        start, end = split
        start = max(0, min(total - 1, int(start)))
        end = max(start, min(total - 1, int(end)))

    sampled_indices = _linearly_sample_indices(start, end, max_frames=max_frames)
    frames = vr.get_batch(sampled_indices).asnumpy()
    height, width = frames.shape[1], frames.shape[2]
    return frames, sampled_indices, int(width), int(height)


def _write_sampled_clip(frames: np.ndarray, out_path: Path, fps: float) -> None:
    frame_dir = out_path.parent / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(frame_dir / f"frame_{idx:06d}.jpg", quality=95)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%06d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_path),
        "-y",
    ]
    subprocess.run(cmd, check=True)


def _extract_json_candidate(text: str) -> Optional[str]:
    if not text:
        return None
    stripped = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    return stripped[start:end + 1].strip()


def _extract_frame_map_candidates(text: str) -> Dict[int, List[float]]:
    frame_map = {}
    if not text:
        return frame_map

    matches = re.findall(r'["\']?(\d+)["\']?\s*:\s*\[\s*([^\]]+?)\s*\]', text)
    for frame_idx_text, coords_text in matches:
        try:
            frame_idx = int(frame_idx_text)
            coords = [float(v.strip()) for v in coords_text.split(",")]
        except ValueError:
            continue
        if len(coords) == 4:
            frame_map[frame_idx] = coords
    return frame_map


def _parse_model_frame_map(raw_text: str) -> Dict[int, List[float]]:
    json_candidate = _extract_json_candidate(raw_text)
    if json_candidate:
        try:
            payload = json.loads(json_candidate)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for _, frame_map in payload.items():
                if isinstance(frame_map, dict):
                    parsed = {}
                    for frame_idx_text, coords in frame_map.items():
                        try:
                            frame_idx = int(str(frame_idx_text).strip())
                            parsed[frame_idx] = [float(v) for v in coords]
                        except (TypeError, ValueError):
                            continue
                    if parsed:
                        return parsed
    return _extract_frame_map_candidates(raw_text)


def _normalize_coords(coords: List[float], width: int, height: int):
    if len(coords) != 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = coords
    if max(abs(v) for v in coords) > 1.5:
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    return [
        max(0.0, min(1.0, float(x1))),
        max(0.0, min(1.0, float(y1))),
        max(0.0, min(1.0, float(x2))),
        max(0.0, min(1.0, float(y2))),
    ]


def _remap_frame_map(frame_map: Dict[int, List[float]], original_indices: List[int], width: int, height: int):
    remapped = {}
    original_index_set = {int(idx) for idx in original_indices}
    for local_frame_idx, coords in frame_map.items():
        normalized = _normalize_coords(coords, width=width, height=height)
        if normalized is None:
            continue
        if 0 <= local_frame_idx < len(original_indices):
            original_frame_idx = int(original_indices[local_frame_idx])
        elif local_frame_idx in original_index_set:
            original_frame_idx = int(local_frame_idx)
        else:
            continue
        remapped[str(original_frame_idx)] = normalized
    return remapped
