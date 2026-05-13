import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fire


def _normalize_video_name(video_name: str) -> str:
    name = Path(str(video_name)).name
    if name.endswith(".mp4"):
        return name[:-4]
    return name


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def _sample_id(item: Dict[str, Any], fallback: str = "unknown") -> str:
    value = item.get("queryid") or item.get("query_id") or item.get("sample_id")
    return str(value or fallback)


def _extract_json_candidate(text: str) -> Optional[str]:
    value = str(text or "").strip()
    if not value:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", value, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    start = value.find("{")
    end = value.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return value[start:end + 1].strip()


def _normalize_frame_box_map(frame_map: Any) -> Dict[str, List[float]]:
    normalized: Dict[str, List[float]] = {}
    if not isinstance(frame_map, dict):
        return normalized

    for frame_idx_raw, coords in frame_map.items():
        try:
            frame_idx = int(str(frame_idx_raw).strip())
        except (TypeError, ValueError):
            continue

        if not isinstance(coords, list) or len(coords) != 4:
            continue

        try:
            values = [float(v) for v in coords]
        except (TypeError, ValueError):
            continue

        normalized[str(frame_idx)] = values

    return normalized


def _tracks_from_box_field(item: Dict[str, Any]) -> List[Tuple[Dict[str, List[float]], Tuple[int, int], str]]:
    raw_box = item.get("box")
    candidate = _extract_json_candidate(str(raw_box or ""))
    if not candidate:
        return []

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return []

    tracks: List[Tuple[Dict[str, List[float]], Tuple[int, int], str]] = []
    if not isinstance(payload, dict):
        return tracks

    objects = payload.get("objects")
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            boxes = _normalize_frame_box_map(obj.get("spatial_bboxes"))
            if not boxes:
                continue
            frames = sorted(int(k) for k in boxes.keys())
            span = obj.get("temporal_span")
            if isinstance(span, (list, tuple)) and len(span) == 2:
                temporal_span = (_safe_int(span[0], frames[0]), _safe_int(span[1], frames[-1]))
            else:
                temporal_span = (frames[0], frames[-1])
            description = str(obj.get("description") or item.get("query") or "")
            tracks.append((boxes, temporal_span, description))
        if tracks:
            return tracks

    for description, frame_map in payload.items():
        if description in {"objects", "prediction", "raw_response", "query", "video_path"}:
            continue
        boxes = _normalize_frame_box_map(frame_map)
        if not boxes:
            continue
        frames = sorted(int(k) for k in boxes.keys())
        tracks.append((boxes, (frames[0], frames[-1]), str(description or item.get("query") or "")))

    return tracks


def _tracks_from_target_members(
    item: Dict[str, Any],
    width: int,
    height: int,
) -> List[Tuple[Dict[str, List[float]], Tuple[int, int], str]]:
    members = item.get("target_members")
    per_target_queries = item.get("per_target_queries") or {}
    tracks: List[Tuple[Dict[str, List[float]], Tuple[int, int], str]] = []
    if not isinstance(members, list):
        return tracks

    for member in members:
        if not isinstance(member, dict):
            continue
        raw_boxes = member.get("boxes") or {}
        if not isinstance(raw_boxes, dict):
            continue

        boxes: Dict[str, List[float]] = {}
        for frame_idx_raw, coords in raw_boxes.items():
            try:
                frame_idx = int(str(frame_idx_raw).strip())
            except (TypeError, ValueError):
                continue
            if not isinstance(coords, list) or len(coords) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in coords]
            except (TypeError, ValueError):
                continue

            if width > 0 and height > 0 and max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.0:
                boxes[str(frame_idx)] = [x1 / width, y1 / height, x2 / width, y2 / height]
            else:
                boxes[str(frame_idx)] = [x1, y1, x2, y2]

        if not boxes:
            continue

        frames = sorted(int(k) for k in boxes.keys())
        target_index = member.get("target_index")
        description = per_target_queries.get(f"target {target_index}") or item.get("query") or ""
        tracks.append((boxes, (frames[0], frames[-1]), str(description)))

    return tracks


def _pick_primary_track(item: Dict[str, Any]) -> Optional[Tuple[Dict[str, List[float]], Tuple[int, int], str]]:
    tracks = item.get("gt_tracks_sampled") or []
    if isinstance(tracks, list):
        for track in tracks:
            boxes = track.get("spatial_bboxes") or {}
            if not isinstance(boxes, dict) or not boxes:
                continue
            span = track.get("temporal_span")
            desc = str(track.get("description") or item.get("query") or "")
            frame_ids = sorted(int(k) for k in boxes.keys())
            if isinstance(span, (list, tuple)) and len(span) == 2:
                temporal_span = (_safe_int(span[0], frame_ids[0]), _safe_int(span[1], frame_ids[-1]))
            else:
                temporal_span = (frame_ids[0], frame_ids[-1])
            return boxes, temporal_span, desc

    width = _safe_int(item.get("video_width") or item.get("Width"), 0)
    height = _safe_int(item.get("video_height") or item.get("Height"), 0)

    for boxes, temporal_span, desc in _tracks_from_box_field(item):
        return boxes, temporal_span, desc

    for boxes, temporal_span, desc in _tracks_from_target_members(item, width, height):
        return boxes, temporal_span, desc

    boxes = item.get("gt_bboxes_sampled") or {}
    span = item.get("gt_temporal_sampled")
    desc = str(item.get("query") or "")
    if not isinstance(boxes, dict) or not boxes:
        return None

    frame_ids = sorted(int(k) for k in boxes.keys())
    if isinstance(span, (list, tuple)) and len(span) == 2:
        temporal_span = (_safe_int(span[0], frame_ids[0]), _safe_int(span[1], frame_ids[-1]))
    else:
        temporal_span = (frame_ids[0], frame_ids[-1])
    return boxes, temporal_span, desc


def _to_bbox_dict(coords: List[float], width: int, height: int) -> Dict[str, int]:
    if isinstance(coords, dict):
        if all(k in coords for k in ("xmin", "ymin", "xmax", "ymax")):
            return {
                "xmin": int(round(float(coords["xmin"]))),
                "ymin": int(round(float(coords["ymin"]))),
                "xmax": int(round(float(coords["xmax"]))),
                "ymax": int(round(float(coords["ymax"]))),
            }
        raise ValueError(f"Expected bbox dict with xmin/ymin/xmax/ymax, got {coords}")

    if len(coords) != 4:
        raise ValueError(f"Expected 4 bbox coords, got {coords}")

    x1, y1, x2, y2 = [float(v) for v in coords]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1 *= width
        y1 *= height
        x2 *= width
        y2 *= height

    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return {
        "xmin": int(round(x1)),
        "ymin": int(round(y1)),
        "xmax": int(round(x2)),
        "ymax": int(round(y2)),
    }


def convert(annotation_path: Path, output_dir: Path, fps: float = 1.0) -> None:
    rows = _load_jsonl(annotation_path)

    data_root = output_dir / "data" / "vidstg"
    sent_annos_dir = data_root / "sent_annos"
    bbox_annos_dir = data_root / "bbox_annos"
    vstg_annos_dir = data_root / "vstg_annos"
    videos_dir = data_root / "videos"

    sent_annos_dir.mkdir(parents=True, exist_ok=True)
    bbox_annos_dir.mkdir(parents=True, exist_ok=True)
    vstg_annos_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    sent_annotations: List[Dict[str, Any]] = []
    per_video: Dict[str, Dict[str, Any]] = {}
    linked_videos = 0
    skipped_no_bbox = 0

    for idx, item in enumerate(rows):
        video_path_raw = item.get("video_input_path") or item.get("video_path")
        if not video_path_raw:
            raise ValueError(f"Sample at index {idx} has no video path")

        video_path = Path(str(video_path_raw).split("::split=", 1)[0])
        video_stem = _normalize_video_name(video_path.name)
        width = _safe_int(item.get("video_width") or item.get("Width"), 0)
        height = _safe_int(item.get("video_height") or item.get("Height"), 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"Sample {video_stem} is missing valid width/height")

        picked = _pick_primary_track(item)
        if picked is None:
            skipped_no_bbox += 1
            print(f"[convert] skip sample without bounding boxes: {_sample_id(item, str(idx))}")
            continue

        boxes, temporal_span, description = picked
        begin_fid, end_fid = temporal_span
        if begin_fid > end_fid:
            begin_fid, end_fid = end_fid, begin_fid

        frame_ids = sorted(int(k) for k in boxes.keys())
        frame_count = max(frame_ids[-1] + 1, end_fid + 1)

        video_state = per_video.setdefault(
            video_stem,
            {
                "width": width,
                "height": height,
                "frame_count": 0,
                "next_tid": 0,
                "trajectories": {},
                "video_path": video_path,
            },
        )
        video_state["width"] = width
        video_state["height"] = height
        video_state["frame_count"] = max(video_state["frame_count"], frame_count)

        target_id = int(video_state["next_tid"])
        video_state["next_tid"] += 1
        sent_annotations.append(
            {
                "vid": video_stem,
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": frame_count,
                "used_segment": {
                    "begin_fid": 0,
                    "end_fid": frame_count - 1,
                },
                "temporal_gt": {
                    "begin_fid": begin_fid,
                    "end_fid": end_fid,
                },
                "captions": [
                    {
                        "description": description,
                        "target_id": target_id,
                    }
                ],
                "questions": [],
                "subject/objects": [
                    {
                        "tid": target_id,
                        "category": "target",
                    }
                ],
            }
        )

        for frame_idx in range(frame_count):
            coords = boxes.get(str(frame_idx))
            if coords is None:
                coords = boxes.get(frame_idx)
            if coords is not None:
                video_state["trajectories"].setdefault(frame_idx, []).append(
                    {
                        "tid": target_id,
                        "bbox": _to_bbox_dict(coords, width, height),
                    }
                )

        link_path = videos_dir / f"{video_stem}.mp4"
        if not link_path.exists():
            try:
                link_path.symlink_to(video_path)
                linked_videos += 1
            except FileExistsError:
                pass

    if not sent_annotations:
        raise ValueError(f"No valid samples were converted from {annotation_path}")

    with (sent_annos_dir / "test_annotations.json").open("w", encoding="utf-8") as f:
        json.dump(sent_annotations, f, ensure_ascii=False, indent=2)

    with (sent_annos_dir / "val_annotations.json").open("w", encoding="utf-8") as f:
        json.dump(sent_annotations, f, ensure_ascii=False, indent=2)

    bbox_count = 0
    for video_stem, state in per_video.items():
        trajectories: List[List[Dict[str, Any]]] = []
        for frame_idx in range(int(state["frame_count"])):
            trajectories.append(state["trajectories"].get(frame_idx, []))

        payload = {
            "vid": video_stem,
            "frame_count": int(state["frame_count"]),
            "trajectories": trajectories,
        }
        with (bbox_annos_dir / f"{video_stem}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        with (vstg_annos_dir / f"{video_stem}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        bbox_count += 1

    print(f"[convert] samples={len(sent_annotations)}")
    print(f"[convert] linked_videos={linked_videos}")
    print(f"[convert] output={data_root}")
    print(f"[convert] sent_file={sent_annos_dir / 'test_annotations.json'}")
    print(f"[convert] bbox_files={bbox_count}")
    print(f"[convert] vstg_files={bbox_count}")
    print(f"[convert] skipped_no_bbox={skipped_no_bbox}")


def main(
    annotation_path: Path,
    output_dir: Path,
    fps: float = 1.0,
) -> None:
    convert(annotation_path, output_dir, fps=fps)


if __name__ == "__main__":
    fire.Fire(main)
