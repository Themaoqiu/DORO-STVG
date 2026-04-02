#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import fire


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _sorted_box_items(boxes: Dict[str, List[Any]]) -> Iterable[Tuple[int, List[Any]]]:
    pairs: List[Tuple[int, List[Any]]] = []
    for k, v in boxes.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        pairs.append((idx, v))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _format_box_entries(
    boxes: Dict[str, List[Any]],
    fps: float,
    video_width: Optional[float] = None,
    video_height: Optional[float] = None,
    coord_decimals: int = 1,
) -> List[str]:
    entries: List[str] = []
    for frame_idx, coords in _sorted_box_items(boxes):
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        x1, y1, x2, y2 = (_to_float(c) for c in coords)
        time_s = round(frame_idx / fps, 1)
        if video_width and video_height and video_width > 0 and video_height > 0:
            x1 = x1 / video_width
            y1 = y1 / video_height
            x2 = x2 / video_width
            y2 = y2 / video_height
        entries.append(
            f"{frame_idx}, {time_s:.1f}, "
            f"{x1:.{coord_decimals}f}, {y1:.{coord_decimals}f}, "
            f"{x2:.{coord_decimals}f}, {y2:.{coord_decimals}f}"
        )
    return entries


def format_box_string(
    boxes: Dict[str, List[Any]],
    fps: float,
    video_width: Optional[float] = None,
    video_height: Optional[float] = None,
    coord_decimals: int = 1,
) -> str:
    entries = _format_box_entries(
        boxes=boxes,
        fps=fps,
        video_width=video_width,
        video_height=video_height,
        coord_decimals=coord_decimals,
    )
    return "<" + "; ".join(entries) + " />"


def _probe_video_size(video_path: str) -> Tuple[Optional[int], Optional[int]]:
    if not video_path:
        return None, None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                video_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return None, None
        stream = streams[0] or {}
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
    except Exception:
        return None, None
    if width <= 0 or height <= 0:
        return None, None
    return width, height


def transform_record(
    obj: Dict[str, Any],
    fps: float,
    size_cache: Dict[str, Tuple[Optional[int], Optional[int]]],
) -> Dict[str, Any]:
    video_path = str(obj.get("video_path", ""))
    boxes = obj.get("boxes")
    if not isinstance(boxes, dict):
        boxes = {}
    video_width = obj.get("video_width")
    video_height = obj.get("video_height")
    if not isinstance(video_width, int) or video_width <= 0 or not isinstance(video_height, int) or video_height <= 0:
        if video_path not in size_cache:
            size_cache[video_path] = _probe_video_size(video_path)
        video_width, video_height = size_cache[video_path]

    return {
        "videopath": Path(video_path).name,
        "queryid": obj.get("query_id"),
        "query": obj.get("query"),
        "Difficulty": {
            "D_s": obj.get("D_s"),
            "D_t": obj.get("D_t"),
            "D": obj.get("D"),
        },
        "Width": video_width,
        "Height": video_height,
        "box": "The object box is: "
        + format_box_string(
            boxes=boxes,
            fps=fps,
            video_width=video_width,
            video_height=video_height,
            coord_decimals=4,
        ),
    }


def main(
    input: str = "query.jsonl",
    output: str = "query_formatted.jsonl",
    fps: float = 2.0,
) -> None:
    input_path = Path(input)
    output_path = Path(output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    size_cache: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out = transform_record(obj, fps=fps, size_cache=size_cache)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} lines -> {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
