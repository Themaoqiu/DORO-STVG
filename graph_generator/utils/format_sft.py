#!/usr/bin/env python3

from __future__ import annotations

import fire
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _sorted_box_items(boxes: Dict[str, List[Any]]) -> Iterable[Tuple[int, List[Any]]]:
    items: List[Tuple[int, List[Any]]] = []
    for k, v in boxes.items():
        idx = _to_int(k, -1)
        if idx < 0:
            continue
        items.append((idx, v))
    items.sort(key=lambda x: x[0])
    return items


def _format_boxes(
    boxes: Dict[str, List[Any]],
    fps: float,
    coord_decimals: int,
    video_width: Optional[int] = None,
    video_height: Optional[int] = None,
) -> str:
    parts: List[str] = []
    for frame_idx, coords in _sorted_box_items(boxes):
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        x1, y1, x2, y2 = (_to_float(v) for v in coords)
        if video_width and video_height and video_width > 0 and video_height > 0:
            x1 /= video_width
            y1 /= video_height
            x2 /= video_width
            y2 /= video_height
        time_s = frame_idx / fps
        parts.append(
            f"{frame_idx}, {time_s:.1f}, "
            f"{x1:.{coord_decimals}f}, {y1:.{coord_decimals}f}, "
            f"{x2:.{coord_decimals}f}, {y2:.{coord_decimals}f}"
        )
    return "<" + "; ".join(parts) + " />"


def _detect_media_dir(video_paths: List[str]) -> Optional[Path]:
    non_empty = [str(Path(p).resolve()) for p in video_paths if p and Path(p).is_absolute()]
    if not non_empty:
        return None
    try:
        parents = [str(Path(p).parent) for p in non_empty]
        return Path(os.path.commonpath(parents))
    except Exception:
        # Fall back to first parent
        return Path(non_empty[0]).parent


def _resolve_video_ref(video_path: str, media_dir: Optional[Path], path_mode: str) -> str:
    p = Path(video_path)
    if path_mode == "absolute":
        return str(p)
    if path_mode == "basename":
        return p.name
    if media_dir is None:
        return p.name
    try:
        return str(p.resolve().relative_to(media_dir.resolve()))
    except Exception:
        return p.name


def build_messages(
    query: str,
    pixel_box_text: Optional[str],
    norm_box_text: Optional[str],
    assistant_format: str,
    include_system: bool,
) -> List[Dict[str, str]]:
    has_multi_targets = any(
        text and str(text).strip().lower().startswith("the object boxes are:")
        for text in (pixel_box_text, norm_box_text)
    )
    user_content = (
        f"<video>{query}\n\n"
        "Localize every referred target across frames and output trajectories in the exact required format.\n"
        "Output requirements:\n"
        "- Use normalized box coordinates in [0, 1].\n"
        "- Do not output explanations, reasoning, or extra text beyond the required answer.\n"
        "- Keep the frame-level coordinate format exactly as <frame_idx, time_sec, x1, y1, x2, y2; ... />.\n"
    )
    if has_multi_targets:
        user_content += (
            "Multi-target format:\n"
            "- Output one trajectory for each referred target.\n"
            "- Before each <... />, write a short target description that identifies which target it is.\n"
            "- Separate different targets with '; '.\n"
            "- Exact format: The object boxes are: target description 1: <...>; target description 2: <...>\n"
        )
    else:
        user_content += (
            "Single-target format:\n"
            "- Exact format: The object box is: <frame_idx, time_sec, x1, y1, x2, y2; ... />\n"
        )

    if assistant_format == "pixel":
        assistant_content = pixel_box_text or "The object box is: < />"
    elif assistant_format == "norm":
        assistant_content = norm_box_text or pixel_box_text or "The object box is: < />"
    else:
        parts: List[str] = []
        if pixel_box_text:
            parts.append(pixel_box_text)
        if norm_box_text:
            parts.append(norm_box_text)
        assistant_content = "\n".join(parts) if parts else "The object box is: < />"

    msgs: List[Dict[str, str]] = []
    if include_system:
        msgs.append(
            {
                "role": "system",
                "content": (
                    "You are an expert video grounding assistant. "
                    "Given a query and a video, return only the target trajectory answer in the required format. "
                    "Do not add explanations. Preserve all referred targets in multi-target queries."
                ),
            }
        )
    msgs.extend(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    )
    return msgs


def main(
    input: str = "/home/wangxingjian/DORO-STVG/graph_generator/output/query_train.jsonl",
    output: str = "/home/wangxingjian/DORO-STVG/trainer/LlamaFactory/data/vidstg_query_train_sharegpt.jsonl",
    fps: float = 2.0,
    media_dir: str = "",
    path_mode: str = "relative",
    assistant_format: str = "pixel",
    include_system: bool = False,
) -> None:
    if path_mode not in {"relative", "basename", "absolute"}:
        raise ValueError("path_mode must be one of: relative, basename, absolute")
    if assistant_format not in {"pixel", "norm", "both"}:
        raise ValueError("assistant_format must be one of: pixel, norm, both")

    in_path = Path(input)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, Any]] = []
    video_paths: List[str] = []
    with in_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw_rows.append(obj)
            video_paths.append(str(obj.get("video_path") or obj.get("videopath") or ""))

    media_dir_path = Path(media_dir).resolve() if media_dir else _detect_media_dir(video_paths)

    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(raw_rows):
            query = str(row.get("query", "")).strip()
            # Prefer query_formatted.jsonl keys.
            video_ref_raw = str(row.get("videopath") or row.get("video_path") or "").strip()
            if not query or not video_ref_raw:
                continue

            video_width = row.get("video_width", row.get("Width"))
            video_height = row.get("video_height", row.get("Height"))

            # `query_train.jsonl` usually provides only `box`; `box_01` is optional.
            pixel_box_text = str(row.get("box", "")).strip() or None
            norm_box_text = str(row.get("box_01", "")).strip() or None

            # Backward compatibility for raw query.jsonl (boxes dict).
            boxes = row.get("boxes")
            if (not pixel_box_text or not norm_box_text) and isinstance(boxes, dict):
                if not pixel_box_text:
                    pixel_box_text = "The object box is: " + _format_boxes(
                        boxes=boxes,
                        fps=fps,
                        coord_decimals=1,
                    )
                if not norm_box_text:
                    norm_box_text = "The object box (0-1 normalized) is: " + _format_boxes(
                        boxes=boxes,
                        fps=fps,
                        coord_decimals=4,
                        video_width=video_width,
                        video_height=video_height,
                    )

            sample = {
                "messages": build_messages(
                    query=query,
                    pixel_box_text=pixel_box_text,
                    norm_box_text=norm_box_text,
                    assistant_format=assistant_format,
                    include_system=include_system,
                ),
                "videos": [_resolve_video_ref(video_ref_raw, media_dir=media_dir_path, path_mode=path_mode)],
                "query_id": row.get("queryid") or row.get("query_id") or f"query_{idx}",
                "video_width": video_width,
                "video_height": video_height,
                "source": "vidstg_query_train_jsonl",
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    print(f"[convert_vidstg] input={in_path}")
    print(f"[convert_vidstg] output={out_path}")
    print(f"[convert_vidstg] samples={written}")
    if media_dir_path is not None:
        print(f"[convert_vidstg] media_dir={media_dir_path}")
    else:
        print("[convert_vidstg] media_dir=(auto detect failed)")


if __name__ == "__main__":
    fire.Fire(main)
