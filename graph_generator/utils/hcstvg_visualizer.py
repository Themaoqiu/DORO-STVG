import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


def load_annotations(json_path: Path) -> Dict[str, dict]:
    if not json_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {json_path}")
    with json_path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Annotation file must be a JSON object mapping video name to metadata")
    return data


def xywh_to_xyxy(box: List[float]) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    return int(x), int(y), int(x + w), int(y + h)


class HCSTVGVisualizer:
    """Visualize HC-STVG style annotations (bbox format: [x, y, w, h])."""

    def __init__(self, ann_path: Path, video_root: Path, output_root: Path):
        self.video_root = video_root
        self.output_root = output_root
        self.annotations = load_annotations(ann_path)

    def _get_annotation(self, video_name: str) -> Optional[dict]:
        # Try exact match
        if video_name in self.annotations:
            return self.annotations[video_name]
        # Try stem match
        stem = Path(video_name).stem
        for key, val in self.annotations.items():
            if Path(key).stem == stem:
                return val
        return None

    def visualize_video(
        self,
        video_name: str,
        output_path: Optional[Path] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_label: bool = True,
    ) -> Path:
        ann = self._get_annotation(video_name)
        if ann is None:
            raise ValueError(f"No annotation found for video: {video_name}")

        bboxes: List[List[float]] = ann.get("bbox", [])
        if not bboxes:
            raise ValueError(f"Annotation has no 'bbox' list for video: {video_name}")

        st_frame = int(ann.get("st_frame", 0))
        ed_frame = int(ann.get("ed_frame", st_frame + len(bboxes) - 1))

        video_path = self.video_root / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if output_path is None:
            output_path = self.output_root / f"{Path(video_name).stem}_vis.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            raise RuntimeError(f"Cannot create output video: {output_path}")

        frame_idx = 0
        written = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if st_frame <= frame_idx <= ed_frame:
                local_idx = frame_idx - st_frame
                if 0 <= local_idx < len(bboxes):
                    box = bboxes[local_idx]
                    if len(box) >= 4:
                        x1, y1, x2, y2 = xywh_to_xyxy(box)
                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(0, min(x2, width - 1))
                        y2 = max(0, min(y2, height - 1))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        if show_label:
                            label = ann.get("sub", "obj")
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
                            cv2.putText(frame, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            out.write(frame)
            written += 1
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames for {video_name}")

        cap.release()
        out.release()
        print(f"Saved visualization to {output_path} (frames written: {written})")
        return output_path

    def visualize_all(self, limit: Optional[int] = None):
        count = 0
        for key in self.annotations.keys():
            try:
                self.visualize_video(key)
                count += 1
                if limit is not None and count >= limit:
                    break
            except Exception as e:
                print(f"[WARN] Failed on {key}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize HC-STVG bounding boxes (xywh)")
    parser.add_argument("--ann", required=True, help="Path to annotation JSON (mapping video -> info)")
    parser.add_argument("--video-root", required=True, help="Directory containing videos")
    parser.add_argument("--output-dir", required=True, help="Directory to save visualized videos")
    parser.add_argument("--video", help="Single video name to visualize (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of videos when visualizing all")
    parser.add_argument("--no-label", action="store_true", help="Do not draw label text")
    return parser.parse_args()


def main():
    args = parse_args()
    ann_path = Path(args.ann)
    video_root = Path(args.video_root)
    output_root = Path(args.output_dir)

    viz = HCSTVGVisualizer(ann_path, video_root, output_root)

    if args.video:
        viz.visualize_video(args.video, show_label=not args.no_label)
    else:
        viz.visualize_all(limit=args.limit)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
