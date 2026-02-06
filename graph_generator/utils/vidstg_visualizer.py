import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse


class VidSTGVisualizer:
    def __init__(self, ann_path: Path, video_root: Path, output_root: Path):
        self.video_root = video_root
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        with open(ann_path, 'r') as f:
            self.annotations = json.load(f)

    def _find_video(self, vid: str) -> Path:
        for ext in ['.flv', '.mp4', '.avi', '.mkv']:
            path = self.video_root / f"{vid}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Video not found: {vid}")

    def draw_bbox(
        self,
        frame: np.ndarray,
        bbox: dict,
        label: str = "",
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        x1, y1 = int(bbox['xmin']), int(bbox['ymin'])
        x2, y2 = int(bbox['xmax']), int(bbox['ymax'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    def visualize(self, ann_id: str) -> Path:
        ann = self.annotations[ann_id]
        vid = ann['vid']
        video_path = self._find_video(str(vid))

        ori_temp_gt = ann['ori_temp_gt']
        begin_fid = ori_temp_gt['begin_fid']
        end_fid = ori_temp_gt['end_fid']
        bboxs = ann['target_bboxs']
        label = ann.get('target_category', '')

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_w, out_h = w - w % 2, h - h % 2
        output_path = self.output_root / f"{ann_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if begin_fid <= idx <= end_fid:
                bbox_idx = idx - begin_fid
                if bbox_idx < len(bboxs):
                    frame = self.draw_bbox(frame, bboxs[bbox_idx], label)
            out.write(frame[:out_h, :out_w])
            idx += 1

        cap.release()
        out.release()
        print(f"Saved: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ann-id", required=True)
    args = parser.parse_args()

    viz = VidSTGVisualizer(Path(args.ann), Path(args.video_root), Path(args.output_dir))
    viz.visualize(args.ann_id)


if __name__ == "__main__":
    main()
