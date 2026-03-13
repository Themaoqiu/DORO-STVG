import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

GRAPH_GENERATOR_ROOT = Path(__file__).resolve().parents[1]
if str(GRAPH_GENERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPH_GENERATOR_ROOT))

from modules.groundingedsam2_tracker import GroundedSAM2Tracker
from modules.scene_detector import SceneDetector, SceneClip

GROUNDEDSAM2_ROOT = GRAPH_GENERATOR_ROOT / "dependence" / "GroundedSAM2"
if str(GROUNDEDSAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(GROUNDEDSAM2_ROOT))

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class SAM2AutoSceneTracker(GroundedSAM2Tracker):
    def __init__(
        self,
        sam2_model_cfg: str,
        sam2_checkpoint: str,
        iou_threshold: float = 0.4,
        overlap_threshold: float = 0.6,
        redetection_interval: int = 15,
        points_per_side: int = 24,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.92,
        box_nms_thresh: float = 0.7,
        min_mask_region_area: int = 128,
        min_mask_area: int = 256,
        max_mask_area_ratio: float = 0.5,
        max_masks_per_frame: int = 24,
    ):
        super().__init__(
            sam2_model_cfg=sam2_model_cfg,
            sam2_checkpoint=sam2_checkpoint,
            iou_threshold=iou_threshold,
            overlap_threshold=overlap_threshold,
            redetection_interval=redetection_interval,
            mask_output_dir=None,
        )
        self.auto_mask_generator = SAM2AutomaticMaskGenerator(
            model=self.image_predictor.model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
            min_mask_region_area=min_mask_region_area,
            output_mode="binary_mask",
            multimask_output=True,
        )
        self.min_mask_area = int(min_mask_area)
        self.max_mask_area_ratio = float(max_mask_area_ratio)
        self.max_masks_per_frame = int(max_masks_per_frame)
        self.last_frame_masks: Dict[int, Dict[int, np.ndarray]] = {}

    def track_video(
        self,
        video_path: str,
        clips: List[SceneClip],
    ):
        temp_dir = Path(tempfile.mkdtemp(prefix="sam2_auto_frames_"))
        try:
            frame_paths = self._extract_frames(video_path, temp_dir)
            inference_state = self.video_predictor.init_state(
                video_path=str(temp_dir),
                offload_video_to_cpu=True,
                async_loading_frames=True,
            )

            all_global_tracks = []
            all_frame_masks: Dict[int, Dict[int, np.ndarray]] = {}
            global_track_id = 0
            for clip in clips:
                print(
                    f"  Processing clip {clip.clip_id} "
                    f"(frames {clip.start_frame}-{clip.end_frame})..."
                )
                clip_tracks, global_track_id, clip_frame_masks = self._track_clip(
                    clip=clip,
                    clip_dets={},
                    start_global_id=global_track_id,
                    frame_paths=frame_paths,
                    inference_state=inference_state,
                )
                all_global_tracks.extend(clip_tracks)
                for frame_idx, frame_obj_masks in clip_frame_masks.items():
                    all_frame_masks.setdefault(frame_idx, {}).update(frame_obj_masks)

            self.last_frame_masks = all_frame_masks
            return all_global_tracks
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _collect_detection_masks(
        self,
        frame_paths: List[Path],
        frame_idx: int,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        del detections
        if frame_idx < 0 or frame_idx >= len(frame_paths):
            return []

        frame = cv2.imread(str(frame_paths[frame_idx]))
        if frame is None:
            return []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask_records = self.auto_mask_generator.generate(frame_rgb)
        if not mask_records:
            return []

        frame_area = frame.shape[0] * frame.shape[1]
        filtered: List[Dict[str, Any]] = []
        for ann in mask_records:
            raw_mask = ann.get("segmentation")
            if raw_mask is None:
                continue
            mask_np = np.array(raw_mask, dtype=np.uint8)
            if mask_np.ndim != 2:
                continue
            area = int(ann.get("area", int(mask_np.sum())))
            if area <= 0 or area < self.min_mask_area:
                continue
            if area > frame_area * self.max_mask_area_ratio:
                continue
            filtered.append(
                {
                    "mask": (mask_np > 0).astype(np.uint8),
                    "class": "object",
                    "area": area,
                    "predicted_iou": float(ann.get("predicted_iou", 0.0)),
                    "stability_score": float(ann.get("stability_score", 0.0)),
                }
            )

        filtered.sort(
            key=lambda item: (
                item["predicted_iou"],
                item["stability_score"],
                item["area"],
            ),
            reverse=True,
        )
        if self.max_masks_per_frame > 0:
            filtered = filtered[: self.max_masks_per_frame]

        return [{"mask": item["mask"], "class": item["class"]} for item in filtered]


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    rng = random.Random(track_id)
    return (
        rng.randint(40, 255),
        rng.randint(40, 255),
        rng.randint(40, 255),
    )


def render_video(
    video_path: str,
    output_path: str,
    frame_masks: Dict[int, Dict[int, np.ndarray]],
    fps: Optional[float] = None,
    alpha: float = 0.45,
    use_ffmpeg: bool = True,
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output: Optional[str] = None
    if use_ffmpeg:
        temp_output = tempfile.NamedTemporaryFile(suffix=".avi", delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot create output video: {output_path}")

    frame_idx = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            obj_masks = frame_masks.get(frame_idx, {})
            annotated = frame.copy()
            for track_id in sorted(obj_masks.keys()):
                mask = np.array(obj_masks[track_id], dtype=np.uint8)
                if mask.ndim != 2 or mask.sum() == 0:
                    continue

                color = np.array(_color_for_track(track_id), dtype=np.uint8)
                overlay = annotated.copy()
                overlay[mask > 0] = color
                annotated = cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0)

                box = GroundedSAM2Tracker._mask_to_box(mask)
                if box is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                color_tuple = tuple(int(v) for v in color.tolist())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_tuple, 2)
                label = f"ID:{track_id}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                y_text_top = max(y1 - th - 8, 0)
                cv2.rectangle(
                    annotated,
                    (x1, y_text_top),
                    (x1 + tw + 6, y_text_top + th + 8),
                    color_tuple,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 3, y_text_top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(annotated)
            written += 1
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Rendered {frame_idx} frames")
    finally:
        cap.release()
        writer.release()

    if use_ffmpeg and temp_output is not None:
        print("Re-encoding with ffmpeg for better compatibility...")
        cmd = [
            "ffmpeg",
            "-i",
            temp_output,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(output_file),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(temp_output)
            print(f"Saved visualization to {output_file} (frames written: {written})")
        except subprocess.CalledProcessError as exc:
            print(f"FFmpeg failed, using temporary file: {temp_output}")
            print(exc.stderr.decode(errors="ignore"))
            shutil.move(temp_output, str(output_file))
            print(f"Saved visualization to {output_file} (frames written: {written})")
        except FileNotFoundError:
            print("FFmpeg not found, falling back to the temporary AVI output.")
            shutil.move(temp_output, str(output_file))
            print(f"Saved visualization to {output_file} (frames written: {written})")
    else:
        print(f"Saved visualization to {output_file} (frames written: {written})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM2 automatic segmentation on scene keyframes, feed the masks "
            "back into SAM2 video tracking, and save an annotated video."
        )
    )
    parser.add_argument("--video-path", required=True, help="Input video path")
    parser.add_argument(
        "--output-path",
        help="Output visualization video path. Defaults to output/<video>_sam2_auto.mp4",
    )
    parser.add_argument(
        "--sam2-model-cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 Hydra config name",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        required=True,
        help="SAM2 checkpoint path",
    )
    parser.add_argument("--scene-threshold", type=float, default=3.0)
    parser.add_argument("--min-scene-duration", type=float, default=0.5)
    parser.add_argument("--redetection-interval", type=int, default=15)
    parser.add_argument("--iou-threshold", type=float, default=0.4)
    parser.add_argument("--overlap-threshold", type=float, default=0.6)
    parser.add_argument("--points-per-side", type=int, default=24)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.7)
    parser.add_argument("--stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--min-mask-region-area", type=int, default=128)
    parser.add_argument("--min-mask-area", type=int, default=256)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.5)
    parser.add_argument("--max-masks-per-frame", type=int, default=24)
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="Disable ffmpeg re-encoding and write the video directly with OpenCV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path = args.output_path
    if not output_path:
        output_path = (
            GRAPH_GENERATOR_ROOT / "output" / f"{video_path.stem}_sam2_auto.mp4"
        )
    output_path = str(Path(output_path).expanduser().resolve())

    print("[1/3] Detecting scenes...")
    scene_detector = SceneDetector(
        str(video_path),
        threshold=args.scene_threshold,
        min_scene_duration=args.min_scene_duration,
    )
    clips = scene_detector.detect()
    print(f"  Found {len(clips)} scenes")

    print("[2/3] Running SAM2 auto-segmentation + tracking...")
    tracker = SAM2AutoSceneTracker(
        sam2_model_cfg=args.sam2_model_cfg,
        sam2_checkpoint=args.sam2_checkpoint,
        iou_threshold=args.iou_threshold,
        overlap_threshold=args.overlap_threshold,
        redetection_interval=args.redetection_interval,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        min_mask_region_area=args.min_mask_region_area,
        min_mask_area=args.min_mask_area,
        max_mask_area_ratio=args.max_mask_area_ratio,
        max_masks_per_frame=args.max_masks_per_frame,
    )
    global_tracks = tracker.track_video(str(video_path), clips)
    print(f"  Tracked {len(global_tracks)} objects")

    print("[3/3] Rendering visualization video...")
    render_video(
        video_path=str(video_path),
        output_path=output_path,
        frame_masks=tracker.last_frame_masks,
        alpha=args.mask_alpha,
        use_ffmpeg=not args.no_ffmpeg,
    )


if __name__ == "__main__":
    main()
