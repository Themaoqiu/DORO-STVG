import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import fire
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.groundingedsam2_tracker import GroundedSAM2Tracker
from modules.scene_detector import SceneClip, SceneDetector
from modules.yolo_detector import YOLOKeyframeDetector

from pycocotools import mask as mask_utils


class GroundedSAM2MaskExporter(GroundedSAM2Tracker):
    def track_video_with_masks(
        self,
        video_path: str,
        clips: List[SceneClip],
        detections: Dict[int, Dict[int, List]],
    ) -> Tuple[List[Any], List[List[Dict[str, Any]]]]:
        temp_dir = Path(tempfile.mkdtemp(prefix="gsam2_frames_"))
        try:
            frame_paths = self._extract_frames(video_path, temp_dir)
            inference_state = self.video_predictor.init_state(
                video_path=str(temp_dir),
                offload_video_to_cpu=True,
                async_loading_frames=True,
            )

            all_global_tracks: List[Any] = []
            all_frame_masks: Dict[int, Dict[int, np.ndarray]] = {}
            global_track_id = 0
            for clip in clips:
                clip_dets = detections.get(clip.clip_id, {}) or {}
                print(
                    f"  Processing clip {clip.clip_id} "
                    f"(frames {clip.start_frame}-{clip.end_frame})..."
                )
                clip_tracks, global_track_id, clip_frame_masks = self._track_clip_with_masks(
                    clip=clip,
                    clip_dets=clip_dets,
                    start_global_id=global_track_id,
                    frame_paths=frame_paths,
                    inference_state=inference_state,
                )
                all_global_tracks.extend(clip_tracks)
                for frame_idx, frame_masks in clip_frame_masks.items():
                    all_frame_masks.setdefault(frame_idx, {}).update(frame_masks)

            return all_global_tracks, self._build_video_rle_masks(
                frame_masks=all_frame_masks,
                num_frames=len(frame_paths),
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _track_clip_with_masks(
        self,
        clip: SceneClip,
        clip_dets: Dict[int, List],
        start_global_id: int,
        frame_paths: List[Path],
        inference_state: Dict[str, Any],
    ) -> Tuple[List[Any], int, Dict[int, Dict[int, np.ndarray]]]:
        tracked_instances: Dict[int, Dict[int, Dict[str, Any]]] = {}
        instance_classes: Dict[int, str] = {}
        frame_masks: Dict[int, Dict[int, Dict[str, Any]]] = {}
        next_obj_id = 0

        if clip.end_frame < clip.start_frame:
            return [], start_global_id, {}

        first_frame = clip.start_frame
        first_prompts: Dict[int, Dict[str, Any]] = {}
        for det_mask in self._collect_detection_masks(
            frame_paths, first_frame, clip_dets.get(first_frame, [])
        ):
            obj_id = next_obj_id
            next_obj_id += 1
            cls_name = det_mask["class"]
            first_prompts[obj_id] = {"mask": det_mask["mask"], "class": cls_name}
            instance_classes[obj_id] = cls_name

        if first_prompts:
            self._propagate_from_frame(
                clip=clip,
                start_frame=first_frame,
                prompt_objects=first_prompts,
                instance_classes=instance_classes,
                tracked_instances=tracked_instances,
                frame_masks=frame_masks,
                store_obj_ids=None,
                inference_state=inference_state,
            )

        step = self.redetection_interval
        for start_frame in range(first_frame + step, clip.end_frame + 1, step):
            det_masks = self._collect_detection_masks(
                frame_paths, start_frame, clip_dets.get(start_frame, [])
            )
            if not det_masks:
                continue

            existing_at_frame = {
                obj_id: {
                    "mask": info["mask"],
                    "class": info.get("class", instance_classes.get(obj_id, "object")),
                }
                for obj_id, info in frame_masks.get(start_frame, {}).items()
            }
            new_masks, updated_existing = self._filter_masks_by_containment(
                new_masks=det_masks,
                existing_masks=existing_at_frame,
            )
            if not new_masks:
                continue

            prompt_objects = dict(updated_existing)
            new_obj_ids: Set[int] = set()
            for det_mask in new_masks:
                obj_id = next_obj_id
                next_obj_id += 1
                cls_name = det_mask["class"]
                prompt_objects[obj_id] = {"mask": det_mask["mask"], "class": cls_name}
                instance_classes[obj_id] = cls_name
                new_obj_ids.add(obj_id)

            frame_masks[start_frame] = {
                obj_id: {
                    "mask": info["mask"],
                    "class": info["class"],
                    "box": self._mask_to_box(info["mask"]),
                }
                for obj_id, info in prompt_objects.items()
            }
            self._propagate_from_frame(
                clip=clip,
                start_frame=start_frame,
                prompt_objects=prompt_objects,
                instance_classes=instance_classes,
                tracked_instances=tracked_instances,
                frame_masks=frame_masks,
                store_obj_ids=new_obj_ids,
                inference_state=inference_state,
            )

        global_tracks = self._convert_to_global_tracks(
            tracked_instances=tracked_instances,
            instance_classes=instance_classes,
            clip=clip,
            start_global_id=start_global_id,
        )
        local_to_global_id = {
            local_track.track_id: g_track.global_id
            for g_track in global_tracks
            for local_track in g_track.local_tracks
        }
        frame_masks_by_global_id = self._remap_frame_masks_to_global_ids(
            frame_masks=frame_masks,
            local_to_global_id=local_to_global_id,
        )
        print(f"    Found {len(global_tracks)} objects in clip {clip.clip_id}")
        return global_tracks, start_global_id + len(global_tracks), frame_masks_by_global_id

    @staticmethod
    def _mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
        if mask_utils is None:
            raise ImportError(
                "pycocotools is required for RLE export. Please install it first."
            )
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {
            "size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": counts,
        }

    def _remap_frame_masks_to_global_ids(
        self,
        frame_masks: Dict[int, Dict[int, Dict[str, Any]]],
        local_to_global_id: Dict[int, int],
    ) -> Dict[int, Dict[int, np.ndarray]]:
        out: Dict[int, Dict[int, np.ndarray]] = {}
        for frame_idx, frame_obj_masks in frame_masks.items():
            remapped_obj_masks: Dict[int, np.ndarray] = {}
            for local_obj_id, info in frame_obj_masks.items():
                global_id = local_to_global_id.get(local_obj_id)
                if global_id is None:
                    continue
                raw_mask = info.get("mask")
                if raw_mask is None:
                    continue
                mask_np = np.array(raw_mask, dtype=np.uint8)
                if mask_np.sum() == 0:
                    continue
                remapped_obj_masks[global_id] = mask_np
            if remapped_obj_masks:
                out[frame_idx] = remapped_obj_masks
        return out

    def _build_video_rle_masks(
        self,
        frame_masks: Dict[int, Dict[int, np.ndarray]],
        num_frames: int,
    ) -> List[List[Dict[str, Any]]]:
        frame_rles: List[List[Dict[str, Any]]] = [[] for _ in range(num_frames)]
        for frame_idx in range(num_frames):
            obj_masks = frame_masks.get(frame_idx, {})
            if not obj_masks:
                continue
            frame_rles[frame_idx] = [
                self._mask_to_rle(obj_masks[global_id])
                for global_id in sorted(obj_masks.keys())
            ]
        return frame_rles


def export_video_masks(
    video_path: str,
    output_dir: str,
    yolo_model: str = "yolo26x.pt",
    scene_threshold: float = 3.0,
    min_scene_duration: float = 1.0,
    conf: float = 0.25,
    iou: float = 0.5,
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    sam2_checkpoint: str = None,
    redetection_interval: int = 15,
    sam2_iou_threshold: float = 0.4,
    sam2_overlap_threshold: float = 0.6,
) -> Path:
    if sam2_checkpoint is None:
        raise ValueError("sam2_checkpoint is required.")

    print("[1/3] Detecting scenes...")
    scene_detector = SceneDetector(
        video_path,
        threshold=scene_threshold,
        min_scene_duration=min_scene_duration,
    )
    clips = scene_detector.detect()
    print(f"  Found {len(clips)} scenes")

    print("[2/3] Detecting keyframes with YOLO...")
    keyframe_detector = YOLOKeyframeDetector(
        model_path=yolo_model,
        conf=conf,
        iou=iou,
        keyframe_interval=redetection_interval,
    )
    all_detections = keyframe_detector.detect_keyframes(video_path, clips)

    print("[3/3] Tracking with Grounded-SAM2 and exporting masks...")
    tracker = GroundedSAM2MaskExporter(
        sam2_model_cfg=sam2_model_cfg,
        sam2_checkpoint=sam2_checkpoint,
        iou_threshold=sam2_iou_threshold,
        overlap_threshold=sam2_overlap_threshold,
        redetection_interval=redetection_interval,
    )
    global_tracks, frame_masks_rle = tracker.track_video_with_masks(
        video_path, clips, all_detections
    )
    print(f"  Tracked {len(global_tracks)} objects")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / f"{Path(video_path).stem}_sam2_masks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(frame_masks_rle, f, ensure_ascii=False)
    print(f"Saved masks to {output_path}")
    return output_path


def run(
    video: str = None,
    video_dir: str = None,
    output_dir: str = "output/sam2_masks",
    yolo_model: str = "yolo26x.pt",
    scene_threshold: float = 3.0,
    min_scene_duration: float = 1.0,
    conf: float = 0.25,
    iou: float = 0.5,
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    sam2_checkpoint: str = None,
    redetection_interval: int = 15,
    sam2_iou_threshold: float = 0.4,
    sam2_overlap_threshold: float = 0.6,
):
    if video:
        export_video_masks(
            video_path=video,
            output_dir=output_dir,
            yolo_model=yolo_model,
            scene_threshold=scene_threshold,
            min_scene_duration=min_scene_duration,
            conf=conf,
            iou=iou,
            sam2_model_cfg=sam2_model_cfg,
            sam2_checkpoint=sam2_checkpoint,
            redetection_interval=redetection_interval,
            sam2_iou_threshold=sam2_iou_threshold,
            sam2_overlap_threshold=sam2_overlap_threshold,
        )
        return

    if video_dir:
        video_dir_path = Path(video_dir)
        videos = list(video_dir_path.glob("*.mp4")) + list(video_dir_path.glob("*.avi"))
        for idx, video_path in enumerate(videos):
            print(f"\n[{idx + 1}/{len(videos)}] Processing {video_path.name}")
            export_video_masks(
                video_path=str(video_path),
                output_dir=output_dir,
                yolo_model=yolo_model,
                scene_threshold=scene_threshold,
                min_scene_duration=min_scene_duration,
                conf=conf,
                iou=iou,
                sam2_model_cfg=sam2_model_cfg,
                sam2_checkpoint=sam2_checkpoint,
                redetection_interval=redetection_interval,
                sam2_iou_threshold=sam2_iou_threshold,
                sam2_overlap_threshold=sam2_overlap_threshold,
            )
        return

    raise ValueError("Please provide either --video or --video_dir.")


if __name__ == "__main__":
    fire.Fire(run)
