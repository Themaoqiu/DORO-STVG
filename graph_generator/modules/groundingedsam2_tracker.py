import shutil
import sys
import tempfile
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from .scene_detector import SceneClip
from .yolo_tracker import GlobalTrack, YOLOTrack

GRAPH_GENERATOR_ROOT = Path(__file__).resolve().parents[1]
GROUNDEDSAM2_ROOT = GRAPH_GENERATOR_ROOT / "dependence" / "GroundedSAM2"
if str(GROUNDEDSAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(GROUNDEDSAM2_ROOT))

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


class GroundedSAM2Tracker:
    def __init__(
        self,
        sam2_model_cfg: str,
        sam2_checkpoint: Optional[str] = None,
        iou_threshold: float = 0.4,
        overlap_threshold: float = 0.6,
        redetection_interval: int = 15,
        mask_output_dir: Optional[str] = None,
    ):
        self.sam2_root = GROUNDEDSAM2_ROOT
        if not self.sam2_root.exists():
            raise FileNotFoundError(f"GroundedSAM2 path not found: {self.sam2_root}")
        if sam2_checkpoint is None:
            raise ValueError("sam2_checkpoint is required for GroundedSAM2Tracker")
        if Path(sam2_model_cfg).is_absolute():
            raise ValueError(
                "sam2_model_cfg must be a Hydra config name, e.g. "
                "'configs/sam2.1/sam2.1_hiera_l.yaml', not an absolute file path."
            )

        self.sam2_model_cfg = sam2_model_cfg
        self.sam2_checkpoint = sam2_checkpoint
        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold
        self.redetection_interval = redetection_interval
        self.mask_output_dir = mask_output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.video_predictor = build_sam2_video_predictor(
            self.sam2_model_cfg,
            self.sam2_checkpoint,
            device=self.device,
        )
        image_model = build_sam2(
            self.sam2_model_cfg,
            self.sam2_checkpoint,
            device=self.device,
        )
        self.image_predictor = SAM2ImagePredictor(image_model)

    def track_video(
        self,
        video_path: str,
        clips: List[SceneClip],
        detections: Dict[int, Dict[int, List]],
    ) -> List[GlobalTrack]:
        temp_dir = Path(tempfile.mkdtemp(prefix="gsam2_frames_"))
        try:
            frame_paths = self._extract_frames(video_path, temp_dir)
            inference_state = self.video_predictor.init_state(
                video_path=str(temp_dir),
                offload_video_to_cpu=True,
                async_loading_frames=True,
            )

            all_global_tracks: List[GlobalTrack] = []
            all_frame_masks: Dict[int, Dict[int, np.ndarray]] = {}
            global_track_id = 0
            for clip in clips:
                clip_dets = detections.get(clip.clip_id, {}) or {}
                print(
                    f"  Processing clip {clip.clip_id} "
                    f"(frames {clip.start_frame}-{clip.end_frame})..."
                )
                clip_tracks, global_track_id, clip_frame_masks = self._track_clip(
                    clip=clip,
                    clip_dets=clip_dets,
                    start_global_id=global_track_id,
                    frame_paths=frame_paths,
                    inference_state=inference_state,
                )
                all_global_tracks.extend(clip_tracks)
                for frame_idx, frame_obj_masks in clip_frame_masks.items():
                    all_frame_masks.setdefault(frame_idx, {}).update(frame_obj_masks)
            if self.mask_output_dir:
                self._export_video_masks(
                    video_path=video_path,
                    frame_masks=all_frame_masks,
                    num_frames=len(frame_paths),
                )
            return all_global_tracks
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _track_clip(
        self,
        clip: SceneClip,
        clip_dets: Dict[int, List],
        start_global_id: int,
        frame_paths: List[Path],
        inference_state: Dict[str, Any],
    ) -> Tuple[List[GlobalTrack], int, Dict[int, Dict[int, np.ndarray]]]:
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
        print(f"    Found {len(global_tracks)} objects in clip {clip.clip_id}")
        local_to_global_id = {
            local_track.track_id: g_track.global_id
            for g_track in global_tracks
            for local_track in g_track.local_tracks
        }
        frame_masks_by_global_id = self._remap_frame_masks_to_global_ids(
            frame_masks=frame_masks,
            local_to_global_id=local_to_global_id,
        )
        return global_tracks, start_global_id + len(global_tracks), frame_masks_by_global_id

    def _collect_detection_masks(
        self,
        frame_paths: List[Path],
        frame_idx: int,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if frame_idx < 0 or frame_idx >= len(frame_paths):
            return []
        frame = cv2.imread(str(frame_paths[frame_idx]))
        if frame is None:
            return []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes: List[List[float]] = []
        classes: List[str] = []
        for det in detections:
            box = det.get("box")
            if box is None:
                continue
            boxes.append([float(v) for v in box])
            classes.append(self._resolve_det_class(det))
        if not boxes:
            return []

        input_boxes = torch.tensor(boxes, device=self.device, dtype=torch.float32)
        self.image_predictor.set_image(frame_rgb)
        masks, _, _ = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 2:
            masks = masks[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        out: List[Dict[str, Any]] = []
        for idx, mask in enumerate(masks):
            mask_np = np.array(mask > 0, dtype=np.uint8)
            if mask_np.sum() == 0:
                continue
            cls_name = classes[idx] if idx < len(classes) else "object"
            out.append({"mask": mask_np, "class": cls_name})
        return out

    def _propagate_from_frame(
        self,
        clip: SceneClip,
        start_frame: int,
        prompt_objects: Dict[int, Dict[str, Any]],
        instance_classes: Dict[int, str],
        tracked_instances: Dict[int, Dict[int, Dict[str, Any]]],
        frame_masks: Dict[int, Dict[int, Dict[str, Any]]],
        store_obj_ids: Optional[Set[int]],
        inference_state: Dict[str, Any],
    ) -> None:
        if not prompt_objects:
            return

        self.video_predictor.reset_state(inference_state)
        valid_obj_ids: List[int] = []
        for obj_id, info in prompt_objects.items():
            mask_np = np.array(info["mask"], dtype=np.uint8)
            if mask_np.sum() == 0:
                continue
            self.video_predictor.add_new_mask(
                inference_state,
                start_frame,
                obj_id,
                torch.from_numpy(mask_np > 0).to(self.device),
            )
            valid_obj_ids.append(obj_id)
            if store_obj_ids is None or obj_id in store_obj_ids:
                box = self._mask_to_box(mask_np)
                if box is not None:
                    tracked_instances.setdefault(obj_id, {})
                    tracked_instances[obj_id][start_frame] = {"box": box}
        if not valid_obj_ids:
            return

        max_len = clip.end_frame - start_frame + 1
        if max_len <= 0:
            return

        for frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame,
            max_frame_num_to_track=max_len,
        ):
            if frame_idx < clip.start_frame or frame_idx > clip.end_frame:
                continue
            target_frame = frame_masks.setdefault(frame_idx, {})
            for i, obj_id in enumerate(out_obj_ids):
                obj_id = int(obj_id)
                if store_obj_ids is not None and obj_id not in store_obj_ids:
                    continue
                out_mask = (out_mask_logits[i] > 0.0)
                if out_mask.sum() == 0:
                    continue
                mask_np = out_mask[0].detach().cpu().numpy().astype(np.uint8)
                box = self._mask_to_box(mask_np)
                if box is None:
                    continue

                cls_name = instance_classes.get(
                    obj_id, prompt_objects.get(obj_id, {}).get("class", "object")
                )
                tracked_instances.setdefault(obj_id, {})
                tracked_instances[obj_id][frame_idx] = {"box": box}
                target_frame[obj_id] = {"mask": mask_np, "class": cls_name, "box": box}

    def _filter_masks_by_containment(
        self,
        new_masks: List[Dict[str, Any]],
        existing_masks: Dict[int, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        filtered_new: List[Dict[str, Any]] = []
        updated_existing = dict(existing_masks)

        for det in new_masks:
            mask_np = det["mask"]
            should_add_new = True
            existing_to_remove: List[int] = []
            for obj_id, existing in list(updated_existing.items()):
                existing_mask = existing["mask"]
                if existing_mask is None or existing_mask.sum() == 0:
                    continue
                intersection = np.logical_and(mask_np, existing_mask)
                inter_cnt = int(intersection.sum())
                if inter_cnt == 0:
                    continue
                union_cnt = int(np.logical_or(mask_np, existing_mask).sum())
                iou = inter_cnt / union_cnt if union_cnt > 0 else 0.0
                new_cnt = int(mask_np.sum())
                exist_cnt = int(existing_mask.sum())
                new_in_exist = inter_cnt / new_cnt if new_cnt > 0 else 0.0
                exist_in_new = inter_cnt / exist_cnt if exist_cnt > 0 else 0.0

                has_high_iou = iou >= self.iou_threshold
                has_containment = (
                    new_in_exist >= self.overlap_threshold
                    or exist_in_new >= self.overlap_threshold
                )
                if has_high_iou:
                    should_add_new = False
                    break
                if has_containment:
                    if new_cnt > exist_cnt:
                        existing_to_remove.append(obj_id)
                    else:
                        should_add_new = False
                        break

            for obj_id in existing_to_remove:
                updated_existing.pop(obj_id, None)
            if should_add_new:
                filtered_new.append(det)
        return filtered_new, updated_existing

    @staticmethod
    def _resolve_det_class(det: Dict[str, Any]) -> str:
        return (
            det.get("mapped_class")
            or det.get("object_class")
            or det.get("class_name")
            or det.get("class")
            or "object"
        )

    @staticmethod
    def _mask_to_box(mask: np.ndarray) -> Optional[List[float]]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    @staticmethod
    def _extract_frames(video_path: str, output_dir: Path) -> List[Path]:
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        frame_paths: List[Path] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out_path = output_dir / f"{frame_idx}.jpg"
            cv2.imwrite(str(out_path), frame)
            frame_paths.append(out_path)
            frame_idx += 1
        cap.release()
        return frame_paths

    def _convert_to_global_tracks(
        self,
        tracked_instances: Dict[int, Dict[int, Dict[str, Any]]],
        instance_classes: Dict[int, str],
        clip: SceneClip,
        start_global_id: int,
    ) -> List[GlobalTrack]:
        global_tracks: List[GlobalTrack] = []
        global_track_id = start_global_id
        for obj_id, frames_dict in tracked_instances.items():
            if not frames_dict:
                continue
            object_class = instance_classes.get(obj_id, "object")
            local_track = YOLOTrack(
                track_id=obj_id,
                object_class=object_class,
                clip_id=clip.clip_id,
                frames={
                    frame_idx: {"box": frame_data["box"], "conf": 1.0}
                    for frame_idx, frame_data in frames_dict.items()
                },
            )
            global_track = GlobalTrack(
                global_id=global_track_id,
                object_class=object_class,
                local_tracks=[local_track],
            )
            global_tracks.append(global_track)
            global_track_id += 1
        return global_tracks

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

    @staticmethod
    def _mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {
            "size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": counts,
        }

    def _build_video_rle_masks_with_ids(
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
                {
                    "global_track_id": int(global_id),
                    "segmentation": self._mask_to_rle(obj_masks[global_id]),
                }
                for global_id in sorted(obj_masks.keys())
            ]
        return frame_rles

    def _export_video_masks(
        self,
        video_path: str,
        frame_masks: Dict[int, Dict[int, np.ndarray]],
        num_frames: int,
    ) -> None:
        output_dir = Path(self.mask_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(video_path).stem}_sam2_masks_indexed.json"
        payload = self._build_video_rle_masks_with_ids(
            frame_masks=frame_masks,
            num_frames=num_frames,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"  Saved Grounded-SAM2 masks to {output_path}")
