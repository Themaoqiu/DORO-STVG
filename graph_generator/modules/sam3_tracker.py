import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch

GRAPH_GENERATOR_ROOT = Path(__file__).resolve().parents[1]
DEPENDENCE_ROOT = GRAPH_GENERATOR_ROOT / "dependence"
if str(DEPENDENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPENDENCE_ROOT))

from modules.scene_detector import SceneClip
from modules.yolo_tracker import GlobalTrack, YOLOTrack
from sam3.sam3.model.sam3_tracker_utils import mask_to_box
from sam3.sam3.model_builder import build_sam3_video_predictor


class SAM3Tracker:
    def __init__(
        self,
        model_path: str = "sam3.pt",
        iou_threshold: float = 0.4,
        overlap_threshold: float = 0.6,
        redetection_interval: int = 15,
        mask_output_dir: Optional[str] = None,
        match_output_dir: Optional[str] = None,
        match_log_path: Optional[str] = None,
    ):
        self.video_predictor = build_sam3_video_predictor(checkpoint_path=model_path)
        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold
        self.redetection_interval = redetection_interval
        self.mask_output_dir = mask_output_dir
        self.match_output_dir = match_output_dir
        self.match_log_path = match_log_path

    def track_video(
        self,
        video_path: str,
        clips: List[SceneClip],
        detections: Dict[int, Dict[int, List]],
    ) -> List[GlobalTrack]:
        cap = cv2.VideoCapture(video_path)
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        all_global_tracks: List[GlobalTrack] = []
        global_track_id = 0

        for clip in clips:
            clip_dets = detections.get(clip.clip_id, {}) or {}
            print(f"  Processing clip {clip.clip_id} (frames {clip.start_frame}-{clip.end_frame})...")
            clip_tracks, global_track_id = self._track_clip(
                video_path=video_path,
                clip=clip,
                clip_dets=clip_dets,
                start_global_id=global_track_id,
            )
            all_global_tracks.extend(clip_tracks)
        return all_global_tracks

    def _track_clip(
        self,
        video_path: str,
        clip: SceneClip,
        clip_dets: Dict[int, List],
        start_global_id: int,
    ) -> Tuple[List[GlobalTrack], int]:
        track_session = self.video_predictor.handle_request(
            request={"type": "start_session", "resource_path": video_path}
        )["session_id"]
        det_session = self.video_predictor.handle_request(
            request={"type": "start_session", "resource_path": video_path}
        )["session_id"]

        tracked_instances: Dict[int, Dict[int, Dict[str, Any]]] = {}
        instance_classes: Dict[int, str] = {}
        frame_masks: Dict[int, Dict[int, Dict[str, Any]]] = {}
        next_obj_id = 0

        try:
            if clip.end_frame < clip.start_frame:
                return [], start_global_id

            first_frame = clip.start_frame

            # Step 1 (stvg-r1 style): detect on first frame and propagate to end of clip.
            first_det_masks = self._collect_detection_masks(det_session, first_frame, clip_dets.get(first_frame, []))
            first_prompts: Dict[int, Dict[str, Any]] = {}
            for det_mask in first_det_masks:
                obj_id = next_obj_id
                next_obj_id += 1
                cls_name = det_mask["class"]
                first_prompts[obj_id] = {"mask": det_mask["mask"], "class": cls_name}
                instance_classes[obj_id] = cls_name
            if first_prompts:
                self._propagate_from_frame(
                    session_id=track_session,
                    clip=clip,
                    start_frame=first_frame,
                    prompt_objects=first_prompts,
                    instance_classes=instance_classes,
                    tracked_instances=tracked_instances,
                    frame_masks=frame_masks,
                    store_obj_ids=None,
                )

            # Step 2 (stvg-r1 style): every N frames, detect new objects and only propagate new IDs.
            step = self.redetection_interval
            for start_frame in range(first_frame + step, clip.end_frame + 1, step):
                det_masks = self._collect_detection_masks(det_session, start_frame, clip_dets.get(start_frame, []))
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
                    session_id=track_session,
                    clip=clip,
                    start_frame=start_frame,
                    prompt_objects=prompt_objects,
                    instance_classes=instance_classes,
                    tracked_instances=tracked_instances,
                    frame_masks=frame_masks,
                    store_obj_ids=new_obj_ids,
                )
        finally:
            self.video_predictor.handle_request(
                request={"type": "close_session", "session_id": track_session}
            )
            self.video_predictor.handle_request(
                request={"type": "close_session", "session_id": det_session}
            )

        global_tracks = self._convert_to_global_tracks(
            tracked_instances=tracked_instances,
            instance_classes=instance_classes,
            clip=clip,
            start_global_id=start_global_id,
        )
        print(f"    Found {len(global_tracks)} objects in clip {clip.clip_id}")
        return global_tracks, start_global_id + len(global_tracks)

    def _collect_detection_masks(
        self,
        det_session: str,
        frame_idx: int,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for det in detections:
            mask = self._infer_mask_from_box(det_session, frame_idx, det)
            if mask is None or mask.sum() == 0:
                continue
            cls_name = self._resolve_det_class(det)
            out.append({"mask": mask, "class": cls_name})
        return out

    def _infer_mask_from_box(
        self,
        session_id: str,
        frame_idx: int,
        det: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        box = det.get("box")
        if box is None:
            return None
        self.video_predictor.handle_request(
            request={"type": "reset_session", "session_id": session_id}
        )
        bbox_xywh = self._xyxy_to_xywh_norm(box)
        response = self.video_predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_idx,
                "text": self._resolve_det_class(det),
                "bounding_boxes": [bbox_xywh],
                "bounding_box_labels": [1],
            }
        )
        outputs = response.get("outputs", {})
        out_masks = outputs.get("out_binary_masks")
        if out_masks is None or len(out_masks) == 0:
            return None
        return np.array(out_masks[0], dtype=np.uint8)

    def _propagate_from_frame(
        self,
        session_id: str,
        clip: SceneClip,
        start_frame: int,
        prompt_objects: Dict[int, Dict[str, Any]],
        instance_classes: Dict[int, str],
        tracked_instances: Dict[int, Dict[int, Dict[str, Any]]],
        frame_masks: Dict[int, Dict[int, Dict[str, Any]]],
        store_obj_ids: Optional[Set[int]],
    ) -> None:
        if not prompt_objects:
            return
        valid_prompt_objects: Dict[int, Dict[str, Any]] = {}
        for obj_id, info in prompt_objects.items():
            box = self._mask_to_box(info["mask"])
            if box is None:
                continue
            prompt_objects[obj_id]["box"] = box
            valid_prompt_objects[obj_id] = info
            if store_obj_ids is None or obj_id in store_obj_ids:
                tracked_instances.setdefault(obj_id, {})
                tracked_instances[obj_id][start_frame] = {"box": box}
        if not valid_prompt_objects:
            return

        # Dettrack-style seeding: use detected masks as stable instance prompts.
        self._seed_tracker_with_masks(
            session_id=session_id,
            frame_idx=start_frame,
            prompt_objects=valid_prompt_objects,
        )

        max_len = clip.end_frame - start_frame + 1
        if max_len <= 0:
            return
        for response in self.video_predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": start_frame,
                "max_frame_num_to_track": max_len,
            }
        ):
            frame_idx = response.get("frame_index")
            if frame_idx is None or frame_idx < clip.start_frame or frame_idx > clip.end_frame:
                continue
            outputs = response.get("outputs", {})
            out_masks = outputs.get("out_binary_masks")
            out_obj_ids = outputs.get("out_obj_ids")
            if out_masks is None or out_obj_ids is None:
                continue

            target_frame = frame_masks.setdefault(frame_idx, {})
            for obj_id, mask in zip(out_obj_ids, out_masks):
                obj_id = int(obj_id)
                if store_obj_ids is not None and obj_id not in store_obj_ids:
                    continue
                mask_np = np.array(mask, dtype=np.uint8)
                if mask_np.sum() == 0:
                    continue
                box = self._mask_to_box(mask_np)
                if box is None:
                    continue

                cls_name = instance_classes.get(obj_id, prompt_objects.get(obj_id, {}).get("class", "object"))
                tracked_instances.setdefault(obj_id, {})
                tracked_instances[obj_id][frame_idx] = {"box": box}
                target_frame[obj_id] = {"mask": mask_np, "class": cls_name, "box": box}

    def _seed_tracker_with_masks(
        self,
        session_id: str,
        frame_idx: int,
        prompt_objects: Dict[int, Dict[str, Any]],
    ) -> None:
        self.video_predictor.handle_request(
            request={"type": "reset_session", "session_id": session_id}
        )
        session = self.video_predictor._get_session(session_id)  # pylint: disable=protected-access
        inference_state = session["state"]
        model = self.video_predictor.model

        # Ensure tracker-side image features for this frame are ready before add_new_mask.
        model._prepare_backbone_feats(inference_state, frame_idx, reverse=False)  # pylint: disable=protected-access

        obj_ids = sorted(int(obj_id) for obj_id in prompt_objects.keys())
        if not obj_ids:
            return

        masks_np: List[np.ndarray] = []
        obj_id_to_mask: Dict[int, torch.Tensor] = {}
        for obj_id in obj_ids:
            mask_np = np.array(prompt_objects[obj_id]["mask"], dtype=np.uint8)
            if mask_np.sum() == 0:
                continue
            masks_np.append(mask_np.astype(np.float32))
            obj_id_to_mask[obj_id] = torch.from_numpy(mask_np > 0).unsqueeze(0).to(model.device)

        if not masks_np:
            return

        obj_ids = [obj_id for obj_id in obj_ids if obj_id in obj_id_to_mask]
        new_obj_masks = torch.from_numpy(np.stack(masks_np, axis=0)).to(model.device).float()

        inference_state["tracker_inference_states"] = model._tracker_add_new_objects(  # pylint: disable=protected-access
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            new_obj_ids=obj_ids,
            new_obj_masks=new_obj_masks,
            tracker_states_local=inference_state["tracker_inference_states"],
            orig_vid_height=inference_state["orig_height"],
            orig_vid_width=inference_state["orig_width"],
            feature_cache=inference_state["feature_cache"],
        )

        tracker_metadata = model._initialize_metadata()  # pylint: disable=protected-access
        world_size = model.world_size
        obj_ids_np = np.array(obj_ids, dtype=np.int64)
        obj_ids_per_gpu = [np.array([], np.int64) for _ in range(world_size)]
        obj_ids_per_gpu[0] = obj_ids_np
        num_obj_per_gpu = np.zeros(world_size, np.int64)
        num_obj_per_gpu[0] = len(obj_ids)

        tracker_metadata["obj_ids_per_gpu"] = obj_ids_per_gpu
        tracker_metadata["obj_ids_all_gpu"] = obj_ids_np
        tracker_metadata["num_obj_per_gpu"] = num_obj_per_gpu
        tracker_metadata["obj_id_to_score"] = {obj_id: 1.0 for obj_id in obj_ids}
        tracker_metadata["max_obj_id"] = max(obj_ids)
        for obj_id in obj_ids:
            tracker_metadata["obj_id_to_tracker_score_frame_wise"][frame_idx][obj_id] = 1.0
        if "rank0_metadata" in tracker_metadata:
            rank0 = tracker_metadata["rank0_metadata"]
            rank0["obj_first_frame_idx"] = {obj_id: frame_idx for obj_id in obj_ids}
            if "masklet_confirmation" in rank0:
                rank0["masklet_confirmation"]["status"] = np.zeros(len(obj_ids), dtype=np.int64)
                rank0["masklet_confirmation"]["consecutive_det_num"] = np.zeros(len(obj_ids), dtype=np.int64)

        inference_state["tracker_metadata"] = tracker_metadata
        model._cache_frame_outputs(inference_state, frame_idx, obj_id_to_mask)  # pylint: disable=protected-access
        model.add_action_history(
            inference_state,
            action_type="add",
            frame_idx=frame_idx,
            obj_ids=obj_ids,
        )

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

    def _xyxy_to_xywh_norm(self, bbox: List[float]) -> List[float]:
        x1, y1, x2, y2 = bbox
        return [
            x1 / self.video_width,
            y1 / self.video_height,
            (x2 - x1) / self.video_width,
            (y2 - y1) / self.video_height,
        ]

    @staticmethod
    def _mask_to_box(mask: np.ndarray) -> Optional[List[float]]:
        mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        bbox_tensor = mask_to_box(mask_tensor)
        bbox = bbox_tensor[0, 0].tolist()
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        return bbox

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
