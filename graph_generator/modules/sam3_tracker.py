from typing import List, Dict, Tuple, Optional, Any
import json
import os
import torch
import cv2
import numpy as np

from modules.scene_detector import SceneClip
from modules.yolo_tracker import YOLOTrack, GlobalTrack
from sam3.sam3.model_builder import build_sam3_video_predictor
from sam3.sam3.model.sam3_tracker_utils import mask_to_box


class SAM3Tracker:
    def __init__(
        self,
        model_path: str = "sam3.pt",
        iou_threshold: float = 0.4,
        overlap_threshold: float = 0.6,
        mask_output_dir: Optional[str] = None,
        match_output_dir: Optional[str] = None,
        match_log_path: Optional[str] = None,
    ):
        self.video_predictor = build_sam3_video_predictor(checkpoint_path=model_path)
        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold
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

        all_global_tracks = []
        global_track_id = 0

        for clip in clips:
            clip_dets = detections.get(clip.clip_id, {})
            if not clip_dets:
                continue

            print(f"  Processing clip {clip.clip_id} (frames {clip.start_frame}-{clip.end_frame})...")
            clip_tracks, global_track_id = self._track_clip(
                video_path, clip, clip_dets, global_track_id
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
        session_response = self.video_predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path)
        )
        session_id = session_response["session_id"]

        tracked_instances: Dict[int, Dict[int, Dict[str, Any]]] = {}
        instance_classes: Dict[int, str] = {}
        current_obj_id = 0

        keyframes = sorted(set(clip_dets.keys()) | {clip.start_frame})
        if not keyframes:
            self.video_predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
            return [], start_global_id

        mask_store_frames = set(keyframes)
        first_frame = clip.start_frame
        first_dets = clip_dets.get(first_frame, [])
        print(f"    First frame {first_frame}: {len(first_dets)} detections")

        for det in first_dets:
            obj_id = current_obj_id
            current_obj_id += 1
            instance_classes[obj_id] = det["class"]
            tracked_instances[obj_id] = {}
            self._track_instance(
                session_id=session_id,
                start_frame=first_frame,
                clip=clip,
                obj_id=obj_id,
                det=det,
                tracked_instances=tracked_instances,
                mask_store_frames=mask_store_frames,
                direction="forward",
            )

        cap = None
        if self.match_output_dir:
            cap = cv2.VideoCapture(video_path)

        for frame_idx in keyframes:
            if frame_idx == first_frame:
                continue
            frame_dets = clip_dets.get(frame_idx, [])
            if not frame_dets:
                continue
            frame_image = None
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame_image = cap.read()
                if not ret:
                    frame_image = None
            sam3_boxes = self._collect_frame_boxes(tracked_instances, frame_idx)
            for det_index, det in enumerate(frame_dets):
                det_mask = self._get_detection_mask(
                    session_id=session_id,
                    clip_id=clip.clip_id,
                    frame_idx=frame_idx,
                    det=det,
                )
                match_id, match_info = self._match_existing_instance(
                    det_box=det["box"],
                    det_mask=det_mask,
                    det_class=det["class"],
                    frame_idx=frame_idx,
                    tracked_instances=tracked_instances,
                    instance_classes=instance_classes,
                )
                if self.match_log_path:
                    self._write_match_log(
                        clip_id=clip.clip_id,
                        frame_idx=frame_idx,
                        det_index=det_index,
                        det=det,
                        match_id=match_id,
                        match_info=match_info,
                        num_instances=len(tracked_instances),
                    )
                if self.match_output_dir and frame_image is not None:
                    self._save_match_debug(
                        frame_image,
                        clip_id=clip.clip_id,
                        frame_idx=frame_idx,
                        det_index=det_index,
                        det_box=det["box"],
                        det_class=det["class"],
                        match_id=match_id,
                        match_info=match_info,
                        sam3_boxes=sam3_boxes,
                    )

                if match_id is not None:
                    continue

                obj_id = current_obj_id
                current_obj_id += 1
                instance_classes[obj_id] = det["class"]
                tracked_instances[obj_id] = {}
                self._track_instance(
                    session_id=session_id,
                    start_frame=frame_idx,
                    clip=clip,
                    obj_id=obj_id,
                    det=det,
                    tracked_instances=tracked_instances,
                    mask_store_frames=mask_store_frames,
                    direction="both",
                )
                print(f"    New {det['class']} at frame {frame_idx}, obj_id={obj_id}")

        if cap is not None:
            cap.release()

        self.video_predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )
        global_tracks = self._convert_to_global_tracks(
            tracked_instances, instance_classes, clip, start_global_id
        )
        print(f"    Found {len(global_tracks)} objects in clip {clip.clip_id}")
        return global_tracks, start_global_id + len(global_tracks)

    def _track_instance(
        self,
        session_id: str,
        start_frame: int,
        clip: SceneClip,
        obj_id: int,
        det: Dict,
        tracked_instances: Dict[int, Dict[int, Dict[str, Any]]],
        mask_store_frames: set,
        direction: str,
    ) -> None:
        bbox = det["box"]
        det_class = det["class"]
        bbox_xywh = self._xyxy_to_xywh_norm(bbox)

        self.video_predictor.handle_request(
            request=dict(type="reset_session", session_id=session_id)
        )
        self.video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=start_frame,
                text=det_class,
                bounding_boxes=[bbox_xywh],
                bounding_box_labels=[1],
            )
        )

        tracked_instances[obj_id][start_frame] = {"box": bbox}

        self._propagate_and_collect(
            session_id=session_id,
            start_frame=start_frame,
            clip=clip,
            obj_id=obj_id,
            tracked_instances=tracked_instances,
            mask_store_frames=mask_store_frames,
            direction=direction,
        )

    def _propagate_and_collect(
        self,
        session_id: str,
        start_frame: int,
        clip: SceneClip,
        obj_id: int,
        tracked_instances: Dict[int, Dict[int, Dict[str, Any]]],
        mask_store_frames: set,
        direction: str,
    ):
        propagate_request = dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction=direction,
            start_frame_index=start_frame,
        )

        for response in self.video_predictor.handle_stream_request(propagate_request):
            frame_idx = response["frame_index"]
            if frame_idx < clip.start_frame or frame_idx > clip.end_frame:
                continue

            outputs = response.get("outputs", {})
            out_masks = outputs.get("out_binary_masks", None)

            if out_masks is None or len(out_masks) == 0:
                continue

            mask = np.array(out_masks[0], dtype=np.uint8)
            mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
            bbox_tensor = mask_to_box(mask_tensor)
            bbox = bbox_tensor[0, 0].tolist()

            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            tracked_instances[obj_id][frame_idx] = {
                "box": bbox,
                "mask": mask if frame_idx in mask_store_frames else None,
            }
            if frame_idx in mask_store_frames and self.mask_output_dir:
                self._save_mask(mask, clip.clip_id, obj_id, frame_idx, "track")

    def _get_detection_mask(
        self,
        session_id: str,
        clip_id: int,
        frame_idx: int,
        det: Dict,
    ) -> Optional[np.ndarray]:
        bbox_xywh = self._xyxy_to_xywh_norm(det["box"])
        self.video_predictor.handle_request(
            request=dict(type="reset_session", session_id=session_id)
        )
        response = self.video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_idx,
                text=det["class"],
                bounding_boxes=[bbox_xywh],
                bounding_box_labels=[1],
            )
        )
        outputs = response.get("outputs", {})
        out_masks = outputs.get("out_binary_masks", None)
        if out_masks is None or len(out_masks) == 0:
            return None
        mask = np.array(out_masks[0], dtype=np.uint8)
        if self.mask_output_dir:
            self._save_mask(mask, clip_id, -1, frame_idx, "det")
        return mask

    def _save_mask(self, mask: np.ndarray, clip_id: int, obj_id: int, frame_idx: int, tag: str) -> None:
        base_dir = self.mask_output_dir
        if not base_dir:
            return
        os.makedirs(base_dir, exist_ok=True)
        filename = f"clip_{clip_id}_obj_{obj_id}_frame_{frame_idx}_{tag}.png"
        path = os.path.join(base_dir, filename)
        cv2.imwrite(path, (mask.astype(np.uint8) * 255))

    def _write_match_log(
        self,
        clip_id: int,
        frame_idx: int,
        det_index: int,
        det: Dict,
        match_id: Optional[int],
        match_info: Dict[str, Any],
        num_instances: int,
    ) -> None:
        if not self.match_log_path:
            return
        os.makedirs(os.path.dirname(self.match_log_path) or ".", exist_ok=True)
        record = {
            "clip_id": clip_id,
            "frame_idx": frame_idx,
            "det_index": det_index,
            "det_class": det.get("class"),
            "det_box": det.get("box"),
            "match_id": match_id,
            "best_box": match_info.get("box"),
            "box_iou": match_info.get("box_iou"),
            "box_overlap": match_info.get("box_overlap"),
            "mask_iou": match_info.get("mask_iou"),
            "mask_overlap": match_info.get("mask_overlap"),
            "iou_threshold": self.iou_threshold,
            "overlap_threshold": self.overlap_threshold,
            "num_instances": num_instances,
        }
        with open(self.match_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _save_match_debug(
        self,
        frame: np.ndarray,
        clip_id: int,
        frame_idx: int,
        det_index: int,
        det_box: List[float],
        det_class: str,
        match_id: Optional[int],
        match_info: Dict[str, Any],
        sam3_boxes: List[Tuple[int, List[float]]],
    ) -> None:
        base_dir = self.match_output_dir
        if not base_dir:
            return
        os.makedirs(base_dir, exist_ok=True)

        image = frame.copy()
        for obj_id, box in sam3_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 128, 0), 1)
            cv2.putText(
                image,
                f"id:{obj_id}",
                (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 128, 0),
                1,
                cv2.LINE_AA,
            )
        det_x1, det_y1, det_x2, det_y2 = map(int, det_box)
        cv2.rectangle(image, (det_x1, det_y1), (det_x2, det_y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"det:{det_class}",
            (det_x1, max(0, det_y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        match_box = match_info.get("box")
        if match_box:
            mx1, my1, mx2, my2 = map(int, match_box)
            cv2.rectangle(image, (mx1, my1), (mx2, my2), (0, 0, 255), 2)
            label = f"id:{match_id} iou:{match_info.get('box_iou', 0):.2f} ov:{match_info.get('box_overlap', 0):.2f}"
            cv2.putText(
                image,
                label,
                (mx1, max(0, my1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        filename = f"clip_{clip_id}_frame_{frame_idx}_det_{det_index}.jpg"
        path = os.path.join(base_dir, filename)
        cv2.imwrite(path, image)

    @staticmethod
    def _collect_frame_boxes(
        tracked_instances: Dict[int, Dict[int, Dict[str, Any]]],
        frame_idx: int,
    ) -> List[Tuple[int, List[float]]]:
        boxes: List[Tuple[int, List[float]]] = []
        for obj_id, frames_dict in tracked_instances.items():
            frame_data = frames_dict.get(frame_idx)
            if not frame_data:
                continue
            box = frame_data.get("box")
            if box is None:
                continue
            boxes.append((obj_id, box))
        return boxes

    def _xyxy_to_xywh_norm(self, bbox: List[float]) -> List[float]:
        x1, y1, x2, y2 = bbox
        return [
            x1 / self.video_width,
            y1 / self.video_height,
            (x2 - x1) / self.video_width,
            (y2 - y1) / self.video_height,
        ]

    def _match_existing_instance(
        self,
        det_box: List[float],
        det_mask: Optional[np.ndarray],
        det_class: str,
        frame_idx: int,
        tracked_instances: Dict[int, Dict[int, Dict[str, Any]]],
        instance_classes: Dict[int, str],
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        best_match_id = None
        best_score = 0.0
        best_info: Dict[str, Any] = {}

        for obj_id, frames_dict in tracked_instances.items():
            if instance_classes[obj_id] != det_class:
                continue

            frame_data = frames_dict.get(frame_idx)
            if not frame_data:
                continue

            box = frame_data.get("box")
            if box is None:
                continue

            box_iou, box_overlap = self._box_box_metrics(box, det_box)
            if box_iou >= self.iou_threshold or box_overlap >= self.overlap_threshold:
                best_match_id = obj_id
            score = max(box_iou, box_overlap)
            if score > best_score:
                best_score = score
                best_info = {
                    "box": box,
                    "box_iou": box_iou,
                    "box_overlap": box_overlap,
                    "mask_iou": None,
                    "mask_overlap": None,
                }

            # mask = frame_data.get("mask")
            # if mask is None or det_mask is None:
            #     continue

            # mask_iou, mask_overlap = self._mask_mask_metrics(mask, det_mask)
            # if mask_iou >= self.iou_threshold or mask_overlap >= self.overlap_threshold:
            #     score = max(mask_iou, mask_overlap)
            #     if score > best_score:
            #         best_score = score
            #         best_match_id = obj_id

        return best_match_id, best_info

    def _mask_mask_metrics(self, mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[float, float]:
        if mask_a.shape != mask_b.shape:
            return 0.0, 0.0

        mask_a = mask_a.astype(bool)
        mask_b = mask_b.astype(bool)

        inter = np.count_nonzero(mask_a & mask_b)
        if inter == 0:
            return 0.0, 0.0

        area_a = np.count_nonzero(mask_a)
        area_b = np.count_nonzero(mask_b)
        union = area_a + area_b - inter

        iou = inter / union if union > 0 else 0.0
        overlap = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
        return iou, overlap

    @staticmethod
    def _box_box_metrics(box_a: List[float], box_b: List[float]) -> Tuple[float, float]:
        x1_min, y1_min, x1_max, y1_max = box_a
        x2_min, y2_min, x2_max, y2_max = box_b

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0.0, inter_xmax - inter_xmin) * max(0.0, inter_ymax - inter_ymin)
        area_a = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
        area_b = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
        union = area_a + area_b - inter_area

        iou = inter_area / union
        overlap = inter_area / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
        return iou, overlap

    def _convert_to_global_tracks(
        self,
        tracked_instances: Dict[int, Dict[int, Dict[str, Any]]],
        instance_classes: Dict[int, str],
        clip: SceneClip,
        start_global_id: int,
    ) -> List[GlobalTrack]:
        global_tracks = []
        global_track_id = start_global_id

        for obj_id, frames_dict in tracked_instances.items():
            if len(frames_dict) < 1:
                continue

            object_class = instance_classes[obj_id]
            print(f"    {object_class} obj_id={obj_id}: tracked {len(frames_dict)} frames")

            local_track = YOLOTrack(
                track_id=obj_id,
                object_class=object_class,
                clip_id=clip.clip_id,
                frames={
                    f_idx: {"box": frame_data["box"], "conf": 1.0}
                    for f_idx, frame_data in frames_dict.items()
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
