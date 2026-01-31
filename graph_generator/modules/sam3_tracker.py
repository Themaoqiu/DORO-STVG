from pathlib import Path
from typing import List, Dict, Tuple
import sys
import os
import torch
import cv2

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

sam3_path = project_root / "sam3"
if str(sam3_path) not in sys.path:
    sys.path.insert(0, str(sam3_path))

from modules.scene_detector import SceneClip
from modules.yolo_tracker import YOLOTrack, GlobalTrack
from sam3.model_builder import build_sam3_video_predictor
from sam3.model.sam3_tracker_utils import mask_to_box


class SAM3Tracker:
    def __init__(self, model_path: str = "sam3.pt", iou_threshold: float = 0.3):
        self.video_predictor = build_sam3_video_predictor(checkpoint_path=model_path)
        self.iou_threshold = iou_threshold

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

        tracked_instances: Dict[int, Dict[int, List[float]]] = {}
        instance_classes: Dict[int, str] = {}
        current_obj_id = 0

        sorted_frames = sorted(clip_dets.keys())
        if not sorted_frames:
            self.video_predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
            return [], start_global_id

        first_frame = sorted_frames[0]
        first_dets = clip_dets[first_frame]
        print(f"    First frame {first_frame}: {len(first_dets)} detections")

        for det in first_dets:
            bbox = det['box']
            det_class = det['class']
            bbox_xywh = self._xyxy_to_xywh_norm(bbox)

            response = self.video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=first_frame,
                    text=det_class,
                    bounding_boxes=[bbox_xywh],
                    bounding_box_labels=[1],
                )
            )

            obj_id = current_obj_id
            current_obj_id += 1
            instance_classes[obj_id] = det_class
            tracked_instances[obj_id] = {first_frame: bbox}

            self._propagate_and_collect(
                session_id, first_frame, clip, obj_id, tracked_instances
            )

        for frame_idx in sorted_frames[1:]:
            frame_dets = clip_dets[frame_idx]
            for det in frame_dets:
                bbox = det['box']
                det_class = det['class']

                match_id = self._find_matching_instance(bbox, det_class, frame_idx, tracked_instances, instance_classes)

                if match_id is None:
                    bbox_xywh = self._xyxy_to_xywh_norm(bbox)
                    self.video_predictor.handle_request(
                        request=dict(type="reset_session", session_id=session_id)
                    )
                    self.video_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=frame_idx,
                            text=det_class,
                            bounding_boxes=[bbox_xywh],
                            bounding_box_labels=[1],
                        )
                    )

                    obj_id = current_obj_id
                    current_obj_id += 1
                    instance_classes[obj_id] = det_class
                    tracked_instances[obj_id] = {frame_idx: bbox}

                    self._propagate_and_collect(
                        session_id, frame_idx, clip, obj_id, tracked_instances
                    )
                    print(f"    New {det_class} at frame {frame_idx}, obj_id={obj_id}")

        self.video_predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )

        global_tracks = self._convert_to_global_tracks(
            tracked_instances, instance_classes, clip, start_global_id
        )
        print(f"    Found {len(global_tracks)} objects in clip {clip.clip_id}")
        return global_tracks, start_global_id + len(global_tracks)

    def _propagate_and_collect(
        self,
        session_id: str,
        start_frame: int,
        clip: SceneClip,
        obj_id: int,
        tracked_instances: Dict[int, Dict[int, List[float]]],
    ):
        propagate_request = dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction="both",
            start_frame_index=start_frame,
        )

        for response in self.video_predictor.handle_stream_request(propagate_request):
            frame_idx = response["frame_index"]
            if frame_idx < clip.start_frame or frame_idx > clip.end_frame:
                continue

            outputs = response.get("outputs", {})
            out_masks = outputs.get("out_binary_masks", None)

            if out_masks is not None and len(out_masks) > 0:
                mask_tensor = torch.tensor(out_masks[0]).unsqueeze(0).unsqueeze(0)
                bbox_tensor = mask_to_box(mask_tensor)
                bbox = bbox_tensor[0, 0].tolist()
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    tracked_instances[obj_id][frame_idx] = bbox

    def _xyxy_to_xywh_norm(self, bbox: List[float]) -> List[float]:
        x1, y1, x2, y2 = bbox
        return [
            x1 / self.video_width,
            y1 / self.video_height,
            (x2 - x1) / self.video_width,
            (y2 - y1) / self.video_height,
        ]

    def _find_matching_instance(
        self,
        det_bbox: List[float],
        det_class: str,
        frame_idx: int,
        tracked_instances: Dict[int, Dict[int, List[float]]],
        instance_classes: Dict[int, str],
    ) -> int:
        best_match_id = None
        best_iou = self.iou_threshold

        for obj_id, frames_dict in tracked_instances.items():
            if instance_classes[obj_id] != det_class:
                continue
            if frame_idx in frames_dict:
                tracked_bbox = frames_dict[frame_idx]
                iou = self._compute_iou(det_bbox, tracked_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = obj_id

        return best_match_id

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - inter_area

        return inter_area / union if union > 0 else 0

    def _convert_to_global_tracks(
        self,
        tracked_instances: Dict[int, Dict[int, List[float]]],
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
                    f_idx: {'box': bbox, 'conf': 1.0}
                    for f_idx, bbox in frames_dict.items()
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
