from pathlib import Path
from typing import List, Dict
import sys
import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.scene_detector import SceneClip
from modules.yolo_tracker import GlobalTrack
from modules.keyframe_clustering import KeyframeClustering


class SAM3Tracker:
    def __init__(
        self,
        model_path: str = "sam3.pt",
        conf: float = 0.25,
        redetection_interval: int = 15,
        iou_threshold: float = 0.4,
        overlap_threshold: float = 0.6,
        smooth_alpha: float = 0.7,
        smooth_window: int = 3,
    ):
        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=model_path,
            half=True,
            save=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.redetection_interval = redetection_interval
        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold
        self.smooth_alpha = smooth_alpha
        self.smooth_window = smooth_window
        self.keyframe_selector = KeyframeClustering(max_interval=redetection_interval)

    def enhance_tracks(
        self,
        video_path: str,
        global_tracks: List[GlobalTrack],
        clips: List[SceneClip],
    ) -> List[GlobalTrack]:
        cap = cv2.VideoCapture(video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        enhanced_tracks = []

        for g_track in global_tracks:
            all_frames_dict = g_track.all_frames

            keyframes = self.keyframe_selector.select_keyframes(
                all_frames_dict, video_width, video_height
            )

            if len(keyframes) < 2:
                enhanced_tracks.append(g_track)
                continue

            sam3_bboxes = self._propagate_track(
                video_path,
                g_track,
                keyframes,
                all_frames_dict,
            )

            enhanced_track = self._merge_yolo_sam3(g_track, sam3_bboxes)
            enhanced_tracks.append(enhanced_track)

        return enhanced_tracks

    def _propagate_track(
        self,
        video_path: str,
        track: GlobalTrack,
        keyframes: List[int],
        yolo_frames: Dict[int, Dict],
    ) -> Dict[int, List[float]]:
        all_bboxes = {}

        for i in range(len(keyframes)):
            keyframe_global = keyframes[i]
            next_keyframe = keyframes[i + 1] if i + 1 < len(keyframes) else track.end_frame

            segment_bboxes = self._propagate_segment(
                video_path,
                keyframe_global,
                next_keyframe,
                yolo_frames,
            )

            all_bboxes.update(segment_bboxes)

        smoothed_bboxes = self._temporal_smooth(all_bboxes)
        return smoothed_bboxes

    def _propagate_segment(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        yolo_frames: Dict[int, Dict],
    ) -> Dict[int, List[float]]:
        cap = cv2.VideoCapture(video_path)
        bboxes = {}

        if start_frame not in yolo_frames:
            cap.release()
            return bboxes

        initial_bbox = yolo_frames[start_frame]['box']
        frame_counter = 0

        for frame_idx in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            if frame_idx == start_frame:
                bboxes[frame_idx] = initial_bbox
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(frame_rgb)

            current_bbox = bboxes.get(frame_idx - 1, initial_bbox)

            try:
                results = self.predictor(bboxes=[current_bbox])

                if results and len(results) > 0 and hasattr(results[0], 'masks'):
                    masks = results[0].masks
                    if masks is not None and len(masks) > 0:
                        mask = masks.data[0].cpu().numpy()
                        sam_bbox = self._mask_to_bbox(mask)

                        frame_counter += 1

                        if frame_counter % self.redetection_interval == 0 and frame_idx in yolo_frames:
                            yolo_box = yolo_frames[frame_idx]['box']

                            if self._should_redetect(sam_bbox, yolo_box):
                                bboxes[frame_idx] = yolo_box
                                continue

                        bboxes[frame_idx] = sam_bbox
                    else:
                        if frame_idx in yolo_frames:
                            bboxes[frame_idx] = yolo_frames[frame_idx]['box']
                else:
                    if frame_idx in yolo_frames:
                        bboxes[frame_idx] = yolo_frames[frame_idx]['box']

            except Exception:
                if frame_idx in yolo_frames:
                    bboxes[frame_idx] = yolo_frames[frame_idx]['box']

        cap.release()
        return bboxes

    def _mask_to_bbox(self, mask: np.ndarray) -> List[float]:
        if mask.sum() == 0:
            return [0, 0, 0, 0]

        coords = np.where(mask > 0.5)
        y_coords, x_coords = coords

        if len(x_coords) == 0 or len(y_coords) == 0:
            return [0, 0, 0, 0]

        x1 = float(x_coords.min())
        y1 = float(y_coords.min())
        x2 = float(x_coords.max())
        y2 = float(y_coords.max())

        return [x1, y1, x2, y2]

    def _should_redetect(self, sam_box: List[float], yolo_box: List[float]) -> bool:
        iou = self._compute_iou(sam_box, yolo_box)
        if iou < self.iou_threshold:
            return True

        overlap_ratio = self._compute_overlap_ratio(sam_box, yolo_box)
        if overlap_ratio < self.overlap_threshold:
            return True

        return False

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _compute_overlap_ratio(self, box1: List[float], box2: List[float]) -> float:
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

        if box1_area <= 0:
            return 0.0

        return inter_area / box1_area

    def _temporal_smooth(self, bboxes: Dict[int, List[float]]) -> Dict[int, List[float]]:
        if len(bboxes) <= 1:
            return bboxes

        sorted_frames = sorted(bboxes.keys())
        smoothed = {}

        for i, frame_idx in enumerate(sorted_frames):
            if i == 0:
                smoothed[frame_idx] = bboxes[frame_idx]
                continue

            recent_frames = sorted_frames[max(0, i - self.smooth_window):i]
            recent_boxes = np.array([bboxes[f] for f in recent_frames])
            avg_box = np.mean(recent_boxes, axis=0)

            current_box = np.array(bboxes[frame_idx])
            smooth_box = self.smooth_alpha * current_box + (1 - self.smooth_alpha) * avg_box

            smoothed[frame_idx] = smooth_box.tolist()

        return smoothed

    def _merge_yolo_sam3(
        self,
        yolo_track: GlobalTrack,
        sam3_bboxes: Dict[int, List[float]],
    ) -> GlobalTrack:
        for local_track in yolo_track.local_tracks:
            for frame_idx in local_track.frames.keys():
                if frame_idx in sam3_bboxes:
                    local_track.frames[frame_idx]['box'] = sam3_bboxes[frame_idx]

        return yolo_track
