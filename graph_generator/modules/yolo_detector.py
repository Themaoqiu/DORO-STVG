from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import sys
import cv2
from ultralytics import YOLO

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.scene_detector import SceneClip


@dataclass
class YOLOTrack:
    track_id: int
    object_class: str
    clip_id: int
    frames: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    @property
    def start_frame(self) -> int:
        return min(self.frames.keys()) if self.frames else 0

    @property
    def end_frame(self) -> int:
        return max(self.frames.keys()) if self.frames else 0


@dataclass
class GlobalTrack:
    global_id: int
    object_class: str
    local_tracks: List[YOLOTrack] = field(default_factory=list)

    @property
    def all_frames(self) -> Dict[int, Dict[str, Any]]:
        merged: Dict[int, Dict[str, Any]] = {}
        for track in self.local_tracks:
            merged.update(track.frames)
        return merged

    @property
    def start_frame(self) -> int:
        return min(self.all_frames.keys()) if self.all_frames else 0

    @property
    def end_frame(self) -> int:
        return max(self.all_frames.keys()) if self.all_frames else 0


class YOLOKeyframeDetector:
    def __init__(
        self,
        model_path: str = "yolo26x.pt",
        conf: float = 0.3,
        iou: float = 0.5,
        keyframe_interval: int = 15,
    ):
        self.model = YOLO(model_path, verbose=False)
        self.conf = conf
        self.iou = iou
        self.keyframe_interval = keyframe_interval
        self.class_names = self.model.names

    def detect_keyframes(
        self,
        video_path: str,
        clips: List[SceneClip],
    ) -> Dict[int, Dict[int, List]]:
        cap = cv2.VideoCapture(video_path)
        all_detections = {}

        for clip in clips:
            clip_detections = {}

            for frame_idx in range(clip.start_frame, clip.end_frame + 1, self.keyframe_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                results = self.model(
                    frame,
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False,
                    imgsz=640,
                )[0]

                if results.boxes is None:
                    continue

                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.int().cpu().tolist()
                confs = results.boxes.conf.cpu().tolist()

                detections = []
                for box, cls, conf in zip(boxes, classes, confs):
                    class_name = self.class_names[cls]
                    x1, y1, x2, y2 = box.tolist()
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'class': class_name,
                        'conf': conf,
                    })

                if detections:
                    clip_detections[frame_idx] = detections

            all_detections[clip.clip_id] = clip_detections

        cap.release()
        return all_detections

    def detections_to_tracks(
        self,
        all_detections: Dict[int, Dict[int, List]],
    ) -> List[GlobalTrack]:
        global_tracks = []
        track_id = 0

        for clip_id, clip_detections in all_detections.items():
            for frame_idx, detections in clip_detections.items():
                for det in detections:
                    local_track = YOLOTrack(
                        track_id=track_id,
                        object_class=det['class'],
                        clip_id=clip_id,
                        frames={
                            frame_idx: {
                                'box': det['box'],
                                'conf': det['conf'],
                            }
                        },
                    )

                    global_track = GlobalTrack(
                        global_id=track_id,
                        object_class=det['class'],
                        local_tracks=[local_track],
                    )

                    global_tracks.append(global_track)
                    track_id += 1

        return global_tracks
