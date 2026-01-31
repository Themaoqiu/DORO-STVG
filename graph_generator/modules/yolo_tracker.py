import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

import cv2
import numpy as np
from .scene_detector import SceneClip
from ultralytics import YOLO

os.environ['YOLO_VERBOSE'] = 'False'


@dataclass
class YOLOTrack:
    track_id: int
    object_class: str
    clip_id: int
    frames: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    def add_frame(self, frame_idx: int, box: List[float], conf: float, mask: Optional[np.ndarray] = None):
        self.frames[frame_idx] = {'box': box, 'conf': conf}
        if mask is not None:
            self.frames[frame_idx]['mask'] = mask
    
    @property
    def start_frame(self) -> int:
        return min(self.frames.keys()) if self.frames else 0
    
    @property
    def end_frame(self) -> int:
        return max(self.frames.keys()) if self.frames else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'track_id': self.track_id,
            'object_class': self.object_class,
            'clip_id': self.clip_id,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'num_frames': len(self.frames),
            'boxes': {k: v['box'] for k, v in self.frames.items()},
        }


@dataclass
class GlobalTrack:
    global_id: int
    object_class: str
    local_tracks: List[YOLOTrack] = field(default_factory=list)
    
    @property
    def all_frames(self) -> Dict[int, Dict]:
        merged = {}
        for t in self.local_tracks:
            merged.update(t.frames)
        return merged
    
    @property
    def start_frame(self) -> int:
        return min(self.all_frames.keys()) if self.all_frames else 0
    
    @property
    def end_frame(self) -> int:
        return max(self.all_frames.keys()) if self.all_frames else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'global_id': self.global_id,
            'object_class': self.object_class,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'num_frames': len(self.all_frames),
            'clip_ids': list(set(t.clip_id for t in self.local_tracks)),
        }


class YOLOTracker:
    def __init__(
        self,
        model_path: str = "yolo26x.pt",
        tracker_config: str = "botsort.yaml",
        conf: float = 0.3,
        iou: float = 0.5,
        use_seg: bool = False,
        gap_threshold: int = 5,
        min_track_length: int = 10,
    ):
        self.model = YOLO(model_path, verbose=False)
        self.tracker_config = tracker_config
        self.conf = conf
        self.iou = iou
        self.use_seg = use_seg
        self.gap_threshold = gap_threshold
        self.min_track_length = min_track_length
        self.class_names = self.model.names
    
    def track_shot(self, video_path: str, clip: SceneClip) -> Dict[int, YOLOTrack]:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame)
        
        tracks = {}
        self.model.predictor = None  # reset tracker state
        
        for offset in range(clip.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_config,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                imgsz=640,
            )[0]
            
            if results.boxes is None or not results.boxes.is_track:
                continue
            
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            classes = results.boxes.cls.int().cpu().tolist()
            confs = results.boxes.conf.cpu().tolist()
            
            masks = None
            if self.use_seg and results.masks is not None:
                masks = results.masks.data.cpu().numpy()
            
            for i, (box, tid, cls, conf) in enumerate(zip(boxes, track_ids, classes, confs)):
                if tid not in tracks:
                    tracks[tid] = YOLOTrack(
                        track_id=tid,
                        object_class=self.class_names[cls],
                        clip_id=clip.clip_id,
                    )
                
                global_frame = clip.start_frame + offset
                mask = masks[i] if masks is not None else None
                tracks[tid].add_frame(global_frame, box.tolist(), conf, mask)
        
        cap.release()

        return tracks


    def track_video(self, video_path: str, clips: List[SceneClip]) -> Dict[int, Dict[int, YOLOTrack]]:
        all_tracks = {}
        for clip in clips:
            all_tracks[clip.clip_id] = self.track_shot(video_path, clip)
        return all_tracks
    
    def merge_tracks(
        self,
        all_shot_tracks: Dict[int, Dict[int, YOLOTrack]],
        fps: float,
        time_gap: float = 1.0,
        iou_thresh: float = 0.3,
    ) -> List[GlobalTrack]:
        global_tracks = []
        global_id = 0
        
        sorted_clips = sorted(all_shot_tracks.keys())
        
        for clip_id in sorted_clips:
            shot_tracks = all_shot_tracks[clip_id]
            
            for local_track in shot_tracks.values():
                matched = False
                
                for g_track in global_tracks:
                    if self._can_merge(g_track, local_track, fps, time_gap, iou_thresh):
                        g_track.local_tracks.append(local_track)
                        matched = True
                        break
                
                if not matched:
                    new_global = GlobalTrack(
                        global_id=global_id,
                        object_class=local_track.object_class,
                        local_tracks=[local_track],
                    )
                    global_tracks.append(new_global)
                    global_id += 1
        
        return global_tracks
    
    def _can_merge(
        self,
        g_track: GlobalTrack,
        local_track: YOLOTrack,
        fps: float,
        time_gap: float,
        iou_thresh: float,
    ) -> bool:
        if g_track.object_class != local_track.object_class:
            return False
        
        g_end = g_track.end_frame
        l_start = local_track.start_frame
        
        gap_frames = l_start - g_end
        if gap_frames < 0 or gap_frames > time_gap * fps:
            return False
        
        g_frames = g_track.all_frames
        if g_end not in g_frames:
            return False
        l_frames = local_track.frames
        if l_start not in l_frames:
            return False
        
        box1 = g_frames[g_end]['box']
        box2 = l_frames[l_start]['box']
        
        return self._compute_iou(box1, box2) > iou_thresh
    
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

    def save_to_jsonl(
        self,
        output_path: str,
        video_path: str,
        global_tracks: List[GlobalTrack],
    ) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video_path).stem
        record = {
            'video': video_name,
            'video_path': video_path,
            'global_tracks': [t.to_dict() for t in global_tracks],
        }
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    import traceback

    from .scene_detector import SceneDetector
    
    TEST_VIDEO = Path(__file__).resolve().parents[1] / "anno_videos" / "50_TM5MPJIq1Is_annotated_100frames.mp4"
    OUTPUT_JSONL = Path(__file__).resolve().parent / "yolo_tracks.jsonl"
    
    try:
        if not TEST_VIDEO.exists():
            print(f"Test video not found: {TEST_VIDEO}")
        else:
            sd = SceneDetector(str(TEST_VIDEO), min_scene_duration=0.5)
            clips = sd.detect()
            print(f"Detected {len(clips)} clips")
            
            tracker = YOLOTracker(model_path="yolo11n.pt", conf=0.3)
            all_tracks = tracker.track_video(str(TEST_VIDEO), clips)
            
            total = sum(len(t) for t in all_tracks.values())
            print(f"Tracked {total} local tracks across {len(clips)} shots")
            
            global_tracks = tracker.merge_tracks(all_tracks, fps=sd._fps)
            print(f"Merged into {len(global_tracks)} global tracks")
            
            tracker.save_to_jsonl(str(OUTPUT_JSONL), str(TEST_VIDEO), global_tracks)
            print(f"Saved to {OUTPUT_JSONL}")
    except Exception:
        print("Error:\n")
        traceback.print_exc()
