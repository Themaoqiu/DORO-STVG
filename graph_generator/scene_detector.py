from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import cv2
from scenedetect import detect, ContentDetector, AdaptiveDetector


@dataclass
class SceneClip:
    clip_id: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


class SceneDetector:
    def __init__(
        self,
        video_path: str,
        detector_type: str = "adaptive",
        threshold: float = 3.0,
        min_scene_duration: float = 1.0,
    ):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        self.detector_type = detector_type.lower()
        self.threshold = threshold
        self.min_scene_duration = min_scene_duration
        self._fps = self._get_fps()
    
    def _get_fps(self) -> float:
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
    
    def _get_video_duration(self) -> float:
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0
    
    def detect(self) -> List[SceneClip]:
        if self.detector_type == "content":
            detector = ContentDetector(threshold=self.threshold)
        else:
            detector = AdaptiveDetector(
                adaptive_threshold=self.threshold,
                min_scene_len=int(self.min_scene_duration * self._fps),
            )
        
        scenes = detect(str(self.video_path), detector)
        return self._to_clips(scenes)
    
    def _to_clips(self, scenes) -> List[SceneClip]:
        if not scenes:
            duration = self._get_video_duration()
            return [SceneClip(
                clip_id=0,
                start_time=0.0,
                end_time=duration,
                start_frame=0,
                end_frame=int(duration * self._fps),
            )]
        
        clips = []
        for i, (start, end) in enumerate(scenes):
            start_time = start.get_seconds()
            end_time = end.get_seconds()
            
            if end_time - start_time >= self.min_scene_duration:
                clips.append(SceneClip(
                    clip_id=i,
                    start_time=start_time,
                    end_time=end_time,
                    start_frame=int(start_time * self._fps),
                    end_frame=int(end_time * self._fps),
                ))
        
        return clips
