from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import cv2
import json
from scenedetect import detect, ContentDetector, AdaptiveDetector


@dataclass
class SceneClip:
    clip_id: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    shot_type: Optional[str] = None
    camera_motion: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            'clip_id': self.clip_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'duration': self.duration,
            'num_frames': self.num_frames,
        }
        if self.shot_type is not None:
            data['shot_type'] = self.shot_type
        if self.camera_motion is not None:
            data['camera_motion'] = self.camera_motion
        if self.metadata:
            data['metadata'] = self.metadata
        return data


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
        self.last_clips = None
    
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
        clips = self._to_clips(scenes)
        self.last_clips = clips
        return clips
    
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
    
    def save_to_jsonl(self, output_path: str, clips: Optional[List[SceneClip]] = None) -> None:
        if clips is None:
            clips = self.last_clips
        if clips is None:
            raise ValueError("没有可保存的片段，请先调用 detect()")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        video_name = self.video_path.stem
        record = {
            'video': video_name,
            'video_path': str(self.video_path),
            'clips': [clip.to_dict() for clip in clips],
        }
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    import traceback
    TEST_VIDEO = Path(__file__).resolve().parents[1] / "anno_videos" / "50_TM5MPJIq1Is_annotated_100frames.mp4"
    OUTPUT_JSONL = Path(__file__).resolve().parent / "scene_results.jsonl"

    try:
        if not TEST_VIDEO.exists():
            print(f"Test video not found: {TEST_VIDEO}")
        else:
            det = SceneDetector(str(TEST_VIDEO), detector_type="adaptive", threshold=3.0, min_scene_duration=0.5)
            clips = det.detect()
            print(f"Detected {len(clips)} clips:")
            for c in clips:
                print(f"  id={c.clip_id} start={c.start_time:.3f}s end={c.end_time:.3f}s dur={c.duration:.3f}s frames={c.num_frames}")
            det.save_to_jsonl(str(OUTPUT_JSONL), clips)
            print(f"Saved JSONL to {OUTPUT_JSONL}")
    except Exception:
        print("Error while running detection:\n")
        traceback.print_exc()
