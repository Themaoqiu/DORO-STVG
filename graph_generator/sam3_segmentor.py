from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import cv2
import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator

from .scene_detector import SceneClip


@dataclass
class ObjectTrack:
    object_id: int
    clip_id: int
    frames: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    def add_frame(self, frame_idx: int, mask: np.ndarray, box: np.ndarray, score: float):
        self.frames[frame_idx] = {
            'mask': mask,
            'box': box.tolist() if isinstance(box, np.ndarray) else box,
            'score': score,
        }
    
    def to_dict(self, save_masks: bool = False) -> Dict[str, Any]:
        data = {
            'object_id': self.object_id,
            'clip_id': self.clip_id,
            'frame_indices': list(self.frames.keys()),
            'num_frames': len(self.frames),
        }
        if save_masks:
            data['frames'] = {
                k: {'box': v['box'], 'score': v['score']} 
                for k, v in self.frames.items()
            }
        return data


class SAM3Segmentor:
    def __init__(
        self,
        model_name: str = "facebook/sam3",
        dtype: torch.dtype = torch.bfloat16,
        text_prompt: str = "person",
        max_frames_per_shot: int = 50,
    ):
        self.device = Accelerator().device
        self.dtype = dtype
        self.text_prompt = text_prompt
        self.max_frames_per_shot = max_frames_per_shot
        
        self.model = Sam3VideoModel.from_pretrained(model_name).to(self.device, dtype=dtype)
        self.processor = Sam3VideoProcessor.from_pretrained(model_name)
    
    def extract_shot_frames(self, video_path: str, clip: SceneClip) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame)
        
        for _ in range(clip.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        return frames
    
    def select_keyframe(self, frames: List[np.ndarray], method: str = "middle") -> int:
        if method == "middle":
            return len(frames) // 2
        elif method == "first":
            return 0
        elif method == "last":
            return len(frames) - 1
        return len(frames) // 2
    
    def process_shot(self, frames: List[np.ndarray], clip: SceneClip) -> List[ObjectTrack]:
        if not frames:
            return []
        
        inference_session = self.processor.init_video_session(
            video=frames,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )
        
        inference_session = self.processor.add_text_prompt(
            inference_session=inference_session,
            text=self.text_prompt,
        )
        
        outputs_per_frame = {}
        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=min(self.max_frames_per_shot, len(frames)),
        ):
            processed = self.processor.postprocess_outputs(inference_session, model_outputs)
            outputs_per_frame[model_outputs.frame_idx] = processed
        
        return self._build_tracks(outputs_per_frame, clip)
    
    def _build_tracks(self, outputs_per_frame: Dict[int, Dict], clip: SceneClip) -> List[ObjectTrack]:
        tracks: Dict[int, ObjectTrack] = {}
        
        for frame_idx, outputs in outputs_per_frame.items():
            object_ids = outputs['object_ids'].tolist()
            masks = outputs['masks'].cpu().numpy()
            boxes = outputs['boxes'].cpu().numpy()
            scores = outputs['scores'].tolist()
            
            for i, obj_id in enumerate(object_ids):
                if obj_id not in tracks:
                    tracks[obj_id] = ObjectTrack(object_id=obj_id, clip_id=clip.clip_id)
                
                global_frame = clip.start_frame + frame_idx
                tracks[obj_id].add_frame(
                    frame_idx=global_frame,
                    mask=masks[i],
                    box=boxes[i],
                    score=scores[i],
                )
        
        return list(tracks.values())
    
    def process_video(self, video_path: str, clips: List[SceneClip]) -> Dict[int, List[ObjectTrack]]:
        all_tracks = {}
        
        for clip in clips:
            frames = self.extract_shot_frames(video_path, clip)
            tracks = self.process_shot(frames, clip)
            all_tracks[clip.clip_id] = tracks
        
        return all_tracks
    
    def save_tracks_to_jsonl(
        self,
        output_path: str,
        video_path: str,
        all_tracks: Dict[int, List[ObjectTrack]],
    ) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video_path).stem
        record = {
            'video': video_name,
            'video_path': video_path,
            'text_prompt': self.text_prompt,
            'clips': {
                clip_id: [track.to_dict() for track in tracks]
                for clip_id, tracks in all_tracks.items()
            },
        }
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    def save_masks(
        self,
        output_dir: str,
        video_path: str,
        all_tracks: Dict[int, List[ObjectTrack]],
    ) -> None:
        output_path = Path(output_dir)
        video_name = Path(video_path).stem
        
        for clip_id, tracks in all_tracks.items():
            for track in tracks:
                for frame_idx, frame_data in track.frames.items():
                    mask_dir = output_path / video_name / f"clip_{clip_id}" / f"obj_{track.object_id}"
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    
                    mask = (frame_data['mask'] * 255).astype(np.uint8)
                    cv2.imwrite(str(mask_dir / f"frame_{frame_idx:06d}.png"), mask)
