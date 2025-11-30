import os
import logging
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont

from qwen_vl_utils import process_vision_info, fetch_video

logger = logging.getLogger(__name__)


def annotate_frame_with_index(
    frame: np.ndarray,
    frame_idx: int,
    position: str = "bottom_right",
    font_size: int = 30,
    color: Tuple[int, int, int] = (255, 0, 0),
    font_path: Optional[str] = None,
) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
    else:
        frame_np = frame.astype(np.uint8)
    
    pil_image = Image.fromarray(frame_np)
    draw = ImageDraw.Draw(pil_image)
    
    if font_path is None:
        possible_fonts = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "C:/Windows/Fonts/arial.ttf",  # Windows
        ]
        font_path = next((f for f in possible_fonts if os.path.exists(f)), None)
    
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception as e:
        logger.warning(f"Failed to load font: {e}, using default")
        font = ImageFont.load_default()
    
    text = str(frame_idx)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    width, height = pil_image.size
    margin = 10
    
    if position == "top_left":
        x, y = margin, margin
    elif position == "top_right":
        x, y = width - text_width - margin, margin
    elif position == "bottom_left":
        x, y = margin, height - text_height - margin
    elif position == "bottom_right":
        x, y = width - text_width - margin, height - text_height - margin
    elif position == "center":
        x, y = (width - text_width) // 2, (height - text_height) // 2
    else:
        raise ValueError(f"Invalid position: {position}")
    
    if position in ["bottom_left", "bottom_right"]:
        y -= text_height // 3
    
    draw.text((x, y), text, font=font, fill=color)
    annotated_frame = np.array(pil_image)
    
    return annotated_frame


def save_annotated_video(
    video_file: str,
    output_file: str,
    frame_indices: Optional[List[int]] = None,
    position: str = "bottom_right",
    font_size: int = 40,
    color: Tuple[int, int, int] = (255, 0, 0),
    font_path: Optional[str] = None,
    resize_to: Optional[Tuple[int, int]] = None,
    fps_out: Optional[float] = None,
) -> Dict[str, Any]:
    try:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_file}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if resize_to:
            width, height = resize_to
        
        if fps_out is None:
            fps_out = fps
        
        if frame_indices is None:
            frame_indices = list(range(total_frames))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps_out, (width, height))
        
        logger.info(f"Processing video: {len(frame_indices)}/{total_frames} frames at {fps:.2f} fps")
        
        for sample_idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}")
                continue
            
            if resize_to:
                frame = cv2.resize(frame, (width, height))
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            annotated_frame = annotate_frame_with_index(
                frame=frame_rgb,
                frame_idx=sample_idx,
                position=position,
                font_size=font_size,
                color=color,
                font_path=font_path,
            )
            
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            out.write(annotated_frame_bgr)
            
            if (sample_idx + 1) % 100 == 0:
                logger.info(f"  Processed {sample_idx + 1}/{len(frame_indices)} frames")
        
        if 'annotated_frame_bgr' in locals():
            out.write(annotated_frame_bgr)
            
        cap.release()
        out.release()
        logger.info(f"Saved annotated video to: {output_file}")
        
        frame_times = [idx / fps for idx in frame_indices]
        video_metadata = {
            'fps': fps,
            'frames_indices': frame_indices,
            'total_num_frames': total_frames,
            'num_frames_sampled': len(frame_indices),
            'frame_times': frame_times,
            'video_duration': (total_frames - 1) / fps,
            'annotated_video_path': output_file,
        }
        
        return video_metadata
        
    except Exception as e:
        logger.error(f"Error processing video {video_file}: {e}")
        raise


def process_video(
    video_path: str,
    output_folder: str,
    num_frames: int = 100,
    annotate_frames: bool = True,
    annotation_position: str = "bottom_right",
    annotation_font_size: int = 40,
    annotation_color: Tuple[int, int, int] = (255, 255, 0),
    annotation_font_path: Optional[str] = None,
    max_pixels: int = 360 * 420,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    
    if annotate_frames:
        os.makedirs(output_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        annotated_video_path = os.path.join(output_folder, f"{base_name}_annotated_{num_frames}frames.mp4")

        if os.path.exists(annotated_video_path):
            logger.info(f"Using existing annotated video: {annotated_video_path}")
            video_path = annotated_video_path
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
            
            video_metadata = save_annotated_video(
                video_file=video_path,
                output_file=annotated_video_path,
                frame_indices=frame_indices,
                position=annotation_position,
                font_size=annotation_font_size,
                color=annotation_color,
                font_path=annotation_font_path,
                fps_out=1.0,
            )
            
            video_path = annotated_video_path
        logger.info(f"Using annotated video: {video_path}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "nframes": num_frames,
                    "max_pixels": max_pixels,
                }
            ]
        }
    ]
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True
    )
    
    
    if not annotate_frames:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        frame_times = [idx / fps for idx in frame_indices]
        
        video_metadata = {
            'fps': fps,
            'frames_indices': frame_indices,
            'total_num_frames': total_frames,
            'num_frames_sampled': num_frames,
            'frame_times': frame_times,
            'video_duration': (total_frames - 1) / fps,
        }
    
    logger.info(f"[VideoUtils] Loaded video: {video_path}")
    
    return video_inputs, video_metadata


def convert_frame_index_to_time(frame_idx: int, fps: float) -> float:
    return frame_idx / fps


def convert_time_to_frame_index(time_sec: float, fps: float) -> int:
    return int(round(time_sec * fps))


def get_frame_time_from_metadata(frame_idx: int, video_metadata: Dict[str, Any]) -> float:
    frame_times = video_metadata['frame_times']
    if frame_idx >= len(frame_times):
        return frame_times[-1]
    return frame_times[frame_idx]


def convert_sampled_frame_to_original(
    sampled_frame_idx: int,
    video_metadata: Dict[str, Any]
) -> int:
    frames_indices = video_metadata['frames_indices']
    if sampled_frame_idx >= len(frames_indices):
        return frames_indices[-1]
    return frames_indices[sampled_frame_idx]
