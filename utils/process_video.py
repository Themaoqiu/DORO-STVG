"""视频处理工具 - 基于 qwen-vl-utils"""
import os
import math
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch

from qwen_vl_utils import fetch_video, calculate_video_frame_range, smart_nframes

logger = logging.getLogger(__name__)


def process_video(
    video_file: str,
    num_frames: int = 100,
    frames_upbound: int = -1,
    force_sample: bool = False,
    image_patch_size: int = 14,
) -> Tuple[np.ndarray, float, str, int, Dict[str, Any]]:
    """
    使用 qwen-vl-utils 处理视频,返回采样帧和时间映射信息
    
    Args:
        video_file: 视频文件路径
        split: (start_frame, end_frame) 视频片段范围,为None时使用整个视频
        num_frames: 基础采样帧数 (100帧)
        fps: 采样频率 (每秒采样多少帧)
        frames_upbound: 采样帧数上限,-1表示不限制
        force_sample: 是否强制使用 frames_upbound
        image_patch_size: 图像patch大小 (默认14)
        
    Returns:
        Tuple[
            video_array (np.ndarray): (T, C, H, W) 格式的视频帧
            video_duration (float): 视频时长 (秒)
            frame_time_str (str): 采样帧的时间戳字符串 "0.00s,0.40s,..."
            num_frames_sampled (int): 实际采样的帧数
            video_metadata (Dict): 包含 fps, frames_indices, total_num_frames 等
        ]
    """
    
    ele = {
        "video": video_file,
        "nframes": num_frames,
    }
    
    # 2. 调用 fetch_video 获得采样后的视频和元数据
    video_tensor, video_metadata = fetch_video(
        ele,
        image_patch_size=image_patch_size,
        return_video_metadata=True
    )
    
    # video_tensor shape: (T, C, H, W)
    # video_metadata 包含: fps, frames_indices, total_num_frames
    
    fps = video_metadata['fps']
    frames_indices = video_metadata['frames_indices']
    total_num_frames = video_metadata['total_num_frames']
    
    
    nframes_final = len(frames_indices)
    if frames_upbound > 0:
        if nframes_final > frames_upbound or force_sample:
            uniform_idx = np.linspace(0, total_num_frames - 1, frames_upbound, dtype=int)
            frames_indices = uniform_idx.tolist()
            nframes_final = len(frames_indices)
            logger.info(f"  Resampled to {nframes_final} frames (upbound: {frames_upbound})")
    
    # 5. 计算帧对应的时间戳 (秒)
    frame_times = [idx / fps for idx in frames_indices]
    
    # 6. 格式化时间戳字符串 "0.00s,0.40s,..."
    frame_time_str = ",".join([f"{t:.2f}s" for t in frame_times])
    
    # 7. 计算视频时长
    video_duration = (total_num_frames - 1) / fps
    
    
    # 9. 更新 metadata 用于后续使用
    video_metadata_updated = {
        'fps': fps,
        'frames_indices': frames_indices,
        'total_num_frames': total_num_frames,
        'num_frames_sampled': nframes_final,
        'frame_times': frame_times,
        'video_duration': video_duration,
        'frame_time_str': frame_time_str,
    }
  
    return video_tensor, video_duration, frame_time_str, nframes_final, video_metadata_updated


def convert_frame_index_to_time(frame_idx: int, fps: float) -> float:
    return frame_idx / fps


def convert_time_to_frame_index(time_sec: float, fps: float) -> int:
    return int(round(time_sec * fps))


def get_frame_time_from_metadata(frame_idx: int, video_metadata: Dict[str, Any]) -> float:
    """
    从元数据中获取采样帧对应的时间
    
    Args:
        frame_idx: 采样帧的索引 (0到num_frames_sampled-1)
        video_metadata: 视频元数据
        
    Returns:
        时间(秒)
    """
    if frame_idx >= len(video_metadata['frame_times']):
        return video_metadata['frame_times'][-1]
    
    return video_metadata['frame_times'][frame_idx]


def normalize_temporal_span_with_metadata(
    temporal_span: Tuple[int, int],
    video_metadata: Dict[str, Any]
) -> Tuple[int, int]:
    """
    使用视频元数据归一化时间跨度
    
    Args:
        temporal_span: (start_frame, end_frame) 原始帧索引
        video_metadata: 视频元数据,包含 fps 和 frames_indices
        
    Returns:
        (start_idx, end_idx) 采样帧索引范围 [0, num_frames_sampled-1]
    """
    start_frame, end_frame = temporal_span
    fps = video_metadata['fps']
    frames_indices = video_metadata['frames_indices']
    
    # 转换为秒数
    start_time = convert_frame_index_to_time(start_frame, fps)
    end_time = convert_frame_index_to_time(end_frame, fps)
    
    # 找到最近的采样帧
    frame_times = video_metadata['frame_times']
    start_idx = min(range(len(frame_times)), key=lambda i: abs(frame_times[i] - start_time))
    end_idx = min(range(len(frame_times)), key=lambda i: abs(frame_times[i] - end_time))
    
    return (start_idx, end_idx)