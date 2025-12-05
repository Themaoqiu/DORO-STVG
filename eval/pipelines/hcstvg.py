import json
import logging
import cv2
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from pipelines.base_pipeline import BasePipeline
from utils.stvg_video_utils import process_video
from prompts import format_prompt, parse_response, SYSTEM_PROMPT
from utils.metrics import compute_metrics


logger = logging.getLogger(__name__)


class HCSTVGPipeline(BasePipeline):
    
    def get_dataset_name(self) -> str:
        return self.data_name
    
    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for video_name, anno in data.items():
            video_path = self.video_dir / video_name
            
            orig2sampled, sampled_indices = self._get_frame_mapping(str(video_path))
            
            st_frame_orig = anno['st_frame']
            ed_frame_orig = anno['ed_frame']
            gt_temporal_sampled = (
                self._map_frame_to_sampled(st_frame_orig, sampled_indices),
                self._map_frame_to_sampled(ed_frame_orig, sampled_indices)
            )
            
            width = anno['width']
            height = anno['height']
            
            gt_bboxes_sampled = {}
            # 检查原始 bbox 是否越界，如果越界则记录警告并跳过整个视频样本
            invalid_bbox = False
            for frame_idx, bbox in enumerate(anno['bbox'], start=st_frame_orig):
                x, y, w, h = bbox
                # 如果左上角小于 0，或右/下边界超过视频宽高，则视为越界
                if x < 0 or y < 0 or (x + w) > width or (y + h) > height:
                    logger.warning(
                        f"Video {video_name}: bbox {bbox} in frame {frame_idx} exceeds video bounds (width={width}, height={height}), skipping this video."
                    )
                    invalid_bbox = True
                    break

                x1_norm = x / width
                y1_norm = y / height
                x2_norm = (x + w) / width
                y2_norm = (y + h) / height
                bbox_normalized = [x1_norm, y1_norm, x2_norm, y2_norm]

                sampled_frame_idx = self._map_frame_to_sampled(frame_idx, sampled_indices)
                gt_bboxes_sampled[sampled_frame_idx] = bbox_normalized

            if invalid_bbox:
                continue
            
            samples.append({
                'video_name': video_name,
                'video_path': str(video_path),
                'query': anno.get('English', ''),
                'gt_temporal_sampled': gt_temporal_sampled,
                'gt_bboxes_sampled': gt_bboxes_sampled,
                'sampled_indices': sampled_indices.tolist(),
                'st_frame_orig': st_frame_orig,
                'ed_frame_orig': ed_frame_orig,
                'metadata': {
                    'img_size': anno['img_size'],
                    'width': anno['width'],
                    'height': anno['height'],
                    'st_time': anno['st_time'],
                    'ed_time': anno['ed_time'],
                }
            })
        
        logger.info(f"Loaded {len(samples)} samples")
        return samples