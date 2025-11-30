import json
from pathlib import Path
from typing import List, Dict

from .base_dataset import BaseSTVGDataset
from ..core.schema import STVGSample


class HCSTVGDataset(BaseSTVGDataset):
    
    def _load_annotations(self) -> List[dict]:
        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = []
        for video_name, anno in data.items():
            annotations.append({
                'video_name': video_name,
                'annotation': anno
            })
        return annotations
    
    def _parse_to_standard_format(self) -> List[STVGSample]:
        samples = []
        for item in self.raw_data:
            video_name = item['video_name']
            anno = item['annotation']
            
            video_path = self._get_video_path(video_name)
            st_frame = anno['st_frame']
            ed_frame = anno['ed_frame']
            
            gt_bboxes = {}
            for frame_idx, bbox in enumerate(anno['bbox'], start=st_frame):
                x, y, w, h = bbox
                gt_bboxes[frame_idx] = [x, y, x + w, y + h]
            
            sample = STVGSample(
                item_id=video_name,
                video_path=video_path,
                query=anno.get('caption', anno.get('English', '')),
                gt_temporal_bound=(st_frame, ed_frame),
                gt_bboxes=gt_bboxes,
                metadata={
                    'dataset': 'hcstvg',
                    'video_name': video_name,
                    'img_size': anno['img_size'],
                    'img_num': anno['img_num'],
                    'st_time': anno['st_time'],
                    'ed_time': anno['ed_time'],
                    'ed_offset': anno['ed_offset'],
                    'width': anno['width'],
                    'height': anno['height'],
                    'Chinese': anno.get('Chinese', ''),
                    'English': anno.get('English', ''),
                    'sub': anno.get('sub', ''),
                    'verb_index_list': anno.get('verb_index_list', []),
                    'adj_index_list': anno.get('adj_index_list', [])
                }
            )
            samples.append(sample)
        return samples
    
    def _get_video_path(self, video_name: str) -> str:
        return str(self.video_dir / video_name)
