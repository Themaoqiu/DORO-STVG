import json
from pathlib import Path
from typing import List

from .base_dataset import BaseSTVGDataset
from ..core.schema import STVGSample


class HCSTVGDataset(BaseSTVGDataset):
    
    def _load_annotations(self) -> List[dict]:
        return []
    
    def _parse_to_standard_format(self) -> List[STVGSample]:
        samples = []
        for item in self.raw_data:
            video_path = self._get_video_path(item.get('video_id', ''))
            sample = STVGSample(
                item_id=item.get('item_id', ''),
                video_path=video_path,
                query=item.get('query', ''),
                gt_temporal_bound=tuple(item.get('temporal_bound', (0, 0))),
                gt_bboxes=item.get('bboxes', {}),
                metadata=item.get('metadata', {})
            )
            samples.append(sample)
        return samples
    
    def _get_video_path(self, video_id: str) -> str:
        return str(self.video_dir / f"{video_id}.mp4")
