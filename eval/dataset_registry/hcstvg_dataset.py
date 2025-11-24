import torch
from pathlib import Path
from typing import List
from .base_dataset import BaseSTVGDataset
from ..core.schema import STVGSample


class HCSTVGDataset(BaseSTVGDataset):
    """HCSTVG数据集适配器 (使用cache格式)"""
    TODO:# 改成json load形式
    def _load_annotations(self) -> List[dict]:
        """加载HCSTVG的cache文件"""
        if self.annotation_path.suffix == '.cache':
            cache_path = self.annotation_path
        else:
            cache_path = self.annotation_path / 'data_cache' / f'hcstvg-{self.subset}-anno.cache'
        
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")
        
        print(f"[HCSTVG] Loading annotations from: {cache_path}")
        return torch.load(cache_path)
    
    def _parse_to_standard_format(self) -> List[STVGSample]:
        """将HCSTVG格式转换为标准格式"""
        samples = []
        
        for item in self.raw_data:
            sample = STVGSample(
                item_id=item['item_id'],
                video_path=self._get_video_path(item['vid']),
                query=item['description'],
                gt_temporal_bound=tuple(item['gt_temp_bound']),
                gt_bboxes={int(k): [v] for k, v in item['bboxs'].items()},
                metadata={
                    'qtype': None,              # HCSTVG没有qtype
                    'video_name': item['vid'],
                    'vid': item['vid'],
                    'dataset': 'hcstvg'
                }
            )
            samples.append(sample)
        
        return samples
    
    def _get_video_path(self, video_id: str) -> str:
        """HCSTVG视频命名规则"""
        video_path = self.video_dir / f"{video_id}.mp4"
        
        if not video_path.exists():
            print(f"[Warning] Video not found: {video_path}")
        
        return str(video_path)