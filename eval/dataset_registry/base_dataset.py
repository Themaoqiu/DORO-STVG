from abc import ABC, abstractmethod
from typing import List, Iterator
from pathlib import Path
from ..core.schema import STVGSample


class BaseSTVGDataset(ABC):
    def __init__(
        self, 
        annotation_path: str,      # JSON标注文件或缓存文件路径
        video_dir: str,            # 视频文件夹路径
        subset: str = "test",      # 数据子集: train/val/test
        **kwargs
    ):
        self.annotation_path = Path(annotation_path)
        self.video_dir = Path(video_dir)
        self.subset = subset
        self.kwargs = kwargs
        
        self.raw_data = self._load_annotations()
        self.samples = self._parse_to_standard_format()
        
        print(f"[{self.__class__.__name__}] Loaded {len(self.samples)} samples from {subset} set")
    
    @abstractmethod
    def _load_annotations(self) -> List[dict]:
        pass
    
    @abstractmethod
    def _parse_to_standard_format(self) -> List[STVGSample]:
        pass
    
    @abstractmethod
    def _get_video_path(self, video_id: str) -> str:
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> STVGSample:
        return self.samples[idx]
    
    def __iter__(self) -> Iterator[STVGSample]:
        return iter(self.samples)