"""数据集基类定义"""
from abc import ABC, abstractmethod
from typing import List, Iterator
from pathlib import Path
from ..core.schema import STVGSample


class BaseSTVGDataset(ABC):
    """STVG任务数据集基类"""
    
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
        
        # 加载并解析数据
        self.raw_data = self._load_annotations()
        self.samples = self._parse_to_standard_format()
        
        print(f"[{self.__class__.__name__}] Loaded {len(self.samples)} samples from {subset} set")
    
    @abstractmethod
    def _load_annotations(self) -> List[dict]:
        """
        加载原始标注数据
        
        Returns:
            原始数据列表
        """
        pass
    
    @abstractmethod
    def _parse_to_standard_format(self) -> List[STVGSample]:
        """
        将原始数据转换为STVGSample标准格式
        
        Returns:
            标准化样本列表
        """
        pass
    
    @abstractmethod
    def _get_video_path(self, video_id: str) -> str:
        """
        根据video_id获取完整视频路径
        
        Args:
            video_id: 视频唯一标识
            
        Returns:
            完整视频文件路径
        """
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> STVGSample:
        return self.samples[idx]
    
    def __iter__(self) -> Iterator[STVGSample]:
        return iter(self.samples)