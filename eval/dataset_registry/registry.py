"""数据集注册器"""
from typing import Dict, Type
from .base_dataset import BaseSTVGDataset
from .vidstg_dataset import VidSTGDataset
from .hcstvg_dataset import HCSTVGDataset


class DatasetRegistry:
    """数据集注册和管理"""
    
    _registry: Dict[str, Type[BaseSTVGDataset]] = {
        'vidstg': VidSTGDataset,
        'hcstvg': HCSTVGDataset,
    }
    
    @classmethod
    def register(cls, name: str):
        """注册新数据集的装饰器"""
        def decorator(dataset_class: Type[BaseSTVGDataset]):
            cls._registry[name.lower()] = dataset_class
            return dataset_class
        return decorator
    
    @classmethod
    def build(cls, name: str, **kwargs) -> BaseSTVGDataset:
        """
        构建数据集实例
        
        Args:
            name: 数据集名称
            **kwargs: 传递给数据集构造函数的参数
            
        Returns:
            数据集实例
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise ValueError(
                f"Dataset '{name}' not found. "
                f"Available datasets: {list(cls._registry.keys())}"
            )
        
        dataset_class = cls._registry[name_lower]
        return dataset_class(**kwargs)
    
    @classmethod
    def list_datasets(cls) -> list:
        """列出所有已注册的数据集"""
        return list(cls._registry.keys())