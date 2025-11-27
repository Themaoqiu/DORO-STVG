from typing import Dict, Type
from .base_dataset import BaseSTVGDataset
from .vidstg_dataset import VidSTGDataset
from .hcstvg_dataset import HCSTVGDataset


class DatasetRegistry:
    _registry: Dict[str, Type[BaseSTVGDataset]] = {
        'vidstg': VidSTGDataset,
        'hcstvg': HCSTVGDataset,
    }
    
    @classmethod
    def register(cls, name: str):
        def decorator(dataset_class: Type[BaseSTVGDataset]):
            cls._registry[name.lower()] = dataset_class
            return dataset_class
        return decorator
    
    @classmethod
    def build(cls, name: str, **kwargs) -> BaseSTVGDataset:
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
        return list(cls._registry.keys())