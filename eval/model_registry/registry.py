from typing import Dict, Type, Callable
from .base_model import BaseModel
from .qwen_family import Qwen3


class ModelRegistry:
    
    _registry: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, pattern: str):
        def decorator(builder_func: Callable):
            cls._registry[pattern.lower()] = builder_func
            return builder_func
        return decorator
    
    @classmethod
    def build(cls, model_name: str, **kwargs) -> BaseModel:
        model_name_lower = model_name.lower()
        
        for pattern, builder in cls._registry.items():
            if pattern in model_name_lower:
                return builder(model_name=model_name, **kwargs)
        
        raise ValueError(
            f"Model '{model_name}' not registered. "
            f"Available patterns: {list(cls._registry.keys())}"
        )
    
    @classmethod
    def list_models(cls) -> list:
        return list(cls._registry.keys())



@ModelRegistry.register("qwen")
def build_qwen_model(model_name: str, **kwargs) -> Qwen3:
    return Qwen3(model_name=model_name, **kwargs)
