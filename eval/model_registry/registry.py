from typing import Dict, Type, Callable
from .base_model import BaseModel
from .qwen_family import Qwen2_5VL, Qwen3VL


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



@ModelRegistry.register("qwen2.5")
@ModelRegistry.register("qwen2_5vl")
def build_qwen_model(model_name: str, **kwargs) -> Qwen2_5VL:
    return Qwen2_5VL(model_name=model_name, **kwargs)


@ModelRegistry.register("qwen3")
def build_qwen3vl_model(model_name: str, **kwargs) -> Qwen3VL:
    return Qwen3VL(model_name=model_name, **kwargs)
