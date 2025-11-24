"""模型注册器"""
from typing import Dict, Type, Callable
from .base_model import BaseSTVGModel
from .qwen_family import Qwen3


class ModelRegistry:
    """模型注册和管理"""
    
    _registry: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, pattern: str):
        """注册模型的装饰器"""
        def decorator(builder_func: Callable):
            cls._registry[pattern.lower()] = builder_func
            return builder_func
        return decorator
    
    @classmethod
    def build(cls, model_name: str, **kwargs) -> BaseSTVGModel:
        """
        构建模型实例
        
        Args:
            model_name: 模型名称
            **kwargs: 模型参数
            
        Returns:
            模型实例
        """
        model_name_lower = model_name.lower()
        
        # 匹配模型类型
        for pattern, builder in cls._registry.items():
            if pattern in model_name_lower:
                return builder(model_name=model_name, **kwargs)
        
        raise ValueError(
            f"Model '{model_name}' not registered. "
            f"Available patterns: {list(cls._registry.keys())}"
        )
    
    @classmethod
    def list_models(cls) -> list:
        """列出已注册的模型模式"""
        return list(cls._registry.keys())



@ModelRegistry.register("qwen")
def build_qwen_model(model_name: str, **kwargs) -> Qwen3:
    """构建Qwen模型"""
    return Qwen3(model_name=model_name, **kwargs)
