"""模型基类定义"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..core.schema import STVGSample, PredictionResult


class BaseSTVGModel(ABC):
    """STVG任务模型基类"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass
    
    @abstractmethod
    def predict_batch(
        self, 
        samples: List[STVGSample],
        **kwargs
    ) -> List[PredictionResult]:
        """
        批量预测
        
        Args:
            samples: 样本列表
            **kwargs: 额外参数
            
        Returns:
            预测结果列表
        """
        pass
    
    def preprocess_sample(self, sample: STVGSample) -> Dict[str, Any]:
        """
        预处理单个样本(可选重写)
        
        Args:
            sample: 输入样本
            
        Returns:
            模型输入字典
        """
        return {
            'video_path': sample.video_path,
            'query': sample.query,
            'item_id': sample.item_id
        }
    
    def postprocess_output(
        self, 
        model_output: Any, 
        sample: STVGSample
    ) -> PredictionResult:
        """
        后处理模型输出(需子类实现具体解析逻辑)
        
        Args:
            model_output: 模型原始输出
            sample: 对应的输入样本
            
        Returns:
            标准化预测结果
        """
        raise NotImplementedError