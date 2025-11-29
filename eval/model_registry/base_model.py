from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..core.schema import STVGSample, Result


class BaseSTVGModel(ABC):
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def predict_batch(
        self, 
        samples: List[STVGSample],
        **kwargs
    ) -> List[Result]:
        pass
    
    def preprocess_sample(self, sample: STVGSample) -> Dict[str, Any]:
        return {
            'video_path': sample.video_path,
            'query': sample.query,
            'item_id': sample.item_id
        }
    
    def postprocess_output(
        self, 
        model_output: Any, 
        sample: STVGSample
    ) -> Result:
        raise NotImplementedError