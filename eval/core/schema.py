from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

@dataclass
class STVGSample:
    
    item_id: str
    video_path: str
    query: str
    gt_temporal_bound: Tuple[int, int]
    gt_bboxes: Dict[int, List[float]]
    
    video_metadata: Optional[Dict[str, Any]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def qtype(self) -> Optional[str]:
        return self.metadata.get('qtype', None)
    
    @property
    def video_name(self) -> str:
        return self.metadata.get('video_name', self.item_id)
    
    @property
    def dataset_name(self) -> str:
        return self.metadata.get('dataset', 'unknown')
    
    def to_dict(self) -> dict:
        return {
            'item_id': self.item_id,
            'video_path': self.video_path,
            'query': self.query,
            'gt_temporal_bound': self.gt_temporal_bound,
            'gt_bboxes': self.gt_bboxes,
            'metadata': self.metadata
        }


@dataclass
class Result:
    item_id: str
    pred_temporal_bound: Optional[Tuple[int, int]] = None
    pred_bboxes: Dict[int, List[float]] = field(default_factory=dict)
    
    video_metadata: Optional[Dict[str, Any]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'item_id': self.item_id,
            "video_metadata": self.video_metadata,
            'pred_temporal_bound': self.pred_temporal_bound,
            'pred_bboxes': self.pred_bboxes,
            'metadata': self.metadata["raw_output"]
        }