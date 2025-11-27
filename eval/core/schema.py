from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

@dataclass
class STVGSample:
    
    item_id: str                                    # 样本唯一ID
    video_path: str                                 # 视频文件完整路径
    query: str                                      # 文本查询描述
    gt_temporal_bound: Tuple[int, int]              # GT时间边界 (start_frame, end_frame)
    gt_bboxes: Dict[int, List[List[float]]]         # GT边界框 {frame_id: [[x1,y1,x2,y2]]}
    frame_time_mapping: Optional[str] = None  # "0.00s,0.40s,..." 时间戳字符串
    
    video_metadata: Optional[Dict[str, Any]] = None  # {fps, frames_indices, frame_times, ...}
    
    qtype: Optional[str] = None
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
    pred_temporal_bound: Optional[Tuple[int, int]] = None  # (start_frame, end_frame)
    pred_bboxes: Dict[int, List[float]] = field(default_factory=dict)  # {frame_idx: [x1,y1,x2,y2]}
    
    video_metadata: Optional[Dict[str, Any]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'item_id': self.item_id,
            'pred_temporal_bound': self.pred_temporal_bound,
            'pred_bboxes': self.pred_bboxes,
            'confidence': self.confidence,
            'metadata': self.metadata
        }