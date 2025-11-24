"""标准化数据格式定义"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

@dataclass
class STVGSample:
    """STVG任务的标准化样本格式"""
    
    item_id: str                                    # 样本唯一ID
    video_path: str                                 # 视频文件完整路径
    query: str                                      # 文本查询描述
    gt_temporal_bound: Tuple[int, int]              # GT时间边界 (start_frame, end_frame)
    gt_bboxes: Dict[int, List[List[float]]]         # GT边界框 {frame_id: [[x1,y1,x2,y2]]}
    
    metadata: Dict[str, Any] = field(default_factory=dict)  # 数据集特有信息
    
    @property
    def qtype(self) -> Optional[str]:
        """兼容VidSTG的qtype字段"""
        return self.metadata.get('qtype', None)
    
    @property
    def video_name(self) -> str:
        """视频文件名(用于结果保存)"""
        return self.metadata.get('video_name', self.item_id)
    
    @property
    def dataset_name(self) -> str:
        """数据集名称"""
        return self.metadata.get('dataset', 'unknown')
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
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
    """模型预测结果的标准格式"""
    
    item_id: str                                    # 对应样本ID
    pred_temporal_bound: Tuple[int, int]            # 预测时间边界
    pred_bboxes: Dict[int, List[List[float]]]       # 预测边界框
    confidence: Optional[float] = None              # 置信度分数
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外信息
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'item_id': self.item_id,
            'pred_temporal_bound': self.pred_temporal_bound,
            'pred_bboxes': self.pred_bboxes,
            'confidence': self.confidence,
            'metadata': self.metadata
        }