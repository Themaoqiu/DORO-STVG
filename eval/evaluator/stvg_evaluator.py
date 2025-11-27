from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from ..core.schema import Result, STVGSample
from .metrics import compute_stvg_metrics


class STVGEvaluator:
    """统一的STVG评估器"""
    
    def __init__(
        self,
        dataset: 'BaseSTVGDataset',
        iou_thresholds: List[float] = [0.3, 0.5, 0.7],
        num_frames: int = 100,
        logger = None
    ):
        """
        Args:
            dataset: 数据集实例
            iou_thresholds: m_vIoU阈值列表
            num_frames: 视频采样帧数 (默认100)
            logger: 日志记录器
        """
        self.dataset = dataset
        self.iou_thresholds = iou_thresholds
        self.num_frames = num_frames
        self.logger = logger
        
        # 构建GT索引
        self.gt_index: Dict[str, STVGSample] = {
            sample.item_id: sample 
            for sample in dataset.samples
        }
        
        # 预测结果缓存
        self.predictions: Dict[str, Result] = {}
        
        self._log(f"[Evaluator] Initialized with {len(self.gt_index)} GT samples")
        self._log(f"[Evaluator] Using {num_frames} frames per video")

    def update(self, predictions: List[Result]):
        """更新预测结果"""
        for pred in predictions:
            self.predictions[pred.item_id] = pred
        
        self._log(f"[Evaluator] Updated {len(predictions)} predictions (total: {len(self.predictions)})")
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        计算评估指标
        
        Returns:
            {
                'm_tIoU': float,           # 平均时间IoU
                'm_sIoU': float,           # 平均空间IoU
                'm_vIoU': float,           # 平均视频级IoU
                'vIoU@0.3': float,         # Recall@0.3
                'vIoU@0.5': float,         # Recall@0.5
                'vIoU@0.7': float          # Recall@0.7
            }
            
            如果是VidSTG还会按qtype分组:
            {
                'declarative_m_tIoU': ...,
                'interrogative_m_tIoU': ...,
                ...
            }
        """
        if not self.predictions:
            self._log("[Warning] No predictions to evaluate!")
            return {}
        
        # 计算每个样本的指标
        metrics_per_sample = []
        for item_id, pred in self.predictions.items():
            if item_id not in self.gt_index:
                self._log(f"[Warning] No GT for prediction: {item_id}")
                continue
            
            gt = self.gt_index[item_id]
            sample_metrics = self._evaluate_single_sample(gt, pred)
            metrics_per_sample.append(sample_metrics)
        
        self._log(f"[Evaluator] Evaluated {len(metrics_per_sample)} samples")
        
        # 聚合指标
        return self._aggregate_metrics(metrics_per_sample)
    
    def _evaluate_single_sample(
        self, 
        gt: STVGSample, 
        pred: Result
    ) -> dict:
        video_metadata = gt.video_metadata
        if video_metadata is None:
            self._log(f"[Warning] No video metadata for GT: {gt.item_id}")
            fps = 30.0
        else:
            fps = video_metadata['fps']

        gt_span = gt.gt_temporal_bound

        gt_bboxes_normalized = self._normalize_spatial_bboxes(
            gt.gt_bboxes,
            gt.metadata.get('width', 1),
            gt.metadata.get('height', 1)
        )
        
        pred_span = pred.pred_temporal_bound
        pred_bboxes = pred.pred_bboxes
        
        # 计算指标
        metrics = compute_stvg_metrics(
            gt_span=gt_span,
            pred_span=pred_span,
            gt_bboxes=gt_bboxes_normalized,
            pred_bboxes=pred_bboxes,
            num_frames=self.num_frames
        )
        
        # 计算Recall指标
        recalls = {
            f"vIoU@{thresh}": int(metrics['m_vIoU'] >= thresh) 
            for thresh in self.iou_thresholds
        }
        
        return {
            'item_id': gt.item_id,
            'qtype': gt.qtype,
            **metrics,
            **recalls
        }
    
    def _normalize_spatial_bboxes(
        self,
        bboxes: Dict[int, List[List[float]]],
        width: int,
        height: int
    ) -> Dict[int, List[float]]:
        normalized_bboxes = {}
        
        for frame_id, boxes in bboxes.items():
            if boxes:
                box = boxes[0]
                normalized_box = [
                    box[0] / width,
                    box[1] / height,
                    box[2] / width,
                    box[3] / height
                ]
                normalized_bboxes[frame_id] = normalized_box
        
        return normalized_bboxes
    
    def _aggregate_metrics(self, metrics_list: List[dict]) -> Dict[str, float]:
        has_qtype = any(m['qtype'] is not None for m in metrics_list)
        
        if not has_qtype:
            return self._simple_aggregate(metrics_list)
        else:
            return self._grouped_aggregate(metrics_list)
    
    def _simple_aggregate(self, metrics_list: List[dict]) -> Dict[str, float]:
        result = {}
        
        keys = ['tIoU', 'sIoU', 'm_vIoU'] + [f"vIoU@{t}" for t in self.iou_thresholds]
        
        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                if key.startswith('vIoU@'):
                    result[key] = float(np.sum(values) / len(values))
                else:
                    # IoU指标: 求平均
                    result[f"m_{key}"] = float(np.mean(values))
            else:
                result[key] = 0.0
        
        return result
    
    def _grouped_aggregate(self, metrics_list: List[dict]) -> Dict[str, float]:
        groups = defaultdict(list)
        for m in metrics_list:
            groups[m['qtype']].append(m)
        
        result = {}
        for qtype, group_metrics in groups.items():
            group_result = self._simple_aggregate(group_metrics)
            for key, value in group_result.items():
                result[f"{qtype}_{key}"] = value
        
        return result
    
    def _log(self, message: str):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)