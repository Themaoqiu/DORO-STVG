"""统一STVG评估器"""
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict

from ..core.schema import STVGSample, PredictionResult
from .metrics import compute_temporal_iou, compute_iou


class UnifiedSTVGEvaluator:
    """统一的STVG评估器,兼容VidSTG和HCSTVG"""
    
    def __init__(
        self,
        dataset: 'BaseSTVGDataset',
        iou_thresholds: List[float] = [0.3, 0.5],
        logger = None
    ):
        """
        Args:
            dataset: 数据集实例(包含GT信息)
            iou_thresholds: IoU阈值列表
            logger: 日志记录器
        """
        self.dataset = dataset
        self.iou_thresholds = iou_thresholds
        self.logger = logger
        
        # 构建GT索引
        self.gt_index: Dict[str, STVGSample] = {
            sample.item_id: sample 
            for sample in dataset.samples
        }
        
        # 预测结果缓存
        self.predictions: Dict[str, PredictionResult] = {}
        
        self._log(f"[Evaluator] Initialized with {len(self.gt_index)} GT samples")
    
    def update(self, predictions: List[PredictionResult]):
        """更新预测结果"""
        for pred in predictions:
            self.predictions[pred.item_id] = pred
        
        self._log(f"[Evaluator] Updated {len(predictions)} predictions (total: {len(self.predictions)})")
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        计算评估指标
        
        Returns:
            指标字典,格式取决于数据集类型:
            - VidSTG: 按qtype分组 {declarative_viou@0.5: 0.xx, interrogative_viou@0.5: 0.xx}
            - HCSTVG: 不分组 {viou@0.5: 0.xx, tiou: 0.xx}
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
        pred: PredictionResult
    ) -> dict:
        """评估单个样本"""
        # 1. 计算时间IoU
        tiou = compute_temporal_iou(
            gt.gt_temporal_bound, 
            pred.pred_temporal_bound
        )
        
        # 2. 计算空间指标
        gt_start, gt_end = gt.gt_temporal_bound
        pred_start, pred_end = pred.pred_temporal_bound
        
        union_frames = set(range(
            min(gt_start, pred_start),
            max(gt_end, pred_end)
        ))
        inter_frames = set(range(
            max(gt_start, pred_start),
            min(gt_end, pred_end)
        ))
        
        # 计算vIoU (video-level IoU)
        viou_sum = 0.0
        gt_viou_sum = 0.0
        
        for fid in gt.gt_bboxes.keys():
            if fid not in pred.pred_bboxes:
                continue
            
            # 计算该帧的IoU
            iou = compute_iou(
                np.array(pred.pred_bboxes[fid]),
                np.array(gt.gt_bboxes[fid])
            )[0][0]
            
            if fid in inter_frames:
                viou_sum += iou
            gt_viou_sum += iou
        
        viou = viou_sum / max(len(union_frames), 1)
        gt_viou = gt_viou_sum / max(len(gt.gt_bboxes), 1)
        
        # 3. 计算Recall指标
        recalls = {
            f"viou@{thresh}": int(viou > thresh) 
            for thresh in self.iou_thresholds
        }
        gt_recalls = {
            f"gt_viou@{thresh}": int(gt_viou > thresh)
            for thresh in self.iou_thresholds
        }
        
        return {
            'item_id': gt.item_id,
            'qtype': gt.qtype,
            'tiou': tiou,
            'viou': viou,
            'gt_viou': gt_viou,
            **recalls,
            **gt_recalls
        }
    
    def _aggregate_metrics(self, metrics_list: List[dict]) -> Dict[str, float]:
        """聚合指标"""
        has_qtype = any(m['qtype'] is not None for m in metrics_list)
        
        if not has_qtype:
            # HCSTVG模式: 直接平均
            return self._simple_aggregate(metrics_list)
        else:
            # VidSTG模式: 按qtype分组
            return self._grouped_aggregate(metrics_list)
    
    def _simple_aggregate(self, metrics_list: List[dict]) -> Dict[str, float]:
        """简单平均(用于HCSTVG)"""
        result = {}
        keys = ['tiou', 'viou', 'gt_viou'] + \
               [f"viou@{t}" for t in self.iou_thresholds] + \
               [f"gt_viou@{t}" for t in self.iou_thresholds]
        
        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            result[key] = float(np.mean(values)) if values else 0.0
        
        return result
    
    def _grouped_aggregate(self, metrics_list: List[dict]) -> Dict[str, float]:
        """按qtype分组聚合(用于VidSTG)"""
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
        """日志输出"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)