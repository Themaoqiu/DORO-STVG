from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from ..core.schema import Result, STVGSample
from .metrics import compute_stvg_metrics


class STVGEvaluator:
    
    def __init__(
        self,
        dataset: 'BaseSTVGDataset',
        iou_thresholds: List[float] = [0.3, 0.5, 0.7],
        num_frames: int = 100,
        logger = None
    ):
        self.dataset = dataset
        self.iou_thresholds = iou_thresholds
        self.num_frames = num_frames
        self.logger = logger
        
        self.gt_index: Dict[str, STVGSample] = {
            sample.item_id: sample 
            for sample in dataset.samples
        }
        
        self.predictions: Dict[str, Result] = {}
        
        self._log(f"[Evaluator] Initialized with {len(self.gt_index)} GT samples")
        self._log(f"[Evaluator] Using {num_frames} frames per video")

    def update(self, predictions: List[Result]):
        for pred in predictions:
            self.predictions[pred.item_id] = pred
        
        self._log(f"[Evaluator] Updated {len(predictions)} predictions (total: {len(self.predictions)})")
    
    def compute_metrics(self) -> Dict[str, float]:
        if not self.predictions:
            self._log("[Warning] No predictions to evaluate!")
            return {}
        
        metrics_per_sample = []
        for item_id, pred in self.predictions.items():
            if item_id not in self.gt_index:
                self._log(f"[Warning] No GT for prediction: {item_id}")
                continue
            
            gt = self.gt_index[item_id]
            sample_metrics = self._evaluate_single_sample(gt, pred)
            metrics_per_sample.append(sample_metrics)
        
        self._log(f"[Evaluator] Evaluated {len(metrics_per_sample)} samples")
        
        return self._aggregate_metrics(metrics_per_sample)
    
    def _evaluate_single_sample(
        self, 
        gt: STVGSample, 
        pred: Result
    ) -> dict:
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
        bboxes: Dict[int, List[float]],
        width: float,
        height: float
    ) -> Dict[int, List[float]]:
        normalized_bboxes = {}
        
        for frame_id, box in bboxes.items():
            if box:
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