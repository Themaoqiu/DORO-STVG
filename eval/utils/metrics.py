import numpy as np
from typing import Tuple, Dict, List


def compute_tiou(
    gt_span: Tuple[int, int], 
    pred_span: Tuple[int, int]
) -> float:
    if pred_span is None or gt_span is None:
        return 0.0
    
    inter_start = max(gt_span[0], pred_span[0])
    inter_end = min(gt_span[1], pred_span[1])
    
    if inter_end <= inter_start:
        return 0.0
    
    intersection = inter_end - inter_start
    
    union_start = min(gt_span[0], pred_span[0])
    union_end = max(gt_span[1], pred_span[1])
    union = union_end - union_start
    
    return intersection / union if union > 0 else 0.0


def compute_siou(box1: List[float], box2: List[float]) -> float:
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def compute_metrics(
    gt_span: Tuple[int, int],
    pred_span: Tuple[int, int],
    gt_bboxes: Dict[int, List[float]],
    pred_bboxes: Dict[int, List[float]],
    num_frames: int = 100
) -> dict:
    tiou = compute_tiou(gt_span, pred_span)
    
    if pred_span is None:
        return {
            'tIoU': 0.0,
            'sIoU': 0.0,
            'm_vIoU': 0.0
        }
    
    gt_timestamps = set(gt_bboxes.keys())
    pred_timestamps = set(pred_bboxes.keys())
    
    timestamps_union = gt_timestamps | pred_timestamps 
    timestamps_inter = gt_timestamps & pred_timestamps
    
    iou_sum_inter = 0.0
    for timestamp in timestamps_inter:
        if timestamp in gt_bboxes and timestamp in pred_bboxes:
            iou = compute_siou(
                gt_bboxes[timestamp], 
                pred_bboxes[timestamp]
            )
            iou_sum_inter += iou
    
    siou = iou_sum_inter / max(len(timestamps_inter), 1)
    m_viou = iou_sum_inter / max(len(timestamps_union), 1)
    
    return {
        'tIoU': tiou,
        'sIoU': siou,
        'm_vIoU': m_viou
    }
