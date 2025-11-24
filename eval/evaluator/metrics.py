"""评估指标计算"""
import numpy as np
from typing import Tuple, List


def compute_temporal_iou(
    gt_bound: Tuple[int, int], 
    pred_bound: Tuple[int, int]
) -> float:
    """
    计算时间IoU (Temporal IoU)
    
    Args:
        gt_bound: GT时间边界 (start, end)
        pred_bound: 预测时间边界 (start, end)
        
    Returns:
        tIoU值 [0, 1]
    """
    max_start = max(gt_bound[0], pred_bound[0])
    min_end = min(gt_bound[1], pred_bound[1])
    min_start = min(gt_bound[0], pred_bound[0])
    max_end = max(gt_bound[1], pred_bound[1])
    
    if min_end <= max_start:
        return 0.0
    
    intersection = min_end - max_start
    union = max_end - min_start
    
    return intersection / union if union > 0 else 0.0


def compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    计算两组边界框的IoU
    
    Args:
        boxes1: shape (N, 4) [x1, y1, x2, y2]
        boxes2: shape (M, 4) [x1, y1, x2, y2]
        
    Returns:
        shape (N, M) IoU矩阵
    """
    boxes1 = np.atleast_2d(boxes1)
    boxes2 = np.atleast_2d(boxes2)
    
    # 计算交集
    xmin = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    ymin = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    xmax = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    ymax = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    
    intersection = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    
    # 计算面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # 计算并集
    union = area1[:, None] + area2[None, :] - intersection
    
    # 计算IoU
    iou = intersection / np.maximum(union, 1e-10)
    
    return iou