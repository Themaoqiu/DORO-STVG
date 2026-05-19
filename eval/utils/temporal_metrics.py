from typing import Dict, Optional, Tuple

from utils.metrics import compute_tiou


TemporalSpan = Optional[Tuple[int, int]]


def compute_temporal_metrics(gt_span: TemporalSpan, pred_span: TemporalSpan) -> Dict[str, float]:
    tiou = compute_tiou(gt_span, pred_span)
    return {
        "tIoU": tiou,
        "m_tIoU": tiou,
        "tIoU@0.3": float(tiou >= 0.3),
        "tIoU@0.5": float(tiou >= 0.5),
    }
