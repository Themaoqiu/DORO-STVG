from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

FrameBoxes = Dict[int, List[float]]
TemporalSpan = Optional[Tuple[int, int]]


@dataclass(frozen=True)
class Track:
    temporal_span: TemporalSpan
    spatial_bboxes: FrameBoxes
    description: str = ""


def compute_tiou(gt_span: TemporalSpan, pred_span: TemporalSpan) -> float:
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


def compute_viou(gt_bboxes: FrameBoxes, pred_bboxes: FrameBoxes) -> float:
    gt_timestamps = set(gt_bboxes.keys())
    pred_timestamps = set(pred_bboxes.keys())
    timestamps_union = gt_timestamps | pred_timestamps
    timestamps_inter = gt_timestamps & pred_timestamps

    iou_sum_inter = 0.0
    for timestamp in timestamps_inter:
        iou_sum_inter += compute_siou(gt_bboxes[timestamp], pred_bboxes[timestamp])

    return iou_sum_inter / max(len(timestamps_union), 1)


def compute_track_metrics(gt_track: Track, pred_track: Track) -> Dict[str, float]:
    tiou = compute_tiou(gt_track.temporal_span, pred_track.temporal_span)
    viou = compute_viou(gt_track.spatial_bboxes, pred_track.spatial_bboxes)

    gt_timestamps = set(gt_track.spatial_bboxes.keys())
    pred_timestamps = set(pred_track.spatial_bboxes.keys())
    timestamps_inter = gt_timestamps & pred_timestamps

    iou_sum_inter = 0.0
    for timestamp in timestamps_inter:
        iou_sum_inter += compute_siou(gt_track.spatial_bboxes[timestamp], pred_track.spatial_bboxes[timestamp])

    siou = iou_sum_inter / max(len(timestamps_inter), 1)
    return {
        "tIoU": tiou,
        "sIoU": siou,
        "vIoU": viou,
        "m_vIoU": viou,
    }


def _build_tracks(
    temporal_spans: Sequence[TemporalSpan],
    spatial_boxes: Sequence[FrameBoxes],
    descriptions: Optional[Sequence[str]] = None,
) -> List[Track]:
    tracks: List[Track] = []
    for idx, (span, boxes) in enumerate(zip(temporal_spans, spatial_boxes)):
        tracks.append(
            Track(
                temporal_span=span,
                spatial_bboxes=dict(boxes),
                description=descriptions[idx] if descriptions and idx < len(descriptions) else "",
            )
        )
    return tracks


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _optimal_assignment(score_matrix: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    num_rows = len(score_matrix)
    num_cols = len(score_matrix[0]) if score_matrix else 0
    if num_rows == 0 or num_cols == 0:
        return []

    size = max(num_rows, num_cols)
    padded = [[0.0 for _ in range(size)] for _ in range(size)]
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            padded[row_idx][col_idx] = float(score_matrix[row_idx][col_idx])

    @lru_cache(maxsize=None)
    def solve(row_idx: int, used_mask: int) -> Tuple[float, Tuple[int, ...]]:
        if row_idx == size:
            return 0.0, ()

        best_score = -1.0
        best_assignment: Tuple[int, ...] = ()
        for col_idx in range(size):
            if used_mask & (1 << col_idx):
                continue
            next_score, next_assignment = solve(row_idx + 1, used_mask | (1 << col_idx))
            total_score = padded[row_idx][col_idx] + next_score
            if total_score > best_score:
                best_score = total_score
                best_assignment = (col_idx,) + next_assignment
        return best_score, best_assignment

    _, assignment = solve(0, 0)
    pairs: List[Tuple[int, int]] = []
    for row_idx, col_idx in enumerate(assignment):
        if row_idx < num_rows and col_idx < num_cols:
            pairs.append((row_idx, col_idx))
    return pairs


def compute_multi_target_metrics(gt_tracks: Sequence[Track], pred_tracks: Sequence[Track]) -> Dict[str, object]:
    num_gt = len(gt_tracks)
    num_pred = len(pred_tracks)
    if num_gt == 0 and num_pred == 0:
        return {
            "m_tIoU": 0.0,
            "m_vIoU": 0.0,
            "vIoU@0.3": 0.0,
            "vIoU@0.5": 0.0,
            "matches": [],
            "match_count": 0,
            "gt_count": 0,
            "pred_count": 0,
        }

    score_matrix = [[0.0 for _ in range(num_pred)] for _ in range(num_gt)]
    pair_metrics: Dict[Tuple[int, int], Dict[str, float]] = {}
    for gt_idx, gt_track in enumerate(gt_tracks):
        for pred_idx, pred_track in enumerate(pred_tracks):
            metrics = compute_track_metrics(gt_track, pred_track)
            pair_metrics[(gt_idx, pred_idx)] = metrics
            score_matrix[gt_idx][pred_idx] = metrics["vIoU"]

    matches = _optimal_assignment(score_matrix)
    divisor = max(num_gt, num_pred, 1)

    tiou_scores: List[float] = []
    viou_scores: List[float] = []
    match_records: List[Dict[str, object]] = []
    matched_gt = set()
    matched_pred = set()
    for gt_idx, pred_idx in matches:
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        metrics = pair_metrics[(gt_idx, pred_idx)]
        tiou_scores.append(metrics["tIoU"])
        viou_scores.append(metrics["vIoU"])
        match_records.append(
            {
                "gt_index": gt_idx,
                "pred_index": pred_idx,
                "tIoU": metrics["tIoU"],
                "vIoU": metrics["vIoU"],
                "gt_description": gt_tracks[gt_idx].description,
                "pred_description": pred_tracks[pred_idx].description,
            }
        )

    unmatched_count = divisor - len(match_records)
    if unmatched_count > 0:
        tiou_scores.extend([0.0] * unmatched_count)
        viou_scores.extend([0.0] * unmatched_count)

    m_tiou = _mean(tiou_scores)
    m_viou = _mean(viou_scores)

    return {
        "m_tIoU": m_tiou,
        "m_vIoU": m_viou,
        "vIoU@0.3": float(sum(score >= 0.3 for score in viou_scores) / divisor),
        "vIoU@0.5": float(sum(score >= 0.5 for score in viou_scores) / divisor),
        "matches": match_records,
        "match_count": len(match_records),
        "gt_count": num_gt,
        "pred_count": num_pred,
        "unmatched_gt_indices": [idx for idx in range(num_gt) if idx not in matched_gt],
        "unmatched_pred_indices": [idx for idx in range(num_pred) if idx not in matched_pred],
    }


def compute_metrics(
    gt_span: TemporalSpan,
    pred_span: TemporalSpan,
    gt_bboxes: FrameBoxes,
    pred_bboxes: FrameBoxes,
    num_frames: int = 100,
) -> Dict[str, object]:
    del num_frames
    gt_tracks = _build_tracks([gt_span], [gt_bboxes])
    pred_tracks = _build_tracks([pred_span], [pred_bboxes])
    multi_metrics = compute_multi_target_metrics(gt_tracks, pred_tracks)
    single_metrics = compute_track_metrics(gt_tracks[0], pred_tracks[0])
    return {
        **single_metrics,
        **multi_metrics,
    }
