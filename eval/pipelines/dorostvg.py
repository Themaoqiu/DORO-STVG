import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from pipelines.base_pipeline import BasePipeline
from prompts import parse_response


logger = logging.getLogger(__name__)


def _normalize_boxes(boxes: Dict[str, Any], width: float, height: float) -> Dict[int, List[float]]:
    normalized: Dict[int, List[float]] = {}
    if not isinstance(boxes, dict) or width <= 0 or height <= 0:
        return normalized

    for frame_idx_text, coords in boxes.items():
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        try:
            frame_idx = int(str(frame_idx_text).strip())
            x1, y1, x2, y2 = [float(v) for v in coords]
        except (TypeError, ValueError):
            continue
        normalized[frame_idx] = [
            max(0.0, min(1.0, x1 / width)),
            max(0.0, min(1.0, y1 / height)),
            max(0.0, min(1.0, x2 / width)),
            max(0.0, min(1.0, y2 / height)),
        ]

    return normalized


def _build_gt_tracks_from_target_members(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    width = float(item.get("video_width") or 0)
    height = float(item.get("video_height") or 0)
    target_members = item.get("target_members")
    per_target_queries = item.get("per_target_queries") or {}
    if not isinstance(target_members, list):
        return []

    gt_tracks_sampled: List[Dict[str, Any]] = []
    for member in target_members:
        if not isinstance(member, dict):
            continue
        target_index = member.get("target_index")
        description = per_target_queries.get(f"target {target_index}") or item.get("query", "")
        spatial_bboxes = _normalize_boxes(member.get("boxes", {}), width, height)
        temporal_span = None
        if spatial_bboxes:
            frames = sorted(spatial_bboxes.keys())
            temporal_span = (frames[0], frames[-1])
        gt_tracks_sampled.append(
            {
                "description": str(description),
                "temporal_span": temporal_span,
                "spatial_bboxes": spatial_bboxes,
            }
        )
    return gt_tracks_sampled


class DOROSTVGPipeline(BasePipeline):

    def get_dataset_name(self) -> str:
        return "DORO-STVG"

    def load_data(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            for line_idx, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                item = json.loads(line)
                video_name = item['video_path']
                video_path = self.video_dir / video_name
                if not video_path.exists():
                    logger.warning(f"Video not found: {video_path}, skipping line {line_idx}")
                    continue

                parsed_gt = parse_response(item.get('box', ''))
                gt_tracks_sampled = parsed_gt.get('objects') or _build_gt_tracks_from_target_members(item)
                if not gt_tracks_sampled:
                    gt_tracks_sampled = [{
                        'description': str(item.get('query', '')),
                        'temporal_span': parsed_gt.get('temporal_span'),
                        'spatial_bboxes': parsed_gt.get('spatial_bboxes', {}),
                    }]

                samples.append({
                    'video_name': video_name,
                    'video_path': str(video_path),
                    'query': item.get('query', ''),
                    'gt_temporal_sampled': gt_tracks_sampled[0].get('temporal_span'),
                    'gt_bboxes_sampled': gt_tracks_sampled[0].get('spatial_bboxes', {}),
                    'gt_tracks_sampled': gt_tracks_sampled,
                    'metadata': {
                        'queryid': item.get('queryid') or item.get('query_id', f'line_{line_idx}'),
                        'difficulty': item.get('Difficulty', {}),
                        'width': item.get('Width') or item.get('video_width'),
                        'height': item.get('Height') or item.get('video_height'),
                        'source_line': line_idx,
                    }
                })

        logger.info(f"Loaded {len(samples)} samples")
        return samples
