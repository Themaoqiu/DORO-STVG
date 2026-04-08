import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from pipelines.base_pipeline import BasePipeline
from prompts import parse_response


logger = logging.getLogger(__name__)


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
                video_name = item['videopath']
                video_path = self.video_dir / video_name
                if not video_path.exists():
                    logger.warning(f"Video not found: {video_path}, skipping line {line_idx}")
                    continue

                parsed_gt = parse_response(item.get('box', ''))
                gt_tracks_sampled = parsed_gt.get('objects') or []
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
                        'queryid': item.get('queryid', f'line_{line_idx}'),
                        'difficulty': item.get('Difficulty', {}),
                        'width': item.get('Width'),
                        'height': item.get('Height'),
                        'source_line': line_idx,
                    }
                })

        logger.info(f"Loaded {len(samples)} samples")
        return samples
