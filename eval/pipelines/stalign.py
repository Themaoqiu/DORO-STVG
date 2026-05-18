import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from pipelines.base_pipeline import BasePipeline
from prompts import parse_response


logger = logging.getLogger(__name__)


class STAlignPipeline(BasePipeline):

    def get_dataset_name(self) -> str:
        return "ST-Align"

    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for item_idx, anno in enumerate(data, start=1):
            video_name = anno.get("video_path")
            if not video_name:
                logger.warning(f"Missing video_path, skipping item {item_idx}")
                continue

            video_path = self.video_dir / video_name
            if not video_path.exists():
                logger.warning(f"Video not found: {video_path}, skipping item {item_idx}")
                continue

            query = _get_query(anno)
            if not query:
                logger.warning(f"Missing query, skipping item {item_idx}")
                continue

            gt_temporal_sampled = _get_time_token_span(anno)
            if gt_temporal_sampled is None:
                logger.warning(f"Missing meta.time_token, skipping item {item_idx}")
                continue

            gt_bboxes_sampled = parse_response(anno.get("box", "")).get("spatial_bboxes", {})
            gt_tracks_sampled = [{
                "description": query,
                "temporal_span": gt_temporal_sampled,
                "spatial_bboxes": gt_bboxes_sampled,
            }]

            meta = anno.get("meta", {})
            split = _get_split(meta)
            video_input_path = str(video_path)
            if split is not None:
                video_input_path = f"{video_path}::split={split[0]}:{split[1]}"

            samples.append({
                "video_name": video_name,
                "video_path": str(video_path),
                "video_input_path": video_input_path,
                "query": query,
                "gt_temporal_sampled": gt_temporal_sampled,
                "gt_bboxes_sampled": gt_bboxes_sampled,
                "gt_tracks_sampled": gt_tracks_sampled,
                "metadata": {
                    "queryid": anno.get("id", f"stalign_{item_idx}"),
                    "time_token": meta.get("time_token"),
                    "clip_split": list(split) if split is not None else None,
                    "source_line": item_idx,
                },
            })

        logger.info(f"Loaded {len(samples)} samples")
        return samples


def _get_query(anno: Dict[str, Any]) -> str:
    qa = anno.get("QA")
    if isinstance(qa, dict):
        return qa.get("q") or qa.get("question") or anno.get("query", "")
    if isinstance(qa, list) and qa:
        first_qa = qa[0]
        if isinstance(first_qa, dict):
            return first_qa.get("q") or first_qa.get("question") or anno.get("query", "")
    return anno.get("query", "")


def _get_time_token_span(anno: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    meta = anno.get("meta", {})
    time_token = meta.get("time_token", {}) if isinstance(meta, dict) else {}
    if "<s>" not in time_token or "<e>" not in time_token:
        return None
    start, end = int(time_token["<s>"]), int(time_token["<e>"])
    return (start, end) if start <= end else (end, start)


def _get_split(meta: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    split = meta.get("split") if isinstance(meta, dict) else None
    if not isinstance(split, list) or len(split) != 2:
        return None
    start, end = int(split[0]), int(split[1])
    return (start, end) if start <= end else (end, start)
