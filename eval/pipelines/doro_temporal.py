import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipelines.dorostvg import DOROSTVGPipeline
from temporal_prompts import TEMPORAL_SYSTEM_PROMPT, format_temporal_prompt, parse_temporal_response
from utils.temporal_metrics import compute_temporal_metrics


logger = logging.getLogger(__name__)


class DOROTemporalPipeline(DOROSTVGPipeline):
    """Temporal-only DORO-STVG evaluation pipeline."""

    def get_dataset_name(self) -> str:
        return "DORO-STVG-Temporal"

    def _predict_temporal_batch(self, queries: List[str], video_paths: List[str]) -> List[str]:
        if hasattr(self.model, "predict_temporal_batch"):
            return self.model.predict_temporal_batch(
                queries=queries,
                video_paths=video_paths,
                system_prompt=TEMPORAL_SYSTEM_PROMPT,
            )
        return self.model.predict_batch(
            queries=queries,
            video_paths=video_paths,
            system_prompt=TEMPORAL_SYSTEM_PROMPT,
        )

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        video_paths = []
        for sample in batch:
            video_path = sample["video_path"]
            if getattr(self.model, "use_video_input_path", False):
                video_paths.append(sample.get("video_input_path", video_path))
            else:
                video_paths.append(video_path)
            logger.info("Using original video for temporal eval: %s", video_path)

        queries = [format_temporal_prompt(sample["query"]) for sample in batch]
        full_responses = self._predict_temporal_batch(queries, video_paths)
        raw_responses = getattr(self.model, "last_raw_responses", full_responses)

        batch_results = []
        for idx, (sample, full_response) in enumerate(zip(batch, full_responses)):
            raw_response = raw_responses[idx] if idx < len(raw_responses) else full_response
            parsed = parse_temporal_response(full_response)
            pred_span = parsed.get("temporal_span")
            if pred_span is not None and hasattr(self.model, "map_temporal_span"):
                pred_span = self.model.map_temporal_span(pred_span, sample)
            gt_span = _normalize_span(sample.get("gt_temporal_sampled"))
            metrics = compute_temporal_metrics(gt_span, pred_span)

            batch_results.append(
                {
                    "video_name": sample["video_name"],
                    "query_en": sample["query"],
                    "raw_response": raw_response,
                    "prediction": {
                        "temporal_span": list(pred_span) if pred_span is not None else None,
                    },
                    "gt_temporal_span": list(gt_span) if gt_span is not None else None,
                    "metrics": metrics,
                    "metadata": sample["metadata"],
                    "evaluation_type": "temporal_only",
                    "coordinate_system": "sampled_2fps_frame_index",
                }
            )

        return batch_results

    def _compute_basic_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {
                "m_tIoU": 0.0,
                "tIoU@0.3": 0.0,
                "tIoU@0.5": 0.0,
            }
        return {
            "m_tIoU": float(sum(r["metrics"]["m_tIoU"] for r in results) / len(results)),
            "tIoU@0.3": float(sum(r["metrics"]["tIoU@0.3"] for r in results) / len(results)),
            "tIoU@0.5": float(sum(r["metrics"]["tIoU@0.5"] for r in results) / len(results)),
        }

    def _save_results(self, results: List[Dict[str, Any]], avg_metrics: Dict[str, Any]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = self.get_dataset_name().lower().replace("-", "").replace(" ", "")
        eval_folder_name = f"{dataset_name}_{self.model_name}_{timestamp}"
        eval_folder = self.output_dir / eval_folder_name
        eval_folder.mkdir(parents=True, exist_ok=True)

        results_file = eval_folder / "temporal_results.jsonl"
        with results_file.open("w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Temporal results saved to %s", results_file)

        summary = {
            "dataset": self.get_dataset_name(),
            "model": self.model_name,
            "evaluation_type": "temporal_only",
            "coordinate_system": "sampled_2fps_frame_index",
            "num_samples": len(results),
            "timestamp": timestamp,
            "average_metrics": avg_metrics,
        }
        summary_file = eval_folder / "temporal_status.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("Temporal summary saved to %s", summary_file)


def _normalize_span(span: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return None
    try:
        start, end = int(span[0]), int(span[1])
    except (TypeError, ValueError):
        return None
    if end < start:
        start, end = end, start
    return start, end
