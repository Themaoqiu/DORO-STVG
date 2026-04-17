import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod

from utils.metrics import Track, compute_metrics, compute_multi_target_metrics
from prompts import SYSTEM_PROMPT, format_prompt, parse_response


logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    
    def __init__(
        self,
        model,
        model_name: str,
        data_name: str,
        annotation_path: str,
        video_dir: str,
        output_dir: str,
        batch_size: int = 1,
    ):
        self.model = model
        self.model_name = model_name
        self.data_name = data_name
        self.annotation_path = Path(annotation_path)
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_dataset_name(self) -> str:
        pass

    def _build_gt_tracks(self, sample: Dict[str, Any]) -> List[Track]:
        tracks = sample.get("gt_tracks_sampled")
        if not isinstance(tracks, list) or not tracks:
            return [
                Track(
                    temporal_span=sample["gt_temporal_sampled"],
                    spatial_bboxes=sample["gt_bboxes_sampled"],
                    description=str(sample.get("query", "")),
                )
            ]

        built_tracks: List[Track] = []
        for item in tracks:
            if not isinstance(item, dict):
                continue
            built_tracks.append(
                Track(
                    temporal_span=item.get("temporal_span"),
                    spatial_bboxes=item.get("spatial_bboxes", {}),
                    description=str(item.get("description", "")),
                )
            )
        return built_tracks

    def _build_pred_tracks(self, parsed: Dict[str, Any]) -> List[Track]:
        objects = parsed.get("objects")
        if not isinstance(objects, list) or not objects:
            return [
                Track(
                    temporal_span=parsed.get("temporal_span"),
                    spatial_bboxes=parsed.get("spatial_bboxes", {}),
                    description="",
                )
            ]

        tracks: List[Track] = []
        for item in objects:
            if not isinstance(item, dict):
                continue
            tracks.append(
                Track(
                    temporal_span=item.get("temporal_span"),
                    spatial_bboxes=item.get("spatial_bboxes", {}),
                    description=str(item.get("description", "")),
                )
            )
        return tracks

    def _serialize_tracks(self, tracks: List[Track]) -> List[Dict[str, Any]]:
        return [
            {
                'description': track.description,
                'temporal_span': track.temporal_span,
                'spatial_bboxes': track.spatial_bboxes,
            }
            for track in tracks
        ]

    def run_evaluation(self):
        logger.info(f"Starting {self.get_dataset_name()} Evaluation")
        
        samples = self.load_data()
        
        all_results = []
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i + self.batch_size]
            logger.info(f"Processing batch {i // self.batch_size + 1}/{(len(samples) + self.batch_size - 1) // self.batch_size}")
            
            batch_results = self._process_batch(batch)
            all_results.extend(batch_results)
        
        avg_metrics = self._compute_average_metrics(all_results)
        
        self._save_results(all_results, avg_metrics)
        
        logger.info("Evaluation completed")
        
        return all_results, avg_metrics
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        video_paths = []
        for sample in batch:
            video_path = sample['video_path']
            video_paths.append(video_path)
            logger.info(f"Using original video: {video_path}")
        
        queries = [format_prompt(sample['query']) for sample in batch]

        full_responses = self.model.predict_batch(
            queries=queries,
            video_paths=video_paths,
            system_prompt=SYSTEM_PROMPT
        )
        raw_responses = getattr(self.model, "last_raw_responses", full_responses)
        
        batch_results = []
        for idx, (sample, full_response) in enumerate(zip(batch, full_responses)):
            raw_response = raw_responses[idx] if idx < len(raw_responses) else full_response
            parsed = parse_response(full_response)
            gt_tracks_sampled = self._build_gt_tracks(sample)
            pred_tracks_sampled = self._build_pred_tracks(parsed)
            if len(gt_tracks_sampled) == 1 and len(pred_tracks_sampled) == 1:
                metrics = compute_metrics(
                    gt_span=gt_tracks_sampled[0].temporal_span,
                    pred_span=pred_tracks_sampled[0].temporal_span,
                    gt_bboxes=gt_tracks_sampled[0].spatial_bboxes,
                    pred_bboxes=pred_tracks_sampled[0].spatial_bboxes,
                )
            else:
                metrics = compute_multi_target_metrics(gt_tracks_sampled, pred_tracks_sampled)

            result = {
                'video_name': sample['video_name'],
                'query_en': sample['query'],
                'raw_response': raw_response,
                'prediction': self._serialize_tracks(pred_tracks_sampled),
                'metrics': metrics,
                'metadata': sample['metadata'],
            }
            
            batch_results.append(result)
            
        return batch_results
    
    def _compute_basic_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {
                'm_tIoU': 0.0,
                'm_vIoU': 0.0,
                'vIoU@0.3': 0.0,
                'vIoU@0.5': 0.0,
            }
        return {
            'm_tIoU': float(sum(r['metrics']['m_tIoU'] for r in results) / len(results)),
            'm_vIoU': float(sum(r['metrics']['m_vIoU'] for r in results) / len(results)),
            'vIoU@0.3': float(sum(r['metrics']['vIoU@0.3'] for r in results) / len(results)),
            'vIoU@0.5': float(sum(r['metrics']['vIoU@0.5'] for r in results) / len(results)),
        }

    def _compute_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = self._compute_basic_average_metrics(results)

        bucket_order = ['very_easy', 'easy', 'medium', 'hard', 'very_hard']
        bucket_groups: Dict[str, List[Dict[str, Any]]] = {}
        for bucket in bucket_order:
            bucket_results = [
                result for result in results
                if (result.get('metadata') or {}).get('difficulty_bucket') == bucket
            ]
            if bucket_results:
                bucket_groups[bucket] = bucket_results

        if bucket_groups:
            metrics['by_difficulty_bucket'] = {
                bucket: {
                    'num_samples': len(bucket_results),
                    **self._compute_basic_average_metrics(bucket_results),
                }
                for bucket, bucket_results in bucket_groups.items()
            }

        return metrics
    
    def _save_results(self, results: List[Dict[str, Any]], avg_metrics: Dict[str, float]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        dataset_name = self.get_dataset_name().lower().replace('-', '').replace(' ', '')
        eval_folder_name = f"{dataset_name}_{self.model_name}_{timestamp}"
        eval_folder = self.output_dir / eval_folder_name
        eval_folder.mkdir(parents=True, exist_ok=True)
        
        results_file = eval_folder / "results.jsonl"
        with open(results_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Detailed results saved to {results_file}")
        
        summary = {
            'dataset': self.get_dataset_name(),
            'model': self.model_name,
            'num_samples': len(results),
            'timestamp': timestamp,
            'average_metrics': avg_metrics,
        }
        summary_file = eval_folder / "status.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Summary saved to {summary_file}")
        
        logger.info(f"Evaluation results saved to: {eval_folder}")
        logger.info(f"{'='*60}")
        logger.info("Average Metrics:")
        logger.info(f"  m_tIoU: {avg_metrics['m_tIoU']:.4f}")
        logger.info(f"  m_vIoU: {avg_metrics['m_vIoU']:.4f}")
        logger.info(f"  vIoU@0.3: {avg_metrics['vIoU@0.3']:.4f}")
        logger.info(f"  vIoU@0.5: {avg_metrics['vIoU@0.5']:.4f}")
