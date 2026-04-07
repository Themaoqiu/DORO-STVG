import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod

import cv2
import numpy as np

from utils.stvg_video_utils import process_video
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
        annotated_video_dir: str,
        num_frames: int = 100,
        batch_size: int = 1,
    ):
        self.model = model
        self.model_name = model_name
        self.data_name = data_name
        self.annotation_path = Path(annotation_path)
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.annotated_video_dir = Path(annotated_video_dir)
        self.num_frames = num_frames
        self.batch_size = batch_size
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_video_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_dataset_name(self) -> str:
        pass
    
    def _get_frame_mapping(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Use full original frames without down-sampling.
        sampled_indices = np.arange(total_frames, dtype=int)
        
        orig2sampled = {}
        for sampled_idx, orig_idx in enumerate(sampled_indices):
            orig2sampled[orig_idx] = sampled_idx
        
        return orig2sampled, sampled_indices
    
    def _map_frame_to_sampled(self, orig_frame: int, sampled_indices: np.ndarray) -> int:
        distances = np.abs(sampled_indices - orig_frame)
        return int(np.argmin(distances))
    
    def _map_sampled_to_original(self, sampled_frame: int, sampled_indices: List[int]) -> int:
        if not sampled_indices:
            return sampled_frame
        clamped_idx = max(0, min(int(sampled_frame), len(sampled_indices) - 1))
        return sampled_indices[clamped_idx]

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

    def _map_tracks_to_original(
        self,
        tracks: List[Track],
        sampled_indices: List[int],
    ) -> List[Dict[str, Any]]:
        mapped_tracks: List[Dict[str, Any]] = []
        for track in tracks:
            temporal_span = track.temporal_span
            if temporal_span is None:
                temporal_orig = None
            else:
                temporal_orig = (
                    self._map_sampled_to_original(temporal_span[0], sampled_indices),
                    self._map_sampled_to_original(temporal_span[1], sampled_indices),
                )

            spatial_orig: Dict[int, List[float]] = {}
            for sampled_frame, bbox in track.spatial_bboxes.items():
                orig_frame = self._map_sampled_to_original(sampled_frame, sampled_indices)
                spatial_orig[orig_frame] = bbox

            mapped_tracks.append(
                {
                    "description": track.description,
                    "temporal_span": temporal_orig,
                    "spatial_bboxes": spatial_orig,
                }
            )
        return mapped_tracks
    
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
            processed_video_path, video_metadata = process_video(
                video_path=sample['video_path'],
                output_folder=str(self.annotated_video_dir),
                num_frames=self.num_frames,
                annotate_frames=False
            )
            video_paths.append(processed_video_path)
            sample['video_metadata'] = video_metadata
            logger.info(f"Using original video: {processed_video_path}")
        
        queries = [format_prompt(sample['query']) for sample in batch]

        full_responses = self.model.predict_batch(
            queries=queries,
            annotated_video_paths=video_paths,
            system_prompt=SYSTEM_PROMPT
        )
        
        batch_results = []
        for sample, full_response in zip(batch, full_responses):
            parsed = parse_response(full_response)
            gt_tracks_sampled = self._build_gt_tracks(sample)
            pred_tracks_sampled = self._build_pred_tracks(parsed)
            if len(gt_tracks_sampled) == 1 and len(pred_tracks_sampled) == 1:
                metrics = compute_metrics(
                    gt_span=gt_tracks_sampled[0].temporal_span,
                    pred_span=pred_tracks_sampled[0].temporal_span,
                    gt_bboxes=gt_tracks_sampled[0].spatial_bboxes,
                    pred_bboxes=pred_tracks_sampled[0].spatial_bboxes,
                    num_frames=len(sample['sampled_indices']),
                )
            else:
                metrics = compute_multi_target_metrics(gt_tracks_sampled, pred_tracks_sampled)

            gt_tracks_orig = self._map_tracks_to_original(gt_tracks_sampled, sample['sampled_indices'])
            pred_tracks_orig = self._map_tracks_to_original(pred_tracks_sampled, sample['sampled_indices'])

            pred_temporal_sampled = pred_tracks_sampled[0].temporal_span if pred_tracks_sampled else None
            pred_bboxes_sampled = pred_tracks_sampled[0].spatial_bboxes if pred_tracks_sampled else {}
            pred_temporal_orig = pred_tracks_orig[0]["temporal_span"] if pred_tracks_orig else None
            pred_bboxes_orig = pred_tracks_orig[0]["spatial_bboxes"] if pred_tracks_orig else {}
            
            result = {
                'video_name': sample['video_name'],
                'query_en': sample['query'],
                'full_response': full_response,
                'parsed': parsed,
                
                'gt_temporal_sampled': gt_tracks_sampled[0].temporal_span if gt_tracks_sampled else None,
                'gt_temporal_orig': gt_tracks_orig[0]['temporal_span'] if gt_tracks_orig else None,
                'gt_bboxes_sampled': gt_tracks_sampled[0].spatial_bboxes if gt_tracks_sampled else {},
                'gt_tracks_sampled': [
                    {
                        'description': track.description,
                        'temporal_span': track.temporal_span,
                        'spatial_bboxes': track.spatial_bboxes,
                    }
                    for track in gt_tracks_sampled
                ],
                'gt_tracks_orig': gt_tracks_orig,
                
                'pred_temporal_sampled': pred_temporal_sampled,
                'pred_temporal_orig': pred_temporal_orig,
                'pred_bboxes_sampled': pred_bboxes_sampled,
                'pred_bboxes_orig': pred_bboxes_orig,
                'pred_tracks_sampled': [
                    {
                        'description': track.description,
                        'temporal_span': track.temporal_span,
                        'spatial_bboxes': track.spatial_bboxes,
                    }
                    for track in pred_tracks_sampled
                ],
                'pred_tracks_orig': pred_tracks_orig,
                
                'metrics': metrics,
                
                'sampled_indices': sample['sampled_indices'],
                'num_frames': self.num_frames,
                
                'metadata': sample['metadata'],
            }
            
            batch_results.append(result)
            
        return batch_results
    
    def _compute_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        return {
            'm_tIoU': float(np.mean([r['metrics']['m_tIoU'] for r in results])),
            'm_vIoU': float(np.mean([r['metrics']['m_vIoU'] for r in results])),
            'vIoU@0.3': float(np.mean([r['metrics']['vIoU@0.3'] for r in results])),
            'vIoU@0.5': float(np.mean([r['metrics']['vIoU@0.5'] for r in results])),
        }
    
    def _save_results(self, results: List[Dict[str, Any]], avg_metrics: Dict[str, float]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        dataset_name = self.get_dataset_name().lower().replace('-', '').replace(' ', '')
        eval_folder_name = f"{dataset_name}_{self.model_name}_{timestamp}"
        eval_folder = self.output_dir / eval_folder_name
        eval_folder.mkdir(parents=True, exist_ok=True)
        
        results_file = eval_folder / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to {results_file}")
        
        summary = {
            'dataset': self.get_dataset_name(),
            'model': self.model_name,
            'num_samples': len(results),
            'num_frames': self.num_frames,
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
    
    def cleanup_annotated_videos(self):
        if self.annotated_video_dir.exists():
            logger.info(f"Cleaning up annotated videos in {self.annotated_video_dir}")
            shutil.rmtree(self.annotated_video_dir)
            self.annotated_video_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleanup completed")
