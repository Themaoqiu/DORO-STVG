import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import cv2
import numpy as np

from utils.stvg_video_utils import process_video
from utils.metrics import compute_metrics
from prompts import SYSTEM_PROMPT, format_prompt, parse_response


logger = logging.getLogger(__name__)


class HCSTVGPipeline:
    
    def __init__(
        self,
        model,
        annotation_path: str,
        video_dir: str,
        output_dir: str,
        annotated_video_dir: str,
        num_frames: int = 100,
        batch_size: int = 1,
    ):
        self.model = model
        self.annotation_path = Path(annotation_path)
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.annotated_video_dir = Path(annotated_video_dir)
        self.num_frames = num_frames
        self.batch_size = batch_size
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_video_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for video_name, anno in data.items():
            video_path = self.video_dir / video_name
            
            orig2sampled, sampled_indices = self._get_frame_mapping(str(video_path))
            
            st_frame_orig = anno['st_frame']
            ed_frame_orig = anno['ed_frame']
            gt_temporal_sampled = (
                self._map_frame_to_sampled(st_frame_orig, sampled_indices),
                self._map_frame_to_sampled(ed_frame_orig, sampled_indices)
            )
            
            gt_bboxes_sampled = {}
            for frame_idx, bbox in enumerate(anno['bbox'], start=st_frame_orig):
                x, y, w, h = bbox
                bbox_normalized = [x, y, x + w, y + h]
                
                sampled_frame_idx = self._map_frame_to_sampled(frame_idx, sampled_indices)
                gt_bboxes_sampled[sampled_frame_idx] = bbox_normalized
            
            samples.append({
                'video_name': video_name,
                'video_path': str(video_path),
                'query_cn': anno.get('Chinese', ''),
                'query_en': anno.get('English', ''),
                'gt_temporal_sampled': gt_temporal_sampled,
                'gt_bboxes_sampled': gt_bboxes_sampled,
                'sampled_indices': sampled_indices.tolist(),
                'st_frame_orig': st_frame_orig,
                'ed_frame_orig': ed_frame_orig,
                'metadata': {
                    'img_size': anno['img_size'],
                    'width': anno['width'],
                    'height': anno['height'],
                    'st_time': anno['st_time'],
                    'ed_time': anno['ed_time'],
                }
            })
        
        logger.info(f"Loaded {len(samples)} samples")
        return samples
    
    def _get_frame_mapping(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        sampled_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        orig2sampled = {}
        for sampled_idx, orig_idx in enumerate(sampled_indices):
            orig2sampled[orig_idx] = sampled_idx
        
        return orig2sampled, sampled_indices
    
    def _map_frame_to_sampled(self, orig_frame: int, sampled_indices: np.ndarray) -> int:
        distances = np.abs(sampled_indices - orig_frame)
        return int(np.argmin(distances))
    
    def _map_sampled_to_original(self, sampled_frame: int, sampled_indices: List[int]) -> int:
        return sampled_indices[sampled_frame]
    
    def run_evaluation(self):
        logger.info("=" * 80)
        logger.info("Starting HC-STVG Evaluation")
        logger.info("=" * 80)
        
        samples = self.load_data()
        
        all_results = []
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i + self.batch_size]
            logger.info(f"Processing batch {i // self.batch_size + 1}/{(len(samples) + self.batch_size - 1) // self.batch_size}")
            
            batch_results = self._process_batch(batch)
            all_results.extend(batch_results)
        
        avg_metrics = self._compute_average_metrics(all_results)
        
        self._save_results(all_results, avg_metrics)
        
        logger.info("Evaluation completed!")
        
        return all_results, avg_metrics
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        annotated_video_paths = []
        for sample in batch:
            annotated_path, _ = process_video(
                video_path=sample['video_path'],
                output_folder=str(self.annotated_video_dir),
                num_frames=self.num_frames,
                annotate_frames=True
            )
            annotated_video_paths.append(annotated_path)
            logger.info(f"Annotated video saved: {annotated_path}")
        
        queries = [format_prompt(sample['query_en']) for sample in batch]

        full_responses = self.model.predict_batch(
            queries=queries,
            annotated_video_paths=annotated_video_paths,
            system_prompt=SYSTEM_PROMPT
        )
        
        batch_results = []
        for sample, full_response in zip(batch, full_responses):
            parsed = parse_response(full_response)

            pred_temporal_sampled = parsed.get('temporal_span')
            pred_bboxes_sampled = parsed.get('spatial_bboxes', {})

            metrics = compute_metrics(
                gt_span=sample['gt_temporal_sampled'],
                pred_span=pred_temporal_sampled,
                gt_bboxes=sample['gt_bboxes_sampled'],
                pred_bboxes=pred_bboxes_sampled,
                num_frames=self.num_frames
            )
            
            pred_temporal_orig = None
            if pred_temporal_sampled:
                pred_temporal_orig = (
                    self._map_sampled_to_original(pred_temporal_sampled[0], sample['sampled_indices']),
                    self._map_sampled_to_original(pred_temporal_sampled[1], sample['sampled_indices'])
                )
            
            pred_bboxes_orig = {}
            for sampled_frame, bbox in pred_bboxes_sampled.items():
                orig_frame = self._map_sampled_to_original(sampled_frame, sample['sampled_indices'])
                pred_bboxes_orig[orig_frame] = bbox
            
            result = {
                'video_name': sample['video_name'],
                'query_en': sample['query_en'],
                'full_response': full_response,
                'parsed': parsed,
                
                'gt_temporal_sampled': sample['gt_temporal_sampled'],
                'gt_temporal_orig': (sample['st_frame_orig'], sample['ed_frame_orig']),
                'gt_bboxes_sampled': sample['gt_bboxes_sampled'],
                
                'pred_temporal_sampled': pred_temporal_sampled,
                'pred_temporal_orig': pred_temporal_orig,
                'pred_bboxes_sampled': pred_bboxes_sampled,
                'pred_bboxes_orig': pred_bboxes_orig,
                
                'metrics': metrics,
                
                'sampled_indices': sample['sampled_indices'],
                'num_frames': self.num_frames,
                
                'metadata': sample['metadata'],
            }
            
            batch_results.append(result)
            
        
        return batch_results
    
    def _compute_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        avg_tiou = np.mean([r['metrics']['tIoU'] for r in results])
        avg_siou = np.mean([r['metrics']['sIoU'] for r in results])
        avg_mviou = np.mean([r['metrics']['m_vIoU'] for r in results])
        
        return {
            'tIoU': float(avg_tiou),
            'sIoU': float(avg_siou),
            'm_vIoU': float(avg_mviou),
        }
    
    def _save_results(self, results: List[Dict[str, Any]], avg_metrics: Dict[str, float]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_file = self.output_dir / f"hcstvg_results_{timestamp}.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        summary = {
            'dataset': 'HC-STVG',
            'num_samples': len(results),
            'num_frames': self.num_frames,
            'timestamp': timestamp,
            'average_metrics': avg_metrics,
        }
        summary_file = self.output_dir / f"hcstvg_status_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\nAverage Metrics:")
        logger.info(f"  tIoU: {avg_metrics['tIoU']:.4f}")
        logger.info(f"  sIoU: {avg_metrics['sIoU']:.4f}")
        logger.info(f"  m_vIoU: {avg_metrics['m_vIoU']:.4f}")
    
    def cleanup_annotated_videos(self):
        if self.annotated_video_dir.exists():
            logger.info(f"Cleaning up annotated videos in {self.annotated_video_dir}")
            shutil.rmtree(self.annotated_video_dir)
            self.annotated_video_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleanup completed")
