import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from pipelines.base_pipeline import BaseSTVGPipeline


logger = logging.getLogger(__name__)


class VidSTGPipeline(BaseSTVGPipeline):
    
    def get_dataset_name(self) -> str:
        return "VidSTG"
    
    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item_id, anno in data.items():
            vid = anno['vid']
            video_path = self.video_dir / f"{vid}.mp4"
            
            if not video_path.exists():
                logger.warning(f"Video not found: {video_path}")
                continue
            
            orig2sampled, sampled_indices = self._get_frame_mapping(str(video_path))
            
            begin_fid = anno['temp_gt']['begin_fid']
            end_fid = anno['temp_gt']['end_fid']
            gt_temporal_sampled = (
                self._map_frame_to_sampled(begin_fid, sampled_indices),
                self._map_frame_to_sampled(end_fid, sampled_indices)
            )
            
            width = anno['width']
            height = anno['height']
            
            gt_bboxes_sampled = {}
            for frame_idx, bbox in enumerate(anno['target_bboxs'], start=begin_fid):
                x1_norm = bbox['xmin'] / width
                y1_norm = bbox['ymin'] / height
                x2_norm = bbox['xmax'] / width
                y2_norm = bbox['ymax'] / height
                bbox_normalized = [x1_norm, y1_norm, x2_norm, y2_norm]
                
                sampled_frame_idx = self._map_frame_to_sampled(frame_idx, sampled_indices)
                gt_bboxes_sampled[sampled_frame_idx] = bbox_normalized
            
            qtype = anno['qtype']
            
            samples.append({
                'video_name': f"{vid}.mp4",
                'video_path': str(video_path),
                'query': anno['sentence']['description'],
                'gt_temporal_sampled': gt_temporal_sampled,
                'gt_temporal_orig': (begin_fid, end_fid),
                'gt_bboxes_sampled': gt_bboxes_sampled,
                'sampled_indices': sampled_indices.tolist(),
                'metadata': {
                    'vid': vid,
                    'item_id': item_id,
                    'qtype': qtype,
                    'fps': anno['fps'],
                    'width': width,
                    'height': height,
                    'frame_count': anno['frame_count'],
                    'target_category': anno.get('target_category', ''),
                }
            })
        
        logger.info(f"Loaded {len(samples)} samples")
        return samples
    
    def _compute_average_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        declar_results = [r for r in results if r['metadata']['qtype'] == 'declar']
        non_declar_results = [r for r in results if r['metadata']['qtype'] != 'declar']
        
        overall_metrics = super()._compute_average_metrics(results)
        
        metrics = {
            'overall': overall_metrics,
        }
        
        if declar_results:
            declar_metrics = super()._compute_average_metrics(declar_results)
            metrics['declarative'] = declar_metrics
        
        if non_declar_results:
            non_declar_metrics = super()._compute_average_metrics(non_declar_results)
            metrics['interrogative'] = non_declar_metrics
        
        return metrics
    
    def _save_results(self, results: List[Dict[str, Any]], avg_metrics: Dict[str, Any]):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        eval_folder_name = f"vidstg_{self.model_name}_{timestamp}"
        eval_folder = self.output_dir / eval_folder_name
        eval_folder.mkdir(parents=True, exist_ok=True)
        
        results_file = eval_folder / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to {results_file}")
        
        summary = {
            'dataset': 'VidSTG',
            'model': self.model_name,
            'num_samples': len(results),
            'num_samples_by_qtype': {
                'declarative': len([r for r in results if r['metadata']['qtype'] == 'declar']),
                'interrogative': len([r for r in results if r['metadata']['qtype'] != 'declar']),
            },
            'num_frames': self.num_frames,
            'timestamp': timestamp,
            'average_metrics': avg_metrics,
        }
        summary_file = eval_folder / "status.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Summary saved to {summary_file}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation results saved to: {eval_folder}")
        logger.info(f"{'='*60}")
        logger.info(f"Overall Metrics:")
        logger.info(f"  tIoU: {avg_metrics['overall']['tIoU']:.4f}")
        logger.info(f"  sIoU: {avg_metrics['overall']['sIoU']:.4f}")
        logger.info(f"  m_vIoU: {avg_metrics['overall']['m_vIoU']:.4f}")
        
        if 'declarative' in avg_metrics:
            logger.info(f"Declarative Metrics:")
            logger.info(f"  tIoU: {avg_metrics['declarative']['tIoU']:.4f}")
            logger.info(f"  sIoU: {avg_metrics['declarative']['sIoU']:.4f}")
            logger.info(f"  m_vIoU: {avg_metrics['declarative']['m_vIoU']:.4f}")
        
        if 'interrogative' in avg_metrics:
            logger.info(f"Interrogative Metrics:")
            logger.info(f"  tIoU: {avg_metrics['interrogative']['tIoU']:.4f}")
            logger.info(f"  sIoU: {avg_metrics['interrogative']['sIoU']:.4f}")
            logger.info(f"  m_vIoU: {avg_metrics['interrogative']['m_vIoU']:.4f}")
        
        logger.info(f"{'='*60}")
