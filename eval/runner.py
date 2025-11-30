import json
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from .dataset_registry import DatasetRegistry
from .model_registry import ModelRegistry
from .core.schema import STVGSample, Result
from .evaluator.stvg_evaluator import STVGEvaluator


class EvalRunner:
    
    def __init__(self, config: dict, logger=None):
        self.config = config
        self.logger = logger
        
        self._log("[Runner] Loading dataset...")
        self.dataset = DatasetRegistry.build(
            name=config['dataset']['name'],
            annotation_path=config['dataset']['annotation_path'],
            video_dir=config['dataset']['video_dir'],
            subset=config['dataset'].get('subset', 'test')
        )
        
        self._log("[Runner] Loading model...")
        self.model = ModelRegistry.build(
            model_name=config['model']['name'],
            **config['model']
        )
        
        self._log("[Runner] Initializing evaluator...")
        self.evaluator = STVGEvaluator(
            dataset=self.dataset,
            iou_thresholds=config['evaluation'].get('iou_thresholds', [0.3, 0.5]),
            logger=logger
        )
        
        self.batch_size = config['model'].get('batch_size', 4)
        self.save_predictions = config['evaluation'].get('save_predictions', False)
        self.output_dir = Path(config['evaluation'].get('output_dir', './results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, float]:
        self._log("[Runner] Starting evaluation...")
        
        all_predictions = []
        num_samples = len(self.dataset)
        
        for batch_start in range(0, num_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_samples)
            batch_samples = [self.dataset[i] for i in range(batch_start, batch_end)]
            
            self._log(f"[Runner] Processing batch {batch_start // self.batch_size + 1} "
                     f"({batch_start + 1}-{batch_end}/{num_samples})")
            
            try:
                batch_predictions = self.model.predict_batch(
                    batch_samples,
                    output_folder=self.output_dir / 'annotated_videos'
                )
                all_predictions.extend(batch_predictions)
            except Exception as e:
                self._log(f"[Error] Batch prediction failed: {e}")

                for sample in batch_samples:
                    all_predictions.append(Result(
                        item_id=sample.item_id,
                        pred_temporal_bound=(0, 0),
                        pred_bboxes={},
                        metadata={'error': str(e)}
                    ))
        
        self._log("[Runner] Computing metrics...")
        self.evaluator.update(all_predictions)
        
        metrics = self.evaluator.compute_metrics()
        
        if self.save_predictions:
            self._save_results(all_predictions, metrics)
        
        self._print_metrics(metrics)
        
        return metrics
    
    def _save_results(
        self, 
        predictions: List[Result], 
        metrics: Dict[str, float]
    ):
        pred_file = self.output_dir / 'response.json'
        with open(pred_file, 'w') as f:
            json.dump(
                [pred.to_dict() for pred in predictions],
                f,
                indent=2
            )
        self._log(f"[Runner] Predictions saved to: {pred_file}")
        
        metrics_file = self.output_dir / 'status.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        self._log(f"[Runner] Metrics saved to: {metrics_file}")
    
    def _print_metrics(self, metrics: Dict[str, float]):
        self._log("\n" + "=" * 60)
        self._log("Evaluation Results:")
        self._log("=" * 60)
        
        for key, value in sorted(metrics.items()):
            self._log(f"{key:30s}: {value:.4f}")
        
        self._log("=" * 60 + "\n")
    
    def _log(self, message: str):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)


def run_from_config(config_path: str, logger=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    runner = EvalRunner(config, logger=logger)
    return runner.run()