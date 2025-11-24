"""统一评测运行器"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from .dataset_registry import DatasetRegistry
from .model_registry import ModelRegistry
from .evaluator import UnifiedSTVGEvaluator
from .core.schema import STVGSample, PredictionResult


class EvalRunner:
    """评测运行器"""
    
    def __init__(self, config: dict, logger=None):
        """
        Args:
            config: 配置字典
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        
        # 1. 加载数据集
        self._log("[Runner] Loading dataset...")
        self.dataset = DatasetRegistry.build(
            name=config['dataset']['name'],
            annotation_path=config['dataset']['annotation_path'],
            video_dir=config['dataset']['video_dir'],
            subset=config['dataset'].get('subset', 'test')
        )
        
        # 2. 加载模型
        self._log("[Runner] Loading model...")
        self.model = ModelRegistry.build(
            model_name=config['model']['name'],
            **config['model']
        )
        
        # 3. 初始化评估器
        self._log("[Runner] Initializing evaluator...")
        self.evaluator = UnifiedSTVGEvaluator(
            dataset=self.dataset,
            iou_thresholds=config['evaluation'].get('iou_thresholds', [0.3, 0.5]),
            logger=logger
        )
        
        self.batch_size = config['model'].get('batch_size', 4)
        self.save_predictions = config['evaluation'].get('save_predictions', False)
        self.output_dir = Path(config['evaluation'].get('output_dir', './results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, float]:
        """
        运行完整评测流程
        
        Returns:
            评估指标字典
        """
        self._log("[Runner] Starting evaluation...")
        
        # 1. 批量推理
        all_predictions = []
        num_samples = len(self.dataset)
        
        for batch_start in range(0, num_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_samples)
            batch_samples = [self.dataset[i] for i in range(batch_start, batch_end)]
            
            self._log(f"[Runner] Processing batch {batch_start // self.batch_size + 1} "
                     f"({batch_start + 1}-{batch_end}/{num_samples})")
            
            try:
                batch_predictions = self.model.predict_batch(batch_samples)
                all_predictions.extend(batch_predictions)
            except Exception as e:
                self._log(f"[Error] Batch prediction failed: {e}")
                # 添加空预测以保持索引对齐
                for sample in batch_samples:
                    all_predictions.append(PredictionResult(
                        item_id=sample.item_id,
                        pred_temporal_bound=(0, 0),
                        pred_bboxes={},
                        metadata={'error': str(e)}
                    ))
        
        # 2. 更新评估器
        self._log("[Runner] Computing metrics...")
        self.evaluator.update(all_predictions)
        
        # 3. 计算指标
        metrics = self.evaluator.compute_metrics()
        
        # 4. 保存结果
        if self.save_predictions:
            self._save_results(all_predictions, metrics)
        
        # 5. 打印结果
        self._print_metrics(metrics)
        
        return metrics
    
    def _save_results(
        self, 
        predictions: List[PredictionResult], 
        metrics: Dict[str, float]
    ):
        """保存预测结果和指标"""
        # 保存预测
        pred_file = self.output_dir / 'response.json'
        with open(pred_file, 'w') as f:
            json.dump(
                [pred.to_dict() for pred in predictions],
                f,
                indent=2
            )
        self._log(f"[Runner] Predictions saved to: {pred_file}")
        
        # 保存指标
        metrics_file = self.output_dir / 'results.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        self._log(f"[Runner] Metrics saved to: {metrics_file}")
    
    def _print_metrics(self, metrics: Dict[str, float]):
        """打印评估指标"""
        self._log("\n" + "=" * 60)
        self._log("Evaluation Results:")
        self._log("=" * 60)
        
        for key, value in sorted(metrics.items()):
            self._log(f"{key:30s}: {value:.4f}")
        
        self._log("=" * 60 + "\n")
    
    def _log(self, message: str):
        """日志输出"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)


def run_from_config(config_path: str, logger=None):
    """
    从配置文件运行评测
    
    Args:
        config_path: YAML配置文件路径
        logger: 日志记录器
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    runner = EvalRunner(config, logger=logger)
    return runner.run()