import logging
import sys

import fire


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class STVGEvaluator:
    
    def run(
        self,
        model_name: str,
        model_path: str,
        data_name: str,
        annotation_path: str,
        video_dir: str,
        output_dir: str = "./res",
        batch_size: int = 1,
        max_tokens: int = 512,
        max_model_len: int = 8192,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        logger.info(f"Model: {model_name}")
        logger.info(f"Model Path: {model_path}")
        logger.info(f"Data Name: {data_name}")
        logger.info(f"Annotation: {annotation_path}")
        logger.info(f"Video Dir: {video_dir}")
        logger.info(f"Output Dir: {output_dir}")
        logger.info(f"Batch Size: {batch_size}")
        
        if model_name.lower() in ['qwen2.5vl', 'qwen2.5-vl']:
            from models.qwen_family import Qwen2_5VL
            model = Qwen2_5VL(
                model_path=model_path,
                batch_size=batch_size,
                max_tokens=max_tokens,
                max_model_len=max_model_len,
                temperature=temperature,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        elif model_name.lower() in ['qwen3vl', 'qwen3-vl', 'qwen3.5', 'qwen3.5vl', 'qwen3.5-vl']:
            from models.qwen_family import Qwen3VL
            model = Qwen3VL(
                model_path=model_path,
                batch_size=batch_size,
                max_tokens=max_tokens,
                max_model_len=max_model_len,
                temperature=temperature,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if data_name.lower() in ['hcstvg', 'hc-stvg', 'hcstvg2', 'hc-stvg2', 'hcstvg1']:
            from pipelines.hcstvg import HCSTVGPipeline
            
            pipeline = HCSTVGPipeline(
                model=model,
                model_name=model_name,
                data_name=data_name,
                annotation_path=annotation_path,
                video_dir=video_dir,
                output_dir=output_dir,
                batch_size=batch_size,
            )
            
            results, avg_metrics = pipeline.run_evaluation()
        
        elif data_name.lower() in ['vidstg', 'vid-stg']:
            from pipelines.vidstg import VidSTGPipeline
            
            pipeline = VidSTGPipeline(
                model=model,
                model_name=model_name,
                data_name=data_name,
                annotation_path=annotation_path,
                video_dir=video_dir,
                output_dir=output_dir,
                batch_size=batch_size,
            )
            
            results, avg_metrics = pipeline.run_evaluation()
        
        elif data_name.lower() in ['dorostvg', 'doro-stvg']:
            from pipelines.dorostvg import DOROSTVGPipeline
            
            pipeline = DOROSTVGPipeline(
                model=model,
                model_name=model_name,
                data_name=data_name,
                annotation_path=annotation_path,
                video_dir=video_dir,
                output_dir=output_dir,
                batch_size=batch_size,
            )
            
            results, avg_metrics = pipeline.run_evaluation()


def main():
    fire.Fire(STVGEvaluator)


if __name__ == "__main__":
    main()
