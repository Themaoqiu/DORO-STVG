import logging
import sys
from pathlib import Path

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
        annotated_video_dir: str = "./annotated_videos",
        num_frames: int = 100,
        batch_size: int = 1,
        max_tokens: int = 512,
        max_model_len: int = 8192,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        cleanup_after: bool = False,
    ):
        logger.info(f"Model: {model_name}")
        logger.info(f"Model Path: {model_path}")
        logger.info(f"Data Name: {data_name}")
        logger.info(f"Annotation: {annotation_path}")
        logger.info(f"Video Dir: {video_dir}")
        logger.info(f"Output Dir: {output_dir}")
        logger.info(f"Num Frames: {num_frames}")
        logger.info(f"Batch Size: {batch_size}")
        
        if model_name.lower() in ['qwen2.5vl', 'qwen2.5-vl']:
            from eval.models.qwen_family import Qwen2_5VL
            model = Qwen2_5VL(
                model_path=model_path,
                batch_size=batch_size,
                nframes=num_frames,
                max_tokens=max_tokens,
                max_model_len=max_model_len,
                temperature=temperature,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        elif model_name.lower() in ['qwen3vl', 'qwen3-vl']:
            from eval.models.qwen_family import Qwen3VL
            model = Qwen3VL(
                model_path=model_path,
                batch_size=batch_size,
                nframes=num_frames,
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
                annotated_video_dir=annotated_video_dir,
                num_frames=num_frames,
                batch_size=batch_size,
            )
            
            results, avg_metrics = pipeline.run_evaluation()
        
            if cleanup_after:
                logger.info("Cleaning up annotated videos...")
                pipeline.cleanup_annotated_videos()
        
    
    def cleanup(self, annotated_video_dir: str = "./annotated_videos"):
        import shutil
        
        video_dir = Path(annotated_video_dir)
        if video_dir.exists():
            logger.info(f"Cleaning up {video_dir}")
            shutil.rmtree(video_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleanup completed!")
        else:
            logger.warning(f"Directory {video_dir} does not exist")


def main():
    fire.Fire(STVGEvaluator)


if __name__ == "__main__":
    main()
