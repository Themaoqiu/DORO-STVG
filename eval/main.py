import logging
import sys

import fire


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _build_model(
    model_name: str,
    model_path: str,
    batch_size: int,
    max_tokens: int,
    max_model_len: int,
    temperature: float,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
):
    name = model_name.lower()

    if name in ['qwen2.5vl', 'qwen2.5-vl']:
        from models.qwen_family import Qwen2_5VL
        return Qwen2_5VL(
            model_path=model_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    if name in ['qwen3vl', 'qwen3-vl', 'qwen3.5', 'qwen3.5vl', 'qwen3.5-vl']:
        from models.qwen_family import Qwen3VL
        return Qwen3VL(
            model_path=model_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    if name in ['llava-st-qwen2', 'llava_st_qwen2', 'llavast', 'llava-st'] or 'llava-st-qwen2' in model_path.lower():
        from models.llava_st import LlavaSTQwen2
        return LlavaSTQwen2(
            model_path=model_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    if name in ['llava16', 'llava-1.6', 'llava_16', 'llava-v1.6']:
        from models.llava16 import Llava16Model
        return Llava16Model(
            model_path=model_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    if name in ['videochat-r1', 'videochat_r1', 'videochatr1']:
        from models.videochat_r1 import VideoChatR1
        return VideoChatR1(
            model_path=model_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    if name in ['groundinggpt', 'grounding-gpt', 'grounding_gpt']:
        from models.groundinggpt import GroundingGPTModel
        return GroundingGPTModel(
            model_path=model_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    if name in ['videomolmo', 'video-molmo', 'video_molmo', 'videomlomo']:
        from models.videomolmo import VideoMolmoModel
        return VideoMolmoModel(
            model_path=model_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    if name in ['cgstvg', 'cg-stvg', 'cg_stvg']:
        from models.cgstvg import CGSTVGModel
        return CGSTVGModel(
            model_path=model_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    if name in ['tastvg', 'ta-stvg', 'ta_stvg']:
        from models.tastvg import TASTVGModel
        return TASTVGModel(
            model_path=model_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    if name in ['tubedetr', 'tube-detr', 'tube_detr']:
        from models.tubedetr import TubeDETRModel
        return TubeDETRModel(
            model_path=model_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    raise ValueError(f"Unknown model: {model_name}")


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

        model = _build_model(
            model_name=model_name,
            model_path=model_path,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        name = data_name.lower()
        if name in {"hcstvg", "hc-stvg", "hcstvg2", "hc-stvg2", "hcstvg1"}:
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
            
            return pipeline.run_evaluation()
        
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
            
            return pipeline.run_evaluation()
        
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
            
            return pipeline.run_evaluation()

        raise ValueError(f"Unknown dataset: {data_name}")


def main():
    fire.Fire(STVGEvaluator)


if __name__ == "__main__":
    main()
