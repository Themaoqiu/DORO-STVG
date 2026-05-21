import logging
import os
import sys

import fire


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _use_llavast_chunked_eval(model_name: str, aggregate_only: bool) -> bool:
    from pipelines.llavast_chunked import should_use_chunked_eval

    if aggregate_only:
        return should_use_chunked_eval(model_name)
    chunk_num = int(os.getenv("EVAL_CHUNK_NUM", "1"))
    chunk_id = int(os.getenv("EVAL_CHUNK_ID", "-1"))
    return should_use_chunked_eval(model_name) and chunk_num > 1 and chunk_id >= 0


class STVGEvaluator:
    def _build_model(
        self,
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
        batching_kwargs = {
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        vllm_kwargs = {
            **batching_kwargs,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        }

        if name in ["qwen2.5vl", "qwen2.5-vl"]:
            from models.qwen_family import Qwen2_5VL

            return Qwen2_5VL(model_path=model_path, **vllm_kwargs)

        if name in ["qwen3vl", "qwen3-vl", "qwen3.5", "qwen3.5vl", "qwen3.5-vl"]:
            from models.qwen_family import Qwen3VL

            return Qwen3VL(model_path=model_path, **vllm_kwargs)

        if name in ["llava-st-qwen2", "llava_st_qwen2", "llavast", "llava-st"] or "llava-st-qwen2" in model_path.lower():
            from models.llava_st import LlavaSTQwen2

            return LlavaSTQwen2(model_path=model_path, max_tokens=max_tokens, temperature=temperature)

        if name in ["vtimellm", "vtime-llm", "vtime_llm"]:
            from models.vtimellm import VTimeLLMModel

            return VTimeLLMModel(model_path=model_path, max_tokens=max_tokens, temperature=temperature)

        if name in ["grounded-video-llm", "grounded_video_llm", "groundedvideollm"]:
            from models.grounded_video_llm import GroundedVideoLLMModel

            return GroundedVideoLLMModel(model_path=model_path, max_tokens=max_tokens, temperature=temperature)

        if name in ["llava16", "llava-1.6", "llava_16", "llava-v1.6"]:
            from models.llava16 import Llava16Model

            return Llava16Model(model_path=model_path, **vllm_kwargs)

        if name in ["stvg-r1", "stvg_r1", "stvgr1"]:
            from models.stvg_r1 import STVGR1

            return STVGR1(model_path=model_path, **vllm_kwargs)

        if name in ["videochat-r1", "videochat_r1", "videochatr1"]:
            from models.qwen_family import Qwen2_5VL

            return Qwen2_5VL(model_path=model_path, **vllm_kwargs)

        if name in ["groundinggpt", "grounding-gpt", "grounding_gpt"]:
            from models.groundinggpt import GroundingGPTModel

            return GroundingGPTModel(model_path=model_path, max_tokens=max_tokens, temperature=temperature)

        if name in ["videomolmo", "video-molmo"]:
            from models.videomolmo import VideoMolmoModel

            return VideoMolmoModel(model_path=model_path)

        if name in ["cgstvg", "cg-stvg", "cg_stvg"]:
            from models.cgstvg import CGSTVGModel

            return CGSTVGModel(model_path=model_path)

        if name in ["tastvg", "ta-stvg", "ta_stvg"]:
            from models.tastvg import TASTVGModel

            return TASTVGModel(model_path=model_path)

        if name in ["devil", "de-vil"]:
            from models.devil import DeViLModel

            return DeViLModel(model_path=model_path, max_tokens=max_tokens, temperature=temperature)

        if name in ["internvl3", "internvl-3", "internvl_3"]:
            from models.internvl_family import InternVL3

            return InternVL3(model_path=model_path, **vllm_kwargs)

        if name in ["internvl3.5", "internvl-3.5", "internvl_3_5", "internvl35"]:
            from models.internvl_family import InternVL3_5

            return InternVL3_5(model_path=model_path, **vllm_kwargs)

        if name in ["llava-next-video", "llava_next_video", "llavanextvideo"]:
            from models.llava_family import LlavaNextVideo

            return LlavaNextVideo(model_path=model_path, **vllm_kwargs)

        if name in ["llava-onevision-1.5", "llava_onevision_1_5", "llavaonevision1.5", "ov1.5"]:
            from models.llava_family import LlavaOneVision1_5

            return LlavaOneVision1_5(model_path=model_path, **vllm_kwargs)

        if name in ["llava-onevision-2", "llava_onevision_2", "llavaonevision2", "ov2"]:
            from models.llava_family import LlavaOneVision2

            return LlavaOneVision2(model_path=model_path, **vllm_kwargs)

        if name in ["tubedetr", "tube-detr", "tube_detr"]:
            from models.tubedetr import TubeDETRModel

            return TubeDETRModel(model_path=model_path)

        raise ValueError(f"Unknown model: {model_name}")

    def _build_pipeline(
        self,
        model,
        model_name: str,
        data_name: str,
        annotation_path: str,
        video_dir: str,
        output_dir: str,
        batch_size: int,
    ):
        if data_name.lower() in ["hcstvg", "hc-stvg", "hcstvg2", "hc-stvg2", "hcstvg1"]:
            from pipelines.hcstvg import HCSTVGPipeline

            return HCSTVGPipeline(
                model=model,
                model_name=model_name,
                data_name=data_name,
                annotation_path=annotation_path,
                video_dir=video_dir,
                output_dir=output_dir,
                batch_size=batch_size,
            )

        if data_name.lower() in ["vidstg", "vid-stg"]:
            from pipelines.vidstg import VidSTGPipeline

            return VidSTGPipeline(
                model=model,
                model_name=model_name,
                data_name=data_name,
                annotation_path=annotation_path,
                video_dir=video_dir,
                output_dir=output_dir,
                batch_size=batch_size,
            )

        if data_name.lower() in ["stalign", "st-align", "st_align"]:
            from pipelines.stalign import STAlignPipeline

            return STAlignPipeline(
                model=model,
                model_name=model_name,
                data_name=data_name,
                annotation_path=annotation_path,
                video_dir=video_dir,
                output_dir=output_dir,
                batch_size=batch_size,
            )

        if data_name.lower() in ["dorostvg", "doro-stvg"]:
            from pipelines.dorostvg import DOROSTVGPipeline

            return DOROSTVGPipeline(
                model=model,
                model_name=model_name,
                data_name=data_name,
                annotation_path=annotation_path,
                video_dir=video_dir,
                output_dir=output_dir,
                batch_size=batch_size,
            )

        raise ValueError(f"Unknown dataset: {data_name}")

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
        aggregate_only: bool = False,
        run_id: str = "",
        chunk_num: int = 1,
    ):
        logger.info("Model: %s", model_name)
        logger.info("Model Path: %s", model_path)
        logger.info("Data Name: %s", data_name)
        logger.info("Annotation: %s", annotation_path)
        logger.info("Video Dir: %s", video_dir)
        logger.info("Output Dir: %s", output_dir)
        logger.info("Batch Size: %s", batch_size)

        model = None
        if not aggregate_only:
            model = self._build_model(
                model_name=model_name,
                model_path=model_path,
                batch_size=batch_size,
                max_tokens=max_tokens,
                max_model_len=max_model_len,
                temperature=temperature,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
            )

        pipeline = self._build_pipeline(
            model=model,
            model_name=model_name,
            data_name=data_name,
            annotation_path=annotation_path,
            video_dir=video_dir,
            output_dir=output_dir,
            batch_size=batch_size,
        )

        if aggregate_only:
            from pipelines.llavast_chunked import aggregate_chunk_results

            return aggregate_chunk_results(pipeline, str(run_id), chunk_num)
        if _use_llavast_chunked_eval(model_name, aggregate_only=False):
            from pipelines.llavast_chunked import run_chunked_worker

            chunk_num_env = int(os.getenv("EVAL_CHUNK_NUM", "1"))
            chunk_id_env = int(os.getenv("EVAL_CHUNK_ID", "-1"))
            run_id_env = str(os.getenv("EVAL_RUN_ID", "").strip())
            return run_chunked_worker(pipeline, chunk_num_env, chunk_id_env, run_id_env)
        return pipeline.run_evaluation()


def main():
    fire.Fire(STVGEvaluator)


if __name__ == "__main__":
    main()
