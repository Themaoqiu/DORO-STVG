import logging
import sys

import fire

from main import STVGEvaluator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class TemporalEvaluator(STVGEvaluator):
    def run(
        self,
        model_name: str,
        model_path: str,
        data_name: str,
        annotation_path: str,
        video_dir: str,
        output_dir: str = "./res_temporal",
        batch_size: int = 1,
        max_tokens: int = 512,
        max_model_len: int = 8192,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        logger.info("Temporal-only evaluation")
        logger.info("Model: %s", model_name)
        logger.info("Data Name: %s", data_name)
        logger.info("Annotation: %s", annotation_path)
        logger.info("Video Dir: %s", video_dir)
        logger.info("Output Dir: %s", output_dir)

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

        if data_name.lower() not in ["dorostvg", "doro-stvg"]:
            raise ValueError(f"Temporal-only evaluation currently supports DORO-STVG only, got: {data_name}")

        from pipelines.doro_temporal import DOROTemporalPipeline

        pipeline = DOROTemporalPipeline(
            model=model,
            model_name=model_name,
            data_name=data_name,
            annotation_path=annotation_path,
            video_dir=video_dir,
            output_dir=output_dir,
            batch_size=batch_size,
        )
        return pipeline.run_evaluation()


def main():
    fire.Fire(TemporalEvaluator)


if __name__ == "__main__":
    main()
