import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


def _temp_root(pipeline) -> Path:
    return pipeline.output_dir / ".chunktmp"


def _chunk_file_path(pipeline, run_id: str, chunk_id: int) -> Path:
    return _temp_root(pipeline) / str(run_id) / f"chunk_{chunk_id:03d}.jsonl"


def should_use_chunked_eval(model_name: str) -> bool:
    name = model_name.lower()
    return name in {"llava-st-qwen2", "llava_st_qwen2", "llavast", "llava-st"}


def run_chunked_worker(pipeline, chunk_num: int, chunk_id: int, run_id: str) -> Dict[str, Any]:
    run_id = str(run_id)
    logger.info("Starting chunked %s evaluation", pipeline.get_dataset_name())
    try:
        samples = pipeline.load_data()
        slice_len = math.ceil(len(samples) / chunk_num) if samples else 0
        start = chunk_id * slice_len
        end = min(start + slice_len, len(samples))
        samples = samples[start:end]
        logger.info("Processing chunk %s/%s with sample range [%s, %s)", chunk_id, chunk_num, start, end)

        all_results: List[Dict[str, Any]] = []
        for i in range(0, len(samples), pipeline.batch_size):
            batch = samples[i : i + pipeline.batch_size]
            logger.info(
                "Processing chunk batch %s/%s",
                i // pipeline.batch_size + 1,
                (len(samples) + pipeline.batch_size - 1) // pipeline.batch_size if samples else 0,
            )
            all_results.extend(pipeline._process_batch(batch))

        chunk_file = _chunk_file_path(pipeline, run_id, chunk_id)
        chunk_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chunk_file, "w", encoding="utf-8") as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Chunk results saved to %s", chunk_file)
        return {"chunk_id": chunk_id, "chunk_num": chunk_num, "num_samples": len(all_results)}
    finally:
        close_fn = getattr(pipeline.model, "close", None)
        if callable(close_fn):
            close_fn()


def aggregate_chunk_results(pipeline, run_id: str, chunk_num: int) -> Dict[str, Any]:
    run_id = str(run_id)
    chunk_dir = _temp_root(pipeline) / run_id
    results: List[Dict[str, Any]] = []
    for chunk_id in range(chunk_num):
        chunk_file = _chunk_file_path(pipeline, run_id, chunk_id)
        if not chunk_file.exists():
            raise FileNotFoundError(f"Missing chunk result: {chunk_file}")
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))

    avg_metrics = pipeline._compute_average_metrics(results)
    original_save_results = pipeline._save_results

    def _save_results_with_run_id(result_items, metrics):
        timestamp = run_id
        dataset_name = pipeline.get_dataset_name().lower().replace("-", "").replace(" ", "")
        eval_folder_name = f"{dataset_name}_{pipeline.model_name}_{timestamp}"
        eval_folder = pipeline.output_dir / eval_folder_name
        eval_folder.mkdir(parents=True, exist_ok=True)

        results_file = eval_folder / "results.jsonl"
        with open(results_file, "w", encoding="utf-8") as f:
            for item in result_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        summary = {
            "dataset": pipeline.get_dataset_name(),
            "model": pipeline.model_name,
            "num_samples": len(result_items),
            "timestamp": timestamp,
            "average_metrics": metrics,
        }
        summary_file = eval_folder / "status.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("Detailed results saved to %s", results_file)
        logger.info("Summary saved to %s", summary_file)
        logger.info("Evaluation results saved to: %s", eval_folder)

    pipeline._save_results = _save_results_with_run_id
    try:
        pipeline._save_results(results, avg_metrics)
    finally:
        pipeline._save_results = original_save_results

    shutil.rmtree(chunk_dir, ignore_errors=True)
    temp_root = _temp_root(pipeline)
    if temp_root.exists() and not any(temp_root.iterdir()):
        temp_root.rmdir()
    return avg_metrics
