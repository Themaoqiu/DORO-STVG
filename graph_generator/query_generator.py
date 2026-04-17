from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Union

import fire

from modules.graph_filter import GraphFilter
from modules.query_generator_cpsat import CPSATQuerySampler, DifficultyWeights


def main(
    input_path: str,
    output_path: str,
    min_interval_len: int = 3,
    max_intervals_per_object: int = 12,
    max_target_arity: int = 3,
    max_multi_intervals_per_group: int = 6,
    max_multi_candidates_total: int = 240,
    strict_time_uniqueness_multi_target: bool = False,
    max_chain_len: int = 6,
    max_queries_per_candidate: int = 1,
    time_limit_sec: float = 2.0,
    seed: int = 7,
    eps: float = 0.01,
    lambda_weight: float = 0.5,
    use_llm_polish: bool = False,
    polish_model_name: str = "gpt-4.1-mini",
    api_keys: Optional[Union[str, Iterable[str]]] = None,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    max_queries_per_video: Optional[int] = None,
    max_queries_per_difficulty_bucket: Optional[int] = None,
    expand_non_temporal_query_tracks: bool = True,
) -> None:
    sampler = CPSATQuerySampler(
        min_interval_len=min_interval_len,
        max_intervals_per_object=max_intervals_per_object,
        max_target_arity=max_target_arity,
        max_multi_intervals_per_group=max_multi_intervals_per_group,
        max_multi_candidates_total=max_multi_candidates_total,
        strict_time_uniqueness_multi_target=strict_time_uniqueness_multi_target,
        max_chain_len=max_chain_len,
        max_queries_per_candidate=max_queries_per_candidate,
        time_limit_sec=time_limit_sec,
        seed=seed,
        weights=DifficultyWeights(
            eps=eps,
            lambda_weight=lambda_weight,
        ),
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".jsonl",
        prefix=f"{output_file.stem}.tmp.",
        dir=str(output_file.parent),
        delete=False,
    ) as tmp_file:
        tmp_output_path = tmp_file.name

    try:
        sampler.process_jsonl(
            input_path=input_path,
            output_path=tmp_output_path,
            use_llm_polish=use_llm_polish,
            polish_model_name=polish_model_name,
            api_keys=api_keys,
            max_concurrent_per_key=max_concurrent_per_key,
            max_retries=max_retries,
            max_queries_per_video=max_queries_per_video,
            max_queries_per_difficulty_bucket=max_queries_per_difficulty_bucket,
        )

        GraphFilter(
            expand_non_temporal_query_tracks=expand_non_temporal_query_tracks,
        ).filter_query_jsonl(
            input_path=tmp_output_path,
            scene_graph_path=input_path,
            output_path=output_path,
        )
    finally:
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)


if __name__ == "__main__":
    fire.Fire(main)
