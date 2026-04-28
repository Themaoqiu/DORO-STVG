import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from modules.query_generator_cpsat import (
    CPSATQuerySampler,
    AtomicClue,
    CandidateMember,
    CandidateProfile,
    CandidateTarget,
    DifficultyWeights,
    GraphIndex,
    _compute_candidate_difficulty,
    _candidate_spatial_confusability,
    _assign_difficulty_buckets_5,
    _difficulty_bucket_from_score,
    _sample_class,
)
from modules.autoresearch.analyze_round import _join_samples


def _candidate(name: str, score: float) -> CandidateTarget:
    candidate = CandidateTarget(
        candidate_id=name,
        members=[CandidateMember(object_id=f"{name}_obj", start=0, end=10)],
    )
    candidate.difficulty = score
    return candidate


class DifficultyBucketTests(unittest.TestCase):
    def test_difficulty_bucket_from_score_uses_absolute_thresholds(self) -> None:
        self.assertEqual(_difficulty_bucket_from_score(0.00), "very_easy")
        self.assertEqual(_difficulty_bucket_from_score(0.19), "very_easy")
        self.assertEqual(_difficulty_bucket_from_score(0.20), "easy")
        self.assertEqual(_difficulty_bucket_from_score(0.39), "easy")
        self.assertEqual(_difficulty_bucket_from_score(0.40), "medium")
        self.assertEqual(_difficulty_bucket_from_score(0.59), "medium")
        self.assertEqual(_difficulty_bucket_from_score(0.60), "hard")
        self.assertEqual(_difficulty_bucket_from_score(0.79), "hard")
        self.assertEqual(_difficulty_bucket_from_score(0.80), "very_hard")
        self.assertEqual(_difficulty_bucket_from_score(1.00), "very_hard")

    def test_assign_difficulty_buckets_uses_score_not_rank(self) -> None:
        low = _candidate("low", 0.18)
        mid = _candidate("mid", 0.52)
        high = _candidate("high", 0.91)

        _assign_difficulty_buckets_5([high, low, mid])

        self.assertEqual(low.difficulty_bucket, "very_easy")
        self.assertEqual(mid.difficulty_bucket, "medium")
        self.assertEqual(high.difficulty_bucket, "very_hard")


class ProcessJsonlPolishTests(unittest.TestCase):
    def test_process_jsonl_polishes_all_graphs_in_one_batch(self) -> None:
        class StubSampler(CPSATQuerySampler):
            def __init__(self) -> None:
                super().__init__()
                self.polish_batch_sizes = []

            def generate_for_graph(self, graph):
                return [
                    {
                        "query_id": graph["video_path"],
                        "query": f"raw {graph['video_path']}",
                        "template": "tpl",
                        "difficulty_bucket": "medium",
                        "D_t": 0.5,
                        "D_s": 0.5,
                        "D": 0.5,
                        "target": {"members": [{"object_id": "obj_1", "start_frame": 0, "end_frame": 0}]},
                    }
                ]

            async def polish_queries_with_llm(self, query_nodes, **kwargs):
                self.polish_batch_sizes.append(len(query_nodes))
                return {
                    q["query_id"]: {
                        "query": f"polished {q['query_id']}",
                        "target_queries": {"target 1": "the object"},
                    }
                    for q in query_nodes
                }

            def _build_minimal_records(self, graph, query_nodes):
                return [{"query_id": q["query_id"], "query": q["query"], "llm_polished": q.get("llm_polished", False)} for q in query_nodes]

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "graphs.jsonl"
            output_path = Path(tmp_dir) / "queries.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        '{"video_path":"video_a.mp4","object_nodes":[]}',
                        '{"video_path":"video_b.mp4","object_nodes":[]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            sampler = StubSampler()
            with mock.patch(
                "modules.query_generator_cpsat._summarize_synonyms_from_graph_file",
                return_value=({}, {}, {"graphs": 2, "relation_values": 0, "attribute_values": 0, "relation_synonyms": 0, "attribute_synonyms": 0}),
            ):
                sampler.process_jsonl(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    use_llm_polish=True,
                    api_keys="dummy",
                )

        self.assertEqual(sampler.polish_batch_sizes, [2])


class AnalyzeRoundJoinTests(unittest.TestCase):
    def test_join_samples_falls_back_to_query_scores_when_eval_metadata_omits_difficulty(self) -> None:
        joined = _join_samples(
            query_rows=[
                {
                    "query_id": "q1",
                    "difficulty_bucket": "hard",
                    "D": 0.7,
                    "D_t": 0.9,
                    "D_s": 0.5,
                    "target_arity": 2,
                    "template": "tpl",
                    "query": "query",
                }
            ],
            result_rows=[
                {
                    "metadata": {
                        "queryid": "q1",
                        "difficulty_bucket": None,
                        "difficulty_score": None,
                        "difficulty_temporal": None,
                        "difficulty_spatial": None,
                        "target_arity": None,
                        "template": None,
                    },
                    "metrics": {"m_tIoU": 0.1, "m_vIoU": 0.2, "vIoU@0.3": 0.0, "vIoU@0.5": 0.0},
                }
            ],
        )

        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0].difficulty_bucket, "hard")
        self.assertEqual(joined[0].D, 0.7)
        self.assertEqual(joined[0].D_t, 0.9)
        self.assertEqual(joined[0].D_s, 0.5)
        self.assertEqual(joined[0].target_arity, 2)
        self.assertEqual(joined[0].template, "tpl")


class DifficultyModelTests(unittest.TestCase):
    def test_sample_class_prefers_tracking_class_over_noisy_dam_category(self) -> None:
        obj = {
            "object_class": "dog",
            "dam_category": "cat",
        }

        self.assertEqual(_sample_class(obj, "dog_1"), "dog")

    def test_spatial_confusability_is_not_inflated_by_time_overlap_alone(self) -> None:
        candidate = CandidateTarget(
            candidate_id="target",
            members=[CandidateMember(object_id="person_1", start=0, end=10)],
        )
        competitor = CandidateTarget(
            candidate_id="competitor",
            members=[CandidateMember(object_id="person_2", start=0, end=10)],
        )
        profile_map = {
            "target": CandidateProfile(
                candidate_id="target",
                arity=1,
                classes=["person"],
                attrs=[{"red"}],
                env=[set()],
                temporal_tags=[set()],
                actions=[set()],
                sequences=[set()],
                spa=[set()],
                inter=[set()],
            ),
            "competitor": CandidateProfile(
                candidate_id="competitor",
                arity=1,
                classes=["person"],
                attrs=[{"blue"}],
                env=[set()],
                temporal_tags=[set()],
                actions=[set()],
                sequences=[set()],
                spa=[set()],
                inter=[set()],
            ),
        }

        score, meta = _candidate_spatial_confusability(candidate, [candidate, competitor], profile_map)

        self.assertAlmostEqual(score, 1.0 / 3.0)
        self.assertEqual(meta["spatial_best_similarity"], 1.0 / 3.0)
        self.assertEqual(meta["spatial_best_interval_overlap"], 1.0)

    def test_compute_candidate_difficulty_uses_temporal_confusability_and_spatial_confusability(self) -> None:
        candidate = _candidate("cand", 0.0)
        weights = DifficultyWeights(lambda_weight=0.25)
        index = GraphIndex(
            objects={f"{candidate.candidate_id}_obj": {"id": f"{candidate.candidate_id}_obj"}},
            frames={f"{candidate.candidate_id}_obj": {1, 2, 3}},
            actions_by_obj={},
            relations_by_subj={},
            temporal_spans=[],
        )

        with mock.patch(
            "modules.query_generator_cpsat._candidate_temporal_confusability",
            return_value=(0.8, {"temporal_source": "temporal_conf"}),
        ) as temporal_mock, mock.patch(
            "modules.query_generator_cpsat._candidate_spatial_confusability",
            return_value=(0.2, {"spatial_source": "conf"}),
        ) as spatial_mock, mock.patch(
            "modules.query_generator_cpsat.build_atomic_clues",
            return_value=["dummy_clue"],
        ), mock.patch(
            "modules.query_generator_cpsat.build_exclusion_matrix",
            return_value=([], ["object_rows"], ["time_rows"]),
        ):
            D_t, D_s, D, meta = _compute_candidate_difficulty(
                candidate=candidate,
                index=index,
                all_candidates=[candidate],
                profile_map={"cand": object()},
                weights=weights,
            )

        temporal_mock.assert_called_once()
        spatial_mock.assert_called_once()
        self.assertEqual(D_t, 0.8)
        self.assertEqual(D_s, 0.2)
        self.assertAlmostEqual(D, 0.35)
        self.assertEqual(meta["difficulty_model"], "hybrid_confusability_v3")
        self.assertEqual(meta["temporal_source"], "temporal_conf")
        self.assertEqual(meta["spatial_source"], "conf")

    def test_compute_candidate_difficulty_uses_temporal_confusability_for_full_track_candidates(self) -> None:
        candidate = CandidateTarget(
            candidate_id="cand",
            members=[CandidateMember(object_id="obj_1", start=0, end=10)],
        )
        index = GraphIndex(
            objects={"obj_1": {"id": "obj_1"}},
            frames={"obj_1": set(range(0, 11))},
            actions_by_obj={},
            relations_by_subj={},
            temporal_spans=[],
        )

        with mock.patch(
            "modules.query_generator_cpsat._candidate_temporal_confusability",
            return_value=(0.7, {"temporal_source": "temporal_conf"}),
        ) as temporal_mock, mock.patch(
            "modules.query_generator_cpsat._candidate_spatial_confusability",
            return_value=(0.4, {"spatial_source": "conf"}),
        ):
            D_t, D_s, D, meta = _compute_candidate_difficulty(
                candidate=candidate,
                index=index,
                all_candidates=[candidate],
                profile_map={"cand": CandidateProfile(
                    candidate_id="cand",
                    arity=1,
                    classes=["person"],
                    attrs=[set()],
                    env=[set()],
                    temporal_tags=[set()],
                    actions=[{"running"}],
                    sequences=[set()],
                    spa=[set()],
                    inter=[set()],
                )},
                weights=DifficultyWeights(lambda_weight=0.5),
            )

        temporal_mock.assert_called_once()
        self.assertEqual(D_t, 0.7)
        self.assertEqual(D_s, 0.4)
        self.assertAlmostEqual(D, 0.55)
        self.assertEqual(meta["difficulty_model"], "hybrid_confusability_v3")
        self.assertEqual(meta["temporal_source"], "temporal_conf")

    def test_compute_candidate_difficulty_keeps_temporal_exclusion_for_local_candidates(self) -> None:
        candidate = CandidateTarget(
            candidate_id="cand",
            members=[CandidateMember(object_id="obj_1", start=2, end=8)],
        )
        index = GraphIndex(
            objects={"obj_1": {"id": "obj_1"}},
            frames={"obj_1": set(range(0, 11))},
            actions_by_obj={},
            relations_by_subj={},
            temporal_spans=[],
        )

        with mock.patch(
            "modules.query_generator_cpsat._candidate_temporal_confusability",
            return_value=(0.8, {"temporal_source": "temporal_conf"}),
        ) as temporal_mock, mock.patch(
            "modules.query_generator_cpsat._candidate_spatial_confusability",
            return_value=(0.4, {"spatial_source": "conf"}),
        ):
            D_t, D_s, D, meta = _compute_candidate_difficulty(
                candidate=candidate,
                index=index,
                all_candidates=[candidate],
                profile_map={"cand": CandidateProfile(
                    candidate_id="cand",
                    arity=1,
                    classes=["person"],
                    attrs=[set()],
                    env=[set()],
                    temporal_tags=[set()],
                    actions=[{"running"}],
                    sequences=[set()],
                    spa=[set()],
                    inter=[set()],
                )},
                weights=DifficultyWeights(lambda_weight=0.5),
            )

        temporal_mock.assert_called_once()
        self.assertEqual(D_t, 0.8)
        self.assertEqual(D_s, 0.4)
        self.assertAlmostEqual(D, 0.6)
        self.assertEqual(meta["temporal_source"], "temporal_conf")
