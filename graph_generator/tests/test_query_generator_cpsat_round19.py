import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


ROUND19_PATH = (
    Path(__file__).resolve().parents[1]
    / "modules"
    / "autoresearch"
    / "round_19"
    / "query_generator_cpsat_round19.py"
)


def _load_round19_module():
    spec = importlib.util.spec_from_file_location("query_generator_cpsat_round19", ROUND19_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load round_19 module from {ROUND19_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Round19DifficultyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_round19_module()

    def test_temporal_ambiguity_prefers_mid_overlap_over_identical_interval(self) -> None:
        candidate = self.mod.CandidateTarget(
            candidate_id="target",
            members=[self.mod.CandidateMember(object_id="person_1", start=0, end=10)],
        )
        identical = self.mod.CandidateTarget(
            candidate_id="same_time",
            members=[self.mod.CandidateMember(object_id="person_1", start=0, end=10)],
        )
        shifted = self.mod.CandidateTarget(
            candidate_id="shifted_time",
            members=[self.mod.CandidateMember(object_id="person_1", start=0, end=4)],
        )

        profile_map = {
            "target": self.mod.CandidateProfile(
                candidate_id="target",
                arity=1,
                classes=["person"],
                attrs=[set()],
                env=[set()],
                temporal_tags=[{"walking"}],
                actions=[{"walking"}],
                sequences=[set()],
                spa=[set()],
                inter=[set()],
            ),
            "same_time": self.mod.CandidateProfile(
                candidate_id="same_time",
                arity=1,
                classes=["person"],
                attrs=[set()],
                env=[set()],
                temporal_tags=[{"walking"}],
                actions=[{"walking"}],
                sequences=[set()],
                spa=[set()],
                inter=[set()],
            ),
            "shifted_time": self.mod.CandidateProfile(
                candidate_id="shifted_time",
                arity=1,
                classes=["person"],
                attrs=[set()],
                env=[set()],
                temporal_tags=[{"walking"}],
                actions=[{"walking"}],
                sequences=[set()],
                spa=[set()],
                inter=[set()],
            ),
        }

        score, meta = self.mod._candidate_temporal_ambiguity(
            candidate,
            [candidate, identical, shifted],
            profile_map,
        )

        self.assertGreater(score, 0.0)
        self.assertEqual(meta["temporal_competitors"], 2)
        self.assertEqual(meta["temporal_peak_candidate_id"], "shifted_time")

    def test_spatial_ambiguity_aggregates_multiple_distractors(self) -> None:
        candidate = self.mod.CandidateTarget(
            candidate_id="target",
            members=[self.mod.CandidateMember(object_id="person_1", start=0, end=10)],
        )
        competitor_a = self.mod.CandidateTarget(
            candidate_id="comp_a",
            members=[self.mod.CandidateMember(object_id="person_2", start=0, end=10)],
        )
        competitor_b = self.mod.CandidateTarget(
            candidate_id="comp_b",
            members=[self.mod.CandidateMember(object_id="person_3", start=0, end=10)],
        )
        profile_map = {
            "target": self.mod.CandidateProfile(
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
            "comp_a": self.mod.CandidateProfile(
                candidate_id="comp_a",
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
            "comp_b": self.mod.CandidateProfile(
                candidate_id="comp_b",
                arity=1,
                classes=["person"],
                attrs=[{"green"}],
                env=[set()],
                temporal_tags=[set()],
                actions=[set()],
                sequences=[set()],
                spa=[set()],
                inter=[set()],
            ),
        }

        score, meta = self.mod._candidate_spatial_ambiguity(
            candidate,
            [candidate, competitor_a, competitor_b],
            profile_map,
        )

        self.assertGreater(score, 1.0 / 3.0)
        self.assertEqual(meta["spatial_competitors"], 2)

    def test_compute_candidate_difficulty_combines_all_monotone_terms(self) -> None:
        candidate = self.mod.CandidateTarget(
            candidate_id="cand",
            members=[
                self.mod.CandidateMember(object_id="obj_1", start=0, end=10),
                self.mod.CandidateMember(object_id="obj_2", start=0, end=10),
            ],
        )
        index = self.mod.GraphIndex(
            objects={"obj_1": {"id": "obj_1"}, "obj_2": {"id": "obj_2"}},
            frames={"obj_1": set(range(11)), "obj_2": set(range(11))},
            actions_by_obj={},
            relations_by_subj={},
            temporal_spans=[],
        )
        weights = self.mod.DifficultyWeights(
            temporal_weight=0.4,
            spatial_weight=0.3,
            evidence_weight=0.2,
            multi_weight=0.1,
        )

        with mock.patch.object(
            self.mod,
            "_candidate_temporal_ambiguity",
            return_value=(0.5, {"temporal_term": 0.5}),
        ) as temporal_mock, mock.patch.object(
            self.mod,
            "_candidate_spatial_ambiguity",
            return_value=(0.25, {"spatial_term": 0.25}),
        ) as spatial_mock, mock.patch.object(
            self.mod,
            "_evidence_burden",
            return_value=(0.75, {"evidence_term": 0.75}),
        ) as evidence_mock, mock.patch.object(
            self.mod,
            "_multi_target_complexity",
            return_value=(0.5, {"multi_term": 0.5}),
        ) as multi_mock:
            D_t, D_s, D, meta = self.mod._compute_candidate_difficulty(
                candidate=candidate,
                index=index,
                all_candidates=[candidate],
                profile_map={"cand": object()},
                weights=weights,
            )

        temporal_mock.assert_called_once()
        spatial_mock.assert_called_once()
        evidence_mock.assert_called_once()
        multi_mock.assert_called_once()
        self.assertEqual(D_t, 0.5)
        self.assertEqual(D_s, 0.25)
        self.assertAlmostEqual(D, 0.475)
        self.assertEqual(meta["difficulty_model"], "monotone_proxy_v1")


if __name__ == "__main__":
    unittest.main()
