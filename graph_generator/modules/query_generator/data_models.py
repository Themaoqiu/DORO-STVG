from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


__all__ = [
    "CLUE_TYPES",
    "CandidateMember",
    "CandidateTarget",
    "AtomicClue",
    "TemplateSpec",
    "DifficultyWeights",
    "GraphIndex",
    "CandidateProfile",
    "_build_default_templates",
]


CLUE_TYPES = ("cls", "app", "act", "seq", "spa", "env", "int")


@dataclass(frozen=True)
class CandidateMember:
    object_id: str
    start: int
    end: int


@dataclass
class CandidateTarget:
    candidate_id: str
    members: List[CandidateMember]
    difficulty: float = 0.0
    difficulty_bucket: str = "medium"
    sampling_bucket: str = "easy"
    D_t: float = 0.0
    D_s: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def arity(self) -> int:
        return len(self.members)

    @property
    def interval(self) -> Tuple[int, int]:
        s = min(m.start for m in self.members)
        e = max(m.end for m in self.members)
        return s, e

    @property
    def primary_object_id(self) -> str:
        return self.members[0].object_id


@dataclass(frozen=True)
class AtomicClue:
    clue_id: str
    clue_type: str
    text: str
    signature: Tuple[Any, ...]
    member_indices: Tuple[int, ...]
    is_temporal_evidence: bool
    chain_len: int
    cost: int = 1


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    bucket: str
    type_min: Dict[str, int]
    type_max: Dict[str, int]
    k_min: int
    k_max: int
    q_min: int
    require_seq: int = 0
    seq_min_chain_len: int = 0
    seq_max_chain_len: int = 0
    min_long_seq_len: int = 0
    min_seq_chain_sum: int = 0
    require_time_uniqueness: int = 0
    weight: float = 1.0
    arity_min: int = 1
    arity_max: int = 1


@dataclass(frozen=True)
class DifficultyWeights:
    eps: float = 0.01
    lambda_weight: float = 0.5


@dataclass
class GraphIndex:
    objects: Dict[str, Dict[str, Any]]
    frames: Dict[str, Set[int]]
    actions_by_obj: Dict[str, List[Dict[str, Any]]]
    relations_by_subj: Dict[str, List[Dict[str, Any]]]
    temporal_spans: List[Dict[str, int]]


@dataclass
class CandidateProfile:
    candidate_id: str
    arity: int
    classes: List[str]
    attrs: List[Set[str]]
    env: List[Set[str]]
    temporal_tags: List[Set[str]]
    actions: List[Set[str]]
    sequences: List[Set[Tuple[str, ...]]]
    spa: List[Set[Tuple[str, str]]]
    inter: List[Set[Tuple[str, str]]]


def _build_default_templates() -> List[TemplateSpec]:
    inf = 8
    return [
        # Single-target templates.
        TemplateSpec(
            name="easy_static_attr",
            bucket="easy",
            type_min={"app": 1},
            type_max={"cls": 1, "app": 2, "act": 1, "seq": 0, "spa": 1, "env": 2, "int": 0},
            k_min=1,
            k_max=3,
            q_min=0,
            require_seq=0,
            require_time_uniqueness=0,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        TemplateSpec(
            name="easy_single_act",
            bucket="easy",
            type_min={"act": 1},
            type_max={"cls": 1, "app": 1, "act": 1, "seq": 0, "spa": 1, "env": 1, "int": 0},
            k_min=1,
            k_max=3,
            q_min=1,
            require_seq=0,
            require_time_uniqueness=0,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        TemplateSpec(
            name="medium_short_seq",
            bucket="medium",
            type_min={"seq": 1},
            type_max={"cls": 1, "app": 1, "act": 1, "seq": 1, "spa": 1, "env": inf, "int": 1},
            k_min=1,
            k_max=2,
            q_min=2,
            require_seq=1,
            seq_min_chain_len=2,
            seq_max_chain_len=2,
            require_time_uniqueness=1,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        TemplateSpec(
            name="medium_short_seq_app",
            bucket="medium",
            type_min={"seq": 1, "app": 1},
            type_max={"cls": 1, "app": 2, "act": 1, "seq": 1, "spa": 1, "env": inf, "int": 1},
            k_min=2,
            k_max=3,
            q_min=2,
            require_seq=1,
            seq_min_chain_len=2,
            seq_max_chain_len=2,
            require_time_uniqueness=1,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        TemplateSpec(
            name="medium_short_seq_spa",
            bucket="medium",
            type_min={"seq": 1, "spa": 1},
            type_max={"cls": 1, "app": 1, "act": 1, "seq": 1, "spa": 2, "env": inf, "int": 1},
            k_min=2,
            k_max=3,
            q_min=2,
            require_seq=1,
            seq_min_chain_len=2,
            seq_max_chain_len=2,
            require_time_uniqueness=1,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        TemplateSpec(
            name="medium_short_seq_int",
            bucket="medium",
            type_min={"seq": 1, "int": 1},
            type_max={"cls": 1, "app": 1, "act": 1, "seq": 1, "spa": 1, "env": inf, "int": 2},
            k_min=2,
            k_max=4,
            q_min=2,
            require_seq=1,
            seq_min_chain_len=2,
            seq_max_chain_len=2,
            require_time_uniqueness=1,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        TemplateSpec(
            name="hard_act_int",
            bucket="hard",
            type_min={"act": 1, "int": 1},
            type_max={"cls": 1, "app": 2, "act": 2, "seq": 0, "spa": 1, "env": inf, "int": 2},
            k_min=3,
            k_max=5,
            q_min=3,
            require_seq=0,
            require_time_uniqueness=1,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        TemplateSpec(
            name="hard_act_spa",
            bucket="hard",
            type_min={"act": 1, "spa": 1},
            type_max={"cls": 1, "app": 2, "act": 2, "seq": 0, "spa": 2, "env": inf, "int": 1},
            k_min=3,
            k_max=5,
            q_min=3,
            require_seq=0,
            require_time_uniqueness=1,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        # Multi-target templates.
        TemplateSpec(
            name="easy_pair_attr_spa",
            bucket="easy",
            type_min={"app": 1, "spa": 1},
            type_max={"cls": 3, "app": 3, "act": 1, "seq": 0, "spa": 2, "env": 2, "int": 1},
            k_min=3,
            k_max=5,
            q_min=0,
            require_seq=0,
            require_time_uniqueness=0,
            weight=0.9,
            arity_min=2,
            arity_max=3,
        ),
        TemplateSpec(
            name="easy_pair_attr_env",
            bucket="easy",
            type_min={"app": 1, "env": 1},
            type_max={"cls": 3, "app": 3, "act": 2, "seq": 0, "spa": 1, "env": 2, "int": 1},
            k_min=3,
            k_max=5,
            q_min=0,
            require_seq=0,
            require_time_uniqueness=0,
            weight=0.9,
            arity_min=2,
            arity_max=3,
        ),
        TemplateSpec(
            name="medium_pair_act_spa",
            bucket="medium",
            type_min={"act": 1, "spa": 1},
            type_max={"cls": 2, "app": 2, "act": 2, "seq": 0, "spa": 2, "env": 1, "int": 1},
            k_min=3,
            k_max=5,
            q_min=2,
            require_seq=0,
            require_time_uniqueness=0,
            weight=0.7,
            arity_min=2,
            arity_max=2,
        ),
        TemplateSpec(
            name="medium_triplet_relational",
            bucket="medium",
            type_min={"app": 1, "spa": 1},
            type_max={"cls": 3, "app": 3, "act": 1, "seq": 0, "spa": 2, "env": 2, "int": 1},
            k_min=4,
            k_max=6,
            q_min=2,
            require_seq=0,
            require_time_uniqueness=0,
            weight=0.5,
            arity_min=3,
            arity_max=3,
        ),
        TemplateSpec(
            name="hard_pair_act_spa",
            bucket="hard",
            type_min={"act": 1, "spa": 1},
            type_max={"cls": 3, "app": 2, "act": 2, "seq": 0, "spa": 2, "env": 2, "int": 1},
            k_min=4,
            k_max=6,
            q_min=3,
            require_seq=0,
            require_time_uniqueness=0,
            weight=0.8,
            arity_min=2,
            arity_max=3,
        ),
        TemplateSpec(
            name="hard_triplet_attr_spa_env",
            bucket="hard",
            type_min={"app": 1, "spa": 1, "env": 1},
            type_max={"cls": 3, "app": 3, "act": 1, "seq": 0, "spa": 2, "env": 2, "int": 1},
            k_min=4,
            k_max=6,
            q_min=2,
            require_seq=0,
            require_time_uniqueness=0,
            weight=0.6,
            arity_min=3,
            arity_max=3,
        ),
    ]
