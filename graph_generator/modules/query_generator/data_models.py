from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


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
