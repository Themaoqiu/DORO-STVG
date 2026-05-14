from __future__ import annotations

import math
from typing import Any, Dict, List, Set, Tuple

from .clues import build_atomic_clues, build_exclusion_matrix
from .data_models import (
    AtomicClue,
    CandidateProfile,
    CandidateTarget,
    DifficultyWeights,
    GraphIndex,
    TemplateSpec,
)
from .intervals import _interval_tiou


_DIFFICULTY_BUCKET_ORDER = {
    "very_easy": 0,
    "easy": 1,
    "medium": 2,
    "hard": 3,
    "very_hard": 4,
}

_DIFFICULTY_BUCKET_THRESHOLDS: Tuple[Tuple[float, str], ...] = (
    (0.20, "very_easy"),
    (0.40, "easy"),
    (0.60, "medium"),
    (0.80, "hard"),
    (1.01, "very_hard"),
)

_TEMPLATE_BUCKET_DIFFICULTY_LABELS = {
    "easy": ("very_easy", "easy"),
    "medium": ("medium",),
    "hard": ("hard", "very_hard"),
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _member_object_signature(candidate: CandidateTarget) -> Tuple[str, ...]:
    return tuple(member.object_id for member in candidate.members)


def _profile_temporal_tokens(profile: CandidateProfile) -> Set[str]:
    tokens: Set[str] = set()
    for idx in range(profile.arity):
        for tag in profile.temporal_tags[idx]:
            tokens.add(f"m{idx}:tag:{tag}")
        for action in profile.actions[idx]:
            tokens.add(f"m{idx}:act:{action}")
        for chain in profile.sequences[idx]:
            tokens.add(f"m{idx}:seq:{'|'.join(chain)}")
    return tokens


def _profile_spatial_tokens(profile: CandidateProfile) -> Set[str]:
    tokens: Set[str] = set()
    for idx in range(profile.arity):
        tokens.add(f"m{idx}:cls:{profile.classes[idx]}")
        for attr in profile.attrs[idx]:
            tokens.add(f"m{idx}:attr:{attr}")
        for env in profile.env[idx]:
            tokens.add(f"m{idx}:env:{env}")
        for pred, ref_cls in profile.spa[idx]:
            tokens.add(f"m{idx}:spa:{pred}:{ref_cls}")
        for pred, ref_cls in profile.inter[idx]:
            tokens.add(f"m{idx}:int:{pred}:{ref_cls}")
    return tokens


def _jaccard_similarity(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right)) / float(len(union))


def _candidate_interval_similarity(left: CandidateTarget, right: CandidateTarget) -> float:
    left_intervals = [(member.start, member.end) for member in left.members]
    right_intervals = [(member.start, member.end) for member in right.members]
    if len(left_intervals) != len(right_intervals) or not left_intervals:
        return 0.0
    scores = [_interval_tiou(a, b) for a, b in zip(left_intervals, right_intervals)]
    return float(sum(scores) / len(scores))


def _candidate_spatial_confusability(
    candidate: CandidateTarget,
    all_candidates: List[CandidateTarget],
    profile_map: Dict[str, CandidateProfile],
) -> Tuple[float, Dict[str, Any]]:
    # D_s as softcount over aggregated spatial confusion mass:
    #   X = sum_c ( sim_s(target, c) * tIoU(interval_t, interval_c) ) ** SPATIAL_EXPONENT
    #   D_s = X / (1 + X)
    SPATIAL_EXPONENT = 2.0
    target_signature = _member_object_signature(candidate)
    target_tokens = _profile_spatial_tokens(profile_map[candidate.candidate_id])
    mass = 0.0
    best_pair = (0.0, "", 0.0, 0.0)
    competitor_count = 0

    for other in all_candidates:
        if other.candidate_id == candidate.candidate_id or other.arity != candidate.arity:
            continue
        if _member_object_signature(other) == target_signature:
            continue
        overlap = _interval_tiou(candidate.interval, other.interval)
        if overlap <= 0.0:
            continue
        competitor_count += 1
        similarity = _jaccard_similarity(target_tokens, _profile_spatial_tokens(profile_map[other.candidate_id]))
        contribution = similarity * overlap
        if contribution <= 0.0:
            continue
        mass += contribution ** SPATIAL_EXPONENT
        if contribution > best_pair[0]:
            best_pair = (contribution, other.candidate_id, similarity, overlap)

    d_s = mass / (1.0 + mass)
    return _clamp01(d_s), {
        "spatial_competitors": competitor_count,
        "spatial_confusion_mass": mass,
        "spatial_best_candidate_id": best_pair[1],
        "spatial_best_similarity": best_pair[2],
        "spatial_best_interval_overlap": best_pair[3],
        "spatial_token_count": len(target_tokens),
    }


def _best_exclusion_difficulty(
    rows: List[List[int]],
    clues: List[AtomicClue],
    *,
    allow_temporal: bool,
) -> Tuple[float, Dict[str, Any]]:
    if not rows:
        return 0.0, {
            "competitor_rows": 0,
            "eligible_clues": 0,
            "best_clue_id": "",
            "best_clue_type": "",
            "best_exclusion_rate": 1.0,
        }

    eligible = [
        clue
        for clue in clues
        if clue.is_temporal_evidence == allow_temporal
    ]
    if not eligible:
        return 1.0, {
            "competitor_rows": len(rows),
            "eligible_clues": 0,
            "best_clue_id": "",
            "best_clue_type": "",
            "best_exclusion_rate": 0.0,
        }

    best_clue = None
    best_rate = -1.0
    for clue in eligible:
        clue_idx = int(clue.clue_id[1:])
        rate = float(sum(row[clue_idx] for row in rows)) / float(len(rows))
        if rate > best_rate:
            best_rate = rate
            best_clue = clue

    return _clamp01(1.0 - best_rate), {
        "competitor_rows": len(rows),
        "eligible_clues": len(eligible),
        "best_clue_id": "" if best_clue is None else best_clue.clue_id,
        "best_clue_type": "" if best_clue is None else best_clue.clue_type,
        "best_exclusion_rate": _clamp01(best_rate),
    }


def _compute_candidate_difficulty(
    candidate: CandidateTarget,
    index: GraphIndex,
    all_candidates: List[CandidateTarget],
    profile_map: Dict[str, CandidateProfile],
    weights: DifficultyWeights,
) -> Tuple[float, float, float, Dict[str, Any]]:
    lambda_weight = min(max(float(weights.lambda_weight), 0.0), 1.0)
    clues = build_atomic_clues(index, candidate)
    _, _, time_rows = build_exclusion_matrix(candidate, all_candidates, clues, profile_map)
    D_t, temporal_meta = _best_exclusion_difficulty(time_rows, clues, allow_temporal=True)
    D_s, spatial_meta = _candidate_spatial_confusability(candidate, all_candidates, profile_map)
    D = _clamp01(lambda_weight * D_t + (1.0 - lambda_weight) * D_s)

    meta = {
        "lambda_weight": lambda_weight,
        "arity": candidate.arity,
        "difficulty_model": "hybrid_confusability_v2_spx2",
        "atomic_clue_count": len(clues),
    }
    meta.update(temporal_meta)
    meta.update(spatial_meta)
    return D_t, D_s, D, meta


def _assign_sampling_buckets(candidates: List[CandidateTarget]) -> None:
    if not candidates:
        return
    ordered = sorted(candidates, key=lambda c: (c.difficulty, c.interval[1] - c.interval[0] + 1))
    n = len(ordered)
    ratios = {"easy": 0.30, "medium": 0.45, "hard": 0.25}

    raw_counts = {k: ratios[k] * n for k in ratios}
    counts = {k: int(math.floor(raw_counts[k])) for k in ratios}
    remain = n - sum(counts.values())
    for k in sorted(ratios.keys(), key=lambda x: raw_counts[x] - counts[x], reverse=True):
        if remain <= 0:
            break
        counts[k] += 1
        remain -= 1

    if n >= 3:
        if counts["medium"] == 0:
            donor = "easy" if counts["easy"] > counts["hard"] else "hard"
            if counts[donor] > 1:
                counts[donor] -= 1
                counts["medium"] += 1
        if counts["easy"] == 0:
            donor = "medium" if counts["medium"] > counts["hard"] else "hard"
            if counts[donor] > 1:
                counts[donor] -= 1
                counts["easy"] += 1
        if counts["hard"] == 0:
            donor = "medium" if counts["medium"] > counts["easy"] else "easy"
            if counts[donor] > 1:
                counts[donor] -= 1
                counts["hard"] += 1

    easy_end = counts["easy"]
    medium_end = counts["easy"] + counts["medium"]
    for i, c in enumerate(ordered):
        if i < easy_end:
            c.sampling_bucket = "easy"
        elif i < medium_end:
            c.sampling_bucket = "medium"
        else:
            c.sampling_bucket = "hard"


def _difficulty_bucket_from_score(score: float) -> str:
    clamped = _clamp01(score)
    for upper_bound, label in _DIFFICULTY_BUCKET_THRESHOLDS:
        if clamped < upper_bound:
            return label
    return "very_hard"


def _assign_difficulty_buckets_5(candidates: List[CandidateTarget]) -> None:
    for cand in candidates:
        cand.difficulty_bucket = _difficulty_bucket_from_score(cand.difficulty)


def _template_candidate_labels(template_bucket: str) -> Tuple[str, ...]:
    return _TEMPLATE_BUCKET_DIFFICULTY_LABELS.get(template_bucket, ("medium",))


def _template_target_labels(template: TemplateSpec) -> Tuple[str, ...]:
    labels = _template_candidate_labels(template.bucket)
    only_app_min = template.type_min == {"app": 1}
    no_temporal_or_relational_min = (
        template.type_min.get("act", 0) == 0
        and template.type_min.get("seq", 0) == 0
        and template.type_min.get("spa", 0) == 0
        and template.type_min.get("int", 0) == 0
    )
    if template.arity_max == 1 and template.bucket == "easy" and only_app_min and no_temporal_or_relational_min:
        return ("very_easy",)
    return labels


def _template_difficulty_rank(candidate: CandidateTarget, template: TemplateSpec) -> Tuple[int, float]:
    preferred_labels = _template_target_labels(template)
    if candidate.difficulty_bucket in preferred_labels:
        bucket_distance = 0
    else:
        candidate_rank = _DIFFICULTY_BUCKET_ORDER.get(candidate.difficulty_bucket, _DIFFICULTY_BUCKET_ORDER["medium"])
        preferred_ranks = [_DIFFICULTY_BUCKET_ORDER[label] for label in preferred_labels]
        bucket_distance = min(abs(candidate_rank - rank) for rank in preferred_ranks)

    if template.bucket == "hard":
        difficulty_order = -candidate.difficulty
    else:
        difficulty_order = candidate.difficulty
    return bucket_distance, difficulty_order
