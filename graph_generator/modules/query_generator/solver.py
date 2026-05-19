from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from ortools.sat.python import cp_model

from .data_models import AtomicClue, CandidateProfile, CandidateTarget, CLUE_TYPES, TemplateSpec
from .text_utils import _norm


def _add_exclusion_constraints(model: Any, x: List[Any], exclusion_matrix: List[List[int]]) -> None:
    for row in exclusion_matrix:
        model.Add(sum(row[j] * x[j] for j in range(len(x))) >= 1)


def _member_clue_indices(clues: List[AtomicClue], member_index: int, *, exclude_cls: bool = False) -> List[int]:
    return [
        clue_index
        for clue_index, clue in enumerate(clues)
        if member_index in clue.member_indices and (not exclude_cls or clue.clue_type != "cls")
    ]


def _add_member_coverage_constraints(model: Any, x: List[Any], target: CandidateTarget, clues: List[AtomicClue]) -> bool:
    for member_index in range(target.arity):
        member_indices = _member_clue_indices(clues, member_index)
        if not member_indices:
            return False
        model.Add(sum(x[j] for j in member_indices) >= 1)

        if target.arity <= 1:
            continue
        grounded_indices = _member_clue_indices(clues, member_index, exclude_cls=True)
        if not grounded_indices:
            return False
        model.Add(sum(x[j] for j in grounded_indices) >= 1)
    return True


def _add_clue_type_constraints(model: Any, x: List[Any], template: TemplateSpec, clues: List[AtomicClue]) -> bool:
    for clue_type in CLUE_TYPES:
        type_indices = [j for j, clue in enumerate(clues) if clue.clue_type == clue_type]
        if not type_indices:
            if template.type_min.get(clue_type, 0) > 0:
                return False
            continue
        model.Add(sum(x[j] for j in type_indices) >= int(template.type_min.get(clue_type, 0)))
        model.Add(sum(x[j] for j in type_indices) <= int(template.type_max.get(clue_type, template.k_max)))
    return True


def _sequence_indices(clues: List[AtomicClue]) -> List[int]:
    return [j for j, clue in enumerate(clues) if clue.clue_type == "seq"]


def _qualified_sequence_indices(template: TemplateSpec, clues: List[AtomicClue], seq_indices: List[int]) -> List[int]:
    seq_min_len = int(template.seq_min_chain_len)
    seq_max_len = int(template.seq_max_chain_len)
    return [
        j
        for j in seq_indices
        if (seq_min_len <= 0 or clues[j].chain_len >= seq_min_len)
        and (seq_max_len <= 0 or clues[j].chain_len <= seq_max_len)
    ]


def _add_required_sequence_constraints(
    model: Any,
    x: List[Any],
    template: TemplateSpec,
    clues: List[AtomicClue],
) -> bool:
    seq_indices = _sequence_indices(clues)
    qualified_indices = _qualified_sequence_indices(template, clues, seq_indices)

    if template.require_seq:
        if not qualified_indices:
            return False
        model.Add(sum(x[j] for j in qualified_indices) >= int(template.require_seq))

    if template.min_long_seq_len <= 0 and template.min_seq_chain_sum <= 0:
        return True
    if not seq_indices:
        return False

    long_seq_indices = [j for j in seq_indices if clues[j].chain_len >= int(template.min_long_seq_len)]
    seq_chain_sum = sum(clues[j].chain_len * x[j] for j in seq_indices)
    if template.min_long_seq_len > 0 and template.min_seq_chain_sum > 0:
        enough_seq_sum = model.NewBoolVar("enough_seq_chain_sum")
        model.Add(seq_chain_sum >= int(template.min_seq_chain_sum)).OnlyEnforceIf(enough_seq_sum)
        model.Add(seq_chain_sum < int(template.min_seq_chain_sum)).OnlyEnforceIf(enough_seq_sum.Not())
        model.AddBoolOr([enough_seq_sum, *[x[j] for j in long_seq_indices]])
        return True
    if template.min_seq_chain_sum > 0:
        model.Add(seq_chain_sum >= int(template.min_seq_chain_sum))
        return True
    if not long_seq_indices:
        return False
    model.Add(sum(x[j] for j in long_seq_indices) >= 1)
    return True


def solve_query_cpsat(
    target: CandidateTarget,
    template: TemplateSpec,
    clues: List[AtomicClue],
    object_exclusion_matrix: List[List[int]],
    time_exclusion_matrix: List[List[int]],
    enforce_time_uniqueness: bool = True,
    time_limit_sec: float = 2.0,
) -> Optional[Dict[str, Any]]:
    if cp_model is None:
        raise RuntimeError("ortools is required. Install it with: uv pip install ortools")

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{j}") for j in range(len(clues))]

    _add_exclusion_constraints(model, x, object_exclusion_matrix)

    if template.require_time_uniqueness and enforce_time_uniqueness:
        _add_exclusion_constraints(model, x, time_exclusion_matrix)

    if not _add_member_coverage_constraints(model, x, target, clues):
        return None
    if not _add_clue_type_constraints(model, x, template, clues):
        return None

    model.Add(sum(x) >= int(template.k_min))
    model.Add(sum(x) <= int(template.k_max))

    model.Add(sum(clue.chain_len * x[j] for j, clue in enumerate(clues)) >= int(template.q_min))

    if not _add_required_sequence_constraints(model, x, template, clues):
        return None

    model.Minimize(sum(x))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    chosen = [j for j in range(len(clues)) if solver.Value(x[j]) == 1]
    return {
        "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        "selected_indices": chosen,
        "objective": int(round(solver.ObjectiveValue())),
    }


# ---------------------------------------------------------------------------
# Query rendering: turn solved clue selections into human-readable text
# ---------------------------------------------------------------------------


def _render_query(profile: CandidateProfile, clues: List[AtomicClue]) -> str:
    cls_phrase = None
    temporal_phrases: List[str] = []
    other_phrases: List[str] = []

    for clue in clues:
        if clue.clue_type == "cls" and cls_phrase is None:
            cls_phrase = clue.text
            continue
        if clue.clue_type == "env" and clue.signature[1] == "temporal":
            temporal_phrases.append(clue.text)
            continue
        other_phrases.append(clue.text)

    if cls_phrase is None:
        cls_phrase = f"the {profile.classes[0] if profile.classes else 'object'}"

    parts = [cls_phrase]
    if temporal_phrases:
        parts.extend(temporal_phrases)
    if other_phrases:
        parts.append(" and ".join(other_phrases))
    return _norm(" ".join(parts))


def decorate_query(core_clues: List[AtomicClue], profile: CandidateProfile) -> Tuple[str, List[str]]:
    return "", []


def _query_core_from_clues(core_clues: List[AtomicClue], profile: CandidateProfile) -> str:
    if profile.arity <= 1:
        parts = [_norm(c.text) for c in core_clues if _norm(c.text)]
        if not parts:
            return ""
        return ", ".join(parts)

    member_cls: Dict[int, str] = {}
    member_props: Dict[int, List[str]] = defaultdict(list)
    shared_parts: List[str] = []

    for clue in core_clues:
        text = _norm(clue.text)
        if not text:
            continue
        if len(clue.member_indices) != 1:
            shared_parts.append(text)
            continue
        k = clue.member_indices[0]
        if clue.clue_type == "cls" and k not in member_cls:
            member_cls[k] = text
            continue
        if clue.clue_type == "seq":
            continue
        s = _norm(text)
        for prefix in ("that is ", "that "):
            if s.startswith(prefix):
                s = _norm(s[len(prefix):])
                break
        member_props[k].append(s)

    parts: List[str] = []
    for k in range(profile.arity):
        cls = member_cls.get(k) or f"the {profile.classes[k] if k < len(profile.classes) else 'object'}"
        qualifiers: List[str] = []
        seen: Set[str] = set()
        for q in member_props.get(k, []):
            if not q or q == cls or q in seen:
                continue
            qualifiers.append(q)
            seen.add(q)
        if qualifiers:
            parts.append(_norm(f"{cls} {' and '.join(qualifiers[:2])}"))
        else:
            parts.append(cls)

    core = _norm(" and ".join(parts))
    if shared_parts:
        core = _norm(core + " and " + " and ".join(shared_parts))
    return core
