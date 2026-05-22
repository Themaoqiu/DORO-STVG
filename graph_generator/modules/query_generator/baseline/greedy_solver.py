"""Greedy set-cover baseline.

Strategy:
  1. Treat every active row of object_E (and time_E if enforced) as an element to cover.
     A row is covered when at least one selected clue j has row[j]==1.
  2. Repeatedly pick the clue maximizing (newly covered rows) - tie-break by
     (-cost, larger chain_len) - subject to type_max / k_max budgets.
  3. After coverage is satisfied, top up to satisfy:
       * per-member coverage (Eq. 3),
       * type_min / k_min / q_min,
       * require_seq, min_long_seq_len / min_seq_chain_sum.
  4. Final feasibility check via shared validator; return None if any constraint
     is still violated (mirrors paper's "no heuristic fallback" stance).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..data_models import AtomicClue, CandidateTarget, CLUE_TYPES, TemplateSpec
from ._common import is_feasible


def _type_count(selected: Set[int], clues: List[AtomicClue], ctype: str) -> int:
    return sum(1 for j in selected if clues[j].clue_type == ctype)


def _can_add(j: int, selected: Set[int], clues: List[AtomicClue], template: TemplateSpec) -> bool:
    if j in selected:
        return False
    if len(selected) + 1 > int(template.k_max):
        return False
    ctype = clues[j].clue_type
    cap = int(template.type_max.get(ctype, template.k_max))
    if _type_count(selected, clues, ctype) + 1 > cap:
        return False
    return True


def _greedy_cover(
    selected: Set[int],
    clues: List[AtomicClue],
    template: TemplateSpec,
    rows: List[List[int]],
) -> None:
    """Pick clues that cover currently-uncovered rows, respecting budgets."""
    uncovered = [i for i, row in enumerate(rows) if sum(row[j] for j in selected) < 1]
    while uncovered:
        best_j = -1
        best_gain = 0
        best_tie = (0, 0)  # (-cost, chain_len)
        for j in range(len(clues)):
            if not _can_add(j, selected, clues, template):
                continue
            gain = sum(1 for i in uncovered if rows[i][j] == 1)
            if gain <= 0:
                continue
            tie = (-int(clues[j].cost), int(clues[j].chain_len))
            if gain > best_gain or (gain == best_gain and tie > best_tie):
                best_gain = gain
                best_tie = tie
                best_j = j
        if best_j < 0:
            return
        selected.add(best_j)
        uncovered = [i for i in uncovered if rows[i][best_j] != 1]


def _topup_member_coverage(
    selected: Set[int],
    target: CandidateTarget,
    clues: List[AtomicClue],
    template: TemplateSpec,
) -> None:
    for member_index in range(target.arity):
        members = [j for j, c in enumerate(clues) if member_index in c.member_indices]
        if members and not any(j in selected for j in members):
            for j in members:
                if _can_add(j, selected, clues, template):
                    selected.add(j)
                    break
        if target.arity > 1:
            grounded = [j for j, c in enumerate(clues) if member_index in c.member_indices and c.clue_type != "cls"]
            if grounded and not any(j in selected for j in grounded):
                for j in grounded:
                    if _can_add(j, selected, clues, template):
                        selected.add(j)
                        break


def _topup_type_min(
    selected: Set[int],
    clues: List[AtomicClue],
    template: TemplateSpec,
) -> None:
    for ctype in CLUE_TYPES:
        need = int(template.type_min.get(ctype, 0))
        while _type_count(selected, clues, ctype) < need:
            cands = [j for j in range(len(clues)) if clues[j].clue_type == ctype and _can_add(j, selected, clues, template)]
            if not cands:
                return
            cands.sort(key=lambda j: -int(clues[j].chain_len))
            selected.add(cands[0])


def _topup_kmin_qmin(
    selected: Set[int],
    clues: List[AtomicClue],
    template: TemplateSpec,
) -> None:
    while len(selected) < int(template.k_min) or sum(clues[j].chain_len for j in selected) < int(template.q_min):
        cands = [j for j in range(len(clues)) if _can_add(j, selected, clues, template)]
        if not cands:
            return
        cands.sort(key=lambda j: -int(clues[j].chain_len))
        selected.add(cands[0])


def _topup_seq(
    selected: Set[int],
    clues: List[AtomicClue],
    template: TemplateSpec,
) -> None:
    if template.require_seq:
        seq_min = int(template.seq_min_chain_len)
        seq_max = int(template.seq_max_chain_len)
        qualified = [
            j for j, c in enumerate(clues)
            if c.clue_type == "seq"
            and (seq_min <= 0 or c.chain_len >= seq_min)
            and (seq_max <= 0 or c.chain_len <= seq_max)
        ]
        while sum(1 for j in qualified if j in selected) < int(template.require_seq):
            cands = [j for j in qualified if _can_add(j, selected, clues, template)]
            if not cands:
                return
            cands.sort(key=lambda j: -int(clues[j].chain_len))
            selected.add(cands[0])

    if template.min_long_seq_len > 0 or template.min_seq_chain_sum > 0:
        long_seq = [j for j, c in enumerate(clues) if c.clue_type == "seq" and c.chain_len >= int(template.min_long_seq_len)]
        if long_seq and not any(j in selected for j in long_seq):
            for j in sorted(long_seq, key=lambda j: -int(clues[j].chain_len)):
                if _can_add(j, selected, clues, template):
                    selected.add(j)
                    break


def solve_query_greedy(
    target: CandidateTarget,
    template: TemplateSpec,
    clues: List[AtomicClue],
    object_exclusion_matrix: List[List[int]],
    time_exclusion_matrix: List[List[int]],
    enforce_time_uniqueness: bool = True,
    time_limit_sec: float = 2.0,
) -> Optional[Dict[str, Any]]:
    _ = time_limit_sec
    if not clues:
        return None

    selected: Set[int] = set()

    rows = list(object_exclusion_matrix)
    if template.require_time_uniqueness and enforce_time_uniqueness:
        rows = rows + list(time_exclusion_matrix)

    _greedy_cover(selected, clues, template, rows)
    _topup_member_coverage(selected, target, clues, template)
    _topup_type_min(selected, clues, template)
    _topup_seq(selected, clues, template)
    _topup_kmin_qmin(selected, clues, template)

    if not is_feasible(
        selected, target, template, clues,
        object_exclusion_matrix, time_exclusion_matrix,
        enforce_time_uniqueness,
    ):
        return None

    return {
        "status": "FEASIBLE",
        "selected_indices": sorted(selected),
        "objective": len(selected),
    }
