"""Shared utilities for baseline solvers (random / greedy / ILP).

All baseline solvers share the same I/O contract as ``solve_query_cpsat``:
    inputs:  target, template, clues, object_E, time_E, enforce_time_uniqueness
    output:  None  OR  {"status", "selected_indices", "objective"}
"""
from __future__ import annotations

from typing import List, Sequence

from ..data_models import AtomicClue, CandidateTarget, CLUE_TYPES, TemplateSpec


def is_feasible(
    selection: Sequence[int],
    target: CandidateTarget,
    template: TemplateSpec,
    clues: List[AtomicClue],
    object_E: List[List[int]],
    time_E: List[List[int]],
    enforce_time_uniqueness: bool,
) -> bool:
    """Return True iff ``selection`` (list of clue indices, 0/1 representation
    by membership) satisfies every constraint in ``solve_query_cpsat``."""
    sel = set(int(j) for j in selection)
    if not sel:
        return False

    # Object exclusion: each row must have sum >= 1 over selected clues.
    for row in object_E:
        if sum(row[j] for j in sel) < 1:
            return False

    # Time exclusion (when enforced).
    if template.require_time_uniqueness and enforce_time_uniqueness:
        for row in time_E:
            if sum(row[j] for j in sel) < 1:
                return False

    # Per-member coverage.
    for member_index in range(target.arity):
        members = [j for j, c in enumerate(clues) if member_index in c.member_indices]
        if not members or sum(1 for j in members if j in sel) < 1:
            return False
        if target.arity > 1:
            grounded = [j for j in members if clues[j].clue_type != "cls"]
            if not grounded or sum(1 for j in grounded if j in sel) < 1:
                return False

    # Type min/max.
    for ctype in CLUE_TYPES:
        idxs = [j for j, c in enumerate(clues) if c.clue_type == ctype]
        cnt = sum(1 for j in idxs if j in sel)
        if cnt < int(template.type_min.get(ctype, 0)):
            return False
        cap = int(template.type_max.get(ctype, template.k_max))
        if cnt > cap:
            return False

    # k_min / k_max.
    if len(sel) < int(template.k_min) or len(sel) > int(template.k_max):
        return False

    # q_min: sum of chain_len.
    if sum(clues[j].chain_len for j in sel) < int(template.q_min):
        return False

    # require_seq + chain length envelope.
    seq_idx = [j for j, c in enumerate(clues) if c.clue_type == "seq"]
    seq_min = int(template.seq_min_chain_len)
    seq_max = int(template.seq_max_chain_len)
    qualified = [
        j for j in seq_idx
        if (seq_min <= 0 or clues[j].chain_len >= seq_min)
        and (seq_max <= 0 or clues[j].chain_len <= seq_max)
    ]
    if template.require_seq:
        if sum(1 for j in qualified if j in sel) < int(template.require_seq):
            return False

    # min_long_seq_len + min_seq_chain_sum (OR semantics, same as CP-SAT).
    if template.min_long_seq_len > 0 or template.min_seq_chain_sum > 0:
        if not seq_idx:
            return False
        long_seq = [j for j in seq_idx if clues[j].chain_len >= int(template.min_long_seq_len)]
        seq_sum = sum(clues[j].chain_len for j in seq_idx if j in sel)
        if template.min_long_seq_len > 0 and template.min_seq_chain_sum > 0:
            ok = (seq_sum >= int(template.min_seq_chain_sum)) or any(j in sel for j in long_seq)
            if not ok:
                return False
        elif template.min_seq_chain_sum > 0:
            if seq_sum < int(template.min_seq_chain_sum):
                return False
        else:
            if not long_seq or not any(j in sel for j in long_seq):
                return False

    return True
