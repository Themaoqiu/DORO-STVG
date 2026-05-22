"""ILP baseline using PuLP + CBC.

Same model as ``solve_query_cpsat`` but expressed as a 0-1 mixed integer
program. Used as sanity check / wall-clock comparison against CP-SAT.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import pulp
except ImportError:  # pragma: no cover
    pulp = None

from ..data_models import AtomicClue, CandidateTarget, CLUE_TYPES, TemplateSpec


def solve_query_ilp(
    target: CandidateTarget,
    template: TemplateSpec,
    clues: List[AtomicClue],
    object_exclusion_matrix: List[List[int]],
    time_exclusion_matrix: List[List[int]],
    enforce_time_uniqueness: bool = True,
    time_limit_sec: float = 2.0,
    *,
    msg: bool = False,
) -> Optional[Dict[str, Any]]:
    if pulp is None:
        raise RuntimeError("pulp is required. Install via: uv pip install pulp")

    n = len(clues)
    if n == 0:
        return None

    prob = pulp.LpProblem("query_ilp", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", cat=pulp.LpBinary) for j in range(n)]

    # ----- Objective: min sum x_j -----
    prob += pulp.lpSum(x)

    # ----- Object exclusion -----
    for row in object_exclusion_matrix:
        prob += pulp.lpSum(row[j] * x[j] for j in range(n)) >= 1

    # ----- Time exclusion -----
    if template.require_time_uniqueness and enforce_time_uniqueness:
        for row in time_exclusion_matrix:
            prob += pulp.lpSum(row[j] * x[j] for j in range(n)) >= 1

    # ----- Per-member coverage -----
    for member_index in range(target.arity):
        members = [j for j, c in enumerate(clues) if member_index in c.member_indices]
        if not members:
            return None
        prob += pulp.lpSum(x[j] for j in members) >= 1
        if target.arity > 1:
            grounded = [j for j in members if clues[j].clue_type != "cls"]
            if not grounded:
                return None
            prob += pulp.lpSum(x[j] for j in grounded) >= 1

    # ----- Type min/max -----
    for ctype in CLUE_TYPES:
        idxs = [j for j, c in enumerate(clues) if c.clue_type == ctype]
        if not idxs:
            if int(template.type_min.get(ctype, 0)) > 0:
                return None
            continue
        prob += pulp.lpSum(x[j] for j in idxs) >= int(template.type_min.get(ctype, 0))
        prob += pulp.lpSum(x[j] for j in idxs) <= int(template.type_max.get(ctype, template.k_max))

    # ----- k_min / k_max -----
    prob += pulp.lpSum(x) >= int(template.k_min)
    prob += pulp.lpSum(x) <= int(template.k_max)

    # ----- q_min: sum chain_len >= q_min -----
    prob += pulp.lpSum(int(c.chain_len) * x[j] for j, c in enumerate(clues)) >= int(template.q_min)

    # ----- Sequence constraints -----
    seq_idx = [j for j, c in enumerate(clues) if c.clue_type == "seq"]
    seq_min = int(template.seq_min_chain_len)
    seq_max = int(template.seq_max_chain_len)
    qualified = [
        j for j in seq_idx
        if (seq_min <= 0 or clues[j].chain_len >= seq_min)
        and (seq_max <= 0 or clues[j].chain_len <= seq_max)
    ]
    if template.require_seq:
        if not qualified:
            return None
        prob += pulp.lpSum(x[j] for j in qualified) >= int(template.require_seq)

    if template.min_long_seq_len > 0 or template.min_seq_chain_sum > 0:
        if not seq_idx:
            return None
        long_seq = [j for j in seq_idx if clues[j].chain_len >= int(template.min_long_seq_len)]
        seq_chain_sum = pulp.lpSum(int(clues[j].chain_len) * x[j] for j in seq_idx)
        if template.min_long_seq_len > 0 and template.min_seq_chain_sum > 0:
            # OR via auxiliary binary: enough_seq_sum=1 -> sum>=S, OR pick a long seq.
            S = int(template.min_seq_chain_sum)
            big_M = sum(int(clues[j].chain_len) for j in seq_idx) + S + 1
            enough = pulp.LpVariable("enough_seq_sum", cat=pulp.LpBinary)
            # enough=1 implies seq_chain_sum >= S
            prob += seq_chain_sum >= S - big_M * (1 - enough)
            # at least one of {enough, any long_seq selected}
            prob += enough + pulp.lpSum(x[j] for j in long_seq) >= 1
        elif template.min_seq_chain_sum > 0:
            prob += seq_chain_sum >= int(template.min_seq_chain_sum)
        else:
            if not long_seq:
                return None
            prob += pulp.lpSum(x[j] for j in long_seq) >= 1

    # ----- Solve with CBC, time-limited -----
    solver = pulp.PULP_CBC_CMD(msg=bool(msg), timeLimit=float(time_limit_sec))
    status = prob.solve(solver)

    status_name = pulp.LpStatus.get(status, "Unknown")
    if status_name not in {"Optimal", "Feasible"}:
        return None

    chosen = [j for j in range(n) if x[j].value() is not None and round(x[j].value()) >= 0.5]
    if not chosen:
        return None

    return {
        "status": "OPTIMAL" if status_name == "Optimal" else "FEASIBLE",
        "selected_indices": sorted(chosen),
        "objective": int(round(pulp.value(prob.objective))),
    }
