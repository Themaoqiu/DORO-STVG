"""Random baseline: uniformly sample subsets of size in [k_min, k_max] until one is feasible."""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from ..data_models import AtomicClue, CandidateTarget, TemplateSpec
from ._common import is_feasible


def solve_query_random(
    target: CandidateTarget,
    template: TemplateSpec,
    clues: List[AtomicClue],
    object_exclusion_matrix: List[List[int]],
    time_exclusion_matrix: List[List[int]],
    enforce_time_uniqueness: bool = True,
    time_limit_sec: float = 2.0,
    *,
    max_trials: int = 500,
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    _ = time_limit_sec
    n = len(clues)
    if n == 0:
        return None
    rng = random.Random(seed)

    k_lo = max(1, int(template.k_min))
    k_hi = max(k_lo, min(int(template.k_max), n))

    indices = list(range(n))
    for _t in range(max_trials):
        k = rng.randint(k_lo, k_hi)
        sel = rng.sample(indices, k)
        if is_feasible(
            sel, target, template, clues,
            object_exclusion_matrix, time_exclusion_matrix,
            enforce_time_uniqueness,
        ):
            return {
                "status": "FEASIBLE",
                "selected_indices": sorted(sel),
                "objective": len(sel),
            }
    return None
