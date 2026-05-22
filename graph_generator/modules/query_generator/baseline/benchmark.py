"""Compare CP-SAT solver against random / greedy / ILP baselines.

Usage:
    uv run python -m graph_generator.modules.query_generator.baseline.benchmark \
        --inputs /Users/themaoqiu/Downloads/preception.jsonl,/Users/themaoqiu/Downloads/mose.jsonl \
        --num_graphs 2 \
        --max_cases 60 \
        --out /tmp/baseline_run.json
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fire

from ..clues import (
    _assign_difficulty_buckets_5,
    _assign_sampling_buckets,
    _compute_candidate_difficulty,
    _profile_for_candidate,
    _template_target_labels,
    build_atomic_clues,
    build_exclusion_matrix,
)
from ..data_models import (
    AtomicClue,
    CandidateTarget,
    DifficultyWeights,
    TemplateSpec,
    _build_default_templates,
)
from ..indexing import build_candidate_intervals, build_graph_index
from ..solver import solve_query_cpsat
from .greedy_solver import solve_query_greedy
from .ilp_solver import solve_query_ilp
from .random_solver import solve_query_random


SolverCase = Tuple[CandidateTarget, TemplateSpec, List[AtomicClue], List[List[int]], List[List[int]], bool]


def _collect_cases_from_graph(
    graph: Dict[str, Any],
    *,
    max_chain_len: int = 4,
    max_intervals_per_object: int = 12,
    max_target_arity: int = 3,
    min_interval_len: int = 3,
    max_multi_intervals_per_group: int = 8,
    max_multi_candidates_total: int = 240,
    weights: DifficultyWeights = DifficultyWeights(),
    strict_time_uniqueness_multi_target: bool = False,
) -> List[SolverCase]:
    """Reproduce the (candidate, template, clues, matrices) tuples that the
    real sampler would feed to ``solve_query_cpsat``."""
    index = build_graph_index(graph)
    candidates = build_candidate_intervals(
        index=index,
        min_interval_len=min_interval_len,
        max_intervals_per_object=max_intervals_per_object,
        include_full_track=True,
        max_target_arity=max_target_arity,
        max_multi_intervals_per_group=max_multi_intervals_per_group,
        max_multi_candidates_total=max_multi_candidates_total,
    )
    if not candidates:
        return []

    profile_map = {
        c.candidate_id: _profile_for_candidate(index, c, max_chain_len=max_chain_len)
        for c in candidates
    }

    for c in candidates:
        D_t, D_s, D, meta = _compute_candidate_difficulty(c, index, candidates, profile_map, weights)
        c.D_t = D_t
        c.D_s = D_s
        c.difficulty = D
        c.meta.update(meta)

    single = [c for c in candidates if c.arity == 1]
    multi = [c for c in candidates if c.arity >= 2]
    _assign_sampling_buckets(single)
    _assign_difficulty_buckets_5(single)
    if multi:
        _assign_sampling_buckets(multi)
        _assign_difficulty_buckets_5(multi)

    templates = _build_default_templates()

    cases: List[SolverCase] = []
    seen = set()
    for tmpl in templates:
        if tmpl.arity_min >= 2:
            pool = multi
        else:
            pool = single
        labels = _template_target_labels(tmpl)
        for cand in pool:
            if not (tmpl.arity_min <= cand.arity <= tmpl.arity_max):
                continue
            if cand.difficulty_bucket not in labels:
                continue
            key = (cand.candidate_id, tmpl.name)
            if key in seen:
                continue
            seen.add(key)
            clues = build_atomic_clues(index, cand, max_chain_len=max_chain_len)
            if not clues:
                continue
            arity_pool = single if cand.arity == 1 else [c for c in candidates if c.arity == cand.arity]
            _, obj_E, time_E = build_exclusion_matrix(cand, arity_pool, clues, profile_map)
            enforce_time = (cand.arity == 1) or strict_time_uniqueness_multi_target
            cases.append((cand, tmpl, clues, obj_E, time_E, enforce_time))
    return cases


def _run_solver(name: str, fn, case: SolverCase, time_limit_sec: float) -> Dict[str, Any]:
    cand, tmpl, clues, obj_E, time_E, enforce = case
    t0 = time.perf_counter()
    try:
        result = fn(
            target=cand,
            template=tmpl,
            clues=clues,
            object_exclusion_matrix=obj_E,
            time_exclusion_matrix=time_E,
            enforce_time_uniqueness=enforce,
            time_limit_sec=time_limit_sec,
        )
        err = None
    except Exception as e:
        result = None
        err = f"{type(e).__name__}: {e}"
    dt = time.perf_counter() - t0
    if result is None:
        return {"solver": name, "feasible": False, "objective": None, "time_sec": dt, "error": err}
    return {
        "solver": name,
        "feasible": True,
        "status": result.get("status"),
        "objective": int(result.get("objective", len(result.get("selected_indices", [])))),
        "selected_indices": list(result.get("selected_indices", [])),
        "time_sec": dt,
        "error": err,
    }


def _summarize(per_case: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_solver: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    cpsat_obj: Dict[int, int] = {}
    for i, case in enumerate(per_case):
        for r in case["results"]:
            by_solver[r["solver"]].append((i, r))
            if r["solver"] == "cpsat" and r["feasible"]:
                cpsat_obj[i] = int(r["objective"])

    summary: Dict[str, Any] = {}
    for name, indexed in by_solver.items():
        n = len(indexed)
        feas = [r for _, r in indexed if r["feasible"]]
        feas_count = len(feas)
        mean_obj = (sum(r["objective"] for r in feas) / feas_count) if feas else None
        mean_time = sum(r["time_sec"] for _, r in indexed) / n if n else 0.0
        gaps = []
        for idx, r in indexed:
            if not r["feasible"] or idx not in cpsat_obj:
                continue
            base = cpsat_obj[idx]
            if base > 0:
                gaps.append((r["objective"] - base) / base)
        summary[name] = {
            "n_cases": n,
            "feasibility_rate": feas_count / n if n else 0.0,
            "mean_objective": mean_obj,
            "mean_time_sec": mean_time,
            "mean_relative_gap_vs_cpsat": (sum(gaps) / len(gaps)) if gaps else None,
        }
    return summary


def _summarize_by_source(per_case: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in per_case:
        # Group by file stem (drop "#gN").
        src = str(c.get("source", "")).split("#")[0] or "unknown"
        by_source[src].append(c)
    return {src: _summarize(rows) for src, rows in by_source.items()}


def _format_table(summary: Dict[str, Any]) -> str:
    header = f"{'solver':<10} {'n':>5} {'feas%':>7} {'mean|x|':>9} {'gap%':>7} {'time(ms)':>10}"
    lines = [header, "-" * len(header)]
    for name in ["cpsat", "ilp", "greedy", "random"]:
        if name not in summary:
            continue
        s = summary[name]
        feas = f"{100 * s['feasibility_rate']:.1f}"
        mo = f"{s['mean_objective']:.2f}" if s['mean_objective'] is not None else "-"
        gap = f"{100 * s['mean_relative_gap_vs_cpsat']:.1f}" if s['mean_relative_gap_vs_cpsat'] is not None else "-"
        ms = f"{1000 * s['mean_time_sec']:.1f}"
        lines.append(f"{name:<10} {s['n_cases']:>5} {feas:>7} {mo:>9} {gap:>7} {ms:>10}")
    return "\n".join(lines)


def _process_one_graph(args):
    """Worker: load graph (or accept dict), build cases, run all 4 solvers, return per-case records."""
    tag, graph_line, time_limit_sec, random_max_trials, random_seed = args
    graph = json.loads(graph_line)
    graph_cases = _collect_cases_from_graph(graph)
    out: List[Dict[str, Any]] = []
    for j, case in enumerate(graph_cases):
        cand, tmpl, clues, obj_E, time_E, enforce = case
        rec = {
            "source": tag,
            "case_in_graph": j,
            "candidate_id": cand.candidate_id,
            "arity": cand.arity,
            "template": tmpl.name,
            "n_clues": len(clues),
            "n_object_rows": len(obj_E),
            "n_time_rows": len(time_E),
            "results": [],
        }
        rec["results"].append(_run_solver("cpsat", solve_query_cpsat, case, time_limit_sec))
        rec["results"].append(_run_solver("ilp", solve_query_ilp, case, time_limit_sec))
        rec["results"].append(_run_solver("greedy", solve_query_greedy, case, time_limit_sec))

        def _random_wrapped(**kw):
            return solve_query_random(max_trials=random_max_trials, seed=random_seed, **kw)
        rec["results"].append(_run_solver("random", _random_wrapped, case, time_limit_sec))
        out.append(rec)
    return tag, out


def run(
    inputs: str,
    num_graphs: int = -1,
    max_cases: int = -1,
    out: Optional[str] = None,
    time_limit_sec: float = 2.0,
    random_max_trials: int = 500,
    random_seed: int = 0,
    num_workers: int = 1,
) -> None:
    paths = [p.strip() for p in inputs.split(",") if p.strip()]
    jobs: List[Tuple[str, str]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for gi, line in enumerate(f):
                if num_graphs >= 0 and gi >= num_graphs:
                    break
                if not line.strip():
                    continue
                tag = f"{Path(path).stem}#g{gi}"
                jobs.append((tag, line))
    print(f"[bench] graphs queued: {len(jobs)} from {len(paths)} files", flush=True)

    per_case: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    payload = [
        (tag, line, time_limit_sec, random_max_trials, random_seed)
        for tag, line in jobs
    ]

    if num_workers <= 1:
        for i, args in enumerate(payload):
            tag, recs = _process_one_graph(args)
            per_case.extend(recs)
            if max_cases >= 0 and len(per_case) >= max_cases:
                per_case = per_case[:max_cases]
                break
            if (i + 1) % 10 == 0 or i + 1 == len(payload):
                print(f"[bench] graphs {i + 1}/{len(payload)} cases={len(per_case)} elapsed={time.perf_counter()-t_start:.1f}s", flush=True)
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futs = [ex.submit(_process_one_graph, a) for a in payload]
            for fut in as_completed(futs):
                tag, recs = fut.result()
                per_case.extend(recs)
                done += 1
                if done % 10 == 0 or done == len(futs):
                    print(f"[bench] graphs {done}/{len(futs)} cases={len(per_case)} elapsed={time.perf_counter()-t_start:.1f}s", flush=True)
                if max_cases >= 0 and len(per_case) >= max_cases:
                    break
        if max_cases >= 0:
            per_case = per_case[:max_cases]

    # Stable indexing
    for i, c in enumerate(per_case):
        c["case_index"] = i

    print(f"[bench] total cases: {len(per_case)}  total time: {time.perf_counter()-t_start:.1f}s", flush=True)

    summary = _summarize(per_case)
    by_source = _summarize_by_source(per_case)

    print()
    print("===== overall =====")
    print(_format_table(summary))
    for src, s in by_source.items():
        print(f"\n===== {src} =====")
        print(_format_table(s))

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "summary_by_source": by_source, "cases": per_case},
                f, ensure_ascii=False, indent=2,
            )
        print(f"[bench] wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(run)
