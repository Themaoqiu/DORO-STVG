from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from api_sync.api import StreamGenerator
from api_sync.utils.parser import JSONParser

from .clues import (
    _DIFFICULTY_BUCKET_ORDER,
    _assign_difficulty_buckets_5,
    _assign_sampling_buckets,
    _compute_candidate_difficulty,
    _profile_for_candidate,
    _template_difficulty_rank,
    _template_target_labels,
    build_atomic_clues,
    build_exclusion_matrix,
)
from .data_models import (
    AtomicClue,
    CandidateProfile,
    CandidateTarget,
    DifficultyWeights,
    GraphIndex,
    TemplateSpec,
    _build_default_templates,
)
from .indexing import build_candidate_intervals, build_graph_index
from .solver import _query_core_from_clues, _render_query, decorate_query, solve_query_cpsat
from .text_utils import (
    _load_env_var_from_project_env,
    _norm,
    _normalize_api_keys,
    _summarize_synonyms_from_graph_file,
    _to_int,
    set_synonyms,
)


def _query_target_signature(node: Dict[str, Any]) -> Tuple[Tuple[str, int, int], ...]:
    target = node.get("target") or {}
    members = target.get("members") or []
    signature: List[Tuple[str, int, int]] = []
    for member in members:
        object_id = str(member.get("object_id", "")).strip()
        if not object_id:
            continue
        start = _to_int(member.get("start_frame"), 0)
        end = _to_int(member.get("end_frame"), start)
        if end < start:
            start, end = end, start
        signature.append((object_id, start, end))
    signature.sort()
    return tuple(signature)


def _dedupe_query_nodes(query_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_signatures: Set[Tuple[Tuple[str, int, int], ...]] = set()
    for node in query_nodes:
        signature = _query_target_signature(node)
        if not signature or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(node)
    return deduped


def _limit_query_nodes_per_video(
    query_nodes: List[Dict[str, Any]],
    *,
    max_queries_per_video: Optional[int] = None,
    max_queries_per_difficulty_bucket: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not query_nodes:
        return []
    if max_queries_per_video is None and max_queries_per_difficulty_bucket is None:
        return query_nodes

    difficulty_order = tuple(_DIFFICULTY_BUCKET_ORDER.keys())
    bucket_counts: Dict[str, int] = defaultdict(int)
    selected: List[Dict[str, Any]] = []

    def can_take(node: Dict[str, Any]) -> bool:
        if max_queries_per_video is not None and len(selected) >= max_queries_per_video:
            return False
        bucket = str(node.get("difficulty_bucket", "medium"))
        if (
            max_queries_per_difficulty_bucket is not None
            and bucket_counts[bucket] >= max_queries_per_difficulty_bucket
        ):
            return False
        return True

    ordered_nodes = sorted(
        query_nodes,
        key=lambda node: (
            _DIFFICULTY_BUCKET_ORDER.get(str(node.get("difficulty_bucket", "medium")), _DIFFICULTY_BUCKET_ORDER["medium"]),
            _to_int(node.get("target", {}).get("members", []).__len__(), 0),
            float(node.get("D", 0.0)),
            str(node.get("query_id", "")),
        ),
    )

    for bucket in difficulty_order:
        for node in ordered_nodes:
            if str(node.get("difficulty_bucket", "medium")) != bucket:
                continue
            if not can_take(node):
                continue
            selected.append(node)
            bucket_counts[bucket] += 1
            break

    if max_queries_per_video is None or len(selected) >= max_queries_per_video:
        return selected

    for node in ordered_nodes:
        if node in selected:
            continue
        if not can_take(node):
            continue
        bucket = str(node.get("difficulty_bucket", "medium"))
        selected.append(node)
        bucket_counts[bucket] += 1
        if len(selected) >= max_queries_per_video:
            break

    return selected


class CPSATQuerySampler:
    def __init__(
        self,
        min_interval_len: int = 3,
        max_intervals_per_object: int = 12,
        max_target_arity: int = 3,
        max_multi_intervals_per_group: int = 8,
        max_multi_candidates_total: int = 240,
        strict_time_uniqueness_multi_target: bool = False,
        max_chain_len: int = 4,
        max_queries_per_candidate: int = 1,
        time_limit_sec: float = 2.0,
        seed: int = 7,
        weights: DifficultyWeights = DifficultyWeights(),
    ) -> None:
        self.min_interval_len = max(1, int(min_interval_len))
        self.max_intervals_per_object = max(1, int(max_intervals_per_object))
        self.max_target_arity = min(3, max(1, int(max_target_arity)))
        self.max_multi_intervals_per_group = max(1, int(max_multi_intervals_per_group))
        self.max_multi_candidates_total = max(1, int(max_multi_candidates_total))
        self.strict_time_uniqueness_multi_target = bool(strict_time_uniqueness_multi_target)
        self.max_chain_len = max(2, int(max_chain_len))
        self.max_queries_per_candidate = max(1, int(max_queries_per_candidate))
        self.time_limit_sec = max(0.1, float(time_limit_sec))
        self.seed = int(seed)
        self.weights = weights
        self.templates = _build_default_templates()

    def _candidate_order(
        self,
        candidates: List[CandidateTarget],
        template: TemplateSpec,
        sampled_intervals_by_sig: Dict[Tuple[str, ...], Set[Tuple[int, int]]],
        candidate_use_count: Dict[str, int],
    ) -> List[CandidateTarget]:
        pool = [
            c
            for c in candidates
            if template.arity_min <= c.arity <= template.arity_max
            and c.difficulty_bucket in _template_target_labels(template)
        ]
        if not pool:
            pool = [
                c
                for c in candidates
                if c.sampling_bucket == template.bucket and template.arity_min <= c.arity <= template.arity_max
            ]

        def rank(c: CandidateTarget) -> Tuple[int, int, int, float, int]:
            sig = tuple(m.object_id for m in c.members)
            interval = c.interval
            sampled_for_sig = sampled_intervals_by_sig.get(sig, set())
            if sampled_for_sig and interval not in sampled_for_sig:
                mode = 0
            elif not sampled_for_sig:
                mode = 1
            else:
                mode = 2
            bucket_distance, difficulty_order = _template_difficulty_rank(c, template)
            return (
                mode,
                candidate_use_count.get(c.candidate_id, 0),
                bucket_distance,
                difficulty_order,
                interval[1] - interval[0] + 1,
            )

        return sorted(pool, key=rank)

    def _solve_for_candidate_template(
        self,
        candidate: CandidateTarget,
        all_candidates: List[CandidateTarget],
        profile_map: Dict[str, CandidateProfile],
        template: TemplateSpec,
    ) -> Optional[Dict[str, Any]]:
        clues = build_atomic_clues(self._index, candidate, max_chain_len=self.max_chain_len)
        if not clues:
            return None

        _, object_E, time_E = build_exclusion_matrix(candidate, all_candidates, clues, profile_map)
        solved = solve_query_cpsat(
            target=candidate,
            template=template,
            clues=clues,
            object_exclusion_matrix=object_E,
            time_exclusion_matrix=time_E,
            enforce_time_uniqueness=(candidate.arity == 1 or self.strict_time_uniqueness_multi_target),
            time_limit_sec=self.time_limit_sec,
        )
        if not solved:
            return None

        selected = [clues[i] for i in solved["selected_indices"]]
        profile = profile_map[candidate.candidate_id]
        core_query = _query_core_from_clues(selected, profile) or _render_query(profile, selected)
        suffix, decorations = decorate_query(selected, profile)
        query = _norm(core_query + suffix)

        return {
            "candidate": candidate,
            "profile": profile,
            "selected_clues": selected,
            "solver": solved,
            "template": template,
            "core_query": core_query,
            "decorations": decorations,
            "query": query,
        }

    def _templates_for_arity(self, multi: bool) -> List[TemplateSpec]:
        if multi:
            return [t for t in self.templates if t.arity_min >= 2]
        return [t for t in self.templates if t.arity_min <= 1 <= t.arity_max and t.arity_max == 1]

    def _node_from_solution(self, picked: Dict[str, Any], query_index: int) -> Dict[str, Any]:
        cand: CandidateTarget = picked["candidate"]
        profile: CandidateProfile = picked["profile"]
        template: TemplateSpec = picked["template"]
        members = [
            {
                "object_id": m.object_id,
                "class": profile.classes[idx],
                "start_frame": m.start,
                "end_frame": m.end,
            }
            for idx, m in enumerate(cand.members)
        ]
        per_target_clues: List[Dict[str, Any]] = []
        for idx, member in enumerate(members, 1):
            grouped = []
            for c in picked["selected_clues"]:
                if tuple(c.member_indices) != (idx - 1,):
                    continue
                grouped.append(
                    {
                        "type": c.clue_type,
                        "text": c.text,
                        "chain_len": c.chain_len,
                        "is_temporal_evidence": c.is_temporal_evidence,
                    }
                )
            per_target_clues.append(
                {
                    "target_index": idx,
                    **member,
                    "clues": grouped,
                }
            )

        shared_clues = [
            {
                "type": c.clue_type,
                "text": c.text,
                "member_indices": list(c.member_indices),
                "chain_len": c.chain_len,
                "is_temporal_evidence": c.is_temporal_evidence,
            }
            for c in picked["selected_clues"]
            if len(c.member_indices) != 1
        ]

        return {
            "query_id": f"cpsat_{query_index}_{cand.candidate_id}",
            "query": picked["query"],
            "query_core": picked["core_query"],
            "query_decorations": picked["decorations"],
            "template": template.name,
            "difficulty_bucket": cand.difficulty_bucket,
            "D_t": cand.D_t,
            "D_s": cand.D_s,
            "D": cand.difficulty,
            "target": {
                "candidate_id": cand.candidate_id,
                "members": members,
            },
            "clues": {
                "per_target": per_target_clues,
                "shared": shared_clues,
            },
            "solver": picked["solver"],
        }

    def _sample_with_templates(
        self,
        *,
        candidates: List[CandidateTarget],
        profile_map: Dict[str, CandidateProfile],
        templates: List[TemplateSpec],
        start_query_index: int,
    ) -> List[Dict[str, Any]]:
        if not candidates or not templates:
            return []

        candidate_use_count: Dict[str, int] = defaultdict(int)
        sampled_intervals_by_sig: Dict[Tuple[str, ...], Set[Tuple[int, int]]] = defaultdict(set)
        used_object_sigs: Set[Tuple[str, ...]] = set()
        solve_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
        nodes: List[Dict[str, Any]] = []

        while True:
            best_pick: Optional[Dict[str, Any]] = None
            best_rank: Optional[Tuple[int, int, float, int, int, int]] = None

            for template_index, template in enumerate(templates):
                picked = None
                for cand in self._candidate_order(candidates, template, sampled_intervals_by_sig, candidate_use_count):
                    obj_sig = tuple(m.object_id for m in cand.members)
                    if obj_sig in used_object_sigs:
                        continue
                    if candidate_use_count[cand.candidate_id] >= self.max_queries_per_candidate:
                        continue
                    cache_key = (cand.candidate_id, template.name)
                    if cache_key not in solve_cache:
                        solve_cache[cache_key] = self._solve_for_candidate_template(
                            candidate=cand,
                            all_candidates=candidates,
                            profile_map=profile_map,
                            template=template,
                        )
                    solved = solve_cache[cache_key]
                    if solved:
                        picked = solved
                        break

                if not picked:
                    continue

                cand = picked["candidate"]
                interval = cand.interval
                rank = (
                    candidate_use_count.get(cand.candidate_id, 0),
                    *_template_difficulty_rank(cand, template),
                    interval[1] - interval[0] + 1,
                    -int(round(template.weight * 1000)),
                    template_index,
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pick = picked

            if not best_pick:
                break

            cand: CandidateTarget = best_pick["candidate"]
            candidate_use_count[cand.candidate_id] += 1
            obj_sig = tuple(m.object_id for m in cand.members)
            sampled_intervals_by_sig[obj_sig].add(cand.interval)
            used_object_sigs.add(obj_sig)
            nodes.append(self._node_from_solution(best_pick, start_query_index + len(nodes)))

        return nodes

    def generate_for_graph(self, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._index = build_graph_index(graph)
        candidates = build_candidate_intervals(
            index=self._index,
            min_interval_len=self.min_interval_len,
            max_intervals_per_object=self.max_intervals_per_object,
            include_full_track=True,
            max_target_arity=self.max_target_arity,
            max_multi_intervals_per_group=self.max_multi_intervals_per_group,
            max_multi_candidates_total=self.max_multi_candidates_total,
        )
        if not candidates:
            return []

        profile_map = {
            c.candidate_id: _profile_for_candidate(self._index, c, max_chain_len=self.max_chain_len)
            for c in candidates
        }

        for c in candidates:
            D_t, D_s, D, meta = _compute_candidate_difficulty(c, self._index, candidates, profile_map, self.weights)
            c.D_t = D_t
            c.D_s = D_s
            c.difficulty = D
            c.meta.update(meta)

        single_candidates = [c for c in candidates if c.arity == 1]
        multi_candidates = [c for c in candidates if c.arity >= 2]
        _assign_sampling_buckets(single_candidates)
        _assign_difficulty_buckets_5(single_candidates)
        if multi_candidates:
            _assign_sampling_buckets(multi_candidates)
            _assign_difficulty_buckets_5(multi_candidates)

        single_templates = self._templates_for_arity(multi=False)
        results = self._sample_with_templates(
            candidates=single_candidates,
            profile_map=profile_map,
            templates=single_templates,
            start_query_index=0,
        )

        multi_templates = self._templates_for_arity(multi=True)
        if self.max_target_arity > 1:
            results.extend(
                self._sample_with_templates(
                    candidates=multi_candidates,
                    profile_map=profile_map,
                    templates=multi_templates,
                    start_query_index=len(results),
                )
            )

        return results

    async def polish_queries_with_llm(
        self,
        query_nodes: List[Dict[str, Any]],
        model_name: str,
        api_keys: Optional[Union[str, Iterable[str]]],
        max_concurrent_per_key: int = 100,
        max_retries: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        return await polish_queries_with_llm(
            query_nodes=query_nodes,
            model_name=model_name,
            api_keys=api_keys,
            max_concurrent_per_key=max_concurrent_per_key,
            max_retries=max_retries,
        )

    def _build_minimal_records(self, graph: Dict[str, Any], query_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        objects = {
            str(obj.get("node_id")): obj
            for obj in (graph.get("object_nodes") or [])
            if str(obj.get("node_id", "")).strip()
        }
        records: List[Dict[str, Any]] = []
        for q in query_nodes:
            target = q.get("target") or {}
            members = target.get("members") or []
            if not members:
                continue

            member_payload: List[Dict[str, Any]] = []
            for m in members:
                oid = str(m.get("object_id", "")).strip()
                if not oid or oid not in objects:
                    continue
                start = _to_int(m.get("start_frame"), 0)
                end = _to_int(m.get("end_frame"), start)
                if end < start:
                    start, end = end, start
                raw_boxes = objects[oid].get("bboxes") or {}
                boxes: Dict[str, Any] = {}
                for frame, box in raw_boxes.items():
                    frame_int = _to_int(frame, -1)
                    if frame_int < 0:
                        continue
                    if start <= frame_int <= end:
                        boxes[str(frame_int)] = box
                if boxes:
                    boxes = {str(f): boxes[str(f)] for f in sorted(int(k) for k in boxes.keys())}
                if not boxes:
                    continue
                member_payload.append(
                    {
                        "object_id": oid,
                        "start_frame": start,
                        "end_frame": end,
                        "target_index": len(member_payload) + 1,
                        "boxes": boxes,
                    }
                )
            if not member_payload:
                continue

            records.append(
                {
                    "video_path": graph.get("video_path"),
                    "video_width": graph.get("video_width"),
                    "video_height": graph.get("video_height"),
                    "query_id": q.get("query_id"),
                    "query": q.get("query"),
                    "llm_polished": q.get("llm_polished", False),
                    "query_core": q.get("query_core"),
                    "template": q.get("template"),
                    "difficulty_bucket": q.get("difficulty_bucket"),
                    "D_t": q.get("D_t"),
                    "D_s": q.get("D_s"),
                    "D": q.get("D"),
                    "target_arity": len(member_payload),
                    "target_members": member_payload,
                    "per_target_queries": q.get("per_target_queries", {}),
                    "clues": q.get("clues", {}),
                    "solver": q.get("solver"),
                }
            )
        return records

    def process_jsonl(
        self,
        input_path: str,
        output_path: str,
        use_llm_polish: bool = False,
        polish_model_name: str = "gpt-4.1-mini",
        api_keys: Optional[Union[str, Iterable[str]]] = None,
        max_concurrent_per_key: int = 100,
        max_retries: int = 5,
        max_queries_per_video: Optional[int] = None,
        max_queries_per_difficulty_bucket: Optional[int] = None,
    ) -> None:
        in_path = Path(input_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rel_map, attr_map, stats = _summarize_synonyms_from_graph_file(in_path)
        set_synonyms(rel_map, attr_map)
        print(
            "[query_cpsat] synonym_summary: "
            f"graphs={stats['graphs']} "
            f"rel_values={stats['relation_values']} attr_values={stats['attribute_values']} "
            f"rel_synonyms={stats['relation_synonyms']} attr_synonyms={stats['attribute_synonyms']}",
            flush=True,
        )

        prepared_graphs: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
        with in_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                graph = json.loads(line)
                query_nodes = self.generate_for_graph(graph)
                raw_count = len(query_nodes)
                query_nodes = _dedupe_query_nodes(query_nodes)
                dropped_duplicates = raw_count - len(query_nodes)
                if dropped_duplicates > 0:
                    print(
                        f"[query_cpsat] dedupe_drop: video={Path(str(graph.get('video_path', ''))).name} "
                        f"dropped={dropped_duplicates}",
                        flush=True,
                    )
                limited_nodes = _limit_query_nodes_per_video(
                    query_nodes,
                    max_queries_per_video=max_queries_per_video,
                    max_queries_per_difficulty_bucket=max_queries_per_difficulty_bucket,
                )
                dropped_by_limit = len(query_nodes) - len(limited_nodes)
                query_nodes = limited_nodes
                if dropped_by_limit > 0:
                    print(
                        f"[query_cpsat] per_video_limit: video={Path(str(graph.get('video_path', ''))).name} "
                        f"kept={len(query_nodes)} dropped={dropped_by_limit} "
                        f"max_queries_per_video={max_queries_per_video} "
                        f"max_queries_per_difficulty_bucket={max_queries_per_difficulty_bucket}",
                        flush=True,
                    )
                prepared_graphs.append((graph, query_nodes))

        if use_llm_polish:
            all_query_nodes = [query for _, query_nodes in prepared_graphs for query in query_nodes]
            polished_map = asyncio.run(
                polish_queries_with_llm(
                    query_nodes=all_query_nodes,
                    model_name=polish_model_name,
                    api_keys=api_keys,
                    max_concurrent_per_key=max_concurrent_per_key,
                    max_retries=max_retries,
                )
            )
            dropped = 0
            for _, query_nodes in prepared_graphs:
                kept_nodes: List[Dict[str, Any]] = []
                for q in query_nodes:
                    qid = str(q.get("query_id", ""))
                    polished = polished_map.get(qid) or {}
                    polished_query = _norm(polished.get("query", ""))
                    if polished_query:
                        q["query"] = polished_query
                        q["per_target_queries"] = polished.get("target_queries", {})
                        q["llm_polished"] = True
                        kept_nodes.append(q)
                    else:
                        dropped += 1
                query_nodes[:] = kept_nodes
            if dropped > 0:
                print(f"[query_cpsat] llm_generation_drop: dropped={dropped}", flush=True)

        graph_count = 0
        record_count = 0
        with out_path.open("w", encoding="utf-8") as fout:
            for graph, query_nodes in prepared_graphs:
                records = self._build_minimal_records(graph, query_nodes)
                for record in records:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                graph_count += 1
                record_count += len(records)

        print(f"[query_cpsat] done. graphs={graph_count}, records={record_count}, output={out_path}", flush=True)


# ---------------------------------------------------------------------------
# LLM-based query polishing (called as the optional last stage of the pipeline)
# ---------------------------------------------------------------------------


_QUERY_POLISH_SYSTEM_PROMPT = """## Role
You are a precise Query Writer for spatiotemporal video grounding. Your task is
to generate one natural, concise grounding query from structured clues while preserving
all grounding semantics.
## Core Principle
- Preserve semantics exactly. Do not add, remove, or alter target identity clues.
- Keep temporal meaning faithful to the original clue set.
- Use clear and fluent wording suitable for human annotation.
- Also produce one standalone natural-language target description for each target object.
## Output Rule
Return strict JSON only:
{
  "query": "<polished full query>",
  "target_queries": {
    "target 1": "<standalone natural-language description for target 1>",
    "target 2": "<standalone natural-language description for target 2>"
  }
}
"""

_QUERY_POLISH_PROMPT_TEMPLATE = """## Task
You will receive visual cues for constructing query text for video localization tasks. You need to correctly use all the cues to generate a single query for locating the target object. Ensure the description is fluent and natural without losing any information.
## Input Format
- target_classes_json: target index to class mapping.
- target_intervals_json: target index to interval mapping.
- clues_per_target_json: clues grouped by target and category.
- category_legend_json: category name mapping.
## The information you need to use:
1. target object class for localization:
{target_classes_text}

2. Visual cues used to localize each object:
{clues_per_target_text}
## Guidelines
- Carefully examine the content in the section Visual cues used to localize each object.
  - If have only one object: output a semantically clear query using all the provided clue information to locate this object.
  - If have multi-target: Using all the provided clues, generate one semantically clear query that can locate both objects simultaneously. You may describe multiple objects separately, such as "the person in black clothing and the person leaning against the wall"; you may also fuse the descriptions of multiple objects, such as "the person in black clothing and the cat at his feet".
- Prefer concise natural English; avoid redundant filler phrases.
- The clues may not be naturally phrased, and the generated query should not follow the language style of the clues.
- In addition to the full query, also output one simple general natural-language target description for each target separately. And keep it brief—just enough to distinguish the target objects.
- Each standalone target description should be a natural referring expression rather than a keyword list.
## Output Format
Return one valid JSON object:
{{
  "query": "...",
  "target_queries": {{
    "target 1": "...",
    "target 2": "..."
  }}
}}
## Output Specifications
- Output strict JSON only.
- "query" must be non-empty and in English.
- "target_queries" must be an object.
- Each target listed in the input must have exactly one entry in "target_queries".
- Each target description must be non-empty and in English.
"""

_PROMPT_CATEGORY_TEXT: Dict[str, str] = {
    "cls": "class",
    "app": "appearance",
    "env": "Interaction with the environment",
    "act": "action",
    "seq": "Sequence of actions and spatial position changes",
    "spa": "Spatial position",
    "int": "Interaction with other objects",
}

_PROMPT_CATEGORY_ORDER: Tuple[str, ...] = ("cls", "app", "env", "act", "seq", "spa", "int")


def _resolve_api_keys(api_keys: Optional[Union[str, Iterable[str]]]) -> List[str]:
    if api_keys is None:
        api_keys = os.getenv("API_KEYS", "")
    keys = _normalize_api_keys(api_keys)
    if keys:
        return keys
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "API_KEYS":
                    keys = _normalize_api_keys(value.strip().strip('"').strip("'"))
                    if keys:
                        return keys
    return []


def _resolve_api_base_url() -> str:
    return (
        os.getenv("MM_API_BASE_URL", "").strip()
        or os.getenv("VISION_API_BASE_URL", "").strip()
        or os.getenv("VIDEO_API_BASE_URL", "").strip()
        or _load_env_var_from_project_env("MM_API_BASE_URL")
        or _load_env_var_from_project_env("VISION_API_BASE_URL")
        or _load_env_var_from_project_env("VIDEO_API_BASE_URL")
    )


def _build_polish_prompt(node: Dict[str, Any], idx: int) -> Dict[str, Any]:
    query_id = str(node.get("query_id", f"cpsat_{idx}"))
    target = node.get("target") or {}
    members = target.get("members") or []
    clues = node.get("clues") or {}

    target_classes = []
    for i, m in enumerate(members, start=1):
        cls = _norm(m.get("class", "object")) or "object"
        s = _to_int(m.get("start_frame"), 0)
        e = _to_int(m.get("end_frame"), s)
        if e < s:
            s, e = e, s
        target_classes.append({"target_index": i, "class": cls, "interval": [s, e]})
    if not target_classes:
        target_classes = [{"target_index": 1, "class": "object", "interval": ["unknown", "unknown"]}]

    target_lines: List[str] = []
    for item in target_classes:
        tid = _to_int(item.get("target_index"), -1)
        cls = _norm(item.get("class", "object")) or "object"
        if tid <= 0:
            tid = len(target_lines) + 1
        target_lines.append(f"- target {tid}: {cls}")
    target_classes_text = "\n".join(target_lines) if target_lines else "- target 1: object"

    categories = ["cls", "app", "act", "seq", "spa", "int", "env"]
    per_target: Dict[str, Dict[str, List[str]]] = {
        str(i + 1): {c: [] for c in categories} for i in range(len(members))
    }
    shared: Dict[str, List[str]] = {c: [] for c in categories}

    def append_unique(bucket: Dict[str, List[str]], ctype: str, text: str) -> None:
        if not text:
            return
        bucket.setdefault(ctype, [])
        if text not in bucket[ctype]:
            bucket[ctype].append(text)

    for item in clues.get("per_target", []) or []:
        tid = _to_int(item.get("target_index"), -1)
        if not (1 <= tid <= len(members)):
            continue
        for clue in item.get("clues", []) or []:
            append_unique(
                per_target[str(tid)],
                str(clue.get("type", "")).strip().lower() or "unknown",
                _norm(clue.get("text", "")),
            )
    for clue in clues.get("shared", []) or []:
        append_unique(
            shared,
            str(clue.get("type", "")).strip().lower() or "unknown",
            _norm(clue.get("text", "")),
        )

    if shared and target_classes:
        for t in target_classes:
            tid = str(_to_int(t.get("target_index"), -1))
            per_target.setdefault(tid, {})
            for ctype, vals in shared.items():
                if not isinstance(vals, list) or not vals:
                    continue
                existing = per_target[tid].get(ctype, [])
                merged = list(existing)
                for v in vals:
                    vv = _norm(v)
                    if vv and vv not in merged:
                        merged.append(vv)
                per_target[tid][ctype] = merged

    def sort_key(tid: str) -> Tuple[int, str]:
        return (_to_int(tid, 10**9), str(tid))

    clue_lines: List[str] = []
    for tid in sorted(per_target.keys(), key=sort_key):
        clue_lines.append(f"  target {tid}:")
        buckets = per_target.get(tid, {})
        if not isinstance(buckets, dict):
            clue_lines.append("    - xxx")
            continue
        wrote = False
        for ctype in _PROMPT_CATEGORY_ORDER:
            vals = buckets.get(ctype, [])
            if not isinstance(vals, list) or not vals:
                continue
            clean_vals = []
            for v in vals:
                vv = _norm(v)
                lower = vv.lower()
                for p in ("that is ", "that ", "the "):
                    if lower.startswith(p):
                        vv = _norm(vv[len(p):])
                        lower = vv.lower()
                if vv:
                    clean_vals.append(vv)
            if not clean_vals:
                continue
            label = _PROMPT_CATEGORY_TEXT.get(ctype, ctype)
            clue_lines.append(f"    - {label}: {', '.join(clean_vals)}")
            wrote = True
        if not wrote:
            clue_lines.append("    - xxx")
        clue_lines.append("")
    while clue_lines and not clue_lines[-1].strip():
        clue_lines.pop()
    clues_per_target_text = "\n".join(clue_lines) if clue_lines else "  target 1:\n    - xxx"

    prompt_text = _QUERY_POLISH_PROMPT_TEMPLATE.format(
        target_classes_text=target_classes_text,
        clues_per_target_text=clues_per_target_text,
    )
    return {"id": query_id, "prompt": prompt_text}


def _extract_polished_query(response: str) -> Optional[Dict[str, Any]]:
    parsed = JSONParser.parse(response)
    if not isinstance(parsed, dict):
        return None
    query = parsed.get("query")
    if not isinstance(query, str):
        return None
    query = _norm(query)
    if not query:
        return None
    target_queries_raw = parsed.get("target_queries")
    if not isinstance(target_queries_raw, dict):
        return None
    target_queries: Dict[str, str] = {}
    for key, value in target_queries_raw.items():
        kk = _norm(key)
        vv = _norm(value).rstrip(".").strip()
        if not kk or not vv:
            continue
        target_queries[kk] = vv
    if not target_queries:
        return None
    return {"query": query, "target_queries": target_queries}


async def polish_queries_with_llm(
    query_nodes: List[Dict[str, Any]],
    *,
    model_name: str,
    api_keys: Optional[Union[str, Iterable[str]]],
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    system_prompt: str = _QUERY_POLISH_SYSTEM_PROMPT,
) -> Dict[str, Dict[str, Any]]:
    if not query_nodes:
        return {}

    keys = _resolve_api_keys(api_keys)
    if not keys:
        raise ValueError("API_KEYS is required when use_llm_polish=True (set env/.env or pass --api_keys).")

    api_base_url = _resolve_api_base_url()
    if api_base_url:
        os.environ.setdefault("MM_API_BASE_URL", api_base_url)

    generator = StreamGenerator(
        model_name=model_name,
        api_keys=keys,
        max_concurrent_per_key=max_concurrent_per_key,
        max_retries=max_retries,
        rational=False,
        with_unique_id=True,
    )

    prompts: List[Dict[str, Any]] = [_build_polish_prompt(node, idx) for idx, node in enumerate(query_nodes)]

    print(
        f"[query_cpsat] llm_polish_start: model={model_name} prompts={len(prompts)} keys={len(keys)} "
        f"base_url={'set' if api_base_url else 'default'}",
        flush=True,
    )
    polished_map: Dict[str, Dict[str, Any]] = {}
    completed = 0

    async for item in generator.generate_stream(
        prompts=prompts,
        system_prompt=system_prompt,
        validate_func=lambda resp: polished if (polished := _extract_polished_query(resp)) is not None else False,
    ):
        result = item.get("result") if item else None
        if item and isinstance(result, dict):
            polished_map[str(item["id"])] = result
            completed += 1
            print(f"[query_cpsat] llm_polish_progress: {completed}/{len(prompts)}", flush=True)
    print(f"[query_cpsat] llm_polish_done: polished={len(polished_map)}", flush=True)
    return polished_map
