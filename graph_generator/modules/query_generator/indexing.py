from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Set, Tuple

from .data_models import CandidateMember, CandidateTarget, GraphIndex
from .text_utils import _display_phrase, _norm, _normalize_phrase, _to_int


# ---------------------------------------------------------------------------
# Interval / temporal utilities
# ---------------------------------------------------------------------------


def _sample_class(obj: Dict[str, Any], node_id: str) -> str:
    cls = _norm(obj.get("dam_category") or obj.get("object_class") or "object").lower()
    if cls == "person" and str(node_id).lower().startswith("man"):
        return "person"
    return cls or "object"


def _frames_from_obj(obj: Dict[str, Any]) -> Set[int]:
    bboxes = obj.get("bboxes") or {}
    frames = {_to_int(k, -1) for k in bboxes.keys()}
    frames.discard(-1)
    if frames:
        return frames
    s = _to_int(obj.get("start_frame"), 0)
    e = _to_int(obj.get("end_frame"), s)
    if e < s:
        s, e = e, s
    return set(range(s, e + 1))


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    if e < s:
        return None
    return s, e


def _interval_tiou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    ov = _overlap(a, b)
    if not ov:
        return 0.0
    inter = ov[1] - ov[0] + 1
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _segments_from_frames(frames: Set[int]) -> List[Tuple[int, int]]:
    if not frames:
        return []
    arr = sorted(frames)
    out: List[Tuple[int, int]] = []
    s = e = arr[0]
    for x in arr[1:]:
        if x == e + 1:
            e = x
        else:
            out.append((s, e))
            s = e = x
    out.append((s, e))
    return out


def _extract_relation_segments(rel: Dict[str, Any]) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    for seg in rel.get("time_spans") or []:
        if isinstance(seg, dict):
            a = _to_int(seg.get("start_frame"), 0)
            b = _to_int(seg.get("end_frame"), a)
            if b < a:
                a, b = b, a
            segments.append((a, b))
    if (not segments) and ("start_frame" in rel or "end_frame" in rel):
        a = _to_int(rel.get("start_frame"), 0)
        b = _to_int(rel.get("end_frame"), a)
        if b < a:
            a, b = b, a
        segments.append((a, b))
    return sorted(set(segments))


def _relation_interval(rel: Dict[str, Any]) -> Tuple[int, int]:
    start = _to_int(rel.get("start"), 0)
    end = _to_int(rel.get("end"), start)
    if end < start:
        start, end = end, start
    return start, end


def _normalized_relation_pair(rel: Dict[str, Any]) -> Tuple[str, str]:
    pred = _normalize_phrase(rel.get("predicate", ""), "rel")
    ref_cls = _normalize_phrase(rel.get("ref_class", ""), "attr") or _normalize_phrase(rel.get("ref_class", ""), "rel")
    return pred, ref_cls


def _add_overlap_boundaries(
    boundaries: Set[int],
    span: Tuple[int, int],
    full_span: Tuple[int, int],
) -> None:
    ov = _overlap(span, full_span)
    if not ov:
        return
    boundaries.add(ov[0])
    boundaries.add(ov[1] + 1)


def _parse_seq_token(token: str) -> Tuple[str, str, str]:
    if token.startswith("act:"):
        return "act", token[4:], ""
    if token.startswith("rel:"):
        body = token[4:]
        if ":" in body:
            pred, ref = body.split(":", 1)
        else:
            pred, ref = body, "object"
        return "rel", pred, ref
    return "raw", token, ""


def _format_seq_part(token: str) -> str:
    kind, a, b = _parse_seq_token(token)
    if kind == "act":
        return _display_phrase(a)
    if kind == "rel":
        if b:
            return _norm(f"{_display_phrase(a)} {_display_phrase(b)}")
        return _display_phrase(a)
    return _display_phrase(a)


def _enumerate_ordered_chains(
    events: List[Tuple[int, int, str]],
    max_chain_len: int,
    max_chains: int = 128,
) -> Set[Tuple[str, ...]]:
    if max_chain_len < 2 or len(events) < 2:
        return set()
    sorted_events = sorted(events, key=lambda x: (x[0], x[1], x[2]))
    out: Set[Tuple[str, ...]] = set()

    def dfs(last_idx: int, chain: List[str]) -> None:
        if len(out) >= max_chains:
            return
        if 2 <= len(chain) <= max_chain_len:
            out.add(tuple(chain))
        if len(chain) >= max_chain_len:
            return
        last_start = sorted_events[last_idx][0]
        for j in range(last_idx + 1, len(sorted_events)):
            s, e, token = sorted_events[j]
            if s <= last_start:
                continue
            if token == chain[-1]:
                continue
            chain.append(token)
            dfs(j, chain)
            chain.pop()
            if len(out) >= max_chains:
                return

    for i in range(len(sorted_events)):
        dfs(i, [sorted_events[i][2]])
        if len(out) >= max_chains:
            break
    return out


# ---------------------------------------------------------------------------
# Graph indexing & candidate enumeration
# ---------------------------------------------------------------------------


def build_graph_index(graph: Dict[str, Any]) -> GraphIndex:
    object_nodes = graph.get("object_nodes") or []
    objects: Dict[str, Dict[str, Any]] = {}
    frames: Dict[str, Set[int]] = {}

    for obj in object_nodes:
        node_id = str(obj.get("node_id", "")).strip()
        if not node_id:
            continue
        objects[node_id] = obj
        frames[node_id] = _frames_from_obj(obj)

    def _resolve_node_id(raw_id: Any) -> str:
        text = str(raw_id).strip()
        return text if text in objects else ""

    actions_by_obj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in graph.get("action_nodes") or []:
        owner = _resolve_node_id(node.get("object_id"))
        if owner not in objects:
            continue
        for item in node.get("actions") or []:
            label = _normalize_phrase(item.get("action_label", ""), "act")
            if not label:
                continue
            s = _to_int(item.get("start_frame"), 0)
            e = _to_int(item.get("end_frame"), s)
            if e < s:
                s, e = e, s
            ov = _overlap((s, e), (min(frames[owner]), max(frames[owner])))
            if not ov:
                continue
            actions_by_obj[owner].append(
                {
                    "label": label,
                    "start": ov[0],
                    "end": ov[1],
                    "targets": [
                        tid
                        for tid in (_resolve_node_id(t) for t in (item.get("target_object_ids") or []))
                        if tid in objects and tid != owner
                    ],
                }
            )

    relations_by_subj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge in graph.get("edges") or []:
        if "subject_id" not in edge or "relationships" not in edge:
            continue
        sid = _resolve_node_id(edge.get("subject_id"))
        if sid not in objects:
            continue
        for rel in edge.get("relationships") or []:
            oid = _resolve_node_id(rel.get("object_id"))
            if oid not in objects:
                continue
            pred = _normalize_phrase(rel.get("predicate_verb", ""), "rel")
            if not pred:
                continue
            segments = _extract_relation_segments(rel)
            if not segments:
                shared = frames[sid] & frames[oid]
                segments = _segments_from_frames(shared)
            edge_type = _norm(rel.get("edge_type", "")).lower() or "spatial"
            rel_type = _norm(rel.get("relationship_type", "")).lower()
            ref_cls = _sample_class(objects[oid], oid)
            for s, e in segments:
                ov = _overlap((s, e), (min(frames[sid]), max(frames[sid])))
                if not ov:
                    continue
                relations_by_subj[sid].append(
                    {
                        "predicate": pred,
                        "start": ov[0],
                        "end": ov[1],
                        "object_id": oid,
                        "ref_class": ref_cls,
                        "edge_type": edge_type,
                        "relationship_type": rel_type,
                    }
                )

    temporal_spans: List[Dict[str, int]] = [
        {
            "clip_id": _to_int(tn.get("clip_id"), 0),
            "start": _to_int(tn.get("start_frame"), 0),
            "end": _to_int(tn.get("end_frame"), 0),
        }
        for tn in (graph.get("temporal_nodes") or [])
    ]

    return GraphIndex(
        objects=objects,
        frames=frames,
        actions_by_obj=actions_by_obj,
        relations_by_subj=relations_by_subj,
        temporal_spans=temporal_spans,
    )


def _downsample_intervals(intervals: Set[Tuple[int, int]], limit: int) -> List[Tuple[int, int]]:
    ordered = sorted(intervals, key=lambda x: (x[0], x[1]))
    if limit <= 0 or len(ordered) <= limit:
        return ordered
    keep = [ordered[0], ordered[-1]] if len(ordered) >= 2 else ordered
    step = max(1, len(ordered) // max(1, limit - len(keep)))
    keep.extend(ordered[1:-1:step])
    return sorted(set(keep))[:limit]


def _ranked_multi_object_ids(index: GraphIndex, max_objects: int = 18) -> List[str]:
    object_ids = sorted(index.objects.keys())
    if len(object_ids) <= max_objects:
        return object_ids
    return sorted(object_ids, key=lambda oid: (-len(index.frames.get(oid, set())), oid))[:max_objects]


def _interval_group_gap(intervals: Tuple[Tuple[int, int], ...]) -> int:
    starts = [s for s, _ in intervals]
    ends = [e for _, e in intervals]
    return max(0, max(starts) - min(ends))


def _interval_group_span(intervals: Tuple[Tuple[int, int], ...]) -> int:
    starts = [s for s, _ in intervals]
    ends = [e for _, e in intervals]
    return max(ends) - min(starts)


def _multi_interval_choices_for_combo(
    combo: Tuple[str, ...],
    object_intervals: Dict[str, List[Tuple[int, int]]],
    limit: int,
) -> List[Tuple[Tuple[int, int], ...]]:
    if limit <= 0:
        return []
    per_obj_cap = max(1, min(3, limit))
    interval_lists: List[List[Tuple[int, int]]] = []
    for oid in combo:
        intervals = object_intervals.get(oid, [])
        if not intervals:
            return []
        interval_lists.append(intervals[:per_obj_cap])

    products = list(product(*interval_lists))
    products = [tuple(p) for p in products]
    products.sort(key=lambda p: (_interval_group_gap(p), _interval_group_span(p), p))
    return products[:limit]


def build_candidate_intervals(
    index: GraphIndex,
    min_interval_len: int = 3,
    max_intervals_per_object: int = 12,
    include_full_track: bool = True,
    max_target_arity: int = 3,
    max_multi_intervals_per_group: int = 8,
    max_multi_candidates_total: int = 240,
) -> List[CandidateTarget]:
    candidates: List[CandidateTarget] = []
    object_intervals_by_obj: Dict[str, List[Tuple[int, int]]] = {}

    for object_id, obj in index.objects.items():
        obj_frames = index.frames.get(object_id, set())
        if not obj_frames:
            continue
        full_s, full_e = min(obj_frames), max(obj_frames)
        boundaries: Set[int] = {full_s, full_e + 1}

        full_span = (full_s, full_e)
        for item in index.actions_by_obj.get(object_id, []):
            _add_overlap_boundaries(boundaries, (item["start"], item["end"]), full_span)
        for rel in index.relations_by_subj.get(object_id, []):
            _add_overlap_boundaries(boundaries, (rel["start"], rel["end"]), full_span)
        for tn in index.temporal_spans:
            _add_overlap_boundaries(boundaries, (tn["start"], tn["end"]), full_span)

        intervals: Set[Tuple[int, int]] = set()
        if include_full_track and full_e - full_s + 1 >= min_interval_len:
            intervals.add((full_s, full_e))

        sorted_bounds = sorted(boundaries)
        for i in range(len(sorted_bounds) - 1):
            s = sorted_bounds[i]
            e = sorted_bounds[i + 1] - 1
            if e < s:
                continue
            seg_frames = [f for f in range(s, e + 1) if f in obj_frames]
            if not seg_frames:
                continue
            ss, ee = min(seg_frames), max(seg_frames)
            if ee - ss + 1 >= min_interval_len:
                intervals.add((ss, ee))

        sampled_intervals = _downsample_intervals(intervals, max_intervals_per_object)
        object_intervals_by_obj[object_id] = sampled_intervals
        for s, e in sampled_intervals:
            candidates.append(
                CandidateTarget(
                    candidate_id=f"{object_id}@{s}-{e}",
                    members=[CandidateMember(object_id=object_id, start=s, end=e)],
                    meta={
                        "object_id": object_id,
                        "target_class": _sample_class(obj, object_id),
                        "is_full_track_interval": int(s == full_s and e == full_e),
                    },
                )
            )

    max_target_arity = min(3, max(1, int(max_target_arity)))
    if max_target_arity <= 1:
        return candidates

    object_ids = [oid for oid in _ranked_multi_object_ids(index, max_objects=18) if object_intervals_by_obj.get(oid)]
    multi_count = 0
    for arity in range(2, max_target_arity + 1):
        for combo in combinations(object_ids, arity):
            interval_choices = _multi_interval_choices_for_combo(
                combo=combo,
                object_intervals=object_intervals_by_obj,
                limit=max_multi_intervals_per_group,
            )
            if not interval_choices:
                continue

            for interval_tuple in interval_choices:
                members = [
                    CandidateMember(object_id=oid, start=s, end=e)
                    for oid, (s, e) in zip(combo, interval_tuple)
                ]
                member_parts = [f"{oid}@{s}-{e}" for oid, (s, e) in zip(combo, interval_tuple)]
                candidates.append(
                    CandidateTarget(
                        candidate_id="|".join(member_parts),
                        members=members,
                        meta={
                            "object_ids": list(combo),
                            "arity": arity,
                            "is_shared_interval": int(len({(m.start, m.end) for m in members}) == 1),
                        },
                    )
                )
                multi_count += 1
                if multi_count >= max_multi_candidates_total:
                    return candidates

    return candidates
