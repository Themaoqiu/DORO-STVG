import asyncio
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from api_sync.utils.parser import JSONParser


SYSTEM_PROMPT = (
    "You are an expert in spatiotemporal video grounding query writing. "
    "Given a structured query spec, generate one natural-language English STVG query. "
    "Return strict JSON only: {\"query\": \"...\"}. "
    "Do not add fields, markdown, or explanations."
)

PROMPT_TEMPLATE = (
    "Generate one concise STVG query from this spec.\n"
    "Requirements:\n"
    "1) Use only the provided clues.\n"
    "2) Keep it natural and unambiguous.\n"
    "3) Include temporal and relational clues when provided.\n"
    "4) Do not mention exact frame numbers.\n"
    "5) Output strict JSON only: {\"query\": \"...\"}.\n\n"
    "Target difficulty d_star: {d_star}\n"
    "Spec JSON:\n{spec_json}"
)

ASYM_INVERSE = {
    "in front of": "behind",
    "behind": "in front of",
    "above": "below",
    "below": "above",
    "left of": "right of",
    "right of": "left of",
    "is in front of": "behind",
    "is behind": "in front of",
}
SYMMETRIC = {"near", "next to", "beside", "talk_to", "talking to", "walk with"}
BUCKET_CENTER = {"easy": 0.18, "medium": 0.45, "hard": 0.70, "very_hard": 0.92}

QUERY_SPEC_KEYS = [
    "target_class",
    "attributes",
    "context_relations",
    "actions",
    "action_targets",
    "action_sequences",
    "relations",
    "time_range",
    "time_segments",
    "full_track_range",
    "cross_shot",
    "same_entity_nodes",
]

QUERY_NODE_KEYS = [
    "query_id",
    "target_node_id",
    "query",
    "query_spec",
    "D_t",
    "D_s",
    "D",
    "d_star",
    "clue_types",
    "type_bucket",
    "difficulty_bucket",
]

GRAPH_OUTPUT_KEYS = [
    "video",
    "video_path",
    "temporal_nodes",
    "object_nodes",
    "attribute_nodes",
    "action_nodes",
    "edges",
    "query_nodes",
]


@dataclass
class EntityProfile:
    canonical_id: str
    node_ids: List[str]
    canonical_class: str
    frames: Set[int]
    shot_ids: Set[int]
    start_frame: int
    end_frame: int
    attributes: List[str] = field(default_factory=list)
    context_relations: List[str] = field(default_factory=list)
    dam_actions: List[str] = field(default_factory=list)


@dataclass
class ActionEvent:
    owner_id: str
    label: str
    start_frame: int
    end_frame: int
    shot_ids: Set[int]
    target_ids: Set[str]


@dataclass
class RelationEvent:
    subj_id: str
    obj_id: str
    predicate: str
    edge_type: str
    relation_type: str
    segments: List[Tuple[int, int]]
    shot_ids: Set[int]



def _to_int(v: Any, d: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return d


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s)).strip(" \n\t,.;:")


def _uniq_phrases(v: Any) -> List[str]:
    if isinstance(v, list):
        raw = [str(x) for x in v]
    elif isinstance(v, str):
        raw = [x.strip() for x in v.split(",")]
    else:
        raw = []
    out, seen = [], set()
    for x in raw:
        t = _norm(x)
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def _segments_to_frames(segs: List[Tuple[int, int]]) -> Set[int]:
    out: Set[int] = set()
    for s, e in segs:
        if e < s:
            s, e = e, s
        out.update(range(s, e + 1))
    return out


def _frames_to_segments(frames: Set[int]) -> List[Tuple[int, int]]:
    if not frames:
        return []
    arr = sorted(frames)
    res: List[Tuple[int, int]] = []
    a = b = arr[0]
    for x in arr[1:]:
        if x == b + 1:
            b = x
            continue
        res.append((a, b))
        a = b = x
    res.append((a, b))
    return res


def _normalize_api_keys(api_keys: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(api_keys, str):
        return [k.strip() for k in api_keys.split(",") if k.strip()]
    return [str(k).strip() for k in api_keys if str(k).strip()]


def _load_api_keys_from_project_env() -> str:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return ""
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "API_KEYS":
                return v.strip().strip('"').strip("'")
    return ""


def _resolve_api_keys(api_keys: Optional[Union[str, Iterable[str]]]) -> List[str]:
    if api_keys is None:
        api_keys = os.getenv("API_KEYS", "")
    if not api_keys:
        api_keys = _load_api_keys_from_project_env()
    keys = _normalize_api_keys(api_keys)
    if not keys:
        raise ValueError("api_keys is required (pass --api_keys or set API_KEYS in env)")
    return keys


def _extract_query(response: str) -> Union[str, bool]:
    parsed = JSONParser.parse(response)
    if not isinstance(parsed, dict):
        return False
    q = parsed.get("query")
    if not isinstance(q, str):
        return False
    q = q.strip()
    return q if q else False


def _class_from_node_id(node_id: str, fallback: str = "object") -> str:
    nid = str(node_id).strip().lower()
    if not nid:
        return fallback
    prefix = nid.split("_", 1)[0].strip()
    return prefix or fallback


def _build_object_index_maps(object_nodes: List[Dict[str, Any]]) -> Dict[int, str]:
    idx_to_node: Dict[int, str] = {}
    for idx, obj in enumerate(object_nodes):
        node_id = str(obj.get("node_id", "")).strip()
        if node_id:
            idx_to_node[idx] = node_id
    return idx_to_node


def _build_entity_profiles(object_nodes: List[Dict[str, Any]]) -> Dict[str, EntityProfile]:
    profiles: Dict[str, EntityProfile] = {}

    for obj in object_nodes:
        node_id = str(obj.get("node_id", "")).strip()
        if not node_id:
            continue
        members = [obj]
        cls = _class_from_node_id(node_id, fallback=_norm(obj.get("object_class", "object")).lower() or "object")
        frames: Set[int] = set()
        shots: Set[int] = set()
        attrs: List[str] = []
        ctx: List[str] = []
        acts: List[str] = []
        attr_seen, ctx_seen, act_seen = set(), set(), set()
        start, end = 10**9, -1

        for m in members:
            bboxes = m.get("bboxes", {})
            if isinstance(bboxes, dict):
                for k in bboxes:
                    frames.add(_to_int(k, -1))
            shots |= {_to_int(x, -1) for x in m.get("shot_ids", []) if _to_int(x, -1) >= 0}
            s, e = _to_int(m.get("start_frame"), 0), _to_int(m.get("end_frame"), 0)
            if e < s:
                s, e = e, s
            start, end = min(start, s), max(end, e)

            for x in _uniq_phrases(m.get("attributes", [])):
                if x.lower() not in attr_seen:
                    attr_seen.add(x.lower())
                    attrs.append(x)
            for x in _uniq_phrases(m.get("relationships", [])):
                if x.lower() not in ctx_seen:
                    ctx_seen.add(x.lower())
                    ctx.append(x)
            for x in _uniq_phrases(m.get("actions", [])):
                if x.lower() not in act_seen:
                    act_seen.add(x.lower())
                    acts.append(x)

        profiles[node_id] = EntityProfile(
            canonical_id=node_id,
            node_ids=[node_id],
            canonical_class=cls,
            frames=frames,
            shot_ids=shots,
            start_frame=start,
            end_frame=end,
            attributes=attrs,
            context_relations=ctx,
            dam_actions=acts,
        )

    return profiles


def _build_ref_description(
    ref_entity: EntityProfile,
    active_frames: Set[int],
    entities: Dict[str, EntityProfile],
    max_attrs: int = 2,
    max_context: int = 1,
) -> Dict[str, Any]:
    peers = [
        e
        for e in entities.values()
        if e.canonical_id != ref_entity.canonical_id
        and e.canonical_class == ref_entity.canonical_class
        and bool(e.frames & active_frames)
    ]

    def score(phrase: str, src: str) -> Tuple[int, int]:
        p = phrase.lower()
        cnt = 0
        for x in peers:
            pool = x.attributes if src == "attr" else x.context_relations
            if any(p == y.lower() for y in pool):
                cnt += 1
        return cnt, len(phrase)

    attrs = sorted(ref_entity.attributes, key=lambda x: score(x, "attr"))[:max_attrs]
    ctx = sorted(ref_entity.context_relations, key=lambda x: score(x, "ctx"))[:max_context]
    return {"class": ref_entity.canonical_class, "attributes": attrs, "context_relations": ctx}


def _flatten_action_events(
    action_nodes: List[Dict[str, Any]],
    node_to_shots: Dict[str, Set[int]],
) -> List[ActionEvent]:
    raw: List[ActionEvent] = []
    for group in action_nodes:
        owner_node = str(group.get("object_id", "")).strip()
        if not owner_node:
            continue
        for item in group.get("actions", []):
            if not isinstance(item, dict):
                continue
            label = _norm(item.get("action_label", ""))
            if not label:
                continue
            s, e = _to_int(item.get("start_frame"), 0), _to_int(item.get("end_frame"), 0)
            if e < s:
                s, e = e, s
            targets: Set[str] = set()
            for t in item.get("target_object_ids", []) or []:
                if isinstance(t, str) and t.strip():
                    targets.add(t.strip())
            raw.append(
                ActionEvent(
                    owner_id=owner_node,
                    label=label,
                    start_frame=s,
                    end_frame=e,
                    shot_ids=set(node_to_shots.get(owner_node, set())),
                    target_ids=targets,
                )
            )

    grouped: Dict[Tuple[str, str], List[ActionEvent]] = defaultdict(list)
    for r in raw:
        grouped[(r.owner_id, r.label)].append(r)

    merged: List[ActionEvent] = []
    for events in grouped.values():
        events.sort(key=lambda x: (x.start_frame, x.end_frame))
        cur = events[0]
        for nxt in events[1:]:
            if nxt.start_frame <= cur.end_frame or nxt.start_frame - cur.end_frame <= 2:
                cur.start_frame = min(cur.start_frame, nxt.start_frame)
                cur.end_frame = max(cur.end_frame, nxt.end_frame)
                cur.shot_ids |= nxt.shot_ids
                cur.target_ids |= nxt.target_ids
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
    return merged


def _flatten_relation_events(
    edges: List[Dict[str, Any]],
    idx_to_node: Dict[int, str],
    node_to_shots: Dict[str, Set[int]],
) -> List[RelationEvent]:
    out: List[RelationEvent] = []
    seen: Set[str] = set()

    def add(subj: str, obj: str, pred: str, et: str, rt: str, segs: List[Tuple[int, int]], shots: Set[int]) -> None:
        if not segs:
            return
        segs = [(min(a, b), max(a, b)) for a, b in segs]
        segs.sort()
        key = json.dumps({"s": subj, "o": obj, "p": pred, "et": et, "rt": rt, "sg": segs}, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        out.append(RelationEvent(subj, obj, pred, et, rt, segs, set(shots)))

    for bundle in edges:
        if "subject_id" not in bundle or "relationships" not in bundle:
            continue
        s_idx = bundle.get("subject_id")
        if not isinstance(s_idx, int):
            continue
        s_node = idx_to_node.get(s_idx)
        if not s_node:
            continue

        for rel in bundle.get("relationships", []):
            if not isinstance(rel, dict):
                continue
            o_idx = rel.get("object_id")
            if not isinstance(o_idx, int):
                continue
            o_node = idx_to_node.get(o_idx)
            if not o_node:
                continue
            pred = _norm(rel.get("predicate_verb", "")).lower()
            if not pred:
                continue
            et = _norm(rel.get("edge_type", ""))
            rt = _norm(rel.get("relationship_type", ""))
            segs = []
            for seg in rel.get("time_frames", []) or []:
                if isinstance(seg, list) and len(seg) == 2:
                    a, b = _to_int(seg[0], 0), _to_int(seg[1], 0)
                    if b < a:
                        a, b = b, a
                    segs.append((a, b))
            if not segs:
                segs = [(0, 0)]
            shots = set(node_to_shots.get(s_node or "", set())) | set(node_to_shots.get(o_node or "", set()))
            add(s_node, o_node, pred, et, rt, segs, shots)
            if pred in ASYM_INVERSE:
                add(o_node, s_node, ASYM_INVERSE[pred], et, rt, segs, shots)
            elif pred in SYMMETRIC:
                add(o_node, s_node, pred, et, rt, segs, shots)

    return out


def _active_distractors(target: EntityProfile, entities: Dict[str, EntityProfile], support: Set[int]) -> List[EntityProfile]:
    return [
        e
        for e in entities.values()
        if e.canonical_id != target.canonical_id
        and e.canonical_class == target.canonical_class
        and bool(e.frames & support)
    ]


def _temporal_level(clue_types: List[str], cross_shot: bool) -> int:
    if cross_shot:
        return 4
    if "action_sequence" in clue_types:
        return 3
    if "action_relation_overlap" in clue_types or "action+context" in clue_types:
        return 2
    if any(x in clue_types for x in ["action", "action_to_unique", "action_to_described", "relation_to_unique", "relation_to_described"]):
        return 1
    return 0


def _spatial_base(clue_types: List[str], cross_shot: bool) -> float:
    if cross_shot:
        return 0.85
    if "action_relation_overlap" in clue_types:
        return 0.85
    if "action_to_described" in clue_types or "relation_to_described" in clue_types:
        return 0.75
    if "action_to_unique" in clue_types or "relation_to_unique" in clue_types:
        return 0.55
    if "context_relation" in clue_types:
        return 0.30
    if "attribute" in clue_types:
        return 0.20
    return 0.25


def _rule_bucket(clue_types: List[str], support_len: int, track_len: int, active_dist: int, cross_shot: bool) -> str:
    long_action = support_len >= 0.4 * max(track_len, 1)
    has_attr = "attribute" in clue_types
    has_ctx = "context_relation" in clue_types
    has_action = any(x in clue_types for x in ["action", "action_to_unique", "action_to_described"])
    has_relation = any(x in clue_types for x in ["relation_to_unique", "relation_to_described"])
    has_described = any(x in clue_types for x in ["action_to_described", "relation_to_described"])

    if cross_shot or "action_sequence" in clue_types:
        return "very_hard"
    if len(clue_types) >= 3 and has_action and (has_relation or has_ctx):
        return "very_hard"
    if "action_to_described" in clue_types and "relation_to_described" in clue_types:
        return "very_hard"

    if "action_relation_overlap" in clue_types or "action+context" in clue_types:
        return "hard"
    if "action_to_described" in clue_types:
        return "hard"
    if active_dist >= 3 and len(clue_types) == 1:
        return "hard"

    if (has_attr and has_action) or (has_attr and has_ctx) or (has_ctx and has_action):
        return "medium"
    if "action_to_unique" in clue_types or "relation_to_described" in clue_types:
        return "medium"
    if "action" in clue_types and not long_action:
        return "medium"

    if len(clue_types) == 1 and "attribute" in clue_types:
        return "easy"
    if len(clue_types) == 1 and "context_relation" in clue_types:
        return "easy"
    if len(clue_types) == 1 and "action" in clue_types and long_action:
        return "easy"
    if len(clue_types) == 1 and "relation_to_unique" in clue_types:
        return "easy"
    return "medium"


def _type_bucket(clue_types: List[str], cross_shot: bool) -> str:
    if cross_shot:
        return "cross_shot"
    if "action_sequence" in clue_types:
        return "action_sequence"
    if "action_to_unique" in clue_types or "action_to_described" in clue_types:
        return "action_target"
    if "relation_to_unique" in clue_types or "relation_to_described" in clue_types:
        return "relation"
    if "action" in clue_types or "action_relation_overlap" in clue_types:
        return "action"
    if "context_relation" in clue_types:
        return "context_relation"
    return "attribute"


def _ordered_query_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in QUERY_SPEC_KEYS:
        out[k] = spec.get(k)
    for k, v in spec.items():
        if k not in out:
            out[k] = v
    return out


def _ordered_query_node(node: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in QUERY_NODE_KEYS:
        out[k] = node.get(k)
    for k, v in node.items():
        if k not in out:
            out[k] = v
    return out


def _ordered_graph_output(graph: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in GRAPH_OUTPUT_KEYS:
        if k not in graph:
            continue
        if k == "query_nodes" and isinstance(graph.get(k), list):
            out[k] = [_ordered_query_node(x) if isinstance(x, dict) else x for x in graph[k]]
        else:
            out[k] = graph[k]
    for k, v in graph.items():
        if k not in out:
            out[k] = v
    return out


class DifficultyAwareSTVGQueryGenerator:
    def __init__(self) -> None:
        pass

    def _normalize_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        objects = graph.get("object_nodes", [])
        actions = graph.get("action_nodes", [])
        edges = graph.get("edges", [])
        idx_to_node = _build_object_index_maps(objects)
        entities = _build_entity_profiles(objects)

        node_to_shots: Dict[str, Set[int]] = {}
        for obj in objects:
            nid = str(obj.get("node_id", "")).strip()
            if nid:
                node_to_shots[nid] = {_to_int(x, -1) for x in obj.get("shot_ids", []) if _to_int(x, -1) >= 0}

        action_events = _flatten_action_events(actions, node_to_shots)
        relation_events = _flatten_relation_events(edges, idx_to_node, node_to_shots)

        return {
            "objects": objects,
            "entities": entities,
            "action_events": action_events,
            "relation_events": relation_events,
        }

    def build_query_plan(
        self,
        graph: Dict[str, Any],
        target_obj: Dict[str, Any],
        d_star: float,
        return_all_candidates: bool = False,
        max_candidates: int = 20,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        n = self._normalize_graph(graph)
        entities: Dict[str, EntityProfile] = n["entities"]
        action_events: List[ActionEvent] = n["action_events"]
        relation_events: List[RelationEvent] = n["relation_events"]

        target_node = str(target_obj.get("node_id", "")).strip()
        target = entities.get(target_node)
        if not target:
            fallback = self._fallback_plan(target_obj, d_star)
            return [fallback] if return_all_candidates else fallback

        t_actions = [x for x in action_events if x.owner_id == target.canonical_id]
        t_relations = [x for x in relation_events if x.subj_id == target.canonical_id]

        candidates: List[Dict[str, Any]] = []
        seen_sig: Set[str] = set()

        def add(clue_types: List[str], clue_data: Dict[str, Any], support: Set[int], cross_shot: bool = False) -> None:
            s = set(support) or set(target.frames) or set(range(target.start_frame, target.end_frame + 1))
            segs = _frames_to_segments(s)
            t0 = min(s) if s else target.start_frame
            t1 = max(s) if s else target.end_frame

            # 找同时间段同类别的干扰物数量
            active = _active_distractors(target, entities, s)
            active_n = len(active)

            # 
            d_t = _temporal_level(clue_types, cross_shot) / 4.0
            d_s = 0.6 * _spatial_base(clue_types, cross_shot) + 0.4 * min(active_n / 3.0, 1.0)
            d = 0.55 * d_t + 0.45 * d_s

            bucket = _rule_bucket(clue_types, len(s), target.end_frame - target.start_frame + 1, active_n, cross_shot)
            type_b = _type_bucket(clue_types, cross_shot)

            spec = {
                "target_class": target.canonical_class,
                "attributes": clue_data.get("attributes", []),
                "context_relations": clue_data.get("context_relations", []),
                "actions": clue_data.get("actions", []),
                "action_targets": clue_data.get("action_targets", []),
                "action_sequences": clue_data.get("action_sequences", []),
                "relations": clue_data.get("relations", []),
                "time_range": {"start": t0, "end": t1},
                "time_segments": [[a, b] for a, b in segs] if segs else [[target.start_frame, target.end_frame]],
                "full_track_range": {"start": target.start_frame, "end": target.end_frame},
                "cross_shot": bool(cross_shot),
                "same_entity_nodes": sorted(target.node_ids),
            }
            spec = _ordered_query_spec(spec)

            sig_payload = {
                "target": target.canonical_id,
                "clue_types": sorted(set(clue_types)),
                "clue_data": clue_data,
                "segs": spec["time_segments"],
                "cross_shot": bool(cross_shot),
            }
            sig = json.dumps(sig_payload, ensure_ascii=False, sort_keys=True)
            if sig in seen_sig:
                return
            seen_sig.add(sig)

            candidates.append(
                {
                    "query_spec": spec,
                    "D_t": d_t,
                    "D_s": d_s,
                    "D": d,
                    "d_star": float(d_star),
                    "target_node": target_node,
                    "segment_len": max(len(s), 1),
                    "clue_types": sorted(set(clue_types)),
                    "type_bucket": type_b,
                    "difficulty_bucket": bucket,
                    "candidate_signature": sig,
                }
            )

        # attribute / context_relation
        for a in target.attributes:
            add(["attribute"], {"attributes": [a]}, set(target.frames))
        for r in target.context_relations:
            add(["context_relation"], {"context_relations": [r]}, set(target.frames))

        # action / action_target
        for ev in t_actions:
            supp = set(range(ev.start_frame, ev.end_frame + 1)) & target.frames
            add(["action"], {"actions": [ev.label]}, supp)
            for ref_id in sorted(ev.target_ids):
                ref = entities.get(ref_id)
                if not ref:
                    continue
                same_cls = [e for e in entities.values() if e.canonical_class == ref.canonical_class and bool(e.frames & supp)]
                if len(same_cls) == 1:
                    add(
                        ["action_to_unique"],
                        {"action_targets": [{"action": ev.label, "ref": {"class": ref.canonical_class, "attributes": [], "context_relations": []}}]},
                        supp,
                    )
                else:
                    add(["action_to_described"], {"action_targets": [{"action": ev.label, "ref": _build_ref_description(ref, supp, entities)}]}, supp)

        # relation
        for rel in t_relations:
            supp = _segments_to_frames(rel.segments) & target.frames
            ref = entities.get(rel.obj_id)
            if not ref:
                continue
            same_cls = [e for e in entities.values() if e.canonical_class == ref.canonical_class and bool(e.frames & supp)]
            if len(same_cls) == 1:
                add(
                    ["relation_to_unique"],
                    {"relations": [{"relation": rel.predicate, "ref": {"class": ref.canonical_class, "attributes": [], "context_relations": []}}]},
                    supp,
                )
            else:
                add(["relation_to_described"], {"relations": [{"relation": rel.predicate, "ref": _build_ref_description(ref, supp, entities)}]}, supp)

        # action_sequence / cross_shot_sequence
        ta = sorted(t_actions, key=lambda x: (x.start_frame, x.end_frame, x.label))
        track_len = max(target.end_frame - target.start_frame + 1, 1)
        for i in range(len(ta)):
            for j in range(i + 1, len(ta)):
                a, b = ta[i], ta[j]
                if a.label == b.label or a.end_frame >= b.start_frame:
                    continue
                if (a.end_frame - a.start_frame + 1) < 2 or (b.end_frame - b.start_frame + 1) < 2:
                    continue
                if b.end_frame - a.start_frame + 1 > 0.8 * track_len:
                    continue
                cross = len(a.shot_ids | b.shot_ids) >= 2
                add(
                    ["action_sequence"],
                    {"action_sequences": [{"first": a.label, "second": b.label, "order": "before", "cross_shot": cross}]},
                    set(range(a.start_frame, b.end_frame + 1)) & target.frames,
                    cross_shot=cross,
                )

        # compact combos
        one_attr = next((c for c in candidates if c["clue_types"] == ["attribute"]), None)
        one_ctx = next((c for c in candidates if c["clue_types"] == ["context_relation"]), None)
        one_act = next((c for c in candidates if c["clue_types"] == ["action"]), None)
        one_rel = next((c for c in candidates if c["clue_types"] in (["relation_to_unique"], ["relation_to_described"])), None)

        def overlap(c1: Dict[str, Any], c2: Dict[str, Any]) -> Set[int]:
            a, b = c1["query_spec"]["time_range"], c2["query_spec"]["time_range"]
            s, e = max(a["start"], b["start"]), min(a["end"], b["end"])
            return set(range(s, e + 1)) if s <= e else set(target.frames)

        if one_attr and one_act:
            add(["attribute", "action"], {"attributes": one_attr["query_spec"]["attributes"], "actions": one_act["query_spec"]["actions"]}, overlap(one_attr, one_act))
        if one_ctx and one_act:
            add(["action+context", "action", "context_relation"], {"context_relations": one_ctx["query_spec"]["context_relations"], "actions": one_act["query_spec"]["actions"]}, overlap(one_ctx, one_act))
        if one_act and one_rel:
            add(["action_relation_overlap", "action"] + one_rel["clue_types"], {"actions": one_act["query_spec"]["actions"], "relations": one_rel["query_spec"]["relations"]}, overlap(one_act, one_rel))

        if not candidates:
            fallback = self._fallback_plan(target_obj, d_star)
            return [fallback] if return_all_candidates else fallback

        # prefer hard-unique; fallback to best exclusions+support naturally by sorting
        hard_unique = [
            c
            for c in candidates
            if len(
                _active_distractors(
                    target,
                    entities,
                    set(range(c["query_spec"]["time_range"]["start"], c["query_spec"]["time_range"]["end"] + 1)),
                )
            )
            == 0
        ]
        pool = hard_unique or candidates
        pool.sort(
            key=lambda x: (
                abs(BUCKET_CENTER.get(str(x.get("difficulty_bucket", "medium")), 0.5) - float(d_star)),
                abs(float(x.get("D", 0.0)) - float(d_star)),
                -int(x.get("segment_len", 1)),
                len(x.get("clue_types", [])),
            )
        )
        plans = pool[: max(1, max_candidates)]
        return plans if return_all_candidates else plans[0]

    def _fallback_plan(self, target_obj: Dict[str, Any], d_star: float) -> Dict[str, Any]:
        nid = str(target_obj.get("node_id", ""))
        cls = _norm(target_obj.get("object_class", "object")).lower() or "object"
        s, e = _to_int(target_obj.get("start_frame"), 0), _to_int(target_obj.get("end_frame"), 0)
        if e < s:
            s, e = e, s
        spec = {
            "target_class": cls,
            "attributes": [],
            "context_relations": [],
            "actions": [],
            "action_targets": [],
            "action_sequences": [],
            "relations": [],
            "time_range": {"start": s, "end": e},
            "time_segments": [[s, e]],
            "full_track_range": {"start": s, "end": e},
            "cross_shot": False,
            "same_entity_nodes": [nid] if nid else [],
        }
        spec = _ordered_query_spec(spec)
        return {
            "query_spec": spec,
            "D_t": 0.0,
            "D_s": 0.0,
            "D": 0.0,
            "d_star": float(d_star),
            "target_node": nid,
            "segment_len": max(e - s + 1, 1),
            "clue_types": ["fallback"],
            "type_bucket": "attribute",
            "difficulty_bucket": "easy",
            "candidate_signature": f"fallback::{nid}::{s}::{e}",
        }

    def build_graph_plans(
        self,
        graph: Dict[str, Any],
        d_star: float,
        sample_size: Optional[int] = None,
        seed: int = 42,
        target_node_ids: Optional[Union[str, Iterable[str]]] = None,
        d_star_list: Optional[Union[str, Iterable[float]]] = None,
        per_target_limit: int = 1,
        generate_all_candidates: bool = False,
        max_candidates_per_target: int = 20,
        queries_per_graph: Optional[int] = None,
        balance_difficulty: bool = True,
        balance_types: bool = True,
        max_queries_per_target: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        objs = [o for o in graph.get("object_nodes", []) if o.get("node_id") and o.get("bboxes")]
        if not objs:
            return []

        if target_node_ids:
            target_set = (
                {x.strip() for x in target_node_ids.split(",") if x.strip()}
                if isinstance(target_node_ids, str)
                else {str(x).strip() for x in target_node_ids if str(x).strip()}
            )
            objs = [o for o in objs if o.get("node_id") in target_set]

        if sample_size and sample_size > 0 and len(objs) > sample_size:
            objs = random.Random(seed).sample(objs, sample_size)

        d_vals: List[float] = []
        if d_star_list:
            raw = [x.strip() for x in d_star_list.split(",")] if isinstance(d_star_list, str) else [str(x).strip() for x in d_star_list]
            for v in raw:
                try:
                    d_vals.append(min(max(float(v), 0.0), 1.0))
                except Exception:
                    pass
        if not d_vals:
            d_vals = [min(max(float(d_star), 0.0), 1.0)]

        all_plans: List[Dict[str, Any]] = []
        global_sigs: Set[str] = set()

        for obj in objs:
            t_candidates = self.build_query_plan(
                graph=graph,
                target_obj=obj,
                d_star=d_vals[0],
                return_all_candidates=True,
                max_candidates=max_candidates_per_target,
            )
            if not isinstance(t_candidates, list) or not t_candidates:
                continue

            chosen: List[Dict[str, Any]]
            if generate_all_candidates:
                chosen = t_candidates
            elif len(d_vals) > 1:
                chosen, local_seen = [], set()
                for dv in d_vals:
                    best = min(
                        t_candidates,
                        key=lambda x: (
                            abs(BUCKET_CENTER.get(str(x.get("difficulty_bucket", "medium")), 0.5) - dv),
                            abs(float(x.get("D", 0.0)) - dv),
                            -int(x.get("segment_len", 1)),
                            len(x.get("clue_types", [])),
                        ),
                    )
                    sig = str(best.get("candidate_signature", ""))
                    if sig and sig in local_seen:
                        continue
                    local_seen.add(sig)
                    chosen.append({**best, "d_star": dv})
                if per_target_limit > 0:
                    chosen = chosen[:per_target_limit]
            else:
                chosen = t_candidates[: max(1, per_target_limit)]

            for p in chosen:
                sig = str(p.get("candidate_signature", ""))
                if sig and sig in global_sigs:
                    continue
                if sig:
                    global_sigs.add(sig)
                item = dict(p)
                item["plan_id"] = f"plan_{item.get('target_node')}_{len(all_plans)}_{int(float(item.get('D', 0.0)) * 1000)}"
                all_plans.append(item)

        if not queries_per_graph or queries_per_graph <= 0 or len(all_plans) <= queries_per_graph:
            return all_plans

        rng = random.Random(seed)
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for p in all_plans:
            diff_key = str(p.get("difficulty_bucket", "medium")) if balance_difficulty else "all"
            type_key = str(p.get("type_bucket", "attribute")) if balance_types else "all"
            groups[(diff_key, type_key)].append(p)

        for k in list(groups.keys()):
            groups[k].sort(
                key=lambda x: (
                    abs(BUCKET_CENTER.get(str(x.get("difficulty_bucket", "medium")), 0.5) - float(d_star)),
                    abs(float(x.get("D", 0.0)) - float(d_star)),
                    -int(x.get("segment_len", 1)),
                )
            )

        selected: List[Dict[str, Any]] = []
        target_cnt: Dict[str, int] = defaultdict(int)
        keys = list(groups.keys())

        while len(selected) < queries_per_graph:
            rng.shuffle(keys)
            moved = False
            for k in keys:
                bucket = groups[k]
                while bucket:
                    c = bucket.pop(0)
                    t = str(c.get("target_node", ""))
                    if max_queries_per_target and max_queries_per_target > 0 and target_cnt[t] >= max_queries_per_target:
                        continue
                    selected.append(c)
                    target_cnt[t] += 1
                    moved = True
                    break
                if len(selected) >= queries_per_graph:
                    break
            if not moved:
                break

        if len(selected) < queries_per_graph:
            picked = {str(x.get("candidate_signature", "")) for x in selected}
            left = [x for x in all_plans if str(x.get("candidate_signature", "")) not in picked]
            left.sort(
                key=lambda x: (
                    abs(BUCKET_CENTER.get(str(x.get("difficulty_bucket", "medium")), 0.5) - float(d_star)),
                    abs(float(x.get("D", 0.0)) - float(d_star)),
                    -int(x.get("segment_len", 1)),
                )
            )
            for x in left:
                selected.append(x)
                if len(selected) >= queries_per_graph:
                    break

        return selected[:queries_per_graph]

    async def generate_queries_with_llm(
        self,
        plans: List[Dict[str, Any]],
        model_name: str,
        api_keys: Optional[Union[str, Iterable[str]]],
        max_concurrent_per_key: int = 100,
        max_retries: int = 5,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> Dict[str, str]:
        if not plans:
            return {}

        from api_sync.api import StreamGenerator

        generator = StreamGenerator(
            model_name=model_name,
            api_keys=_resolve_api_keys(api_keys),
            max_concurrent_per_key=max_concurrent_per_key,
            max_retries=max_retries,
            rational=False,
            with_unique_id=True,
        )

        prompts = [
            {
                "id": str(p.get("plan_id") or p.get("target_node")),
                "prompt": PROMPT_TEMPLATE.format(d_star=p.get("d_star"), spec_json=json.dumps(p.get("query_spec", {}), ensure_ascii=False)),
            }
            for p in plans
        ]

        query_map: Dict[str, str] = {}
        async for item in generator.generate_stream(prompts=prompts, system_prompt=system_prompt, validate_func=_extract_query):
            if item and isinstance(item.get("result"), str) and item["result"].strip():
                query_map[str(item.get("id"))] = item["result"].strip()
        return query_map

    def _fallback_query(self, query_spec: Dict[str, Any]) -> str:
        chunks = [f"the {query_spec.get('target_class', 'object')}" ]
        if query_spec.get("attributes"):
            chunks.append("with " + ", ".join(query_spec["attributes"]))
        if query_spec.get("context_relations"):
            chunks.append("that is " + " and ".join(query_spec["context_relations"]))
        if query_spec.get("actions"):
            chunks.append("that is " + " and ".join(query_spec["actions"]))
        for x in query_spec.get("action_targets", []):
            if isinstance(x, dict):
                a = x.get("action")
                ref = x.get("ref", {})
                cls = ref.get("class") if isinstance(ref, dict) else None
                if a and cls:
                    chunks.append(f"that {a} the {cls}")
        for x in query_spec.get("action_sequences", []):
            if isinstance(x, dict) and x.get("first") and x.get("second"):
                chunks.append(f"that {x['first']} before {x['second']}")
        for x in query_spec.get("relations", []):
            if isinstance(x, dict):
                r = x.get("relation")
                ref = x.get("ref", {})
                cls = ref.get("class") if isinstance(ref, dict) else None
                if r and cls:
                    chunks.append(f"that is {r} the {cls}")
        return " ".join(chunks)

    def process_graph(
        self,
        graph: Dict[str, Any],
        d_star: float,
        model_name: Optional[str] = None,
        api_keys: Optional[Union[str, Iterable[str]]] = None,
        sample_size: Optional[int] = None,
        seed: int = 42,
        target_node_ids: Optional[Union[str, Iterable[str]]] = None,
        d_star_list: Optional[Union[str, Iterable[float]]] = None,
        per_target_limit: int = 1,
        generate_all_candidates: bool = False,
        max_candidates_per_target: int = 20,
        queries_per_graph: Optional[int] = None,
        balance_difficulty: bool = True,
        balance_types: bool = True,
        max_queries_per_target: Optional[int] = None,
        use_llm: bool = True,
        max_concurrent_per_key: int = 100,
        max_retries: int = 5,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        plans = self.build_graph_plans(
            graph=graph,
            d_star=d_star,
            sample_size=sample_size,
            seed=seed,
            target_node_ids=target_node_ids,
            d_star_list=d_star_list,
            per_target_limit=per_target_limit,
            generate_all_candidates=generate_all_candidates,
            max_candidates_per_target=max_candidates_per_target,
            queries_per_graph=queries_per_graph,
            balance_difficulty=balance_difficulty,
            balance_types=balance_types,
            max_queries_per_target=max_queries_per_target,
        )

        query_map: Dict[str, str] = {}
        if use_llm and plans:
            if not model_name:
                raise ValueError("model_name is required when use_llm=True")
            query_map = asyncio.run(
                self.generate_queries_with_llm(
                    plans=plans,
                    model_name=model_name,
                    api_keys=api_keys,
                    max_concurrent_per_key=max_concurrent_per_key,
                    max_retries=max_retries,
                )
            )

        query_nodes = []
        obj_map = {obj.get("node_id"): obj for obj in graph.get("object_nodes", [])}

        for i, plan in enumerate(plans):
            pid = str(plan.get("plan_id") or f"plan_{i}")
            target = str(plan.get("target_node"))
            query = query_map.get(pid) or self._fallback_query(plan.get("query_spec", {}))
            item = {
                "query_id": f"query_{target}_{i}_{pid}",
                "target_node_id": target,
                "query": query,
                "query_spec": plan.get("query_spec"),
                "D_t": plan.get("D_t"),
                "D_s": plan.get("D_s"),
                "D": plan.get("D"),
                "d_star": plan.get("d_star"),
                "clue_types": plan.get("clue_types", []),
                "type_bucket": plan.get("type_bucket"),
                "difficulty_bucket": plan.get("difficulty_bucket"),
            }
            item["query_spec"] = _ordered_query_spec(item.get("query_spec") or {})
            query_nodes.append(_ordered_query_node(item))

            if target in obj_map:
                obj = obj_map[target]
                obj.setdefault("queries", []).append(query)
                obj.setdefault("query_difficulties", []).append(
                    {
                        "D_t": plan.get("D_t"),
                        "D_s": plan.get("D_s"),
                        "D": plan.get("D"),
                        "difficulty_bucket": plan.get("difficulty_bucket"),
                    }
                )
                if "query" not in obj:
                    obj["query"] = query
                    obj["query_difficulty"] = {
                        "D_t": plan.get("D_t"),
                        "D_s": plan.get("D_s"),
                        "D": plan.get("D"),
                        "difficulty_bucket": plan.get("difficulty_bucket"),
                    }

        if overwrite:
            graph["query_nodes"] = query_nodes
        else:
            graph.setdefault("query_nodes", []).extend(query_nodes)
        return graph

    def process_jsonl(
        self,
        input_path: str,
        output_path: str,
        d_star: float,
        model_name: Optional[str] = None,
        api_keys: Optional[Union[str, Iterable[str]]] = None,
        sample_size: Optional[int] = None,
        seed: int = 42,
        target_node_ids: Optional[Union[str, Iterable[str]]] = None,
        d_star_list: Optional[Union[str, Iterable[float]]] = None,
        per_target_limit: int = 1,
        generate_all_candidates: bool = False,
        max_candidates_per_target: int = 20,
        queries_per_graph: Optional[int] = None,
        balance_difficulty: bool = True,
        balance_types: bool = True,
        max_queries_per_target: Optional[int] = None,
        use_llm: bool = True,
        max_concurrent_per_key: int = 100,
        max_retries: int = 5,
        overwrite: bool = True,
    ) -> None:
        in_path = Path(input_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                graph = json.loads(line)
                graph = self.process_graph(
                    graph=graph,
                    d_star=d_star,
                    model_name=model_name,
                    api_keys=api_keys,
                    sample_size=sample_size,
                    seed=seed,
                    target_node_ids=target_node_ids,
                    d_star_list=d_star_list,
                    per_target_limit=per_target_limit,
                    generate_all_candidates=generate_all_candidates,
                    max_candidates_per_target=max_candidates_per_target,
                    queries_per_graph=queries_per_graph,
                    balance_difficulty=balance_difficulty,
                    balance_types=balance_types,
                    max_queries_per_target=max_queries_per_target,
                    use_llm=use_llm,
                    max_concurrent_per_key=max_concurrent_per_key,
                    max_retries=max_retries,
                    overwrite=overwrite,
                )
                graph = _ordered_graph_output(graph)
                fout.write(json.dumps(graph, ensure_ascii=False) + "\n")


def run(
    input_path: str,
    output_path: str,
    d_star: float = 0.5,
    model_name: Optional[str] = None,
    api_keys: Optional[Union[str, Iterable[str]]] = None,
    sample_size: Optional[int] = None,
    seed: int = 42,
    target_node_ids: Optional[Union[str, Iterable[str]]] = None,
    d_star_list: Optional[Union[str, Iterable[float]]] = None,
    per_target_limit: int = 1,
    generate_all_candidates: bool = False,
    max_candidates_per_target: int = 20,
    queries_per_graph: Optional[int] = None,
    balance_difficulty: bool = True,
    balance_types: bool = True,
    max_queries_per_target: Optional[int] = None,
    use_llm: bool = True,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    overwrite: bool = True,
) -> None:
    g = DifficultyAwareSTVGQueryGenerator()
    g.process_jsonl(
        input_path=input_path,
        output_path=output_path,
        d_star=d_star,
        model_name=model_name,
        api_keys=api_keys,
        sample_size=sample_size,
        seed=seed,
        target_node_ids=target_node_ids,
        d_star_list=d_star_list,
        per_target_limit=per_target_limit,
        generate_all_candidates=generate_all_candidates,
        max_candidates_per_target=max_candidates_per_target,
        queries_per_graph=queries_per_graph,
        balance_difficulty=balance_difficulty,
        balance_types=balance_types,
        max_queries_per_target=max_queries_per_target,
        use_llm=use_llm,
        max_concurrent_per_key=max_concurrent_per_key,
        max_retries=max_retries,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(run)
