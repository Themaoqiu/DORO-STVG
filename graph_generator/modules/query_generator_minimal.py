import asyncio
import json
import os
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import fire
from api_sync.utils.parser import JSONParser


SYSTEM_PROMPT = (
    "You are an expert in spatiotemporal video grounding query writing. "
    "Given a structured clue summary, write exactly one natural English STVG referring expression for the target object. "
    "Use only the provided clues and preserve their semantics faithfully. "
    "Do not invent unseen attributes, relations, actions, objects, identities, temporal facts, or scene details. "
    "Prefer fluent human phrasing over schema-like wording, but keep the query precise and discriminative. "
    "When multiple clue types are provided, prioritize the combination that best disambiguates the target. "
    "Mention attributes, contextual relations, actions, action targets, temporal order, and cross-shot identity only when they are supplied and useful. "
    "Do not mention exact frame numbers, raw field names, node ids, or implementation metadata. "
    "Do not add explanations, notes, markdown, or extra fields. "
    "Return strict JSON only: {\"query\": \"...\"}."
)

PROMPT_TEMPLATE = (
    "Generate one concise but unambiguous STVG query for the target object.\n"
    "Requirements:\n"
    "1) Use only the clue summary below; do not add any new facts.\n"
    "2) Write natural English, not keyword lists or schema labels.\n"
    "3) The query should identify the target as uniquely as possible within the described time span.\n"
    "4) Include relation clues, action clues, action-target clues, temporal order clues, and cross-shot identity clues when they are provided and helpful.\n"
    "5) Prefer the minimal set of clues that still makes the target clear, but do not omit important disambiguating details.\n"
    "6) Do not mention exact frame numbers, node ids, JSON keys, or words like target/context/action_targets.\n"
    "7) If the clue summary contains temporal order such as 'before', express it naturally in the query.\n"
    "8) If the clue summary indicates cross-shot identity, phrase it naturally without exposing annotation terminology.\n"
    "9) Output exactly one query string in strict JSON only: {{\"query\": \"...\"}}.\n\n"
    "Clue summary:\n"
    "{clue_text}"
)

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


def _norm(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x)).strip(" \n\t,.;:")


def _to_int(x: Any, d: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return d


def _uniq(values: Any) -> List[str]:
    raw = values if isinstance(values, list) else ([values] if isinstance(values, str) else [])
    out, seen = [], set()
    for item in raw:
        text = _norm(item)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _frames_from_obj(obj: Dict[str, Any]) -> Set[int]:
    frames = {int(k) for k in (obj.get("bboxes") or {}).keys()}
    if frames:
        return frames
    s = _to_int(obj.get("start_frame"), 0)
    e = _to_int(obj.get("end_frame"), s)
    if e < s:
        s, e = e, s
    return set(range(s, e + 1))


def _frames_to_segments(frames: Set[int]) -> List[Tuple[int, int]]:
    if not frames:
        return []
    arr = sorted(frames)
    out: List[Tuple[int, int]] = []
    a = b = arr[0]
    for x in arr[1:]:
        if x == b + 1:
            b = x
        else:
            out.append((a, b))
            a = b = x
    out.append((a, b))
    return out


def _segments_to_frames(segments: List[Tuple[int, int]]) -> Set[int]:
    frames: Set[int] = set()
    for a, b in segments:
        if b < a:
            a, b = b, a
        frames.update(range(a, b + 1))
    return frames


def _normalize_api_keys(api_keys: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(api_keys, str):
        return [key.strip() for key in api_keys.split(",") if key.strip()]
    return [str(key).strip() for key in api_keys if str(key).strip()]


def _load_api_keys_from_project_env() -> str:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return ""
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "API_KEYS":
                return value.strip().strip('"').strip("'")
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


def _same_entity_map(edges: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for edge in edges:
        if edge.get("reference_relationship") != "same_entity":
            continue
        a = str(edge.get("source_id", "")).strip()
        b = str(edge.get("target_id", "")).strip()
        if a and b:
            out.setdefault(a, set()).add(b)
            out.setdefault(b, set()).add(a)
    return out


def _identity_component(same_entity: Dict[str, Set[str]], node_id: str) -> List[str]:
    node_id = str(node_id).strip()
    if not node_id:
        return []
    stack = [node_id]
    seen: Set[str] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(sorted(same_entity.get(cur, set()) - seen))
    return sorted(seen)


def _ordered_query_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    ordered = {k: spec.get(k) for k in QUERY_SPEC_KEYS}
    for k, v in spec.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def _ordered_query_node(node: Dict[str, Any]) -> Dict[str, Any]:
    ordered = {k: node.get(k) for k in QUERY_NODE_KEYS}
    for k, v in node.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def _ordered_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    ordered: Dict[str, Any] = {}
    for k in GRAPH_OUTPUT_KEYS:
        if k in graph:
            ordered[k] = graph[k]
    for k, v in graph.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def _render_ref(ref: Dict[str, Any]) -> str:
    ref = ref or {}
    cls = _norm(ref.get("class", "object")) or "object"
    attrs = _uniq(ref.get("attributes", []))
    ctx = _uniq(ref.get("context_relations", []))
    parts = [f"the {cls}"]
    if attrs:
        parts.append("with " + ", ".join(attrs))
    if ctx:
        parts.append("that is " + " and ".join(ctx))
    return " ".join(parts)


def _prompt_from_spec(spec: Dict[str, Any]) -> str:
    lines = [f"Target type: {spec.get('target_class', 'object')}", "Required clues:"]
    if spec.get("attributes"):
        lines.append("- attributes: " + "; ".join(_uniq(spec["attributes"])))
    if spec.get("context_relations"):
        lines.append("- context: " + "; ".join(_uniq(spec["context_relations"])))
    if spec.get("actions"):
        lines.append("- actions: " + "; ".join(_uniq(spec["actions"])))
    if spec.get("action_targets"):
        rendered = []
        for item in spec.get("action_targets", []):
            if isinstance(item, dict):
                rendered.append(f"{_norm(item.get('action'))} {_render_ref(item.get('ref', {}))}")
        if rendered:
            lines.append("- action targets: " + "; ".join(rendered))
    if spec.get("relations"):
        rendered = []
        for item in spec.get("relations", []):
            if isinstance(item, dict):
                rendered.append(f"{_norm(item.get('relation'))} {_render_ref(item.get('ref', {}))}")
        if rendered:
            lines.append("- relations: " + "; ".join(rendered))
    if spec.get("action_sequences"):
        rendered = []
        for item in spec.get("action_sequences", []):
            if isinstance(item, dict):
                text = f"{_norm(item.get('first'))} before {_norm(item.get('second'))}"
                if item.get("cross_shot"):
                    text += " (cross-shot)"
                rendered.append(text)
        if rendered:
            lines.append("- sequences: " + "; ".join(rendered))
    if spec.get("cross_shot"):
        lines.append("- cross-shot identity: yes")
    return "\n".join(lines)


def _normalized_ref_signature(ref: Dict[str, Any]) -> Dict[str, Any]:
    ref = ref or {}
    return {
        "class": _norm(ref.get("class", "object")).lower() or "object",
        "sample_class": _norm(ref.get("sample_class", "")).lower(),
        "attributes": sorted(x.lower() for x in _uniq(ref.get("attributes", []))),
        "context_relations": sorted(x.lower() for x in _uniq(ref.get("context_relations", []))),
    }


def _sorted_signature_items(items: Any) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))


def _clue_signature(spec: Dict[str, Any]) -> str:
    spec = spec or {}
    signature = {
        "target_class": _norm(spec.get("target_class", "object")).lower() or "object",
        "attributes": sorted(x.lower() for x in _uniq(spec.get("attributes", []))),
        "context_relations": sorted(x.lower() for x in _uniq(spec.get("context_relations", []))),
        "actions": sorted(x.lower() for x in _uniq(spec.get("actions", []))),
        "action_targets": _sorted_signature_items([
            {
                "action": _norm(item.get("action", "")).lower(),
                "ref": _normalized_ref_signature(item.get("ref", {})),
            }
            for item in spec.get("action_targets", [])
            if isinstance(item, dict)
        ]),
        "relations": _sorted_signature_items([
            {
                "relation": _norm(item.get("relation", "")).lower(),
                "ref": _normalized_ref_signature(item.get("ref", {})),
            }
            for item in spec.get("relations", [])
            if isinstance(item, dict)
        ]),
        "action_sequences": _sorted_signature_items([
            {
                "first": _norm(item.get("first", "")).lower(),
                "second": _norm(item.get("second", "")).lower(),
                "cross_shot": bool(item.get("cross_shot", False)),
            }
            for item in spec.get("action_sequences", [])
            if isinstance(item, dict)
        ]),
        "cross_shot": bool(spec.get("cross_shot", False)),
        "same_entity_nodes": sorted(
            _norm(x).lower() for x in (spec.get("same_entity_nodes") or []) if _norm(x)
        ),
    }
    return json.dumps(signature, ensure_ascii=False, sort_keys=True)


def _sample_class(obj: Dict[str, Any], node_id: Optional[str] = None) -> str:
    node_id = str(node_id or obj.get("node_id", "")).strip().lower()
    if node_id.startswith("person"):
        return "person"
    return _norm(obj.get("object_class", "object")).lower() or "object"


def _surface_class(obj: Dict[str, Any]) -> str:
    return _norm(obj.get("object_class", "object")).lower() or "object"


def _ref_desc(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "class": _surface_class(obj),
        "sample_class": _sample_class(obj),
        "attributes": _uniq(obj.get("attributes", []))[:2],
        "context_relations": _uniq(obj.get("relationships", []))[:1],
    }


def _minimal_ref_desc(
    ref_id: str,
    ref_frames: Set[int],
    objects: Dict[str, Dict[str, Any]],
    object_frames: Dict[str, Set[int]],
) -> Dict[str, Any]:
    ref_obj = objects[ref_id]
    ref_class = _surface_class(ref_obj)
    ref_sample_class = _sample_class(ref_obj, ref_id)
    ref_attrs = _uniq(ref_obj.get("attributes", []))
    ref_ctx = _uniq(ref_obj.get("relationships", []))
    distractors = [
        obj_id
        for obj_id, obj in objects.items()
        if obj_id != ref_id
        and _sample_class(obj, obj_id) == ref_sample_class
        and bool(object_frames[obj_id] & ref_frames)
    ]
    if not distractors:
        return {
            "class": ref_class,
            "sample_class": ref_sample_class,
            "attributes": ref_attrs[:1],
            "context_relations": [] if ref_attrs else ref_ctx[:1],
        }

    for phrase in ref_attrs:
        if all(phrase.lower() not in {x.lower() for x in _uniq(objects[d].get("attributes", []))} for d in distractors):
            return {"class": ref_class, "sample_class": ref_sample_class, "attributes": [phrase], "context_relations": []}

    for phrase in ref_ctx:
        if all(phrase.lower() not in {x.lower() for x in _uniq(objects[d].get("relationships", []))} for d in distractors):
            return {"class": ref_class, "sample_class": ref_sample_class, "attributes": [], "context_relations": [phrase]}

    desc = _ref_desc(ref_obj)
    return {
        "class": ref_class,
        "sample_class": ref_sample_class,
        "attributes": desc["attributes"][:1],
        "context_relations": desc["context_relations"][:1],
    }


def _extract_query(response: str) -> Union[str, bool]:
    parsed = JSONParser.parse(response)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("query"), str):
        return False
    query = parsed["query"].strip()
    return query if query else False


class MinimalSTVGQueryGenerator:
    def __init__(self, max_combo_size: int = 8) -> None:
        self.max_combo_size = max_combo_size

    def _resolve_model_name(self, model_name: Optional[str]) -> str:
        if model_name and str(model_name).strip():
            return str(model_name).strip()
        for key in ["QUERY_MODEL_NAME", "MODEL_NAME", "OPENAI_MODEL"]:
            value = os.getenv(key, "").strip()
            if value:
                return value
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text or "=" not in text:
                    continue
                k, v = text.split("=", 1)
                if k.strip() in {"QUERY_MODEL_NAME", "MODEL_NAME", "OPENAI_MODEL"}:
                    value = v.strip().strip('"').strip("'")
                    if value:
                        return value
        return "gpt-4.1-mini"

    def build_query_plan(
        self,
        graph: Dict[str, Any],
        target_obj: Dict[str, Any],
        d_star: float = 0.5,
        total_frames: Optional[int] = None,
        return_all_candidates: bool = False,
        max_candidates: int = 20,
        identity_node_ids: Optional[Iterable[str]] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        _ = total_frames
        object_nodes = graph.get("object_nodes", []) or []
        action_nodes = graph.get("action_nodes", []) or []
        edges = graph.get("edges", []) or []
        idx_to_node = {i: str(obj.get("node_id")) for i, obj in enumerate(object_nodes) if obj.get("node_id")}
        objects = {str(obj.get("node_id")): obj for obj in object_nodes if obj.get("node_id")}
        same_entity = _same_entity_map(edges)
        object_frames = {obj_id: _frames_from_obj(obj) for obj_id, obj in objects.items()}

        target_id = str(target_obj.get("node_id", ""))
        if target_id not in objects:
            return self._fallback_plan(target_obj, d_star)

        target = objects[target_id]
        target_class = _surface_class(target)
        target_sample_class = _sample_class(target, target_id)
        if identity_node_ids is None:
            target_group = [target_id]
        else:
            target_group = [str(node_id).strip() for node_id in identity_node_ids if str(node_id).strip() in objects]
            if target_id not in target_group:
                target_group.append(target_id)
        target_group = sorted(dict.fromkeys(target_group))
        combo_limit = 5
        target_frames: Set[int] = set()
        for node_id in target_group:
            target_frames |= object_frames.get(node_id, set())
        if not target_frames:
            return self._fallback_plan(target_obj, d_star)
        track_start, track_end = min(target_frames), max(target_frames)

        distractor_ids = [
            obj_id
            for obj_id, obj in objects.items()
            if obj_id not in set(target_group)
            and _sample_class(obj, obj_id) == target_sample_class
            and bool(object_frames[obj_id] & target_frames)
        ]

        action_frames: Dict[str, Dict[str, Set[int]]] = {obj_id: {} for obj_id in objects}
        action_targets: Dict[str, Dict[Tuple[str, str], Set[int]]] = {obj_id: {} for obj_id in objects}
        relation_events: List[Dict[str, Any]] = []

        for group in action_nodes:
            owner_id = str(group.get("object_id", "")).strip()
            if owner_id not in objects:
                continue
            for item in group.get("actions", []) or []:
                label = _norm(item.get("action_label", ""))
                if not label:
                    continue
                s = _to_int(item.get("start_frame"), 0)
                e = _to_int(item.get("end_frame"), s)
                frames = set(range(min(s, e), max(s, e) + 1)) & object_frames[owner_id]
                if not frames:
                    continue
                action_frames[owner_id].setdefault(label, set()).update(frames)
                for ref_id in item.get("target_object_ids", []) or []:
                    ref_id = str(ref_id).strip()
                    if ref_id in objects:
                        action_targets[owner_id].setdefault((label, ref_id), set()).update(frames)

        for bundle in edges:
            if "subject_id" not in bundle or "relationships" not in bundle:
                continue
            subject_id = idx_to_node.get(bundle.get("subject_id"))
            if subject_id not in objects:
                continue
            for rel in bundle.get("relationships", []) or []:
                object_id = idx_to_node.get(rel.get("object_id"))
                predicate = _norm(rel.get("predicate_verb", ""))
                if object_id not in objects or not predicate:
                    continue
                segments = []
                for seg in rel.get("time_frames", []) or []:
                    if isinstance(seg, list) and len(seg) == 2:
                        a, b = _to_int(seg[0], 0), _to_int(seg[1], 0)
                        segments.append((min(a, b), max(a, b)))
                if not segments:
                    segments = _frames_to_segments(object_frames[subject_id] & object_frames[object_id])
                frames = _segments_to_frames(segments) & object_frames[subject_id]
                rel_type = _norm(rel.get("relationship_type", "")).lower()
                relation_events.append(
                    {
                        "subject_id": subject_id,
                        "object_id": object_id,
                        "predicate": predicate,
                        "relationship_type": rel_type,
                        "frames": frames,
                    }
                )
                if rel_type:
                    action_frames[subject_id].setdefault(predicate, set()).update(frames)
                    action_targets[subject_id].setdefault((predicate, object_id), set()).update(frames)

        elements: List[Dict[str, Any]] = []
        seen = set()

        def add(elem_type: str, clue_data: Dict[str, Any], support: Set[int], d_support: Dict[str, Set[int]]) -> None:
            support &= target_frames
            if not support:
                return
            key = json.dumps({"type": elem_type, "clue_data": clue_data}, ensure_ascii=False, sort_keys=True)
            if key in seen:
                return
            seen.add(key)
            elements.append(
                {
                    "type": elem_type,
                    "clue_data": clue_data,
                    "support": set(support),
                    "d_support": {d: set(d_support.get(d, set())) for d in distractor_ids},
                }
            )

        if not distractor_ids:
            add("bare", {}, set(target_frames), {})

        for node_id in target_group:
            node = objects[node_id]
            node_frames = object_frames[node_id]
            for phrase in _uniq(node.get("attributes", [])):
                add(
                    "attribute",
                    {"attributes": [phrase]},
                    set(node_frames),
                    {
                        d: set(object_frames[d]) if phrase.lower() in {x.lower() for x in _uniq(objects[d].get("attributes", []))} else set()
                        for d in distractor_ids
                    },
                )
            for phrase in _uniq(node.get("relationships", [])):
                add(
                    "context_relation",
                    {"context_relations": [phrase]},
                    set(node_frames),
                    {
                        d: set(object_frames[d]) if phrase.lower() in {x.lower() for x in _uniq(objects[d].get("relationships", []))} else set()
                        for d in distractor_ids
                    },
                )
            for label, frames in action_frames[node_id].items():
                if len(frames) >= 2:
                    add("action", {"actions": [label]}, set(frames), {d: action_frames[d].get(label, set()) for d in distractor_ids})
            for (label, ref_id), frames in action_targets[node_id].items():
                if len(frames) < 2 or ref_id not in objects:
                    continue
                ref = _minimal_ref_desc(ref_id, set(frames) & object_frames[ref_id], objects, object_frames)
                d_support = {d: set() for d in distractor_ids}
                for d in distractor_ids:
                    for (d_label, d_ref_id), d_frames in action_targets[d].items():
                        if d_label != label or d_ref_id not in objects:
                            continue
                        d_ref = _ref_desc(objects[d_ref_id])
                        if d_ref.get("sample_class") == ref.get("sample_class") and set(ref["attributes"]) <= set(d_ref["attributes"]) and set(ref["context_relations"]) <= set(d_ref["context_relations"]):
                            d_support[d] |= d_frames
                add("action_target", {"action_targets": [{"action": label, "ref": ref}]}, set(frames), d_support)

        for rel in relation_events:
            if rel["subject_id"] not in target_group or len(rel["frames"]) < 1:
                continue
            ref = _minimal_ref_desc(rel["object_id"], set(rel["frames"]) & object_frames[rel["object_id"]], objects, object_frames)
            elem_type = "relation_described" if len([d for d in distractor_ids if _sample_class(objects[d], d) == ref.get("sample_class")]) else "relation"
            d_support = {d: set() for d in distractor_ids}
            for d in distractor_ids:
                for other in relation_events:
                    if other["subject_id"] != d or other["predicate"] != rel["predicate"]:
                        continue
                    d_ref = _ref_desc(objects[other["object_id"]])
                    if d_ref.get("sample_class") == ref.get("sample_class") and set(ref["attributes"]) <= set(d_ref["attributes"]) and set(ref["context_relations"]) <= set(d_ref["context_relations"]):
                        d_support[d] |= other["frames"]
            add(elem_type, {"relations": [{"relation": rel["predicate"], "ref": ref}]}, set(rel["frames"]), d_support)

        for node_id in target_group:
            labels = sorted(action_frames[node_id])
            for a1, a2 in combinations(labels, 2):
                f1, f2 = action_frames[node_id][a1], action_frames[node_id][a2]
                if not f1 or not f2 or max(f1) >= min(f2):
                    continue
                d_support = {}
                for d in distractor_ids:
                    df1, df2 = action_frames[d].get(a1, set()), action_frames[d].get(a2, set())
                    d_support[d] = df1 | df2 if df1 and df2 and max(df1) < min(df2) else set()
                add("action_sequence", {"action_sequences": [{"first": a1, "second": a2, "order": "before", "cross_shot": False}]}, f1 | f2, d_support)

        cross_sources = target_group if len(target_group) > 1 else [target_id]
        for second_id in cross_sources:
            second_obj = objects[second_id]
            ref_candidates = sorted(set(same_entity.get(second_id, set())) & set(target_group if len(target_group) > 1 else same_entity.get(second_id, set())))
            if not ref_candidates:
                ref_candidates = sorted(same_entity.get(second_id, set()))
            for ref_id in ref_candidates:
                if ref_id not in objects or ref_id == second_id:
                    continue
                if set(second_obj.get("shot_ids", [])) & set(objects[ref_id].get("shot_ids", [])):
                    continue
                for first, f1 in action_frames[ref_id].items():
                    if len(f1) < 2:
                        continue
                    for second, f2 in action_frames[second_id].items():
                        if len(f2) < 2 or first == second or max(f1) >= min(f2):
                            continue
                        d_support = {d: set() for d in distractor_ids}
                        for d in distractor_ids:
                            second_match = action_frames[d].get(second, set())
                            if not second_match:
                                continue
                            for d_ref in same_entity.get(d, set()):
                                if d_ref in objects and not (set(objects[d].get("shot_ids", [])) & set(objects[d_ref].get("shot_ids", []))):
                                    first_match = action_frames[d_ref].get(first, set())
                                    if first_match and max(first_match) < min(second_match):
                                        d_support[d] |= second_match
                        add(
                            "cross_shot",
                            {
                                "action_sequences": [{"first": first, "second": second, "order": "before", "cross_shot": True}],
                                "same_entity_nodes": sorted(set(target_group) | {second_id, ref_id}),
                            },
                            set(f2),
                            d_support,
                        )

        if not elements:
            print(
                f"[query_minimal] build_query_plan: target={target_id} group={target_group} distractors={len(distractor_ids)} elements=0 fallback",
                flush=True,
            )
            return self._fallback_plan(target_obj, d_star, same_entity_nodes=target_group, frames=target_frames)

        print(
            f"[query_minimal] build_query_plan: target={target_id} group={target_group} distractors={len(distractor_ids)} elements={len(elements)} max_combo_size={combo_limit}",
            flush=True,
        )

        segment_pool: List[Tuple[Tuple[int, int], bool]] = [((track_start, track_end), False)]
        seen_segments = {(track_start, track_end)}
        for elem in elements:
            is_local = elem["type"] in {"action", "action_target", "relation", "relation_described", "action_sequence", "cross_shot"}
            for seg in _frames_to_segments(elem["support"]):
                if seg not in seen_segments:
                    seen_segments.add(seg)
                    segment_pool.append((seg, is_local))
        segment_pool.sort(key=lambda item: (item[0][0], item[0][1] - item[0][0], item[1]))

        candidates: List[Dict[str, Any]] = []
        for seg, must_use_local in segment_pool:
            a, b = seg
            seg_frames = set(range(a, b + 1))
            for k in range(1, min(combo_limit, len(elements)) + 1):
                found = []
                for combo in combinations(elements, k):
                    support = set.intersection(*(set(elem["support"]) for elem in combo)) if combo else set()
                    if not seg_frames <= support:
                        continue
                    if must_use_local and not any(elem["type"] in {"action", "action_target", "relation", "relation_described", "action_sequence", "cross_shot"} and seg_frames <= elem["support"] for elem in combo):
                        continue
                    ok = True
                    for d in distractor_ids:
                        d_frames = set.intersection(*(set(elem["d_support"].get(d, set())) for elem in combo)) if combo else set()
                        if d_frames & seg_frames:
                            ok = False
                            break
                    if ok:
                        found.append({"combo": list(combo), "segment": seg, "combo_size": k})
                if found:
                    candidates.extend(found)
                    break

        if not candidates:
            best = None
            for k in range(1, min(combo_limit, len(elements)) + 1):
                for combo in combinations(elements, k):
                    support = set.intersection(*(set(elem["support"]) for elem in combo)) if combo else set()
                    if not support:
                        continue
                    score = sum(1 for d in distractor_ids if not (set.intersection(*(set(elem["d_support"].get(d, set())) for elem in combo)) if combo else set()))
                    item = {"combo": list(combo), "segment": _frames_to_segments(support)[0], "score": score, "support_len": len(support)}
                    if not best or (item["score"], item["support_len"]) > (best["score"], best["support_len"]):
                        best = item
            candidates = [best] if best else []

        plans = []
        seen_plans = set()
        default_same_entity_nodes = sorted(target_group)
        for item in candidates:
            combo = item["combo"]
            seg_start, seg_end = item["segment"]
            clue_types = sorted({elem["type"] for elem in combo if elem["type"] != "bare"}) or ["bare"]
            spec = {
                "target_class": target_class,
                "attributes": [],
                "context_relations": [],
                "actions": [],
                "action_targets": [],
                "action_sequences": [],
                "relations": [],
                "time_range": {"start": seg_start, "end": seg_end},
                "time_segments": [[seg_start, seg_end]],
                "full_track_range": {"start": track_start, "end": track_end},
                "cross_shot": len(default_same_entity_nodes) > 1,
                "same_entity_nodes": list(default_same_entity_nodes),
            }
            for elem in combo:
                clue = elem["clue_data"]
                for key in ["attributes", "context_relations", "actions", "action_targets", "action_sequences", "relations"]:
                    spec[key].extend(clue.get(key, []))
                for nid in clue.get("same_entity_nodes", []):
                    if nid not in spec["same_entity_nodes"]:
                        spec["same_entity_nodes"].append(nid)
                if clue.get("action_sequences"):
                    spec["cross_shot"] = spec["cross_shot"] or bool(clue["action_sequences"][0].get("cross_shot", False))
            spec["same_entity_nodes"] = sorted(dict.fromkeys(spec["same_entity_nodes"]))
            spec = _ordered_query_spec(spec)

            has_sequence = bool(spec["action_sequences"])
            has_cross_shot = bool(spec["cross_shot"])
            has_action = bool(spec["actions"] or spec["action_targets"])
            has_relation = bool(spec["relations"])
            clue_count = sum(bool(spec[k]) for k in ["attributes", "context_relations", "actions", "action_targets", "action_sequences", "relations"])

            if has_cross_shot:
                t_level = 4
            elif has_sequence:
                t_level = 3
            elif clue_count >= 2 and (has_action or has_relation):
                t_level = 2
            elif has_action:
                t_level = 1
            else:
                t_level = 0

            if not distractor_ids and not has_relation and not spec["action_targets"]:
                s_level = 0
            elif len(spec["relations"]) >= 3:
                s_level = 3
            elif any(item.get("ref", {}).get("attributes") or item.get("ref", {}).get("context_relations") for item in spec["relations"] + spec["action_targets"]):
                s_level = 2
            elif has_relation or spec["action_targets"]:
                s_level = 1
            else:
                s_level = 0
            same_class_level = 0 if not distractor_ids else (1 if len(distractor_ids) == 1 else 2)
            s_level = min(max(s_level, same_class_level), 3)

            d_t = t_level / 4.0
            d_s = s_level / 3.0
            d = 0.5 * d_t + 0.5 * d_s

            if d >= 0.75:
                bucket = "very_hard"
            elif d >= 0.5:
                bucket = "hard"
            elif d >= 0.25:
                bucket = "medium"
            else:
                bucket = "easy"

            if has_cross_shot:
                type_bucket = "cross_shot"
            elif has_sequence:
                type_bucket = "action_sequence"
            elif spec["action_targets"]:
                type_bucket = "action_target"
            elif has_relation:
                type_bucket = "relation"
            elif has_action:
                type_bucket = "action"
            elif spec["context_relations"]:
                type_bucket = "context_relation"
            else:
                type_bucket = "attribute"

            signature = json.dumps({"target_node": target_id, "query_spec": spec}, ensure_ascii=False, sort_keys=True)
            if signature in seen_plans:
                continue
            seen_plans.add(signature)
            plans.append(
                {
                    "query_spec": spec,
                    "T": t_level,
                    "S": s_level,
                    "D_t": d_t,
                    "D_s": d_s,
                    "D": d,
                    "d_star": float(d_star),
                    "target_node": target_id,
                    "segment_len": seg_end - seg_start + 1,
                    "combo_size": item.get("combo_size", len(combo)),
                    "clue_types": clue_types,
                    "type_bucket": type_bucket,
                    "difficulty_bucket": bucket,
                    "candidate_signature": signature,
                }
            )

        plans.sort(key=lambda x: (int(x.get("combo_size", len(x["clue_types"]))), x["D"], -x["segment_len"], abs(x["D"] - float(d_star))))
        if return_all_candidates:
            return plans[:max_candidates] if plans else [self._fallback_plan(target_obj, d_star, same_entity_nodes=target_group, frames=target_frames)]
        return plans[0] if plans else self._fallback_plan(target_obj, d_star, same_entity_nodes=target_group, frames=target_frames)

    def _fallback_plan(
        self,
        target_obj: Dict[str, Any],
        d_star: float,
        same_entity_nodes: Optional[Iterable[str]] = None,
        frames: Optional[Set[int]] = None,
    ) -> Dict[str, Any]:
        target_id = str(target_obj.get("node_id", ""))
        target_class = _surface_class(target_obj)
        node_group = [str(node_id).strip() for node_id in (same_entity_nodes or [target_id]) if str(node_id).strip()]
        node_group = sorted(dict.fromkeys(node_group))
        frames = set(frames or _frames_from_obj(target_obj))
        s, e = min(frames), max(frames)
        spec = _ordered_query_spec(
            {
                "target_class": target_class,
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
                "same_entity_nodes": node_group,
            }
        )
        return {
            "query_spec": spec,
            "D_t": 0.0,
            "D_s": 0.0,
            "D": 0.0,
            "d_star": float(d_star),
            "target_node": target_id,
            "segment_len": e - s + 1,
            "clue_types": ["bare"],
            "type_bucket": "attribute",
            "difficulty_bucket": "easy",
            "candidate_signature": f"fallback::{target_id}::{'+'.join(node_group)}",
        }

    def build_graph_plans(
        self,
        graph: Dict[str, Any],
        d_star: float,
        total_frames: Optional[int] = None,
        sample_size: Optional[int] = None,
        seed: int = 42,
        target_node_ids: Optional[Union[str, Iterable[str]]] = None,
        d_star_list: Optional[Union[str, Iterable[float]]] = None,
        per_target_limit: int = 3,
        generate_all_candidates: bool = False,
        max_candidates_per_target: int = 20,
        queries_per_graph: Optional[int] = None,
        balance_difficulty: bool = True,
        balance_types: bool = True,
        difficulty_bins: int = 4,
        max_queries_per_target: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        _ = total_frames, d_star_list, balance_difficulty, balance_types, difficulty_bins, max_queries_per_target
        object_nodes = [obj for obj in graph.get("object_nodes", []) if obj.get("node_id") and obj.get("bboxes")]
        objects = {str(obj.get("node_id")): obj for obj in object_nodes}
        same_entity = _same_entity_map(graph.get("edges", []) or [])
        wanted = None
        if target_node_ids:
            wanted = {x.strip() for x in (target_node_ids.split(",") if isinstance(target_node_ids, str) else target_node_ids) if str(x).strip()}
        candidates = list(object_nodes)
        merged_candidates = []
        seen_groups: Set[Tuple[str, ...]] = set()
        for obj in object_nodes:
            group = tuple(node_id for node_id in _identity_component(same_entity, str(obj.get("node_id"))) if node_id in objects)
            if len(group) <= 1 or group in seen_groups:
                continue
            if wanted and not (set(group) & wanted):
                continue
            seen_groups.add(group)
            merged_obj = dict(objects[group[0]])
            merged_obj["same_entity_nodes"] = list(group)
            merged_candidates.append(merged_obj)
        if wanted:
            candidates = [obj for obj in candidates if str(obj.get("node_id")) in wanted]
        candidates.extend(merged_candidates)
        if sample_size and len(candidates) > sample_size:
            import random
            candidates = random.Random(seed).sample(candidates, sample_size)

        print(
            f"[query_minimal] build_graph_plans: single_candidates={len(object_nodes)} merged_candidates={len(merged_candidates)} active_candidates={len(candidates)}",
            flush=True,
        )

        plans = []
        for idx, obj in enumerate(candidates, start=1):
            node_id = str(obj.get("node_id", ""))
            group = obj.get("same_entity_nodes") or [node_id]
            print(
                f"[query_minimal] candidate_start: {idx}/{len(candidates)} target={node_id} group={group}",
                flush=True,
            )
            result = self.build_query_plan(
                graph=graph,
                target_obj=obj,
                d_star=d_star,
                return_all_candidates=True,
                max_candidates=max_candidates_per_target,
                identity_node_ids=obj.get("same_entity_nodes"),
            )
            result_list = result if isinstance(result, list) else [result]
            result_list = sorted(
                result_list,
                key=lambda item: (
                    abs(float(item.get("D", 0.0)) - float(d_star)),
                    int(item.get("combo_size", len(item.get("clue_types", [])))),
                    -int(item.get("segment_len", 1)),
                    int(item.get("segment_len", 0)) == int(item.get("query_spec", {}).get("full_track_range", {}).get("end", 0)) - int(item.get("query_spec", {}).get("full_track_range", {}).get("start", 0)) + 1,
                ),
            )

            def _non_overlapping(candidate, chosen):
                start = int(candidate.get("query_spec", {}).get("time_range", {}).get("start", 0))
                end = int(candidate.get("query_spec", {}).get("time_range", {}).get("end", start))
                for prev in chosen:
                    ps = int(prev.get("query_spec", {}).get("time_range", {}).get("start", 0))
                    pe = int(prev.get("query_spec", {}).get("time_range", {}).get("end", ps))
                    overlap = max(0, min(end, pe) - max(start, ps) + 1)
                    union = max(end, pe) - min(start, ps) + 1
                    len_cur = end - start + 1
                    len_prev = pe - ps + 1
                    similar = min(len_cur, len_prev) / max(len_cur, len_prev) > 0.6 if max(len_cur, len_prev) > 0 else True
                    if union > 0 and overlap / union > 0.5 and similar:
                        return False
                return True

            selected = []
            if not generate_all_candidates:
                bucket_centers = {
                    "easy": 0.125,
                    "medium": 0.375,
                    "hard": 0.625,
                    "very_hard": 0.875,
                }
                bucket_order = sorted(
                    bucket_centers,
                    key=lambda bucket: abs(bucket_centers[bucket] - float(d_star)),
                )
                bucketed = {bucket: [] for bucket in bucket_order}
                for item in result_list:
                    bucketed.setdefault(str(item.get("difficulty_bucket", "medium")), []).append(item)

                while len(selected) < max(per_target_limit, 1):
                    added = False
                    for bucket in bucket_order:
                        for item in bucketed.get(bucket, []):
                            if item in selected:
                                continue
                            if not _non_overlapping(item, selected):
                                continue
                            selected.append(item)
                            added = True
                            break
                        if len(selected) >= max(per_target_limit, 1):
                            break
                    if not added:
                        break

                if len(selected) < max(per_target_limit, 1):
                    for item in result_list:
                        if item in selected:
                            continue
                        if not _non_overlapping(item, selected):
                            continue
                        selected.append(item)
                        if len(selected) >= max(per_target_limit, 1):
                            break

            emitted = result_list if generate_all_candidates else selected
            for item in emitted:
                item = dict(item)
                item["plan_id"] = f"plan_{item.get('target_node')}_{len(plans)}_{int(float(item.get('D', 0.0)) * 1000)}"
                plans.append(item)
            print(
                f"[query_minimal] candidate_done: {idx}/{len(candidates)} target={node_id} raw_candidates={len(result_list)} selected={len(emitted)} total_plans={len(plans)}",
                flush=True,
            )
        if queries_per_graph and queries_per_graph > 0:
            return plans[:queries_per_graph]
        return plans

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

        keys = _resolve_api_keys(api_keys)

        generator = StreamGenerator(
            model_name=model_name,
            api_keys=keys,
            max_concurrent_per_key=max_concurrent_per_key,
            max_retries=max_retries,
            rational=False,
            with_unique_id=True,
        )
        prompts = [
            {
                "id": str(plan.get("plan_id") or plan.get("target_node")),
                "prompt": PROMPT_TEMPLATE.format(clue_text=_prompt_from_spec(plan.get("query_spec", {}))),
            }
            for plan in plans
        ]
        print(
            f"[query_minimal] generate_queries_with_llm: model={model_name} prompts={len(prompts)} keys={len(keys)}",
            flush=True,
        )
        query_map: Dict[str, str] = {}
        completed = 0
        async for item in generator.generate_stream(prompts=prompts, system_prompt=system_prompt, validate_func=_extract_query):
            if item and isinstance(item.get("result"), str) and item["result"].strip():
                query_map[str(item["id"])] = item["result"].strip()
                completed += 1
                print(f"[query_minimal] llm_progress: {completed}/{len(prompts)}", flush=True)
        print(f"[query_minimal] llm_done: received={len(query_map)}", flush=True)
        return query_map

    def _fallback_query(self, query_spec: Dict[str, Any]) -> str:
        chunks = [f"the {query_spec.get('target_class', 'object')}"]
        if query_spec.get("attributes"):
            chunks.append("with " + ", ".join(query_spec["attributes"]))
        if query_spec.get("context_relations"):
            chunks.append("that is " + " and ".join(query_spec["context_relations"]))
        if query_spec.get("actions"):
            chunks.append("that is " + " and ".join(query_spec["actions"]))
        for item in query_spec.get("action_targets", []):
            if isinstance(item, dict):
                chunks.append(f"that {item.get('action')} {_render_ref(item.get('ref', {}))}")
        for item in query_spec.get("action_sequences", []):
            if isinstance(item, dict):
                chunks.append(f"that {item.get('first')} before {item.get('second')}")
        for item in query_spec.get("relations", []):
            if isinstance(item, dict):
                chunks.append(f"that is {item.get('relation')} {_render_ref(item.get('ref', {}))}")
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
        per_target_limit: int = 3,
        generate_all_candidates: bool = False,
        max_candidates_per_target: int = 20,
        queries_per_graph: Optional[int] = None,
        balance_difficulty: bool = True,
        balance_types: bool = True,
        difficulty_bins: int = 4,
        max_queries_per_target: Optional[int] = None,
        use_llm: bool = True,
        max_concurrent_per_key: int = 100,
        max_retries: int = 5,
        overwrite: bool = True,
    ) -> List[Dict[str, Any]]:
        _ = overwrite
        video_path = str(graph.get("video_path", ""))
        print(f"[query_minimal] process_graph: video={video_path}", flush=True)
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
            difficulty_bins=difficulty_bins,
            max_queries_per_target=max_queries_per_target,
        )
        print(f"[query_minimal] process_graph: plans={len(plans)} use_llm={use_llm}", flush=True)
        query_map = {}
        if use_llm and plans:
            query_map = asyncio.run(
                self.generate_queries_with_llm(
                    plans,
                    self._resolve_model_name(model_name),
                    api_keys,
                    max_concurrent_per_key,
                    max_retries,
                )
            )

        objects = {str(obj.get("node_id")): obj for obj in graph.get("object_nodes", []) if obj.get("node_id")}
        records: List[Dict[str, Any]] = []
        for i, plan in enumerate(plans):
            pid = str(plan.get("plan_id") or f"plan_{i}")
            target_id = str(plan.get("target_node"))
            query = query_map.get(pid) or self._fallback_query(plan.get("query_spec", {}))
            spec = _ordered_query_spec(plan.get("query_spec") or {})
            start = int(spec.get("time_range", {}).get("start", 0))
            end = int(spec.get("time_range", {}).get("end", start))
            node_group = [str(node_id).strip() for node_id in spec.get("same_entity_nodes", []) if str(node_id).strip() in objects] or [target_id]
            boxes: Dict[str, Any] = {}
            use_identity_boxes = len(node_group) > 1
            for node_id in node_group:
                for frame, box in (objects.get(node_id, {}).get("bboxes") or {}).items():
                    frame_int = _to_int(frame, -1)
                    if frame_int < 0:
                        continue
                    if not use_identity_boxes and not (start <= frame_int <= end):
                        continue
                    if str(frame_int) not in boxes:
                        boxes[str(frame_int)] = box
            if use_identity_boxes:
                boxes = {str(frame): boxes[str(frame)] for frame in sorted(int(k) for k in boxes.keys())}
            records.append(
                {
                    "video_path": graph.get("video_path"),
                    "target_node_id": target_id,
                    "same_entity_nodes": node_group,
                    "query": query,
                    "boxes": boxes,
                    "D_t": plan.get("D_t"),
                    "D_s": plan.get("D_s"),
                    "D": plan.get("D"),
                    "clue_types": plan.get("clue_types", []),
                    "difficulty_bucket": plan.get("difficulty_bucket"),
                }
            )
        deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record, plan in zip(records, plans):
            key = (
                str(record.get("target_node_id", "")),
                _clue_signature(plan.get("query_spec") or {}),
            )
            seg_len = len(record.get("boxes") or {})
            prev = deduped.get(key)
            if not prev:
                deduped[key] = record
                continue
            prev_len = len(prev.get("boxes") or {})
            cur_score = (seg_len, -float(record.get("D", 0.0)))
            prev_score = (prev_len, -float(prev.get("D", 0.0)))
            if cur_score > prev_score:
                deduped[key] = record
        print(f"[query_minimal] process_graph: records={len(records)} deduped={len(deduped)}", flush=True)
        return list(deduped.values())

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
        per_target_limit: int = 3,
        generate_all_candidates: bool = False,
        max_candidates_per_target: int = 20,
        queries_per_graph: Optional[int] = None,
        balance_difficulty: bool = True,
        balance_types: bool = True,
        difficulty_bins: int = 4,
        max_queries_per_target: Optional[int] = None,
        use_llm: bool = True,
        max_concurrent_per_key: int = 100,
        max_retries: int = 5,
        overwrite: bool = True,
    ) -> None:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[query_minimal] process_jsonl: input={input_path} output={output_path}", flush=True)
        with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
            graph_idx = 0
            total_records = 0
            for line in fin:
                if not line.strip():
                    continue
                graph_idx += 1
                print(f"[query_minimal] process_jsonl: graph_index={graph_idx}", flush=True)
                graph = json.loads(line)
                records = self.process_graph(
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
                    difficulty_bins=difficulty_bins,
                    max_queries_per_target=max_queries_per_target,
                    use_llm=use_llm,
                    max_concurrent_per_key=max_concurrent_per_key,
                    max_retries=max_retries,
                    overwrite=overwrite,
                )
                for record in records:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_records += len(records)
                print(f"[query_minimal] process_jsonl: graph_index={graph_idx} wrote={len(records)} total={total_records}", flush=True)
        print(f"[query_minimal] process_jsonl: done graphs={graph_idx} total_records={total_records}", flush=True)


def run(input_path: str, output_path: str, d_star: float = 0.5, **kwargs: Any) -> None:
    MinimalSTVGQueryGenerator().process_jsonl(input_path=input_path, output_path=output_path, d_star=d_star, **kwargs)

if __name__ == "__main__":
    fire.Fire(run)
