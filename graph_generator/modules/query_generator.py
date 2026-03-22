import asyncio
import json
import os
import random
import re
from itertools import combinations
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
    "3) Include relation clues and action clues when provided.\n"
    "4) Do not mention exact frame numbers.\n"
    "5) Output strict JSON only: {\"query\": \"...\"}.\n\n"
    "Target difficulty d_star: {d_star}\n"
    "Spec JSON:\n{spec_json}"
)

TYPE_DIFFICULTY = {
    "attribute": 1,
    "context_relation": 2,
    "action": 3,
    "action_target": 4,
    "relation_to_unique": 4,
    "relation_to_described": 5,
    "action_sequence": 6,
    "cross_shot": 7,
}

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


def _to_int(v: Any, d: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return d


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s)).strip(" \n\t,.;:")


def _uniq_phrases(values: Any) -> List[str]:
    if isinstance(values, list):
        raw = [str(x) for x in values]
    elif isinstance(values, str):
        raw = [x.strip() for x in values.split(",")]
    else:
        raw = []
    out: List[str] = []
    seen: Set[str] = set()
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


def _frames_to_segments(frames: Set[int]) -> List[Tuple[int, int]]:
    if not frames:
        return []
    arr = sorted(frames)
    out: List[Tuple[int, int]] = []
    a = b = arr[0]
    for x in arr[1:]:
        if x == b + 1:
            b = x
            continue
        out.append((a, b))
        a = b = x
    out.append((a, b))
    return out


def _segments_to_frames(segments: List[Tuple[int, int]]) -> Set[int]:
    out: Set[int] = set()
    for a, b in segments:
        if b < a:
            a, b = b, a
        out.update(range(a, b + 1))
    return out


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


def _extract_query(response: str) -> Union[str, bool]:
    parsed = JSONParser.parse(response)
    if not isinstance(parsed, dict):
        return False
    query = parsed.get("query")
    if not isinstance(query, str):
        return False
    query = query.strip()
    return query if query else False


def _build_idx_to_node(object_nodes: List[Dict[str, Any]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for idx, obj in enumerate(object_nodes):
        node_id = str(obj.get("node_id", "")).strip()
        if node_id:
            out[idx] = node_id
    return out


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


def _ref_desc(ref_obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "class": _norm(ref_obj.get("object_class", "object")).lower() or "object",
        "attributes": _uniq_phrases(ref_obj.get("attributes", []))[:2],
        "context_relations": _uniq_phrases(ref_obj.get("relationships", []))[:1],
    }


def _build_same_entity_map(edges: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    same_entity: Dict[str, Set[str]] = {}
    for edge in edges:
        if edge.get("reference_relationship") != "same_entity":
            continue
        a = str(edge.get("source_id", "")).strip()
        b = str(edge.get("target_id", "")).strip()
        if not a or not b:
            continue
        same_entity.setdefault(a, set()).add(b)
        same_entity.setdefault(b, set()).add(a)
    return same_entity


class DifficultyAwareSTVGQueryGenerator:
    def __init__(
        self,
        n_action_cap: int = 5,
        n_density_cap: int = 8,
        w1: float = 0.4,
        w2: float = 0.2,
        w3: float = 0.1,
        w4: float = 0.3,
        lam: float = 0.5,
        max_combo_size: int = 3,
    ) -> None:
        self.n_action_cap = n_action_cap
        self.n_density_cap = n_density_cap
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.lam = lam
        self.max_combo_size = max_combo_size

    def build_query_plan(
        self,
        graph: Dict[str, Any],
        target_obj: Dict[str, Any],
        d_star: float,
        total_frames: Optional[int] = None,
        return_all_candidates: bool = False,
        max_candidates: int = 20,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        _ = total_frames
        object_nodes = graph.get("object_nodes", [])
        action_nodes = graph.get("action_nodes", [])
        edges = graph.get("edges", [])
        objects_by_id = {str(obj.get("node_id")): obj for obj in object_nodes if obj.get("node_id")}
        idx_to_node = _build_idx_to_node(object_nodes)
        same_entity_map = _build_same_entity_map(edges)

        target_id = str(target_obj.get("node_id", ""))
        if target_id not in objects_by_id:
            return self._fallback_plan(target_obj, d_star)

        target = objects_by_id[target_id]
        target_class = _norm(target.get("object_class", "object")).lower() or "object"
        target_frames = {int(k) for k in (target.get("bboxes") or {}).keys()}
        if not target_frames:
            s = _to_int(target.get("start_frame"), 0)
            e = _to_int(target.get("end_frame"), 0)
            target_frames = set(range(min(s, e), max(s, e) + 1))
        track_start = min(target_frames) if target_frames else _to_int(target.get("start_frame"), 0)
        track_end = max(target_frames) if target_frames else _to_int(target.get("end_frame"), 0)

        object_frames: Dict[str, Set[int]] = {}
        frame_objects: Dict[int, Set[str]] = {}
        for obj_id, obj in objects_by_id.items():
            frames = {int(k) for k in (obj.get("bboxes") or {}).keys()}
            if not frames:
                s = _to_int(obj.get("start_frame"), 0)
                e = _to_int(obj.get("end_frame"), 0)
                frames = set(range(min(s, e), max(s, e) + 1))
            object_frames[obj_id] = frames
            for f in frames:
                frame_objects.setdefault(f, set()).add(obj_id)

        distractors = [
            obj
            for obj in object_nodes
            if str(obj.get("node_id")) != target_id
            and _norm(obj.get("object_class", "")).lower() == target_class
            and bool(object_frames.get(str(obj.get("node_id")), set()) & target_frames)
        ]
        distractor_ids = [str(obj.get("node_id")) for obj in distractors if obj.get("node_id")]
        distractor_id_set = set(distractor_ids)

        others = [obj for obj in object_nodes if str(obj.get("node_id")) != target_id]
        if not others:
            c_tiou = 0.0
        else:
            tious: List[float] = []
            for obj in others:
                frames = object_frames.get(str(obj.get("node_id")), set())
                union = target_frames | frames
                tious.append((len(target_frames & frames) / len(union)) if union else 0.0)
            c_tiou = 1.0 - (sum(tious) / len(tious))

        shot_ids = set(target.get("shot_ids", []))
        total_shots = len(graph.get("temporal_nodes", []))
        c_bg = (len(shot_ids) - 1) / max(total_shots - 1, 1) if total_shots else 0.0

        counts: List[int] = []
        for frame_idx in target_frames:
            n_others = len(frame_objects.get(frame_idx, set()) - {target_id})
            counts.append(max(n_others, 0))
        c_density = min((sum(counts) / len(counts)) / self.n_density_cap, 1.0) if counts else 0.0

        target_action_frames: Dict[str, Set[int]] = {}
        target_action_targets: Dict[Tuple[str, str], Set[int]] = {}
        distractor_action_frames: Dict[str, Dict[str, Set[int]]] = {d: {} for d in distractor_ids}
        distractor_action_targets: Dict[str, Dict[Tuple[str, str], Set[int]]] = {d: {} for d in distractor_ids}
        all_action_frames: Dict[str, Dict[str, Set[int]]] = {obj_id: {} for obj_id in objects_by_id}
        all_action_targets: Dict[str, Dict[Tuple[str, str], Set[int]]] = {obj_id: {} for obj_id in objects_by_id}

        for group in action_nodes:
            owner_id = str(group.get("object_id", "")).strip()
            if not owner_id:
                continue
            for item in group.get("actions", []) or []:
                if not isinstance(item, dict):
                    continue
                label = _norm(item.get("action_label", ""))
                if not label:
                    continue
                s = _to_int(item.get("start_frame"), 0)
                e = _to_int(item.get("end_frame"), s)
                frames = set(range(min(s, e), max(s, e) + 1))
                frames &= object_frames.get(owner_id, set())
                all_action_frames.setdefault(owner_id, {}).setdefault(label, set()).update(frames)
                if owner_id == target_id:
                    target_action_frames.setdefault(label, set()).update(frames)
                    for ref_id in item.get("target_object_ids", []) or []:
                        ref_id = str(ref_id).strip()
                        if ref_id and ref_id in objects_by_id:
                            all_action_targets.setdefault(owner_id, {}).setdefault((label, ref_id), set()).update(frames)
                            target_action_targets.setdefault((label, ref_id), set()).update(frames)
                elif owner_id in distractor_id_set:
                    distractor_action_frames[owner_id].setdefault(label, set()).update(frames)
                    for ref_id in item.get("target_object_ids", []) or []:
                        ref_id = str(ref_id).strip()
                        if ref_id and ref_id in objects_by_id:
                            all_action_targets.setdefault(owner_id, {}).setdefault((label, ref_id), set()).update(frames)
                            distractor_action_targets[owner_id].setdefault((label, ref_id), set()).update(frames)
                else:
                    for ref_id in item.get("target_object_ids", []) or []:
                        ref_id = str(ref_id).strip()
                        if ref_id and ref_id in objects_by_id:
                            all_action_targets.setdefault(owner_id, {}).setdefault((label, ref_id), set()).update(frames)

        relation_events: List[Dict[str, Any]] = []
        for bundle in edges:
            if "subject_id" not in bundle or "relationships" not in bundle:
                continue
            subject_id = idx_to_node.get(bundle.get("subject_id"))
            if not subject_id:
                continue
            for rel in bundle.get("relationships", []) or []:
                if not isinstance(rel, dict):
                    continue
                object_id = idx_to_node.get(rel.get("object_id"))
                predicate = _norm(rel.get("predicate_verb", ""))
                if not object_id or not predicate:
                    continue
                segments: List[Tuple[int, int]] = []
                for seg in rel.get("time_frames", []) or []:
                    if isinstance(seg, list) and len(seg) == 2:
                        a, b = _to_int(seg[0], 0), _to_int(seg[1], 0)
                        segments.append((min(a, b), max(a, b)))
                if not segments:
                    frames = object_frames.get(subject_id, set()) & object_frames.get(object_id, set())
                    segments = _frames_to_segments(frames)
                relation_events.append(
                    {
                        "subject_id": subject_id,
                        "object_id": object_id,
                        "predicate": predicate,
                        "relationship_type": _norm(rel.get("relationship_type", "")).lower(),
                        "segments": segments,
                    }
                )

        same_class_count: Dict[str, int] = {}
        for obj in object_nodes:
            obj_id = str(obj.get("node_id", ""))
            obj_class = _norm(obj.get("object_class", "")).lower()
            if not obj_id or not obj_class:
                continue
            count = 0
            for other in object_nodes:
                if _norm(other.get("object_class", "")).lower() != obj_class:
                    continue
                if object_frames.get(str(other.get("node_id")), set()) & target_frames:
                    count += 1
            same_class_count[obj_id] = count

        element_pool: List[Dict[str, Any]] = []
        element_key_to_index: Dict[str, int] = {}

        def add_element(
            elem_type: str,
            clue_data: Dict[str, Any],
            support_frames: Set[int],
            distractor_support: Dict[str, Set[int]],
        ) -> None:
            support_frames &= target_frames
            if not support_frames:
                return
            norm_dist = {d_id: set(distractor_support.get(d_id, set())) for d_id in distractor_ids}
            excludes = {d_id for d_id in distractor_ids if not norm_dist[d_id]}
            key = json.dumps({"type": elem_type, "clue_data": clue_data}, ensure_ascii=False, sort_keys=True)
            if key in element_key_to_index:
                idx = element_key_to_index[key]
                element_pool[idx]["support_frames"] |= set(support_frames)
                for d_id in distractor_ids:
                    element_pool[idx]["distractor_support"][d_id] |= norm_dist[d_id]
                element_pool[idx]["excludes"] = {
                    d_id for d_id in distractor_ids if not element_pool[idx]["distractor_support"][d_id]
                }
                return
            element_key_to_index[key] = len(element_pool)
            element_pool.append(
                {
                    "type": elem_type,
                    "type_difficulty": TYPE_DIFFICULTY[elem_type],
                    "clue_data": clue_data,
                    "support_frames": set(support_frames),
                    "distractor_support": norm_dist,
                    "excludes": excludes,
                }
            )

        if not distractor_ids:
            add_element("attribute", {"attributes": _uniq_phrases(target.get("attributes", []))[:1]}, set(target_frames), {})

        for phrase in _uniq_phrases(target.get("attributes", [])):
            d_support = {
                d_id: set(object_frames.get(d_id, set()))
                if phrase.lower() in {x.lower() for x in _uniq_phrases(objects_by_id[d_id].get("attributes", []))}
                else set()
                for d_id in distractor_ids
            }
            add_element("attribute", {"attributes": [phrase]}, set(target_frames), d_support)

        for phrase in _uniq_phrases(target.get("relationships", [])):
            d_support = {
                d_id: set(object_frames.get(d_id, set()))
                if phrase.lower() in {x.lower() for x in _uniq_phrases(objects_by_id[d_id].get("relationships", []))}
                else set()
                for d_id in distractor_ids
            }
            add_element("context_relation", {"context_relations": [phrase]}, set(target_frames), d_support)

        for label, frames in target_action_frames.items():
            if len(frames) < 2:
                continue
            d_support = {d_id: distractor_action_frames[d_id].get(label, set()) for d_id in distractor_ids}
            add_element("action", {"actions": [label]}, set(frames), d_support)

        for (label, ref_id), frames in target_action_targets.items():
            if len(frames) < 2 or ref_id not in objects_by_id:
                continue
            ref_obj = objects_by_id[ref_id]
            ref_desc = _ref_desc(ref_obj)
            d_support: Dict[str, Set[int]] = {}
            for d_id in distractor_ids:
                match_frames: Set[int] = set()
                for (d_label, d_ref_id), d_frames in distractor_action_targets[d_id].items():
                    if d_label != label:
                        continue
                    if d_ref_id not in objects_by_id:
                        continue
                    d_ref = objects_by_id[d_ref_id]
                    if ref_desc["class"] != _norm(d_ref.get("object_class", "")).lower():
                        continue
                    if set(ref_desc["attributes"]) - set(_uniq_phrases(d_ref.get("attributes", []))):
                        continue
                    if set(ref_desc["context_relations"]) - set(_uniq_phrases(d_ref.get("relationships", []))):
                        continue
                    match_frames |= d_frames
                d_support[d_id] = match_frames
            add_element("action_target", {"action_targets": [{"action": label, "ref": ref_desc}]}, set(frames), d_support)

        for rel in relation_events:
            rel_type = rel.get("relationship_type", "")
            if not rel_type:
                continue
            frames = _segments_to_frames(rel["segments"]) & object_frames.get(rel["subject_id"], set())
            all_action_frames.setdefault(rel["subject_id"], {}).setdefault(rel["predicate"], set()).update(frames)
            all_action_targets.setdefault(rel["subject_id"], {}).setdefault((rel["predicate"], rel["object_id"]), set()).update(frames)
            if rel["subject_id"] == target_id:
                target_action_frames.setdefault(rel["predicate"], set()).update(frames)
                target_action_targets.setdefault((rel["predicate"], rel["object_id"]), set()).update(frames)
            elif rel["subject_id"] in distractor_id_set:
                distractor_action_frames[rel["subject_id"]].setdefault(rel["predicate"], set()).update(frames)
                distractor_action_targets[rel["subject_id"]].setdefault((rel["predicate"], rel["object_id"]), set()).update(frames)

        for rel in relation_events:
            if rel["subject_id"] != target_id or rel["object_id"] not in objects_by_id:
                continue
            ref_obj = objects_by_id[rel["object_id"]]
            ref_desc = _ref_desc(ref_obj)
            elem_type = "relation_to_unique" if same_class_count.get(rel["object_id"], 0) == 1 else "relation_to_described"
            support = _segments_to_frames(rel["segments"]) & target_frames
            if len(support) < 2:
                continue
            d_support: Dict[str, Set[int]] = {d_id: set() for d_id in distractor_ids}
            for d_id in distractor_ids:
                for e in relation_events:
                    if e["subject_id"] != d_id:
                        continue
                    if e["predicate"] != rel["predicate"]:
                        continue
                    d_ref = objects_by_id.get(e["object_id"])
                    if not d_ref:
                        continue
                    if ref_desc["class"] != _norm(d_ref.get("object_class", "")).lower():
                        continue
                    if set(ref_desc["attributes"]) - set(_uniq_phrases(d_ref.get("attributes", []))):
                        continue
                    if set(ref_desc["context_relations"]) - set(_uniq_phrases(d_ref.get("relationships", []))):
                        continue
                    d_support[d_id] |= _segments_to_frames(e["segments"]) & object_frames.get(d_id, set())
            add_element(elem_type, {"relations": [{"relation": rel["predicate"], "ref": ref_desc}]}, support, d_support)

        labels = sorted(target_action_frames)
        for a1, a2 in combinations(labels, 2):
            f1 = target_action_frames.get(a1, set())
            f2 = target_action_frames.get(a2, set())
            if not f1 or not f2 or max(f1) >= min(f2):
                continue
            support = f1 | f2
            if len(support) < 4:
                continue
            d_support: Dict[str, Set[int]] = {}
            for d_id in distractor_ids:
                df1 = distractor_action_frames[d_id].get(a1, set())
                df2 = distractor_action_frames[d_id].get(a2, set())
                if not df1 or not df2 or max(df1) >= min(df2):
                    d_support[d_id] = set()
                else:
                    d_support[d_id] = df1 | df2
            add_element(
                "action_sequence",
                {"action_sequences": [{"first": a1, "second": a2, "order": "before", "cross_shot": False}]},
                support,
                d_support,
            )

        for ref_id in sorted(same_entity_map.get(target_id, set())):
            ref_obj = objects_by_id.get(ref_id)
            if not ref_obj:
                continue
            if set(target.get("shot_ids", [])) & set(ref_obj.get("shot_ids", [])):
                continue
            ref_actions = all_action_frames.get(ref_id, {})
            for first_label, first_frames in ref_actions.items():
                if len(first_frames) < 2:
                    continue
                for second_label, second_frames in target_action_frames.items():
                    if first_label == second_label or len(second_frames) < 2:
                        continue
                    if max(first_frames) >= min(second_frames):
                        continue
                    d_support: Dict[str, Set[int]] = {}
                    for d_id in distractor_ids:
                        d_frames = set()
                        second_match = distractor_action_frames[d_id].get(second_label, set())
                        if second_match:
                            for d_ref_id in same_entity_map.get(d_id, set()):
                                d_ref_obj = objects_by_id.get(d_ref_id)
                                if not d_ref_obj:
                                    continue
                                if set(objects_by_id[d_id].get("shot_ids", [])) & set(d_ref_obj.get("shot_ids", [])):
                                    continue
                                first_match = all_action_frames.get(d_ref_id, {}).get(first_label, set())
                                if first_match and max(first_match) < min(second_match):
                                    d_frames |= second_match
                        d_support[d_id] = d_frames
                    add_element(
                        "cross_shot",
                        {
                            "action_sequences": [
                                {"first": first_label, "second": second_label, "order": "before", "cross_shot": True}
                            ],
                            "same_entity_nodes": [target_id, ref_id],
                        },
                        set(second_frames),
                        d_support,
                    )

        if not element_pool:
            fallback = self._fallback_plan(target_obj, d_star)
            return [fallback] if return_all_candidates else fallback

        valid_combinations: List[Dict[str, Any]] = []
        max_k = min(self.max_combo_size, len(element_pool))

        def combo_support_frames(combo: List[Dict[str, Any]]) -> Set[int]:
            frames: Optional[Set[int]] = None
            for elem in combo:
                frames = set(elem["support_frames"]) if frames is None else (frames & elem["support_frames"])
                if not frames:
                    return set()
            return frames or set()

        def distractor_combo_frames(combo: List[Dict[str, Any]], d_id: str) -> Set[int]:
            frames: Optional[Set[int]] = None
            for elem in combo:
                d_frames = elem["distractor_support"].get(d_id, set())
                frames = set(d_frames) if frames is None else (frames & d_frames)
                if not frames:
                    return set()
            return frames or set()

        for k in range(1, max_k + 1):
            for combo_tpl in combinations(element_pool, k):
                combo = list(combo_tpl)
                support = combo_support_frames(combo)
                if not support:
                    continue
                segments = _frames_to_segments(support)
                if not segments:
                    continue
                unique_segments: List[Tuple[int, int]] = []
                for a, b in segments:
                    seg_frames = set(range(a, b + 1))
                    unique = True
                    for d_id in distractor_ids:
                        if distractor_combo_frames(combo, d_id) & seg_frames:
                            unique = False
                            break
                    if unique:
                        unique_segments.append((a, b))
                if unique_segments:
                    valid_combinations.append({"combo": combo, "segments": unique_segments})
            if len(valid_combinations) >= 50:
                break

        if not valid_combinations:
            best_fallback: Optional[Dict[str, Any]] = None
            for k in range(1, max_k + 1):
                for combo_tpl in combinations(element_pool, k):
                    combo = list(combo_tpl)
                    support = combo_support_frames(combo)
                    if not support:
                        continue
                    covered = len(set().union(*(elem["excludes"] for elem in combo)))
                    item = {"combo": combo, "segments": _frames_to_segments(support), "covered": covered, "support_len": len(support)}
                    if not best_fallback or (item["covered"], item["support_len"]) > (best_fallback["covered"], best_fallback["support_len"]):
                        best_fallback = item
            if best_fallback and best_fallback["segments"]:
                valid_combinations.append({"combo": best_fallback["combo"], "segments": best_fallback["segments"]})

        scored_candidates: List[Dict[str, Any]] = []
        max_type_score = max(TYPE_DIFFICULTY.values())
        for item in valid_combinations:
            combo = item["combo"]
            temporal_count = sum(1 for elem in combo if elem["type"] in {"action", "action_target", "relation_to_unique", "relation_to_described"})
            if any(elem["type"] == "action_sequence" for elem in combo):
                temporal_count += 2
            if any(elem["type"] == "cross_shot" for elem in combo):
                temporal_count += 3
            d_t = min(temporal_count / self.n_action_cap, 1.0)
            avg_type_diff = sum(int(elem["type_difficulty"]) for elem in combo) / max(len(combo), 1)
            d_type = avg_type_diff / max_type_score if max_type_score > 0 else 0.0
            d_s = self.w1 * d_type + self.w2 * c_tiou + self.w3 * c_bg + self.w4 * c_density
            d = self.lam * d_t + (1.0 - self.lam) * d_s
            for seg_start, seg_end in item["segments"]:
                scored_candidates.append(
                    {
                        "combo": combo,
                        "segment": (seg_start, seg_end),
                        "segment_len": seg_end - seg_start + 1,
                        "D_t": d_t,
                        "D_s": d_s,
                        "D": d,
                    }
                )

        if not scored_candidates:
            fallback = self._fallback_plan(target_obj, d_star)
            return [fallback] if return_all_candidates else fallback

        scored_candidates.sort(key=lambda x: (abs(x["D"] - d_star), -x["segment_len"], len(x["combo"])))

        plans: List[Dict[str, Any]] = []
        seen_signatures: Set[str] = set()
        for item in scored_candidates:
            combo = item["combo"]
            seg_start, seg_end = item["segment"]
            clue_types = sorted({elem["type"] for elem in combo})
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
                "cross_shot": False,
                "same_entity_nodes": [target_id],
            }
            for elem in combo:
                clue = elem["clue_data"]
                spec["attributes"].extend(clue.get("attributes", []))
                spec["context_relations"].extend(clue.get("context_relations", []))
                spec["actions"].extend(clue.get("actions", []))
                spec["action_targets"].extend(clue.get("action_targets", []))
                spec["action_sequences"].extend(clue.get("action_sequences", []))
                spec["relations"].extend(clue.get("relations", []))
                for same_id in clue.get("same_entity_nodes", []) or []:
                    same_id = str(same_id).strip()
                    if same_id and same_id not in spec["same_entity_nodes"]:
                        spec["same_entity_nodes"].append(same_id)
                if clue.get("action_sequences"):
                    spec["cross_shot"] = spec["cross_shot"] or bool(clue["action_sequences"][0].get("cross_shot", False))
            spec = _ordered_query_spec(spec)

            if "cross_shot" in clue_types:
                type_bucket = "cross_shot"
            elif "action_sequence" in clue_types:
                type_bucket = "action_sequence"
            elif "action_target" in clue_types:
                type_bucket = "action_target"
            elif "relation_to_unique" in clue_types or "relation_to_described" in clue_types:
                type_bucket = "relation"
            elif "action" in clue_types:
                type_bucket = "action"
            elif "context_relation" in clue_types:
                type_bucket = "context_relation"
            else:
                type_bucket = "attribute"

            if "cross_shot" in clue_types:
                difficulty_bucket = "very_hard"
            elif item["D"] < 0.25:
                difficulty_bucket = "easy"
            elif item["D"] < 0.5:
                difficulty_bucket = "medium"
            elif item["D"] < 0.75:
                difficulty_bucket = "hard"
            else:
                difficulty_bucket = "very_hard"

            signature = json.dumps({"target_node": target_id, "query_spec": spec}, ensure_ascii=False, sort_keys=True)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            plans.append(
                {
                    "query_spec": spec,
                    "D_t": item["D_t"],
                    "D_s": item["D_s"],
                    "D": item["D"],
                    "d_star": float(d_star),
                    "target_node": target_id,
                    "segment_len": item["segment_len"],
                    "clue_types": clue_types,
                    "type_bucket": type_bucket,
                    "difficulty_bucket": difficulty_bucket,
                    "candidate_signature": signature,
                }
            )
            if not return_all_candidates:
                break
            if max_candidates and len(plans) >= max_candidates:
                break

        if return_all_candidates:
            return plans or [self._fallback_plan(target_obj, d_star)]
        return plans[0] if plans else self._fallback_plan(target_obj, d_star)

    def _fallback_plan(self, target_obj: Dict[str, Any], d_star: float) -> Dict[str, Any]:
        nid = str(target_obj.get("node_id", ""))
        cls = _norm(target_obj.get("object_class", "object")).lower() or "object"
        s = _to_int(target_obj.get("start_frame"), 0)
        e = _to_int(target_obj.get("end_frame"), s)
        if e < s:
            s, e = e, s
        spec = _ordered_query_spec(
            {
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
        )
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
        total_frames: Optional[int] = None,
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
        difficulty_bins: int = 4,
        max_queries_per_target: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        _ = total_frames
        candidates = [obj for obj in graph.get("object_nodes", []) if obj.get("node_id") and obj.get("bboxes")]
        if not candidates:
            return []

        if target_node_ids:
            if isinstance(target_node_ids, str):
                target_set = {x.strip() for x in target_node_ids.split(",") if x.strip()}
            else:
                target_set = {str(x).strip() for x in target_node_ids if str(x).strip()}
            candidates = [obj for obj in candidates if obj.get("node_id") in target_set]

        if sample_size and sample_size > 0 and len(candidates) > sample_size:
            candidates = random.Random(seed).sample(candidates, sample_size)

        d_star_values: List[float] = []
        if d_star_list:
            raw = [x.strip() for x in d_star_list.split(",")] if isinstance(d_star_list, str) else [str(x).strip() for x in d_star_list]
            for value in raw:
                try:
                    d_star_values.append(min(max(float(value), 0.0), 1.0))
                except Exception:
                    pass
        if not d_star_values:
            d_star_values = [min(max(float(d_star), 0.0), 1.0)]

        all_plans: List[Dict[str, Any]] = []
        global_seen_signatures: Set[str] = set()
        for obj in candidates:
            target_candidates = self.build_query_plan(
                graph=graph,
                target_obj=obj,
                d_star=d_star_values[0],
                return_all_candidates=True,
                max_candidates=max_candidates_per_target,
            )
            if not isinstance(target_candidates, list) or not target_candidates:
                continue

            if generate_all_candidates:
                selected_for_target = target_candidates
            elif len(d_star_values) > 1:
                selected_for_target = []
                local_seen: Set[str] = set()
                for d_value in d_star_values:
                    best = min(
                        target_candidates,
                        key=lambda item: (
                            abs(float(item.get("D", 0.0)) - d_value),
                            -int(item.get("segment_len", 1)),
                            len(item.get("clue_types", [])),
                        ),
                    )
                    signature = str(best.get("candidate_signature", ""))
                    if signature in local_seen:
                        continue
                    local_seen.add(signature)
                    selected_for_target.append({**best, "d_star": d_value})
                selected_for_target = selected_for_target[: max(per_target_limit, 1)]
            else:
                selected_for_target = target_candidates[: max(per_target_limit, 1)]

            for plan in selected_for_target:
                signature = str(plan.get("candidate_signature", ""))
                if signature and signature in global_seen_signatures:
                    continue
                if signature:
                    global_seen_signatures.add(signature)
                item = dict(plan)
                item["plan_id"] = f"plan_{item.get('target_node')}_{len(all_plans)}_{int(float(item.get('D', 0.0)) * 1000)}"
                all_plans.append(item)

        if not queries_per_graph or queries_per_graph <= 0 or len(all_plans) <= queries_per_graph:
            return all_plans

        rng = random.Random(seed)
        n_bins = max(int(difficulty_bins), 1)
        groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
        for plan in all_plans:
            d_value = min(max(float(plan.get("D", 0.0)), 0.0), 1.0)
            bin_idx = min(int(d_value * n_bins), n_bins - 1) if balance_difficulty else 0
            type_key = str(plan.get("type_bucket", "attribute")) if balance_types else "all"
            groups.setdefault((bin_idx, type_key), []).append(plan)

        for key in groups:
            groups[key].sort(
                key=lambda item: (
                    abs(float(item.get("D", 0.0)) - d_star),
                    -int(item.get("segment_len", 1)),
                    len(item.get("clue_types", [])),
                )
            )

        selected: List[Dict[str, Any]] = []
        target_counter: Dict[str, int] = {}
        keys = list(groups.keys())
        while len(selected) < queries_per_graph:
            rng.shuffle(keys)
            moved = False
            for key in keys:
                bucket = groups[key]
                while bucket:
                    item = bucket.pop(0)
                    target_node = str(item.get("target_node", ""))
                    if max_queries_per_target and target_counter.get(target_node, 0) >= max_queries_per_target:
                        continue
                    selected.append(item)
                    target_counter[target_node] = target_counter.get(target_node, 0) + 1
                    moved = True
                    break
                if len(selected) >= queries_per_graph:
                    break
            if not moved:
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
                "id": str(plan.get("plan_id") or plan.get("target_node")),
                "prompt": PROMPT_TEMPLATE.format(
                    d_star=plan.get("d_star"),
                    spec_json=json.dumps(plan.get("query_spec", {}), ensure_ascii=False),
                ),
            }
            for plan in plans
        ]

        query_map: Dict[str, str] = {}
        async for item in generator.generate_stream(prompts=prompts, system_prompt=system_prompt, validate_func=_extract_query):
            if item and isinstance(item.get("result"), str) and item["result"].strip():
                query_map[str(item.get("id"))] = item["result"].strip()
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
                action = item.get("action")
                ref = item.get("ref", {})
                if action and isinstance(ref, dict):
                    chunks.append(f"that {action} the {ref.get('class', 'object')}")
        for item in query_spec.get("action_sequences", []):
            if isinstance(item, dict) and item.get("first") and item.get("second"):
                chunks.append(f"that {item['first']} before {item['second']}")
        for item in query_spec.get("relations", []):
            if isinstance(item, dict):
                rel = item.get("relation")
                ref = item.get("ref", {})
                if rel and isinstance(ref, dict):
                    chunks.append(f"that is {rel} the {ref.get('class', 'object')}")
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
        difficulty_bins: int = 4,
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
            difficulty_bins=difficulty_bins,
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
            node = _ordered_query_node(
                {
                    "query_id": f"query_{target}_{i}_{pid}",
                    "target_node_id": target,
                    "query": query,
                    "query_spec": _ordered_query_spec(plan.get("query_spec") or {}),
                    "D_t": plan.get("D_t"),
                    "D_s": plan.get("D_s"),
                    "D": plan.get("D"),
                    "d_star": plan.get("d_star"),
                    "clue_types": plan.get("clue_types", []),
                    "type_bucket": plan.get("type_bucket"),
                    "difficulty_bucket": plan.get("difficulty_bucket"),
                }
            )
            query_nodes.append(node)
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
        difficulty_bins: int = 4,
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
                    difficulty_bins=difficulty_bins,
                    max_queries_per_target=max_queries_per_target,
                    use_llm=use_llm,
                    max_concurrent_per_key=max_concurrent_per_key,
                    max_retries=max_retries,
                    overwrite=overwrite,
                )
                fout.write(json.dumps(_ordered_graph_output(graph), ensure_ascii=False) + "\n")


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
    difficulty_bins: int = 4,
    max_queries_per_target: Optional[int] = None,
    use_llm: bool = True,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    overwrite: bool = True,
) -> None:
    generator = DifficultyAwareSTVGQueryGenerator()
    generator.process_jsonl(
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
        difficulty_bins=difficulty_bins,
        max_queries_per_target=max_queries_per_target,
        use_llm=use_llm,
        max_concurrent_per_key=max_concurrent_per_key,
        max_retries=max_retries,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(run)
