from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _norm_span(start: int, end: int) -> Tuple[int, int]:
    s = int(start)
    e = int(end)
    if e < s:
        s, e = e, s
    return s, e


def _span_len(span: Tuple[int, int]) -> int:
    return span[1] - span[0] + 1


def _normalize_text_for_merge(text: Any) -> str:
    cleaned = str(text).strip().lower().replace("_", " ").replace("-", " ")
    out_chars: List[str] = []
    for ch in cleaned:
        out_chars.append(ch if ch.isalnum() or ch.isspace() else " ")
    return " ".join("".join(out_chars).split())


REL_KEY_REGEX_PATTERNS: Dict[str, List[str]] = {
    "watch": [
        r"\bwatch(?:ing)?\b",
        r"\blook(?:ing)?\s*at\b",
        r"lookat",
    ],
    "listen to": [r"\blisten(?:ing)?\s*to\b"],
    "talk to": [r"\btalk(?:ing)?\s*to\b", r"\bspeak(?:ing)?\s*to\b", r"\bchat(?:ting)?\s*with\b"],
    "carry hold": [r"\bcarry(?:ing)?\b", r"\bhold(?:ing)?\b"],
    "touch": [r"\btouch(?:ing)?\b"],
    "push": [r"\bpush(?:ing)?\b"],
    "pull": [r"\bpull(?:ing)?\b"],
    "point to": [r"\bpoint(?:ing)?\s*(?:to|at)\b"],
    "hit": [r"\bhit(?:ting)?\b", r"\bstrike\b"],
    "fight": [r"\bfight(?:ing)?\b"],
    "give": [r"\bgive\b", r"\bgiving\b", r"\bserve\b"],
    "hand": [r"\bhand\b", r"\bhand\s*over\b"],
    "grab": [r"\bgrab(?:bing)?\b"],
    "hand shake": [r"\bhand\s*shake\b", r"\bshake\s*hands?\b"],
    "hug": [r"\bhug(?:ging)?\b"],
    "kiss": [r"\bkiss(?:ing)?\b"],
    "lift": [r"\blift(?:ing)?\b", r"\bpick\s*up\b"],
    "sing to": [r"\bsing(?:ing)?\s*to\b"],
    "take from": [r"\btak(?:e|ing)\s+.*\s+from\b"],
    "ride": [r"\bride\b", r"\briding\b"],
    "take a photo": [r"\btake\b.*\bphoto\b", r"\bphotograph\b"],
    "work on": [r"\bwork(?:ing)?\s*on\b"],
    "use": [r"\buse\b", r"\busing\b"],
    "answer phone": [r"\banswer(?:ing)?\s*phone\b"],
    "look at": [r"\blook(?:ing)?\s*at\b", r"lookat"],
}


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _match_rel_keys_by_regex(text: str) -> List[str]:
    norm = _normalize_text_for_merge(text)
    compact = _compact_text(text)
    matched: List[str] = []
    for key, patterns in REL_KEY_REGEX_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, norm) or re.search(pat, compact):
                matched.append(key)
                break
    return matched


def _merge_action_key(label: Any) -> str:
    text = _normalize_text_for_merge(label)
    if not text:
        return ""
    regex_keys = _match_rel_keys_by_regex(text)
    if regex_keys:
        key = regex_keys[0]
        if key == "look at":
            return "watch"
        return key
    return text


def _merge_spans(spans: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ordered = sorted(_norm_span(s, e) for s, e in spans)
    if not ordered:
        return []
    merged: List[Tuple[int, int]] = [ordered[0]]
    for s, e in ordered[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _spans_to_dicts(spans: List[Tuple[int, int]]) -> List[Dict[str, int]]:
    return [{"start_frame": s, "end_frame": e} for s, e in spans]


def _build_valid_node_ids(object_nodes: List[Dict[str, Any]]) -> Set[str]:
    valid_node_ids: Set[str] = set()
    for obj in object_nodes:
        node_id = str(obj.get("node_id", "")).strip()
        if node_id:
            valid_node_ids.add(node_id)
    return valid_node_ids


def _resolve_node_id(raw_id: Any, *, valid_node_ids: Set[str]) -> Optional[str]:
    node_id = str(raw_id).strip()
    if node_id in valid_node_ids:
        return node_id
    return None


class ObjectNodeFilter:
    def __init__(self, min_frames: int = 5) -> None:
        self.min_frames = int(min_frames)

    def should_keep(self, obj_node: Dict[str, Any]) -> bool:
        bboxes = obj_node.get("bboxes", {})
        total_frames = len(bboxes) if isinstance(bboxes, dict) and bboxes else 0
        if total_frames <= 0:
            start = _to_int(obj_node.get("start_frame"), 0)
            end = _to_int(obj_node.get("end_frame"), start)
            start, end = _norm_span(start, end)
            total_frames = end - start + 1

        if total_frames < self.min_frames:
            print(
                f"    Filtered {obj_node.get('node_id', '<unknown>')}: "
                f"too few frames ({total_frames} < {self.min_frames})"
            )
            return False
        return True

    def normalize_nodes(self, object_nodes: List[Dict[str, Any]], *, filter_objects: bool) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for obj in object_nodes:
            if not isinstance(obj, dict):
                continue

            node_id = str(obj.get("node_id", "")).strip()
            if not node_id:
                continue

            item = dict(obj)
            item["node_id"] = node_id

            raw_bboxes = item.get("bboxes")
            normalized_bboxes: Dict[str, Any] = {}
            if isinstance(raw_bboxes, dict):
                for key, value in raw_bboxes.items():
                    frame_idx = _to_int(key, -1)
                    if frame_idx < 0:
                        continue
                    normalized_bboxes[str(frame_idx)] = value
            if normalized_bboxes:
                ordered_frames = sorted(int(key) for key in normalized_bboxes.keys())
                item["bboxes"] = {str(frame): normalized_bboxes[str(frame)] for frame in ordered_frames}

            frame_keys = [int(key) for key in (item.get("bboxes") or {}).keys()]
            if frame_keys:
                item["start_frame"] = min(frame_keys)
                item["end_frame"] = max(frame_keys)
            else:
                start = _to_int(item.get("start_frame"), 0)
                end = _to_int(item.get("end_frame"), start)
                item["start_frame"], item["end_frame"] = _norm_span(start, end)

            if filter_objects and not self.should_keep(item):
                continue
            normalized.append(item)
        return normalized


class SameEntityObjectMerger:
    def merge(self, graph: Dict[str, Any]) -> None:
        object_nodes = graph.get("object_nodes") or []
        if not object_nodes:
            return

        obj_by_id: Dict[str, Dict[str, Any]] = {}
        for obj in object_nodes:
            node_id = str(obj.get("node_id", "")).strip()
            if node_id:
                obj_by_id[node_id] = obj
        if not obj_by_id:
            return

        parent: Dict[str, str] = {node_id: node_id for node_id in obj_by_id.keys()}

        def find(node_id: str) -> str:
            cur = parent.get(node_id, node_id)
            while cur != parent.get(cur, cur):
                cur = parent[cur]
            while node_id != cur:
                nxt = parent[node_id]
                parent[node_id] = cur
                node_id = nxt
            return cur

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for edge in graph.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            if str(edge.get("reference_relationship", "")).strip() != "same_entity":
                continue
            source_id = str(edge.get("source_id", "")).strip()
            target_id = str(edge.get("target_id", "")).strip()
            if source_id in obj_by_id and target_id in obj_by_id and source_id != target_id:
                union(source_id, target_id)

        components: Dict[str, List[str]] = {}
        for node_id in obj_by_id.keys():
            components.setdefault(find(node_id), []).append(node_id)

        def _node_sort_key(node_id: str) -> Tuple[int, int, str]:
            obj = obj_by_id[node_id]
            start = _to_int(obj.get("start_frame"), 0)
            global_track_id = _to_int(obj.get("global_track_id"), 10**9)
            return (start, global_track_id, node_id)

        remap: Dict[str, str] = {}
        merged_nodes: List[Dict[str, Any]] = []

        for members in components.values():
            members_sorted = sorted(members, key=_node_sort_key)
            canonical = members_sorted[0]
            for member_id in members_sorted:
                remap[member_id] = canonical

            base = dict(obj_by_id[canonical])
            base["node_id"] = canonical

            global_track_ids = [
                _to_int(obj_by_id[member_id].get("global_track_id"), -1)
                for member_id in members_sorted
                if _to_int(obj_by_id[member_id].get("global_track_id"), -1) >= 0
            ]
            if global_track_ids:
                base["global_track_id"] = min(global_track_ids)

            for field in ("object_class", "dam_category"):
                if str(base.get(field, "")).strip():
                    continue
                for member_id in members_sorted:
                    value = str(obj_by_id[member_id].get(field, "")).strip()
                    if value:
                        base[field] = value
                        break

            merged_bboxes: Dict[str, Any] = {}
            for member_id in members_sorted:
                bboxes = obj_by_id[member_id].get("bboxes") or {}
                if not isinstance(bboxes, dict):
                    continue
                for frame_key, box in bboxes.items():
                    frame_idx = _to_int(frame_key, -1)
                    if frame_idx < 0:
                        continue
                    normalized_key = str(frame_idx)
                    if normalized_key not in merged_bboxes:
                        merged_bboxes[normalized_key] = box

            if merged_bboxes:
                ordered_frames = sorted(int(key) for key in merged_bboxes.keys())
                base["bboxes"] = {str(frame): merged_bboxes[str(frame)] for frame in ordered_frames}
                base["start_frame"] = ordered_frames[0]
                base["end_frame"] = ordered_frames[-1]
            else:
                starts = [_to_int(obj_by_id[member_id].get("start_frame"), 0) for member_id in members_sorted]
                ends = [
                    _to_int(obj_by_id[member_id].get("end_frame"), starts[idx])
                    for idx, member_id in enumerate(members_sorted)
                ]
                if starts and ends:
                    base["start_frame"] = min(starts)
                    base["end_frame"] = max(ends)

            shot_ids: Set[int] = set()
            for member_id in members_sorted:
                for shot_id in obj_by_id[member_id].get("shot_ids") or []:
                    shot_ids.add(_to_int(shot_id, -1))
            shot_ids.discard(-1)
            if shot_ids:
                base["shot_ids"] = sorted(shot_ids)

            for field in ("attributes", "environment", "actions"):
                seen: Set[str] = set()
                merged_values: List[str] = []
                for member_id in members_sorted:
                    for value in obj_by_id[member_id].get(field) or []:
                        text = str(value).strip()
                        if not text:
                            continue
                        lowered = text.lower()
                        if lowered in seen:
                            continue
                        seen.add(lowered)
                        merged_values.append(text)
                if merged_values:
                    base[field] = merged_values
                elif field in base:
                    base[field] = []

            merged_nodes.append(base)

        remapped_action_nodes: List[Dict[str, Any]] = []
        for group in graph.get("action_nodes") or []:
            if not isinstance(group, dict):
                continue
            owner = remap.get(str(group.get("object_id", "")).strip(), str(group.get("object_id", "")).strip())
            if owner not in obj_by_id:
                continue

            new_group = dict(group)
            new_group["object_id"] = owner
            actions: List[Dict[str, Any]] = []
            for action in group.get("actions") or []:
                if not isinstance(action, dict):
                    continue
                new_action = dict(action)
                targets: List[str] = []
                for target_id in action.get("target_object_ids") or []:
                    remapped_target = remap.get(str(target_id).strip(), str(target_id).strip())
                    if remapped_target and remapped_target != owner:
                        targets.append(remapped_target)
                if targets:
                    new_action["target_object_ids"] = sorted(set(targets))
                else:
                    new_action.pop("target_object_ids", None)
                actions.append(new_action)
            new_group["actions"] = actions
            remapped_action_nodes.append(new_group)

        remapped_edges: List[Dict[str, Any]] = []
        for edge in graph.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            if str(edge.get("reference_relationship", "")).strip() == "same_entity":
                continue

            if "subject_id" in edge and "relationships" in edge:
                subject_id = remap.get(str(edge.get("subject_id", "")).strip(), str(edge.get("subject_id", "")).strip())
                if not subject_id:
                    continue
                relationships: List[Dict[str, Any]] = []
                for rel in edge.get("relationships") or []:
                    if not isinstance(rel, dict):
                        continue
                    object_id = remap.get(str(rel.get("object_id", "")).strip(), str(rel.get("object_id", "")).strip())
                    if not object_id or object_id == subject_id:
                        continue
                    new_rel = dict(rel)
                    new_rel["object_id"] = object_id
                    relationships.append(new_rel)
                if relationships:
                    remapped_edges.append({"subject_id": subject_id, "relationships": relationships})
                continue

            if "source_id" in edge and "target_id" in edge:
                source_id = remap.get(str(edge.get("source_id", "")).strip(), str(edge.get("source_id", "")).strip())
                target_id = remap.get(str(edge.get("target_id", "")).strip(), str(edge.get("target_id", "")).strip())
                if not source_id or not target_id or source_id == target_id:
                    continue
                new_edge = dict(edge)
                new_edge["source_id"] = source_id
                new_edge["target_id"] = target_id
                remapped_edges.append(new_edge)
                continue

            remapped_edges.append(edge)

        merged_nodes.sort(key=lambda item: (_to_int(item.get("start_frame"), 0), str(item.get("node_id", ""))))
        graph["object_nodes"] = merged_nodes
        graph["action_nodes"] = remapped_action_nodes
        graph["edges"] = remapped_edges


class RelationEdgeFilter:
    def __init__(self, min_relation_frames: int = 3) -> None:
        self.min_relation_frames = int(min_relation_frames)

    def _parse_time_spans(self, item: Dict[str, Any]) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []

        raw_spans = item.get("time_spans")
        if isinstance(raw_spans, list):
            for span in raw_spans:
                if not isinstance(span, dict):
                    continue
                start = _to_int(span.get("start_frame"), 0)
                end = _to_int(span.get("end_frame"), start)
                spans.append(_norm_span(start, end))

        if "start_frame" in item or "end_frame" in item:
            start = _to_int(item.get("start_frame"), 0)
            end = _to_int(item.get("end_frame"), start)
            spans.append(_norm_span(start, end))

        return _merge_spans(spans)

    def normalize_edges(
        self,
        graph: Dict[str, Any],
        *,
        valid_node_ids: Set[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        relation_edge_groups: List[Dict[str, Any]] = []
        passthrough_edges: List[Dict[str, Any]] = []
        motion_actions: Dict[str, List[Dict[str, Any]]] = {}

        for edge in graph.get("edges") or []:
            if not isinstance(edge, dict):
                continue

            if "subject_id" in edge and "relationships" in edge:
                subject_id = _resolve_node_id(edge.get("subject_id"), valid_node_ids=valid_node_ids)
                if not subject_id:
                    continue

                relationships: List[Dict[str, Any]] = []
                for rel in edge.get("relationships") or []:
                    if not isinstance(rel, dict):
                        continue

                    predicate = str(rel.get("predicate_verb", "")).strip()
                    if not predicate:
                        continue

                    object_id = _resolve_node_id(rel.get("object_id"), valid_node_ids=valid_node_ids)
                    if not object_id or object_id == subject_id:
                        continue

                    spans = [
                        span
                        for span in self._parse_time_spans(rel)
                        if _span_len(span) >= self.min_relation_frames
                    ]
                    if not spans:
                        continue

                    relationship_type = str(rel.get("relationship_type", "")).strip().lower()
                    edge_type = str(rel.get("edge_type", "")).strip().lower() or "spatial"

                    if relationship_type == "motion":
                        for start, end in spans:
                            motion_actions.setdefault(subject_id, []).append(
                                {
                                    "action_label": predicate,
                                    "frame_idx": start,
                                    "start_frame": start,
                                    "end_frame": end,
                                    "_source": "motion",
                                    "target_object_ids": [object_id],
                                }
                            )
                        continue

                    out_rel = {
                        "object_id": object_id,
                        "predicate_verb": predicate,
                        "edge_type": edge_type,
                        "time_spans": _spans_to_dicts(spans),
                    }
                    if relationship_type:
                        out_rel["relationship_type"] = relationship_type
                    relationships.append(out_rel)

                if relationships:
                    relationships.sort(
                        key=lambda item: (
                            str(item.get("edge_type", "")),
                            str(item.get("predicate_verb", "")),
                            str(item.get("object_id", "")),
                        )
                    )
                    relation_edge_groups.append({"subject_id": subject_id, "relationships": relationships})
                continue

            if "reference_relationship" in edge:
                source_id = _resolve_node_id(edge.get("source_id"), valid_node_ids=valid_node_ids)
                target_id = _resolve_node_id(edge.get("target_id"), valid_node_ids=valid_node_ids)
                if not source_id or not target_id or source_id == target_id:
                    continue
                normalized_edge = dict(edge)
                normalized_edge["source_id"] = source_id
                normalized_edge["target_id"] = target_id
                passthrough_edges.append(normalized_edge)
                continue

            source_id = edge.get("source_id")
            target_id = edge.get("target_id")
            if source_id is not None and target_id is not None:
                normalized_source = _resolve_node_id(source_id, valid_node_ids=valid_node_ids)
                normalized_target = _resolve_node_id(target_id, valid_node_ids=valid_node_ids)
                if not normalized_source or not normalized_target or normalized_source == normalized_target:
                    continue
                normalized_edge = dict(edge)
                normalized_edge["source_id"] = normalized_source
                normalized_edge["target_id"] = normalized_target
                passthrough_edges.append(normalized_edge)

        relation_edge_groups.sort(key=lambda item: str(item.get("subject_id", "")))
        return relation_edge_groups + passthrough_edges, motion_actions


class ActionNodeFilter:
    def __init__(self, min_action_frames: int = 3) -> None:
        self.min_action_frames = int(min_action_frames)

    def _parse_time_spans(self, item: Dict[str, Any]) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []

        raw_spans = item.get("time_spans")
        if isinstance(raw_spans, list):
            for span in raw_spans:
                if not isinstance(span, dict):
                    continue
                start = _to_int(span.get("start_frame"), 0)
                end = _to_int(span.get("end_frame"), start)
                spans.append(_norm_span(start, end))

        if "start_frame" in item or "end_frame" in item:
            start = _to_int(item.get("start_frame"), 0)
            end = _to_int(item.get("end_frame"), start)
            spans.append(_norm_span(start, end))

        return _merge_spans(spans)

    def _dedup_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _parse_targets(raw: Any) -> List[str]:
            targets: List[str] = []
            for item in raw or []:
                if isinstance(item, str) and item.strip():
                    targets.append(item.strip())
            return sorted(set(targets))

        normalized: List[Dict[str, Any]] = []
        for item in actions:
            label = str(item.get("action_label", "")).strip()
            if not label:
                continue
            merge_key = _merge_action_key(label)
            if not merge_key:
                continue

            start = _to_int(item.get("start_frame"), 0)
            end = _to_int(item.get("end_frame"), start)
            start, end = _norm_span(start, end)
            if _span_len((start, end)) < self.min_action_frames:
                continue

            frame_idx = _to_int(item.get("frame_idx"), start)
            frame_idx = min(max(frame_idx, start), end)
            normalized.append(
                {
                    "action_label": label,
                    "merge_key": merge_key,
                    "start_frame": start,
                    "end_frame": end,
                    "frame_idx": frame_idx,
                    "target_object_ids": _parse_targets(item.get("target_object_ids")),
                    "source": "action" if str(item.get("_source", "")).strip() != "motion" else "motion",
                }
            )

        normalized.sort(key=lambda item: (item["merge_key"], item["start_frame"], item["end_frame"]))
        merged: List[Dict[str, Any]] = []

        for item in normalized:
            if not merged:
                merged.append(item)
                continue

            last = merged[-1]
            same_label = last["merge_key"] == item["merge_key"]
            overlap = item["start_frame"] <= last["end_frame"] and last["start_frame"] <= item["end_frame"]
            if not (same_label and overlap):
                item["has_action_source"] = item.get("source") == "action"
                if item["has_action_source"]:
                    item["action_start"] = item["start_frame"]
                    item["action_end"] = item["end_frame"]
                merged.append(item)
                continue

            last_has_action = bool(last.get("has_action_source")) or last.get("source") == "action"
            item_has_action = item.get("source") == "action"

            last_action_start = last.get("action_start")
            last_action_end = last.get("action_end")
            if (last_action_start is None or last_action_end is None) and last_has_action:
                last_action_start = last["start_frame"]
                last_action_end = last["end_frame"]

            if item_has_action:
                if last_action_start is None or last_action_end is None:
                    last_action_start = item["start_frame"]
                    last_action_end = item["end_frame"]
                else:
                    last_action_start = min(last_action_start, item["start_frame"])
                    last_action_end = max(last_action_end, item["end_frame"])

            if last_action_start is not None and last_action_end is not None:
                new_start = int(last_action_start)
                new_end = int(last_action_end)
            else:
                new_start = min(last["start_frame"], item["start_frame"])
                new_end = max(last["end_frame"], item["end_frame"])

            new_targets = sorted(set((last.get("target_object_ids") or []) + (item.get("target_object_ids") or [])))
            new_frame_idx = last["frame_idx"]
            if new_frame_idx < new_start or new_frame_idx > new_end:
                new_frame_idx = min(max(item["frame_idx"], new_start), new_end)

            last["start_frame"] = new_start
            last["end_frame"] = new_end
            last["frame_idx"] = new_frame_idx
            last["has_action_source"] = bool(last_has_action or item_has_action)
            last["action_start"] = last_action_start
            last["action_end"] = last_action_end
            if new_targets:
                last["target_object_ids"] = new_targets
            else:
                last.pop("target_object_ids", None)

        packed: List[Dict[str, Any]] = []
        for item in merged:
            action = {
                "action_label": item["action_label"],
                "frame_idx": item["frame_idx"],
                "start_frame": item["start_frame"],
                "end_frame": item["end_frame"],
            }
            targets = item.get("target_object_ids") or []
            if targets:
                action["target_object_ids"] = targets
            packed.append(action)

        packed.sort(key=lambda item: (item["start_frame"], item["end_frame"], item["action_label"]))
        return packed

    def normalize_nodes(
        self,
        graph: Dict[str, Any],
        *,
        valid_node_ids: Set[str],
        extra_actions: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        owner_to_actions: Dict[str, List[Dict[str, Any]]] = {}
        owner_to_group_node_id: Dict[str, str] = {}

        for group in graph.get("action_nodes") or []:
            if not isinstance(group, dict):
                continue

            owner_id = _resolve_node_id(group.get("object_id"), valid_node_ids=valid_node_ids)
            if not owner_id:
                continue

            group_id = str(group.get("node_id", "")).strip()
            if group_id:
                owner_to_group_node_id[owner_id] = group_id

            parsed_actions: List[Dict[str, Any]] = []
            for action in group.get("actions") or []:
                if not isinstance(action, dict):
                    continue

                label = str(action.get("action_label", "")).strip()
                if not label:
                    continue

                spans = self._parse_time_spans(action)
                if not spans and "frame_idx" in action:
                    frame_idx = _to_int(action.get("frame_idx"), -1)
                    if frame_idx >= 0:
                        spans = [(frame_idx, frame_idx)]
                if not spans:
                    continue

                raw_targets = action.get("target_object_ids") or []
                normalized_targets: List[str] = []
                if isinstance(raw_targets, list):
                    for raw_target in raw_targets:
                        target_id = _resolve_node_id(raw_target, valid_node_ids=valid_node_ids)
                        if target_id and target_id != owner_id:
                            normalized_targets.append(target_id)
                normalized_targets = sorted(set(normalized_targets))

                for start, end in spans:
                    parsed = {
                        "action_label": label,
                        "frame_idx": _to_int(action.get("frame_idx"), start),
                        "start_frame": start,
                        "end_frame": end,
                        "_source": "action",
                    }
                    if normalized_targets:
                        parsed["target_object_ids"] = normalized_targets
                    parsed_actions.append(parsed)

            owner_to_actions.setdefault(owner_id, []).extend(parsed_actions)

        if extra_actions:
            for owner_id, actions in extra_actions.items():
                if owner_id not in valid_node_ids:
                    continue
                owner_to_actions.setdefault(owner_id, []).extend(actions)

        action_nodes: List[Dict[str, Any]] = []
        for index, owner_id in enumerate(sorted(owner_to_actions.keys())):
            actions = self._dedup_actions(owner_to_actions[owner_id])
            if not actions:
                continue
            action_nodes.append(
                {
                    "node_id": owner_to_group_node_id.get(owner_id, f"action_group_{index}"),
                    "object_id": owner_id,
                    "actions": actions,
                }
            )
        return action_nodes


class QueryTrackFilter:
    def __init__(self, expand_non_temporal_tracks: bool = True) -> None:
        self.expand_non_temporal_tracks = bool(expand_non_temporal_tracks)

    def _load_track_map(self, scene_graph_path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
        track_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        with Path(scene_graph_path).open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                graph = json.loads(line)
                video_path = graph.get("video_path", "")
                for node in graph.get("object_nodes") or []:
                    object_id = str(node.get("node_id", "")).strip()
                    if not object_id:
                        continue
                    track_map[(video_path, object_id)] = {
                        "start_frame": node.get("start_frame"),
                        "end_frame": node.get("end_frame"),
                        "boxes": node.get("bboxes", {}) or {},
                    }
        return track_map

    def _should_expand_member(self, clue_item: Dict[str, Any]) -> bool:
        clues = clue_item.get("clues", []) or []
        return not any(bool(clue.get("is_temporal_evidence")) for clue in clues if isinstance(clue, dict))

    def expand_record_tracks(
        self,
        record: Dict[str, Any],
        *,
        track_map: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> bool:
        if not self.expand_non_temporal_tracks:
            return False

        changed = False
        video_path = record.get("video_path", "")
        clues_by_index = {
            item.get("target_index"): item
            for item in (record.get("clues", {}).get("per_target", []) or [])
            if isinstance(item, dict)
        }

        for member in record.get("target_members", []) or []:
            if not isinstance(member, dict):
                continue

            clue_item = clues_by_index.get(member.get("target_index"), {})
            if not self._should_expand_member(clue_item):
                continue

            object_id = str(member.get("object_id", "")).strip()
            full_track = track_map.get((video_path, object_id))
            if not full_track:
                continue

            new_start = full_track["start_frame"]
            new_end = full_track["end_frame"]
            new_boxes = full_track["boxes"]

            if (
                member.get("start_frame") == new_start
                and member.get("end_frame") == new_end
                and (member.get("boxes", {}) or {}) == new_boxes
            ):
                continue

            member["start_frame"] = new_start
            member["end_frame"] = new_end
            member["boxes"] = new_boxes
            if clue_item:
                clue_item["start_frame"] = new_start
                clue_item["end_frame"] = new_end
            changed = True

        return changed

    def expand_records_tracks(
        self,
        records: List[Dict[str, Any]],
        *,
        track_map: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> int:
        changed_count = 0
        for record in records:
            if self.expand_record_tracks(record, track_map=track_map):
                changed_count += 1
        return changed_count

    def filter_jsonl(self, input_path: str, scene_graph_path: str, output_path: str) -> None:
        track_map = self._load_track_map(scene_graph_path)
        input_file = Path(input_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        total_records = 0
        changed_records = 0
        with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                total_records += 1
                if self.expand_record_tracks(record, track_map=track_map):
                    changed_records += 1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"Query tracks normalized: records={total_records}, "
            f"expanded_non_temporal={changed_records}"
        )


class GraphFilter:
    def __init__(
        self,
        min_frames: int = 5,
        min_action_frames: int = 3,
        min_relation_frames: int = 3,
        expand_non_temporal_query_tracks: bool = True,
    ) -> None:
        self.object_filter = ObjectNodeFilter(min_frames=min_frames)
        self.same_entity_merger = SameEntityObjectMerger()
        self.relation_filter = RelationEdgeFilter(min_relation_frames=min_relation_frames)
        self.action_filter = ActionNodeFilter(min_action_frames=min_action_frames)
        self.query_filter = QueryTrackFilter(expand_non_temporal_tracks=expand_non_temporal_query_tracks)

    def should_keep(self, obj_node: Dict[str, Any]) -> bool:
        return self.object_filter.should_keep(obj_node)

    def normalize_graph(self, graph: Dict[str, Any], *, filter_objects: bool = False) -> Dict[str, Any]:
        graph["object_nodes"] = self.object_filter.normalize_nodes(
            graph.get("object_nodes") or [],
            filter_objects=False,
        )
        self.same_entity_merger.merge(graph)

        if filter_objects:
            filtered_nodes: List[Dict[str, Any]] = []
            for obj in graph.get("object_nodes") or []:
                if self.object_filter.should_keep(obj):
                    filtered_nodes.append(obj)
            graph["object_nodes"] = filtered_nodes

        valid_node_ids = _build_valid_node_ids(graph.get("object_nodes") or [])
        edges, motion_actions = self.relation_filter.normalize_edges(
            graph,
            valid_node_ids=valid_node_ids,
        )
        graph["edges"] = edges
        graph["action_nodes"] = self.action_filter.normalize_nodes(
            graph,
            valid_node_ids=valid_node_ids,
            extra_actions=motion_actions,
        )
        return graph

    def filter_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        return self.normalize_graph(graph, filter_objects=True)

    def filter_jsonl(self, input_path: str, output_path: str) -> None:
        input_file = Path(input_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        total_objects_before = 0
        total_objects_after = 0
        with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                graph = json.loads(line)
                total_objects_before += len(graph.get("object_nodes", []))
                filtered_graph = self.filter_graph(graph)
                total_objects_after += len(filtered_graph.get("object_nodes", []))
                fout.write(json.dumps(filtered_graph, ensure_ascii=False) + "\n")

        print(
            f"Filtered: {total_objects_before} -> {total_objects_after} objects "
            f"({total_objects_before - total_objects_after} removed)"
        )

    def normalize_jsonl(self, input_path: str, output_path: str) -> None:
        input_file = Path(input_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        total_objects_before = 0
        total_objects_after = 0
        with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                graph = json.loads(line)
                total_objects_before += len(graph.get("object_nodes", []))
                normalized_graph = self.normalize_graph(graph, filter_objects=False)
                total_objects_after += len(normalized_graph.get("object_nodes", []))
                fout.write(json.dumps(normalized_graph, ensure_ascii=False) + "\n")

        print(
            f"Normalized: {total_objects_before} -> {total_objects_after} objects "
            f"({total_objects_before - total_objects_after} merged/removed)"
        )

    def filter_video_folder(self, input_jsonl: str, video_folder: str, output_path: str) -> None:
        input_file = Path(input_jsonl)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        folder = Path(video_folder)
        video_paths: Set[str] = set()
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.flv", "*.wmv"):
            video_paths.update(str(path) for path in folder.glob(ext))
        if not video_paths:
            print(f"No videos found in {folder}")
            return

        video_names = {Path(path).name for path in video_paths}
        print(f"Found {len(video_paths)} videos in folder")

        total_objects_before = 0
        total_objects_after = 0
        graphs_processed = 0
        graphs_kept = 0

        with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                graph = json.loads(line)
                graph_video_path = graph.get("video_path", "")
                graphs_processed += 1

                if graph_video_path not in video_paths and Path(graph_video_path).name not in video_names:
                    continue

                graphs_kept += 1
                total_objects_before += len(graph.get("object_nodes", []))
                filtered_graph = self.filter_graph(graph)
                total_objects_after += len(filtered_graph.get("object_nodes", []))
                fout.write(json.dumps(filtered_graph, ensure_ascii=False) + "\n")

        print(f"Processed: {graphs_processed} graphs, kept {graphs_kept} graphs from video folder")
        print(
            f"Filtered: {total_objects_before} -> {total_objects_after} objects "
            f"({total_objects_before - total_objects_after} removed)"
        )

    def filter_query_jsonl(self, input_path: str, scene_graph_path: str, output_path: str) -> None:
        self.query_filter.filter_jsonl(
            input_path=input_path,
            scene_graph_path=scene_graph_path,
            output_path=output_path,
        )


def filter_scene_graphs(
    input_path: str,
    output_path: str,
    video_folder: Optional[str] = None,
    min_frames: int = 5,
    min_action_frames: int = 3,
    min_relation_frames: int = 3,
    filter_objects: bool = True,
) -> None:
    graph_filter = GraphFilter(
        min_frames=min_frames,
        min_action_frames=min_action_frames,
        min_relation_frames=min_relation_frames,
    )
    if video_folder:
        if not filter_objects:
            raise ValueError("filter_objects=False is not supported with video_folder filtering.")
        graph_filter.filter_video_folder(input_path, video_folder, output_path)
        return

    if filter_objects:
        graph_filter.filter_jsonl(input_path, output_path)
    else:
        graph_filter.normalize_jsonl(input_path, output_path)


def filter_query_tracks(
    input_path: str,
    scene_graph_path: str,
    output_path: str,
    expand_non_temporal_query_tracks: bool = True,
) -> None:
    graph_filter = GraphFilter(
        expand_non_temporal_query_tracks=expand_non_temporal_query_tracks,
    )
    graph_filter.filter_query_jsonl(
        input_path=input_path,
        scene_graph_path=scene_graph_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "filter_scene_graphs": filter_scene_graphs,
            "filter_query_tracks": filter_query_tracks,
        }
    )
