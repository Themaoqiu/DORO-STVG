import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


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


class GraphFilter:
    def __init__(
        self,
        min_frames: int = 5,
        min_action_frames: int = 3,
        min_relation_frames: int = 3,
    ) -> None:
        self.min_frames = int(min_frames)
        self.min_action_frames = int(min_action_frames)
        self.min_relation_frames = int(min_relation_frames)

    def should_keep(self, obj_node: Dict[str, Any]) -> bool:
        bboxes = obj_node.get("bboxes", {})
        total_frames = len(bboxes) if isinstance(bboxes, dict) and bboxes else 0
        if total_frames <= 0:
            s = _to_int(obj_node.get("start_frame"), 0)
            e = _to_int(obj_node.get("end_frame"), s)
            s, e = _norm_span(s, e)
            total_frames = e - s + 1

        if total_frames < self.min_frames:
            print(
                f"    Filtered {obj_node.get('node_id', '<unknown>')}: "
                f"too few frames ({total_frames} < {self.min_frames})"
            )
            return False
        return True

    def _parse_time_spans(self, item: Dict[str, Any]) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []

        raw_spans = item.get("time_spans")
        if isinstance(raw_spans, list):
            for span in raw_spans:
                if isinstance(span, dict):
                    s = _to_int(span.get("start_frame"), 0)
                    e = _to_int(span.get("end_frame"), s)
                    spans.append(_norm_span(s, e))

        if "start_frame" in item or "end_frame" in item:
            s = _to_int(item.get("start_frame"), 0)
            e = _to_int(item.get("end_frame"), s)
            spans.append(_norm_span(s, e))

        return _merge_spans(spans)

    def _normalize_object_nodes(
        self,
        object_nodes: List[Dict[str, Any]],
        *,
        filter_objects: bool,
    ) -> List[Dict[str, Any]]:
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
            norm_bboxes: Dict[str, Any] = {}
            if isinstance(raw_bboxes, dict):
                for key, value in raw_bboxes.items():
                    frame_idx = _to_int(key, -1)
                    if frame_idx < 0:
                        continue
                    norm_bboxes[str(frame_idx)] = value
                if norm_bboxes:
                    item["bboxes"] = {
                        str(f): norm_bboxes[str(f)] for f in sorted(int(k) for k in norm_bboxes.keys())
                    }

            frame_keys = [int(k) for k in (item.get("bboxes") or {}).keys()]
            if frame_keys:
                item["start_frame"] = min(frame_keys)
                item["end_frame"] = max(frame_keys)
            else:
                s = _to_int(item.get("start_frame"), 0)
                e = _to_int(item.get("end_frame"), s)
                s, e = _norm_span(s, e)
                item["start_frame"] = s
                item["end_frame"] = e

            if filter_objects and not self.should_keep(item):
                continue
            normalized.append(item)
        return normalized

    def _build_object_maps(
        self,
        object_nodes: List[Dict[str, Any]],
    ) -> Set[str]:
        valid_node_ids: Set[str] = set()
        for obj in object_nodes:
            node_id = str(obj.get("node_id", "")).strip()
            if not node_id:
                continue
            valid_node_ids.add(node_id)
        return valid_node_ids

    def _resolve_node_id(
        self,
        raw_id: Any,
        *,
        valid_node_ids: Set[str],
    ) -> Optional[str]:
        node_id = str(raw_id).strip()
        if node_id in valid_node_ids:
            return node_id
        return None

    def _dedup_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _parse_targets(raw: Any) -> List[str]:
            out: List[str] = []
            for item in raw or []:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
            return sorted(set(out))

        normalized: List[Dict[str, Any]] = []
        for item in actions:
            label = str(item.get("action_label", "")).strip()
            if not label:
                continue
            merge_key = _merge_action_key(label)
            if not merge_key:
                continue
            s = _to_int(item.get("start_frame"), 0)
            e = _to_int(item.get("end_frame"), s)
            s, e = _norm_span(s, e)
            if _span_len((s, e)) < self.min_action_frames:
                continue
            frame_idx = _to_int(item.get("frame_idx"), s)
            frame_idx = min(max(frame_idx, s), e)
            parsed = {
                "action_label": label,
                "merge_key": merge_key,
                "start_frame": s,
                "end_frame": e,
                "frame_idx": frame_idx,
                "target_object_ids": _parse_targets(item.get("target_object_ids")),
                "source": "action" if str(item.get("_source", "")).strip() != "motion" else "motion",
            }
            normalized.append(parsed)

        # Merge same-label actions with temporal overlap and union target ids.
        normalized.sort(key=lambda x: (x["merge_key"], x["start_frame"], x["end_frame"]))
        merged: List[Dict[str, Any]] = []
        for item in normalized:
            if not merged:
                merged.append(item)
                continue
            last = merged[-1]
            same_label = last["merge_key"] == item["merge_key"]
            overlap = item["start_frame"] <= last["end_frame"] and last["start_frame"] <= item["end_frame"]
            if same_label and overlap:
                last_has_action = bool(last.get("has_action_source")) or last.get("source") == "action"
                item_has_action = item.get("source") == "action"

                last_action_start = last.get("action_start")
                last_action_end = last.get("action_end")
                if last_action_start is None or last_action_end is None:
                    if last_has_action:
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
                    # Keep interval anchored to original action-node times.
                    new_start = int(last_action_start)
                    new_end = int(last_action_end)
                else:
                    # No original action exists in this merged cluster: fallback to union.
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
            else:
                item["has_action_source"] = item.get("source") == "action"
                if item["has_action_source"]:
                    item["action_start"] = item["start_frame"]
                    item["action_end"] = item["end_frame"]
                merged.append(item)

        out: List[Dict[str, Any]] = []
        for item in merged:
            packed = {
                "action_label": item["action_label"],
                "frame_idx": item["frame_idx"],
                "start_frame": item["start_frame"],
                "end_frame": item["end_frame"],
            }
            targets = item.get("target_object_ids") or []
            if targets:
                packed["target_object_ids"] = targets
            out.append(packed)
        out.sort(key=lambda x: (x["start_frame"], x["end_frame"], x["action_label"]))
        return out

    def _normalize_action_nodes(
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
            owner_id = self._resolve_node_id(
                group.get("object_id"),
                valid_node_ids=valid_node_ids,
            )
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
                norm_targets: List[str] = []
                if isinstance(raw_targets, list):
                    for raw_target in raw_targets:
                        target_node_id = self._resolve_node_id(
                            raw_target,
                            valid_node_ids=valid_node_ids,
                        )
                        if target_node_id and target_node_id != owner_id:
                            norm_targets.append(target_node_id)
                norm_targets = sorted(set(norm_targets))

                for s, e in spans:
                    parsed = {
                        "action_label": label,
                        "frame_idx": _to_int(action.get("frame_idx"), s),
                        "start_frame": s,
                        "end_frame": e,
                        "_source": "action",
                    }
                    if norm_targets:
                        parsed["target_object_ids"] = norm_targets
                    parsed_actions.append(parsed)

            owner_to_actions.setdefault(owner_id, []).extend(parsed_actions)

        if extra_actions:
            for owner_id, actions in extra_actions.items():
                if owner_id not in valid_node_ids:
                    continue
                owner_to_actions.setdefault(owner_id, []).extend(actions)

        action_nodes: List[Dict[str, Any]] = []
        for idx, owner_id in enumerate(sorted(owner_to_actions.keys())):
            actions = self._dedup_actions(owner_to_actions[owner_id])
            if not actions:
                continue
            action_nodes.append(
                {
                    "node_id": owner_to_group_node_id.get(owner_id, f"action_group_{idx}"),
                    "object_id": owner_id,
                    "actions": actions,
                }
            )
        return action_nodes

    def _normalize_relation_edges(
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
                subject_node_id = self._resolve_node_id(
                    edge.get("subject_id"),
                    valid_node_ids=valid_node_ids,
                )
                if not subject_node_id:
                    continue

                out_relationships: List[Dict[str, Any]] = []
                for rel in edge.get("relationships") or []:
                    if not isinstance(rel, dict):
                        continue
                    predicate = str(rel.get("predicate_verb", "")).strip()
                    if not predicate:
                        continue

                    object_node_id = self._resolve_node_id(
                        rel.get("object_id"),
                        valid_node_ids=valid_node_ids,
                    )
                    if not object_node_id or object_node_id == subject_node_id:
                        continue

                    spans = [
                        span
                        for span in self._parse_time_spans(rel)
                        if _span_len(span) >= self.min_relation_frames
                    ]
                    if not spans:
                        continue

                    rel_type = str(rel.get("relationship_type", "")).strip().lower()
                    edge_type = str(rel.get("edge_type", "")).strip().lower() or "spatial"

                    if rel_type == "motion":
                        for s, e in spans:
                            action_item = {
                                "action_label": predicate,
                                "frame_idx": s,
                                "start_frame": s,
                                "end_frame": e,
                                "_source": "motion",
                            }
                            action_item["target_object_ids"] = [object_node_id]
                            motion_actions.setdefault(subject_node_id, []).append(action_item)
                        continue

                    out_rel = {
                        "object_id": object_node_id,
                        "predicate_verb": predicate,
                        "edge_type": edge_type,
                        "time_spans": _spans_to_dicts(spans),
                    }
                    if rel_type:
                        out_rel["relationship_type"] = rel_type
                    out_relationships.append(out_rel)

                if out_relationships:
                    out_relationships.sort(
                        key=lambda x: (
                            str(x.get("edge_type", "")),
                            str(x.get("predicate_verb", "")),
                            str(x.get("object_id", "")),
                            _to_int(x.get("start_frame"), 0),
                        )
                    )
                    relation_edge_groups.append(
                        {
                            "subject_id": subject_node_id,
                            "relationships": out_relationships,
                        }
                    )
                continue

            if "reference_relationship" in edge:
                src = self._resolve_node_id(
                    edge.get("source_id"),
                    valid_node_ids=valid_node_ids,
                )
                tgt = self._resolve_node_id(
                    edge.get("target_id"),
                    valid_node_ids=valid_node_ids,
                )
                if not src or not tgt or src == tgt:
                    continue
                normalized_edge = dict(edge)
                normalized_edge["source_id"] = src
                normalized_edge["target_id"] = tgt
                passthrough_edges.append(normalized_edge)
                continue

            # Keep unknown edge schemas only when their endpoints are still valid.
            src = edge.get("source_id")
            tgt = edge.get("target_id")
            if src is not None and tgt is not None:
                src_node = self._resolve_node_id(
                    src,
                    valid_node_ids=valid_node_ids,
                )
                tgt_node = self._resolve_node_id(
                    tgt,
                    valid_node_ids=valid_node_ids,
                )
                if not src_node or not tgt_node or src_node == tgt_node:
                    continue
                normalized_edge = dict(edge)
                normalized_edge["source_id"] = src_node
                normalized_edge["target_id"] = tgt_node
                passthrough_edges.append(normalized_edge)

        relation_edge_groups.sort(key=lambda x: str(x.get("subject_id", "")))
        return relation_edge_groups + passthrough_edges, motion_actions

    def normalize_graph(self, graph: Dict[str, Any], *, filter_objects: bool = False) -> Dict[str, Any]:
        object_nodes = self._normalize_object_nodes(
            graph.get("object_nodes") or [],
            filter_objects=filter_objects,
        )
        graph["object_nodes"] = object_nodes

        valid_node_ids = self._build_object_maps(object_nodes)
        edges, motion_actions = self._normalize_relation_edges(
            graph,
            valid_node_ids=valid_node_ids,
        )
        graph["edges"] = edges
        graph["action_nodes"] = self._normalize_action_nodes(
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

        with input_file.open("r", encoding="utf-8") as f_in, output_file.open(
            "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                graph = json.loads(line.strip())
                total_objects_before += len(graph.get("object_nodes", []))

                filtered_graph = self.filter_graph(graph)
                total_objects_after += len(filtered_graph.get("object_nodes", []))

                f_out.write(json.dumps(filtered_graph, ensure_ascii=False) + "\n")

        print(
            f"Filtered: {total_objects_before} -> {total_objects_after} objects "
            f"({total_objects_before - total_objects_after} removed)"
        )

    def filter_video_folder(self, input_jsonl: str, video_folder: str, output_path: str) -> None:
        input_file = Path(input_jsonl)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        folder = Path(video_folder)
        video_paths: Set[str] = set()
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.flv", "*.wmv"]:
            video_paths.update(str(p) for p in folder.glob(ext))
        if not video_paths:
            print(f"No videos found in {folder}")
            return

        video_names = {Path(p).name for p in video_paths}
        print(f"Found {len(video_paths)} videos in folder")

        total_objects_before = 0
        total_objects_after = 0
        graphs_processed = 0
        graphs_kept = 0

        with input_file.open("r", encoding="utf-8") as f_in, output_file.open(
            "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                graph = json.loads(line.strip())
                graph_video_path = graph.get("video_path", "")
                graphs_processed += 1

                if graph_video_path not in video_paths and Path(graph_video_path).name not in video_names:
                    continue

                graphs_kept += 1
                total_objects_before += len(graph.get("object_nodes", []))
                filtered_graph = self.filter_graph(graph)
                total_objects_after += len(filtered_graph.get("object_nodes", []))
                f_out.write(json.dumps(filtered_graph, ensure_ascii=False) + "\n")

        print(f"Processed: {graphs_processed} graphs, kept {graphs_kept} graphs from video folder")
        print(
            f"Filtered: {total_objects_before} -> {total_objects_after} objects "
            f"({total_objects_before - total_objects_after} removed)"
        )


def filter_scene_graphs(
    input_path: str,
    output_path: str,
    video_folder: Optional[str] = None,
    min_frames: int = 5,
    min_action_frames: int = 3,
    min_relation_frames: int = 3,
) -> None:
    graph_filter = GraphFilter(
        min_frames=min_frames,
        min_action_frames=min_action_frames,
        min_relation_frames=min_relation_frames,
    )
    if video_folder:
        graph_filter.filter_video_folder(input_path, video_folder, output_path)
    else:
        graph_filter.filter_jsonl(input_path, output_path)


if __name__ == "__main__":
    import fire

    fire.Fire(filter_scene_graphs)
