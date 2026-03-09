import asyncio
import json
import os
import random
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

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
    # Clue type difficulty levels used by D_type; larger means harder clue type.
    "unique_class": 1,
    "attribute": 2,
    "action": 3,
    "relation_to_unique": 4,
    "relation_to_described": 5,
    "action_combination": 6,
}


def _normalize_api_keys(api_keys: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(api_keys, str):
        return [key.strip() for key in api_keys.split(",") if key.strip()]
    return [str(key).strip() for key in api_keys if str(key).strip()]


def _resolve_api_keys(api_keys: Optional[Union[str, Iterable[str]]]) -> List[str]:
    if api_keys is None:
        api_keys = os.getenv("API_KEYS", "")
    keys = _normalize_api_keys(api_keys)
    if not keys:
        raise ValueError("api_keys is required (pass --api_keys or set API_KEYS in env)")
    return keys


def _split_and_strip(s: str, delimiter: str = ",") -> List[str]:
    return [phrase.strip() for phrase in str(s).split(delimiter) if phrase.strip()]


def _extract_query(response: str) -> Union[str, bool]:
    parsed = JSONParser.parse(response)
    if not isinstance(parsed, dict):
        return False
    query = parsed.get("query")
    if not isinstance(query, str):
        return False
    query = query.strip()
    if not query:
        return False
    return query


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
        # Step 0) 初始化：支持 same_entity 合并，并建立帧级索引
        _ = total_frames
        object_nodes = graph.get("object_nodes", [])
        action_nodes = graph.get("action_nodes", [])
        edges = graph.get("edges", [])
        objects_by_id = {obj.get("node_id"): obj for obj in object_nodes if obj.get("node_id")}
        target_id = str(target_obj.get("node_id", ""))
        target_class = target_obj.get("object_class", "object")

        object_frames: Dict[str, Set[int]] = {
            obj_id: {int(k) for k in obj.get("bboxes", {}).keys()}
            for obj_id, obj in objects_by_id.items()
        }
        frame_objects: Dict[int, Set[str]] = {}
        for obj_id, frames in object_frames.items():
            for frame_idx in frames:
                frame_objects.setdefault(frame_idx, set()).add(obj_id)

        def relation_label(edge: Dict[str, Any]) -> Optional[str]:
            return (
                edge.get("spatial_relationship")
                or edge.get("contacting_relationship")
                or edge.get("attention_relationship")
            )

        def interval_frames(start: Any, end: Any) -> Set[int]:
            if start is None and end is None:
                return set()
            if start is None:
                start = end
            if end is None:
                end = start
            s = int(start)
            e = int(end)
            if e < s:
                s, e = e, s
            return set(range(s, e + 1))

        def frames_to_segments(frames: Set[int]) -> List[tuple[int, int]]:
            if not frames:
                return []
            sorted_frames = sorted(frames)
            segments: List[tuple[int, int]] = []
            seg_start = sorted_frames[0]
            prev = sorted_frames[0]
            for cur in sorted_frames[1:]:
                if cur == prev + 1:
                    prev = cur
                    continue
                segments.append((seg_start, prev))
                seg_start = cur
                prev = cur
            segments.append((seg_start, prev))
            return segments

        # Step 0.1) reference_relationship: same_entity 跨镜头合并
        same_entity_graph: Dict[str, Set[str]] = {}
        for edge in edges:
            if edge.get("reference_relationship") != "same_entity":
                continue
            source_id = edge.get("source_id")
            target_id_in_edge = edge.get("target_id")
            if source_id not in objects_by_id or target_id_in_edge not in objects_by_id:
                continue
            if objects_by_id[source_id].get("object_class") != target_class:
                continue
            if objects_by_id[target_id_in_edge].get("object_class") != target_class:
                continue
            same_entity_graph.setdefault(source_id, set()).add(target_id_in_edge)
            same_entity_graph.setdefault(target_id_in_edge, set()).add(source_id)

        merged_target_ids: Set[str] = set()
        if target_id:
            stack = [target_id]
            while stack:
                node = stack.pop()
                if node in merged_target_ids:
                    continue
                merged_target_ids.add(node)
                for nxt in same_entity_graph.get(node, set()):
                    if nxt not in merged_target_ids:
                        stack.append(nxt)
        if not merged_target_ids and target_id:
            merged_target_ids = {target_id}
        merged_target_objs = [objects_by_id[obj_id] for obj_id in merged_target_ids if obj_id in objects_by_id]
        if not merged_target_objs:
            merged_target_objs = [target_obj]
            merged_target_ids = {target_id} if target_id else set()

        # F_o: 合并后的目标出现帧（用于时序定位）
        f_o: Set[int] = set()
        for obj in merged_target_objs:
            f_o |= {int(k) for k in obj.get("bboxes", {}).keys()}
        merged_start_frame = min((obj.get("start_frame", 0) for obj in merged_target_objs), default=0)
        merged_end_frame = max((obj.get("end_frame", 0) for obj in merged_target_objs), default=0)
        if not f_o and merged_end_frame >= merged_start_frame:
            f_o = set(range(int(merged_start_frame), int(merged_end_frame) + 1))
        merged_shot_ids: Set[int] = set()
        for obj in merged_target_objs:
            merged_shot_ids |= set(obj.get("shot_ids", []))
        merged_appearances: List[str] = []
        for obj in merged_target_objs:
            appearance = str(obj.get("appearance", "")).strip()
            if appearance and appearance not in merged_appearances:
                merged_appearances.append(appearance)

        # Step 1) 场景固定项 + 干扰物集合
        distractors: List[Dict[str, Any]] = []
        for obj in object_nodes:
            obj_id = obj.get("node_id")
            if obj_id in merged_target_ids:
                continue
            if obj.get("object_class") != target_class:
                continue
            if f_o & object_frames.get(obj_id, set()):
                distractors.append(obj)
        distractor_ids = [str(d.get("node_id")) for d in distractors if d.get("node_id")]
        distractor_id_set = set(distractor_ids)

        others = [obj for obj in object_nodes if obj.get("node_id") not in merged_target_ids]
        if not others:
            c_tiou = 0.0
        else:
            tious: List[float] = []
            for obj in others:
                f_j = object_frames.get(str(obj.get("node_id")), set())
                union = f_o | f_j
                tious.append((len(f_o & f_j) / len(union)) if union else 0.0)
            c_tiou = 1.0 - (sum(tious) / len(tious))

        n_shots = len(merged_shot_ids)
        total_shots = len(graph.get("temporal_nodes", []))
        c_bg = (n_shots - 1) / max(total_shots - 1, 1)

        counts: List[int] = []
        for frame_idx in f_o:
            n_targets_in_frame = len(frame_objects.get(frame_idx, set()) & merged_target_ids)
            n_others = len(frame_objects.get(frame_idx, set())) - n_targets_in_frame
            counts.append(max(n_others, 0))
        c_density = min((sum(counts) / len(counts)) / self.n_density_cap, 1.0) if counts else 0.0

        # Step 2) 预计算动作帧支持（目标+干扰物）
        target_action_frames: Dict[str, Set[int]] = {}
        distractor_action_frames: Dict[str, Dict[str, Set[int]]] = {d: {} for d in distractor_ids}
        for action in action_nodes:
            obj_id = action.get("object_node_id")
            label = action.get("action_label")
            if not obj_id or not label:
                continue
            frames = interval_frames(action.get("start_frame"), action.get("end_frame"))
            if not frames and action.get("frame_idx") is not None:
                frames = {int(action["frame_idx"])}
            if obj_id in merged_target_ids:
                target_action_frames.setdefault(label, set()).update(frames & f_o)
            elif obj_id in distractor_id_set:
                distractor_action_frames[obj_id].setdefault(label, set()).update(
                    frames & object_frames.get(obj_id, set())
                )

        # Step 2) 构建要素池。每个要素新增:
        # support_frames: 该线索在目标上的有效帧
        # distractor_support: 每个干扰物在该线索上的有效帧
        element_pool: List[Dict[str, Any]] = []
        element_key_to_index: Dict[str, int] = {}

        def add_element(
            elem_type: str,
            content: Any,
            support_frames: Set[int],
            distractor_support: Dict[str, Set[int]],
            ref_object: Optional[Dict[str, Any]] = None,
        ) -> None:
            if not support_frames:
                return
            normalized_distractor_support = {
                d_id: set(distractor_support.get(d_id, set())) for d_id in distractor_ids
            }
            excludes = {d_id for d_id in distractor_ids if not normalized_distractor_support[d_id]}
            key = json.dumps(
                {"type": elem_type, "content": content},
                ensure_ascii=False,
                sort_keys=True,
            )
            if key in element_key_to_index:
                idx = element_key_to_index[key]
                element_pool[idx]["support_frames"] |= set(support_frames)
                for d_id in distractor_ids:
                    element_pool[idx]["distractor_support"][d_id] |= normalized_distractor_support[d_id]
                element_pool[idx]["excludes"] = {
                    d_id for d_id in distractor_ids if not element_pool[idx]["distractor_support"][d_id]
                }
                return
            element_key_to_index[key] = len(element_pool)
            element_pool.append(
                {
                    "type": elem_type,
                    "type_difficulty": TYPE_DIFFICULTY[elem_type],
                    "content": content,
                    "excludes": excludes,
                    "ref_object": ref_object,
                    "support_frames": set(support_frames),
                    "distractor_support": normalized_distractor_support,
                }
            )

        if not distractor_ids:
            add_element(
                elem_type="unique_class",
                content=target_class,
                support_frames=set(f_o),
                distractor_support={},
            )

        # 外观要素
        phrases: List[str] = []
        for appearance in merged_appearances:
            for phrase in _split_and_strip(appearance, ","):
                if phrase not in phrases:
                    phrases.append(phrase)
        for phrase in phrases:
            d_support: Dict[str, Set[int]] = {}
            for d_id in distractor_ids:
                d_obj = objects_by_id.get(d_id, {})
                d_phrases = _split_and_strip(d_obj.get("appearance", ""), ",")
                d_support[d_id] = object_frames.get(d_id, set()) if phrase in d_phrases else set()
            add_element(
                elem_type="attribute",
                content=phrase,
                support_frames=set(f_o),
                distractor_support=d_support,
            )

        # 动作要素
        for label, t_frames in target_action_frames.items():
            d_support = {d_id: distractor_action_frames[d_id].get(label, set()) for d_id in distractor_ids}
            add_element(
                elem_type="action",
                content=label,
                support_frames=set(t_frames),
                distractor_support=d_support,
            )

        # 关系要素（显式）
        unique_class_objects = set()
        for obj in object_nodes:
            obj_id = obj.get("node_id")
            obj_class = obj.get("object_class")
            if not obj_id or not obj_class:
                continue
            same_class_count = 0
            for o2 in object_nodes:
                if o2.get("object_class") != obj_class:
                    continue
                if object_frames.get(o2.get("node_id"), set()) & f_o:
                    same_class_count += 1
            if same_class_count == 1:
                unique_class_objects.add(obj_id)

        for edge in edges:
            if edge.get("edge_scope") == "implicit":
                continue
            source_id = edge.get("source_id")
            target_id_in_edge = edge.get("target_id")
            if source_id not in merged_target_ids and target_id_in_edge not in merged_target_ids:
                continue
            rel = relation_label(edge)
            if not rel:
                continue
            ref_id = (target_id_in_edge or edge.get("target_object")) if source_id in merged_target_ids else source_id
            if not ref_id:
                continue
            ref_obj = objects_by_id.get(ref_id)
            elem_type = "relation_to_unique" if ref_id in unique_class_objects else "relation_to_described"
            t_support = interval_frames(edge.get("start_frame"), edge.get("end_frame")) & f_o
            if not t_support:
                if ref_id in object_frames:
                    t_support = f_o & object_frames.get(ref_id, set())
                else:
                    t_support = set(f_o)
            d_support: Dict[str, Set[int]] = {d_id: set() for d_id in distractor_ids}
            for d_id in distractor_ids:
                for e in edges:
                    if e.get("edge_scope") == "implicit":
                        continue
                    if not (e.get("source_id") == d_id or e.get("target_id") == d_id):
                        continue
                    if not (
                        e.get("target_id") == ref_id
                        or e.get("source_id") == ref_id
                        or e.get("target_object") == ref_id
                    ):
                        continue
                    if relation_label(e) != rel:
                        continue
                    d_frames = interval_frames(e.get("start_frame"), e.get("end_frame"))
                    if not d_frames:
                        if ref_id in object_frames:
                            d_frames = object_frames.get(d_id, set()) & object_frames.get(ref_id, set())
                        else:
                            d_frames = set(object_frames.get(d_id, set()))
                    d_support[d_id] |= d_frames
                d_support[d_id] &= object_frames.get(d_id, set())
            add_element(
                elem_type=elem_type,
                content={"relation": rel, "ref": ref_id},
                support_frames=t_support,
                distractor_support=d_support,
                ref_object=ref_obj,
            )

        # 动作组合要素（两两动作）
        action_labels = list(target_action_frames.keys())
        if len(action_labels) >= 2:
            for a1, a2 in combinations(action_labels, 2):
                t_support = target_action_frames[a1] & target_action_frames[a2]
                d_support = {
                    d_id: distractor_action_frames[d_id].get(a1, set()) & distractor_action_frames[d_id].get(a2, set())
                    for d_id in distractor_ids
                }
                add_element(
                    elem_type="action_combination",
                    content=[a1, a2],
                    support_frames=t_support,
                    distractor_support=d_support,
                )

        # 隐式关系要素
        for edge in edges:
            if edge.get("edge_scope") != "implicit":
                continue
            if edge.get("source_id") not in merged_target_ids:
                continue
            rel = relation_label(edge)
            target_object = edge.get("target_object")
            if not rel or not target_object:
                continue
            t_support = interval_frames(edge.get("start_frame"), edge.get("end_frame")) & f_o
            if not t_support:
                t_support = set(f_o)
            d_support: Dict[str, Set[int]] = {d_id: set() for d_id in distractor_ids}
            for d_id in distractor_ids:
                for e in edges:
                    if e.get("edge_scope") != "implicit":
                        continue
                    if e.get("source_id") != d_id:
                        continue
                    if e.get("target_object") != target_object:
                        continue
                    if relation_label(e) != rel:
                        continue
                    d_frames = interval_frames(e.get("start_frame"), e.get("end_frame"))
                    if not d_frames:
                        d_frames = set(object_frames.get(d_id, set()))
                    d_support[d_id] |= d_frames
                d_support[d_id] &= object_frames.get(d_id, set())
            add_element(
                elem_type="relation_to_unique",
                content={"relation": rel, "ref": target_object},
                support_frames=t_support,
                distractor_support=d_support,
            )

        # Step 3) 组合枚举 + 时间段唯一性校验
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
                union_excluded: Set[str] = set()
                for elem in combo:
                    union_excluded |= elem["excludes"]
                if distractor_ids and not set(distractor_ids).issubset(union_excluded):
                    continue

                support = combo_support_frames(combo)
                if not support:
                    continue
                segments = frames_to_segments(support)
                if not segments:
                    continue

                unique_segments: List[tuple[int, int]] = []
                for seg_start, seg_end in segments:
                    seg_frames = set(range(seg_start, seg_end + 1))
                    is_unique = True
                    for d_id in distractor_ids:
                        if distractor_combo_frames(combo, d_id) & seg_frames:
                            is_unique = False
                            break
                    if is_unique:
                        unique_segments.append((seg_start, seg_end))
                if unique_segments:
                    valid_combinations.append({"combo": combo, "segments": unique_segments})
            if len(valid_combinations) >= 50:
                break

        if not valid_combinations:
            # 兜底：找排除最多且支持帧最长的组合
            best_fallback: Optional[Dict[str, Any]] = None
            for k in range(1, max_k + 1):
                for combo_tpl in combinations(element_pool, k):
                    combo = list(combo_tpl)
                    support = combo_support_frames(combo)
                    if not support:
                        continue
                    union_excluded: Set[str] = set()
                    for elem in combo:
                        union_excluded |= elem["excludes"]
                    candidate = {
                        "combo": combo,
                        "segments": frames_to_segments(support),
                        "covered": len(union_excluded),
                        "support_len": len(support),
                    }
                    if not best_fallback:
                        best_fallback = candidate
                        continue
                    if (candidate["covered"], candidate["support_len"]) > (
                        best_fallback["covered"],
                        best_fallback["support_len"],
                    ):
                        best_fallback = candidate
            if best_fallback and best_fallback["segments"]:
                valid_combinations.append(
                    {"combo": best_fallback["combo"], "segments": best_fallback["segments"]}
                )
            elif element_pool:
                valid_combinations.append(
                    {
                        "combo": [element_pool[0]],
                        "segments": [(merged_start_frame, merged_end_frame)],
                    }
                )

        # Step 4) 组合打分（每个组合的每个唯一时间段都是候选）
        scored_candidates: List[Dict[str, Any]] = []
        max_type_score = max(TYPE_DIFFICULTY.values())
        for item in valid_combinations:
            combo = item["combo"]
            n_actions = 0
            for elem in combo:
                if elem["type"] == "action":
                    n_actions += 1
                elif elem["type"] == "action_combination":
                    n_actions += 2
            d_t = min(n_actions / self.n_action_cap, 1.0)
            type_diffs = [int(elem["type_difficulty"]) for elem in combo]
            avg_type_diff = (sum(type_diffs) / len(type_diffs)) if type_diffs else 0.0
            d_type = avg_type_diff / max_type_score if max_type_score > 0 else 0.0
            d_s = self.w1 * d_type + self.w2 * c_tiou + self.w3 * c_bg + self.w4 * c_density
            d = self.lam * d_t + (1 - self.lam) * d_s
            for seg_start, seg_end in item["segments"]:
                scored_candidates.append(
                    {
                        "combo": combo,
                        "D_t": d_t,
                        "D_s": d_s,
                        "D": d,
                        "segment": (seg_start, seg_end),
                        "segment_len": seg_end - seg_start + 1,
                    }
                )

        if not scored_candidates:
            scored_candidates = [
                {
                    "combo": [element_pool[0]],
                    "D_t": 0.0,
                    "D_s": self.w2 * c_tiou + self.w3 * c_bg + self.w4 * c_density,
                    "D": self.w2 * c_tiou + self.w3 * c_bg + self.w4 * c_density,
                    "segment": (merged_start_frame, merged_end_frame),
                    "segment_len": max(merged_end_frame - merged_start_frame + 1, 1),
                }
            ]

        # Step 5) 按贴近 d_star 的顺序排列候选（支持返回多候选用于均衡采样）
        scored_candidates.sort(key=lambda item: (abs(item["D"] - d_star), -item["segment_len"], len(item["combo"])))
        plans: List[Dict[str, Any]] = []
        seen_signatures: Set[str] = set()

        for candidate in scored_candidates:
            selected_elements = candidate["combo"]
            seg_start, seg_end = candidate["segment"]
            query_spec: Dict[str, Any] = {
                "target_class": target_class,
                "attributes": [e["content"] for e in selected_elements if e["type"] == "attribute"],
                "actions": [e["content"] for e in selected_elements if e["type"] == "action"],
                "action_combos": [e["content"] for e in selected_elements if e["type"] == "action_combination"],
                "relations": [
                    e["content"]
                    for e in selected_elements
                    if e["type"] in {"relation_to_unique", "relation_to_described"}
                ],
                "time_range": {"start": seg_start, "end": seg_end},
                "full_track_range": {"start": merged_start_frame, "end": merged_end_frame},
            }
            if len(merged_target_ids) > 1:
                query_spec["same_entity_nodes"] = sorted(merged_target_ids)
                query_spec["cross_shot"] = True

            ref_objects = []
            for elem in selected_elements:
                if elem["type"] != "relation_to_described":
                    continue
                ref_obj = elem.get("ref_object")
                if not isinstance(ref_obj, dict):
                    continue
                ref_objects.append(
                    {
                        "class": ref_obj.get("object_class", "object"),
                        "appearance": ref_obj.get("appearance", ""),
                        "relation": elem.get("content", {}).get("relation"),
                    }
                )
            if ref_objects:
                query_spec["ref_objects"] = ref_objects

            clue_types = sorted({str(e.get("type")) for e in selected_elements if e.get("type")})
            if query_spec.get("cross_shot"):
                clue_types.append("cross_shot")
            type_bucket = "unique_class"
            if "cross_shot" in clue_types:
                type_bucket = "cross_shot"
            elif any(t in {"relation_to_unique", "relation_to_described"} for t in clue_types):
                type_bucket = "relation"
            elif "action_combination" in clue_types:
                type_bucket = "action_combination"
            elif "action" in clue_types:
                type_bucket = "action"
            elif "attribute" in clue_types:
                type_bucket = "attribute"

            signature = json.dumps(
                {
                    "target_node": target_obj.get("node_id"),
                    "query_spec": query_spec,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            plans.append(
                {
                    "query_spec": query_spec,
                    "D_t": candidate["D_t"],
                    "D_s": candidate["D_s"],
                    "D": candidate["D"],
                    "d_star": d_star,
                    "target_node": target_obj.get("node_id"),
                    "segment_len": candidate["segment_len"],
                    "clue_types": clue_types,
                    "type_bucket": type_bucket,
                    "candidate_signature": signature,
                }
            )
            if not return_all_candidates:
                break
            if max_candidates is not None and max_candidates > 0 and len(plans) >= max_candidates:
                break

        if not plans:
            plans = [
                {
                    "query_spec": {
                        "target_class": target_class,
                        "attributes": [],
                        "actions": [],
                        "action_combos": [],
                        "relations": [],
                        "time_range": {"start": merged_start_frame, "end": merged_end_frame},
                        "full_track_range": {"start": merged_start_frame, "end": merged_end_frame},
                    },
                    "D_t": 0.0,
                    "D_s": 0.0,
                    "D": 0.0,
                    "d_star": d_star,
                    "target_node": target_obj.get("node_id"),
                    "segment_len": max(merged_end_frame - merged_start_frame + 1, 1),
                    "clue_types": ["fallback"],
                    "type_bucket": "fallback",
                    "candidate_signature": f"fallback_{target_obj.get('node_id')}_{merged_start_frame}_{merged_end_frame}",
                }
            ]

        if return_all_candidates:
            return plans
        return plans[0]

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
        candidates = [obj for obj in graph.get("object_nodes", []) if obj.get("node_id") and obj.get("bboxes")]
        if not candidates:
            return []

        if target_node_ids:
            if isinstance(target_node_ids, str):
                target_set = {x.strip() for x in target_node_ids.split(",") if x.strip()}
            else:
                target_set = {str(x).strip() for x in target_node_ids if str(x).strip()}
            candidates = [obj for obj in candidates if obj.get("node_id") in target_set]

        if sample_size is not None and sample_size > 0 and len(candidates) > sample_size:
            rng = random.Random(seed)
            candidates = rng.sample(candidates, sample_size)

        d_star_values: List[float] = []
        if d_star_list:
            if isinstance(d_star_list, str):
                raw_values = [x.strip() for x in d_star_list.split(",") if x.strip()]
            else:
                raw_values = [str(x).strip() for x in d_star_list if str(x).strip()]
            for value in raw_values:
                try:
                    d_value = float(value)
                except ValueError:
                    continue
                d_star_values.append(min(max(d_value, 0.0), 1.0))
        if not d_star_values:
            d_star_values = [min(max(float(d_star), 0.0), 1.0)]

        all_plans: List[Dict[str, Any]] = []
        global_seen_signatures: Set[str] = set()

        for obj in candidates:
            target_candidates = self.build_query_plan(
                graph=graph,
                target_obj=obj,
                d_star=d_star_values[0],
                total_frames=total_frames,
                return_all_candidates=True,
                max_candidates=max_candidates_per_target,
            )
            if not isinstance(target_candidates, list) or not target_candidates:
                continue

            selected_for_target: List[Dict[str, Any]] = []
            if generate_all_candidates:
                selected_for_target = target_candidates
            elif len(d_star_values) > 1:
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
                    if signature and signature in local_seen:
                        continue
                    local_seen.add(signature)
                    selected_for_target.append({**best, "d_star": d_value})
                if per_target_limit > 0:
                    selected_for_target = selected_for_target[:per_target_limit]
            else:
                if per_target_limit <= 0:
                    per_target_limit = 1
                selected_for_target = target_candidates[:per_target_limit]

            for plan in selected_for_target:
                signature = str(plan.get("candidate_signature", ""))
                if signature and signature in global_seen_signatures:
                    continue
                if signature:
                    global_seen_signatures.add(signature)
                plan_item = dict(plan)
                plan_item["plan_id"] = (
                    f"plan_{plan_item.get('target_node')}_{len(all_plans)}_{int(float(plan_item.get('D', 0.0)) * 1000)}"
                )
                all_plans.append(plan_item)

        if queries_per_graph is None or queries_per_graph <= 0 or len(all_plans) <= queries_per_graph:
            return all_plans

        # 对候选池做均衡采样：难度分桶 + 类型分桶 + 轮询抽样
        rng = random.Random(seed)
        n_bins = max(int(difficulty_bins), 1)
        groups: Dict[tuple[int, str], List[Dict[str, Any]]] = {}
        for plan in all_plans:
            d_value = min(max(float(plan.get("D", 0.0)), 0.0), 1.0)
            bin_idx = min(int(d_value * n_bins), n_bins - 1) if balance_difficulty else 0
            type_key = str(plan.get("type_bucket", "other")) if balance_types else "all"
            groups.setdefault((bin_idx, type_key), []).append(plan)

        for group_key, items in groups.items():
            items.sort(
                key=lambda item: (
                    abs(float(item.get("D", 0.0)) - d_star),
                    -int(item.get("segment_len", 1)),
                    len(item.get("clue_types", [])),
                )
            )
            rng.shuffle(items)
            items.sort(
                key=lambda item: (
                    abs(float(item.get("D", 0.0)) - d_star),
                    -int(item.get("segment_len", 1)),
                    len(item.get("clue_types", [])),
                )
            )
            groups[group_key] = items

        selected: List[Dict[str, Any]] = []
        target_counter: Dict[str, int] = {}
        group_keys = list(groups.keys())
        while len(selected) < queries_per_graph:
            rng.shuffle(group_keys)
            progressed = False
            for group_key in group_keys:
                bucket = groups.get(group_key, [])
                while bucket:
                    candidate = bucket.pop(0)
                    target_node = str(candidate.get("target_node", ""))
                    if max_queries_per_target is not None and max_queries_per_target > 0:
                        if target_counter.get(target_node, 0) >= max_queries_per_target:
                            continue
                    selected.append(candidate)
                    target_counter[target_node] = target_counter.get(target_node, 0) + 1
                    progressed = True
                    break
                groups[group_key] = bucket
                if len(selected) >= queries_per_graph:
                    break
            if not progressed:
                break

        if len(selected) < queries_per_graph:
            selected_signatures = {str(item.get("candidate_signature", "")) for item in selected}
            leftovers = [plan for plan in all_plans if str(plan.get("candidate_signature", "")) not in selected_signatures]
            leftovers.sort(
                key=lambda item: (
                    abs(float(item.get("D", 0.0)) - d_star),
                    -int(item.get("segment_len", 1)),
                    len(item.get("clue_types", [])),
                )
            )
            for item in leftovers:
                selected.append(item)
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

        prompts: List[Dict[str, Any]] = []
        for plan in plans:
            plan_id = str(plan.get("plan_id") or plan.get("target_node"))
            prompt = PROMPT_TEMPLATE.format(
                d_star=plan.get("d_star"),
                spec_json=json.dumps(plan.get("query_spec", {}), ensure_ascii=False),
            )
            prompts.append({"id": plan_id, "prompt": prompt})

        query_map: Dict[str, str] = {}
        async for item in generator.generate_stream(
            prompts=prompts,
            system_prompt=system_prompt,
            validate_func=_extract_query,
        ):
            if not item:
                continue
            plan_id = str(item.get("id"))
            query = item.get("result")
            if isinstance(query, str) and query.strip():
                query_map[plan_id] = query.strip()
        return query_map

    def _fallback_query(self, query_spec: Dict[str, Any]) -> str:
        target_class = query_spec.get("target_class", "object")
        attributes = query_spec.get("attributes", [])
        actions = query_spec.get("actions", [])
        action_combos = query_spec.get("action_combos", [])
        relations = query_spec.get("relations", [])

        chunks = [f"the {target_class}"]
        if attributes:
            chunks.append("with " + ", ".join(attributes))
        if actions:
            chunks.append("that is " + " and ".join(actions))
        for combo in action_combos:
            if isinstance(combo, list) and len(combo) == 2:
                chunks.append(f"that {combo[0]} and {combo[1]}")
        if relations:
            rel_chunks = []
            for rel in relations:
                if isinstance(rel, dict):
                    relation = rel.get("relation")
                    ref = rel.get("ref")
                    if relation and ref:
                        rel_chunks.append(f"{relation} {ref}")
            if rel_chunks:
                chunks.append("that is " + " and ".join(rel_chunks))
        return " ".join(chunks)

    def process_graph(
        self,
        graph: Dict[str, Any],
        d_star: float,
        model_name: Optional[str] = None,
        api_keys: Optional[Union[str, Iterable[str]]] = None,
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
        use_llm: bool = True,
        max_concurrent_per_key: int = 100,
        max_retries: int = 5,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        plans = self.build_graph_plans(
            graph=graph,
            d_star=d_star,
            total_frames=total_frames,
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
        object_map = {obj.get("node_id"): obj for obj in graph.get("object_nodes", [])}
        for idx, plan in enumerate(plans):
            plan_id = str(plan.get("plan_id") or f"plan_{idx}")
            target_node = str(plan.get("target_node"))
            query = query_map.get(plan_id) or self._fallback_query(plan.get("query_spec", {}))
            query_item = {
                "query_id": f"query_{target_node}_{idx}_{plan_id}",
                "target_node_id": target_node,
                "query": query,
                "query_spec": plan.get("query_spec"),
                "D_t": plan.get("D_t"),
                "D_s": plan.get("D_s"),
                "D": plan.get("D"),
                "d_star": plan.get("d_star"),
                "clue_types": plan.get("clue_types", []),
                "type_bucket": plan.get("type_bucket"),
            }
            query_nodes.append(query_item)
            if target_node in object_map:
                obj_item = object_map[target_node]
                obj_item.setdefault("queries", [])
                obj_item["queries"].append(query)
                obj_item.setdefault("query_difficulties", [])
                obj_item["query_difficulties"].append(
                    {
                        "D_t": plan.get("D_t"),
                        "D_s": plan.get("D_s"),
                        "D": plan.get("D"),
                    }
                )
                if "query" not in obj_item:
                    obj_item["query"] = query
                    obj_item["query_difficulty"] = {
                        "D_t": plan.get("D_t"),
                        "D_s": plan.get("D_s"),
                        "D": plan.get("D"),
                    }

        if overwrite:
            graph["query_nodes"] = query_nodes
        else:
            graph.setdefault("query_nodes", [])
            graph["query_nodes"].extend(query_nodes)

        return graph

    def process_jsonl(
        self,
        input_path: str,
        output_path: str,
        d_star: float,
        model_name: Optional[str] = None,
        api_keys: Optional[Union[str, Iterable[str]]] = None,
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
        use_llm: bool = True,
        max_concurrent_per_key: int = 100,
        max_retries: int = 5,
        overwrite: bool = True,
    ) -> None:
        input_file = Path(input_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                graph = json.loads(line.strip())
                graph = self.process_graph(
                    graph=graph,
                    d_star=d_star,
                    model_name=model_name,
                    api_keys=api_keys,
                    total_frames=total_frames,
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
                f_out.write(json.dumps(graph, ensure_ascii=False) + "\n")


def run(
    input_path: str,
    output_path: str,
    d_star: float = 0.5,
    model_name: Optional[str] = None,
    api_keys: Optional[Union[str, Iterable[str]]] = None,
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
    use_llm: bool = True,
    n_action_cap: int = 5,
    n_density_cap: int = 8,
    w1: float = 0.4,
    w2: float = 0.2,
    w3: float = 0.4,
    w4: float = 0.3,
    lam: float = 0.5,
    max_combo_size: int = 3,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    overwrite: bool = True,
) -> None:
    generator = DifficultyAwareSTVGQueryGenerator(
        n_action_cap=n_action_cap,
        n_density_cap=n_density_cap,
        w1=w1,
        w2=w2,
        w3=w3,
        w4=w4,
        lam=lam,
        max_combo_size=max_combo_size,
    )
    generator.process_jsonl(
        input_path=input_path,
        output_path=output_path,
        d_star=d_star,
        model_name=model_name,
        api_keys=api_keys,
        total_frames=total_frames,
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
