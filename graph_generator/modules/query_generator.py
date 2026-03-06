import asyncio
import json
import os
import random
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

from api_sync.utils.parser import JSONParser
from api_sync.api import StreamGenerator


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
    ) -> Dict[str, Any]:
        # Step 0) 初始化输入与基础索引
        # d_star: 目标难度，后续从候选组合里选最接近 d_star 的组合
        _ = total_frames
        object_nodes = graph.get("object_nodes", [])
        action_nodes = graph.get("action_nodes", [])
        edges = graph.get("edges", [])
        objects_by_id = {obj.get("node_id"): obj for obj in object_nodes}
        target_id = target_obj.get("node_id", "")
        target_class = target_obj.get("object_class", "object")
        # F_o: 目标物体出现帧集合
        f_o = {int(k) for k in target_obj.get("bboxes", {}).keys()}

        # Step 1.1) 构建帧级索引 frame_objects: frame -> 该帧出现的对象id集合
        frame_objects: Dict[int, Set[str]] = {}
        for obj in object_nodes:
            obj_id = obj.get("node_id")
            if not obj_id:
                continue
            for frame in obj.get("bboxes", {}).keys():
                frame_idx = int(frame)
                if frame_idx not in frame_objects:
                    frame_objects[frame_idx] = set()
                frame_objects[frame_idx].add(obj_id)

        # Step 1.3) 找同类且时间重叠的干扰物 distractors
        distractors: List[Dict[str, Any]] = []
        for obj in object_nodes:
            if obj.get("node_id") == target_id:
                continue
            if obj.get("object_class") != target_class:
                continue
            # F_j: 其他对象 j 的出现帧集合
            f_j = {int(k) for k in obj.get("bboxes", {}).keys()}
            if f_o & f_j:
                distractors.append(obj)

        # Step 1.4) 计算 C_tIoU = 1 - mean(tIoU(target, other))
        others = [obj for obj in object_nodes if obj.get("node_id") != target_id]
        if not others:
            c_tiou = 0.0
        else:
            tious: List[float] = []
            for obj in others:
                f_j = {int(k) for k in obj.get("bboxes", {}).keys()}
                union = f_o | f_j
                tiou = len(f_o & f_j) / len(union) if union else 0.0
                tious.append(tiou)
            c_tiou = 1.0 - (sum(tious) / len(tious))

        # Step 1.5) 计算 C_bg（跨 shot 背景复杂度）
        # C_bg = (n_shots - 1) / max(total_shots - 1, 1)
        n_shots = len(target_obj.get("shot_ids", []))
        total_shots = len(graph.get("temporal_nodes", []))
        c_bg = (n_shots - 1) / max(total_shots - 1, 1)

        # Step 1.6) 计算 C_density（目标出现帧内的平均画面密度）
        # 每帧取除目标外对象数，均值后按 n_density_cap 归一化
        counts: List[int] = []
        for frame_idx in f_o:
            n_others = len(frame_objects.get(frame_idx, set())) - 1
            counts.append(max(n_others, 0))
        c_density = min((sum(counts) / len(counts)) / self.n_density_cap, 1.0) if counts else 0.0

        # Step 2) 构建要素池 element_pool
        # 每个要素包含: type/type_difficulty/content/excludes/ref_object
        element_pool: List[Dict[str, Any]] = []
        seen_keys: Set[str] = set()

        def add_element(elem: Dict[str, Any]) -> None:
            # 去重：相同 type + content + excludes 只保留一份
            key = json.dumps(
                {
                    "type": elem.get("type"),
                    "type_difficulty": elem.get("type_difficulty"),
                    "content": elem.get("content"),
                    "excludes": sorted(elem.get("excludes", [])),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if key in seen_keys:
                return
            seen_keys.add(key)
            element_pool.append(elem)

        # 全部干扰物id集合，用于后续判定组合是否“可唯一定位”
        all_distractor_ids = {d.get("node_id") for d in distractors if d.get("node_id")}

        # Step 2.1) 若无同类干扰物，可用 unique_class 作为线索
        if not distractors:
            add_element(
                {
                    "type": "unique_class",
                    "type_difficulty": TYPE_DIFFICULTY["unique_class"],
                    "content": target_class,
                    "excludes": set(),
                    "ref_object": None,
                }
            )

        # Step 2.2) 外观属性要素：按逗号拆 appearance，逐短语计算 excludes
        phrases = _split_and_strip(target_obj.get("appearance", ""), ",")
        for phrase in phrases:
            excluded = set()
            for d in distractors:
                d_id = d.get("node_id")
                if not d_id:
                    continue
                d_phrases = _split_and_strip(d.get("appearance", ""), ",")
                if phrase not in d_phrases:
                    excluded.add(d_id)
            add_element(
                {
                    "type": "attribute",
                    "type_difficulty": TYPE_DIFFICULTY["attribute"],
                    "content": phrase,
                    "excludes": excluded,
                    "ref_object": None,
                }
            )

        # Step 2.3) 动作要素：目标动作标签 -> 逐标签计算 excludes
        target_actions: Dict[str, Dict[str, Any]] = {}
        for a in action_nodes:
            if a.get("object_node_id") == target_id and a.get("action_label"):
                target_actions[a["action_label"]] = a

        for label in target_actions.keys():
            excluded = set()
            for d in distractors:
                d_id = d.get("node_id")
                if not d_id:
                    continue
                d_has_action = any(
                    a.get("action_label") == label and a.get("object_node_id") == d_id
                    for a in action_nodes
                )
                if not d_has_action:
                    excluded.add(d_id)
            add_element(
                {
                    "type": "action",
                    "type_difficulty": TYPE_DIFFICULTY["action"],
                    "content": label,
                    "excludes": excluded,
                    "ref_object": None,
                }
            )

        # Step 2.4) 关系要素：先找在目标时段内“类别唯一”的对象，区分 relation_to_unique / relation_to_described
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
                f_2 = {int(k) for k in o2.get("bboxes", {}).keys()}
                if f_2 & f_o:
                    same_class_count += 1
            if same_class_count == 1:
                unique_class_objects.add(obj_id)

        for edge in edges:
            if edge.get("edge_scope") == "implicit":
                continue
            source_id = edge.get("source_id")
            target_id_in_edge = edge.get("target_id")
            if source_id != target_id and target_id_in_edge != target_id:
                continue

            rel_label = (
                edge.get("spatial_relationship")
                or edge.get("contacting_relationship")
                or edge.get("attention_relationship")
            )
            if not rel_label:
                continue

            if source_id == target_id:
                ref_id = target_id_in_edge or edge.get("target_object")
            else:
                ref_id = source_id
            if not ref_id:
                continue

            ref_obj = objects_by_id.get(ref_id)
            elem_type = "relation_to_unique" if ref_id in unique_class_objects else "relation_to_described"

            excluded = set()
            for d in distractors:
                d_id = d.get("node_id")
                if not d_id:
                    continue
                d_has_relation = any(
                    (e.get("source_id") == d_id or e.get("target_id") == d_id)
                    and (
                        e.get("target_id") == ref_id
                        or e.get("source_id") == ref_id
                        or e.get("target_object") == ref_id
                    )
                    and (
                        e.get("spatial_relationship") == rel_label
                        or e.get("contacting_relationship") == rel_label
                        or e.get("attention_relationship") == rel_label
                    )
                    for e in edges
                )
                if not d_has_relation:
                    excluded.add(d_id)

            add_element(
                {
                    "type": elem_type,
                    "type_difficulty": TYPE_DIFFICULTY[elem_type],
                    "content": {"relation": rel_label, "ref": ref_id},
                    "excludes": excluded,
                    "ref_object": ref_obj,
                }
            )

        # Step 2.5) 动作组合要素（两两动作）
        action_labels = list(target_actions.keys())
        if len(action_labels) >= 2:
            for a1, a2 in combinations(action_labels, 2):
                excluded = set()
                for d in distractors:
                    d_id = d.get("node_id")
                    if not d_id:
                        continue
                    d_actions = {
                        a.get("action_label")
                        for a in action_nodes
                        if a.get("object_node_id") == d_id and a.get("action_label")
                    }
                    if a1 not in d_actions or a2 not in d_actions:
                        excluded.add(d_id)
                add_element(
                    {
                        "type": "action_combination",
                        "type_difficulty": TYPE_DIFFICULTY["action_combination"],
                        "content": [a1, a2],
                        "excludes": excluded,
                        "ref_object": None,
                    }
                )

        # Step 2.6) 隐式关系要素（edge_scope == implicit）
        for edge in edges:
            if edge.get("edge_scope") != "implicit":
                continue
            if edge.get("source_id") != target_id:
                continue
            rel_label = (
                edge.get("contacting_relationship")
                or edge.get("spatial_relationship")
                or edge.get("attention_relationship")
            )
            target_object = edge.get("target_object")
            if not rel_label or not target_object:
                continue

            excluded = set()
            for d in distractors:
                d_id = d.get("node_id")
                if not d_id:
                    continue
                d_has = any(
                    e.get("edge_scope") == "implicit"
                    and e.get("source_id") == d_id
                    and e.get("target_object") == target_object
                    and (
                        e.get("contacting_relationship") == rel_label
                        or e.get("spatial_relationship") == rel_label
                        or e.get("attention_relationship") == rel_label
                    )
                    for e in edges
                )
                if not d_has:
                    excluded.add(d_id)

            add_element(
                {
                    "type": "relation_to_unique",
                    "type_difficulty": TYPE_DIFFICULTY["relation_to_unique"],
                    "content": {"relation": rel_label, "ref": target_object},
                    "excludes": excluded,
                    "ref_object": None,
                }
            )

        # 极端兜底：若没有任何要素，至少放一个 attribute（目标类别）
        if not element_pool:
            add_element(
                {
                    "type": "attribute",
                    "type_difficulty": TYPE_DIFFICULTY["attribute"],
                    "content": target_class,
                    "excludes": set(),
                    "ref_object": None,
                }
            )

        # Step 3) 枚举有效组合 valid_combinations
        # 有效定义：组合内 excludes 的并集能够覆盖全部干扰物
        valid_combinations: List[List[Dict[str, Any]]] = []
        if not all_distractor_ids:
            # 无干扰物时，任意单要素都可唯一定位
            for elem in element_pool:
                valid_combinations.append([elem])
        else:
            max_k = min(self.max_combo_size, len(element_pool))
            for k in range(1, max_k + 1):
                for combo in combinations(element_pool, k):
                    union_excluded: Set[str] = set()
                    for elem in combo:
                        union_excluded |= elem.get("excludes", set())
                    if all_distractor_ids.issubset(union_excluded):
                        valid_combinations.append(list(combo))
                if len(valid_combinations) >= 50:
                    break

            if not valid_combinations:
                # 兜底：若找不到可全排除组合，取“排除最多干扰物”的组合
                best_combo: Optional[List[Dict[str, Any]]] = None
                best_covered = -1
                for k in range(1, max_k + 1):
                    for combo in combinations(element_pool, k):
                        union_excluded: Set[str] = set()
                        for elem in combo:
                            union_excluded |= elem.get("excludes", set())
                        covered = len(union_excluded)
                        if covered > best_covered:
                            best_covered = covered
                            best_combo = list(combo)
                valid_combinations.append(best_combo or [element_pool[0]])

        # Step 4) 对每个有效组合打分，得到 D_t / D_s / D
        scored_combinations: List[Dict[str, Any]] = []
        max_type_score = max(TYPE_DIFFICULTY.values())
        for combo in valid_combinations:
            # D_t: 动作复杂度；action 算 1，action_combination 算 2
            n_actions = 0
            for elem in combo:
                elem_type = elem.get("type")
                if elem_type == "action":
                    n_actions += 1
                elif elem_type == "action_combination":
                    n_actions += 2
            d_t = min(n_actions / self.n_action_cap, 1.0)

            # D_type: 线索类型难度，= 组合 type_difficulty 均值 / 最大难度等级
            type_diffs = [int(elem.get("type_difficulty", 0)) for elem in combo]
            avg_type_diff = (sum(type_diffs) / len(type_diffs)) if type_diffs else 0.0
            d_type = avg_type_diff / max_type_score if max_type_score > 0 else 0.0

            # D_s: 空间难度综合项
            d_s = self.w1 * d_type + self.w2 * c_tiou + self.w3 * c_bg + self.w4 * c_density
            # D: 时间/空间综合难度
            d = self.lam * d_t + (1 - self.lam) * d_s

            scored_combinations.append({"combo": combo, "D_t": d_t, "D_s": d_s, "D": d})

        # Step 5) 选最佳组合：最小化 |D - d_star|，并偏好更短组合
        best = min(scored_combinations, key=lambda item: (abs(item["D"] - d_star), len(item["combo"])))
        selected_elements = best["combo"]

        # Step 6) 组装 query_spec（给 LLM/模板生成自然语言 query）
        # 说明：当前实现不把 combo 绑定到子时间段，time_range 仍使用目标全时段
        query_spec: Dict[str, Any] = {
            "target_class": target_class,
            "attributes": [e["content"] for e in selected_elements if e.get("type") == "attribute"],
            "actions": [e["content"] for e in selected_elements if e.get("type") == "action"],
            "action_combos": [e["content"] for e in selected_elements if e.get("type") == "action_combination"],
            "relations": [
                e["content"]
                for e in selected_elements
                if e.get("type") in {"relation_to_unique", "relation_to_described"}
            ],
            "time_range": {
                "start": target_obj.get("start_frame"),
                "end": target_obj.get("end_frame"),
            },
        }

        ref_objects = []
        for elem in selected_elements:
            if elem.get("type") != "relation_to_described":
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

        # 输出: 结构化 query 规范 + 实际难度 + 目标节点
        return {
            "query_spec": query_spec,
            "D_t": best["D_t"],
            "D_s": best["D_s"],
            "D": best["D"],
            "d_star": d_star,
            "target_node": target_obj.get("node_id"),
        }

    def build_graph_plans(
        self,
        graph: Dict[str, Any],
        d_star: float,
        total_frames: Optional[int] = None,
        sample_size: Optional[int] = None,
        seed: int = 42,
        target_node_ids: Optional[Union[str, Iterable[str]]] = None,
    ) -> List[Dict[str, Any]]:
        # Build one plan per selected object node, then return all plans.
        # sample_size/seed controls random sampling of targets at graph level.
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

        plans: List[Dict[str, Any]] = []
        for obj in candidates:
            plans.append(
                self.build_query_plan(
                    graph=graph,
                    target_obj=obj,
                    d_star=d_star,
                    total_frames=total_frames,
                )
            )
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
            target_node = str(plan.get("target_node"))
            prompt = PROMPT_TEMPLATE.format(
                d_star=plan.get("d_star"),
                spec_json=json.dumps(plan.get("query_spec", {}), ensure_ascii=False),
            )
            prompts.append({"id": target_node, "prompt": prompt})

        query_map: Dict[str, str] = {}
        async for item in generator.generate_stream(
            prompts=prompts,
            system_prompt=system_prompt,
            validate_func=_extract_query,
        ):
            if not item:
                continue
            target_node = str(item.get("id"))
            query = item.get("result")
            if isinstance(query, str) and query.strip():
                query_map[target_node] = query.strip()
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
            target_node = str(plan.get("target_node"))
            query = query_map.get(target_node) or self._fallback_query(plan.get("query_spec", {}))
            query_item = {
                "query_id": f"query_{target_node}_{idx}",
                "target_node_id": target_node,
                "query": query,
                "query_spec": plan.get("query_spec"),
                "D_t": plan.get("D_t"),
                "D_s": plan.get("D_s"),
                "D": plan.get("D"),
                "d_star": plan.get("d_star"),
            }
            query_nodes.append(query_item)
            if target_node in object_map:
                object_map[target_node]["query"] = query
                object_map[target_node]["query_difficulty"] = {
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
    use_llm: bool = True,
    n_action_cap: int = 5,
    n_density_cap: int = 8,
    w1: float = 0.4,
    w2: float = 0.2,
    w3: float = 0.1,
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
        use_llm=use_llm,
        max_concurrent_per_key=max_concurrent_per_key,
        max_retries=max_retries,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(run)
