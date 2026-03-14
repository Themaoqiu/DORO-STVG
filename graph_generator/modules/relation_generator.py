import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2

from api_sync.api import StreamGenerator
from api_sync.utils.parser import JSONParser


ATTENTION_OPTIONS = ["looking at", "not looking at", "unsure"]
SPATIAL_OPTIONS = ["above", "beneath", "in front of", "behind", "on the side of", "in"]
CONTACTING_OPTIONS = [
    "carrying",
    "covered by",
    "drinking from",
    "eating",
    "have it on the back",
    "holding",
    "leaning on",
    "lying on",
    "not contacting",
    "other relationship",
    "sitting on",
    "standing on",
    "touching",
    "twisting",
    "wearing",
    "wiping",
    "writing on",
]


PROMPT_TEMPLATE = (
    "You will see one full video frame. "
    "Two target objects are marked with red boxes and red numbers: 1 and 2.\n"
    "Decide their relationships and output all three types below. "
    "If you cannot determine a type, output null for that field. "
    "Avoid generic attention labels like 'looking at'/'not looking at' whenever a more specific relation can be inferred; prefer finer-grained relations when possible."
    "The relationship types and options are defined as follows: \n"
    "Attention options: {attention_list}. "
    "Spatial options: {spatial_list}. "
    "Contacting options: {contacting_list}. "
    "Only output strict JSON only:\n"
    "{{\"attention_relationship\": \"...\", "
    "\"spatial_relationship\": \"...\", "
    "\"contacting_relationship\": \"...\"}}.\n"
    "Object 1 class: {class_a}. Object 2 class: {class_b}."
)
TEST_VISION_PROMPT = (
    "Please describe what you can actually see in this image. "
    "Identify the two marked objects (1 and 2), their appearance, and their relative position. "
    "If the image is unclear or unreadable, say it explicitly."
)
IMPLICIT_RELATION_PROMPT = (
    "You will see one full video frame with one target object marked by a red box. "
    "Please identify the special relationships between the target object and other small objects or background elements that are difficult for YOLO models to detect, based on the given image.\n"
    "Examples:\n"
        "- cat (target), floor, lying on\n"
        "- person (target), small suitcase, holding\n"
    "Note: Only label relatively rare or special relationships."
    "Common and trivial ones (such as a person standing on the ground or a person is on the side of another person) should NOT be labeled."
    "If none can be inferred confidently, return that type as an none. \n"
    "For each relation, you must provide relation_type from: attention_relationship, spatial_relationship, contacting_relationship. "
    "The relationship value must come from the corresponding option list:\n"
    "Attention options: {attention_list}.\n"
    "Spatial options: {spatial_list}.\n"
    "Contacting options: {contacting_list}.\n"
    "Return strict JSON only:\n"
    "{{\"relations\":[{{\"other_object\":\"...\",\"relation_type\":\"...\",\"relationship\":\"...\"}}]}}.\n"
    "Target object class: {object_class}."
)

@dataclass
class PairFrameInfo:
    pair_id: str
    object_a: str
    object_b: str
    keyframes: List[int]
    overlap_start_frame: int
    overlap_end_frame: int


@dataclass
class SingleFrameInfo:
    object_id: str
    object_class: str
    object_start_frame: int
    object_end_frame: int
    frame_idx: int
    image_path: str


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
            if key.strip() != "API_KEYS":
                continue
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


def _load_graph(jsonl_path: str, video_path: str) -> Dict[str, Any]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            graph_video = data.get("video_path", "")
            if graph_video == video_path or Path(graph_video).stem == Path(video_path).stem:
                return data
    raise ValueError(f"No graph found for video {video_path}")


def _match_video(graph: Dict[str, Any], video_path: str) -> bool:
    graph_video = graph.get("video_path", "")
    if graph_video == video_path:
        return True
    return Path(graph_video).stem == Path(video_path).stem


def _update_jsonl_inplace(jsonl_path: str, video_path: str, graph: Dict[str, Any]) -> None:
    input_path = Path(jsonl_path)
    temp_path = input_path.with_suffix(input_path.suffix + ".tmp")

    replaced = False
    with open(input_path, "r", encoding="utf-8") as f_in, open(temp_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            if _match_video(data, video_path):
                f_out.write(json.dumps(graph, ensure_ascii=False) + "\n")
                replaced = True
            else:
                f_out.write(line)

    if not replaced:
        with open(temp_path, "a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(graph, ensure_ascii=False) + "\n")

    os.replace(temp_path, input_path)


def _normalize_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _normalize_relation_value(raw: Any, options: List[str]) -> Optional[str]:
    if raw is None:
        return None
    raw_text = str(raw).strip()
    if raw_text.lower() in {"null", "none", ""}:
        return None
    norm = _normalize_token(raw_text)
    if not norm:
        return None
    option_map = {_normalize_token(opt): opt for opt in options}
    return option_map.get(norm)


def _extract_relations(response: str) -> Optional[Dict[str, Optional[str]]]:
    parsed = JSONParser.parse(response)
    if not isinstance(parsed, dict):
        return None
    attention = _normalize_relation_value(parsed.get("attention_relationship"), ATTENTION_OPTIONS)
    spatial = _normalize_relation_value(parsed.get("spatial_relationship"), SPATIAL_OPTIONS)
    contacting = _normalize_relation_value(parsed.get("contacting_relationship"), CONTACTING_OPTIONS)
    if attention is None and spatial is None and contacting is None:
        return None
    return {
        "attention_relationship": attention,
        "spatial_relationship": spatial,
        "contacting_relationship": contacting,
    }


def _extract_implicit_relations(response: str) -> Optional[List[Dict[str, str]]]:
    parsed = JSONParser.parse(response)
    if not isinstance(parsed, dict):
        return None
    relations = parsed.get("relations")
    if not isinstance(relations, list):
        return None

    out: List[Dict[str, str]] = []
    relation_options: Dict[str, List[str]] = {
        "attention_relationship": ATTENTION_OPTIONS,
        "spatial_relationship": SPATIAL_OPTIONS,
        "contacting_relationship": CONTACTING_OPTIONS,
    }
    value_to_type: Dict[str, str] = {}
    for rel_type, options in relation_options.items():
        for option in options:
            value_to_type[_normalize_token(option)] = rel_type

    for item in relations:
        if not isinstance(item, dict):
            continue
        other = str(item.get("other_object", "")).strip()
        rel_type = str(item.get("relation_type", "")).strip()
        rel_raw = item.get("relationship")
        if not other or rel_raw is None:
            continue

        norm_type = _normalize_token(rel_type)
        if norm_type in {"attentionrelationship", "attention"}:
            rel_type = "attention_relationship"
        elif norm_type in {"spatialrelationship", "spatial"}:
            rel_type = "spatial_relationship"
        elif norm_type in {"contactingrelationship", "contacting", "contact"}:
            rel_type = "contacting_relationship"
        else:
            rel_type = ""

        if rel_type:
            rel_value = _normalize_relation_value(rel_raw, relation_options[rel_type])
        else:
            rel_value = None

        if rel_value is None:
            guessed_type = value_to_type.get(_normalize_token(str(rel_raw)))
            if guessed_type:
                rel_type = guessed_type
                rel_value = _normalize_relation_value(rel_raw, relation_options[rel_type])

        if not rel_type or rel_value is None:
            continue

        out.append(
            {
                "other_object": other,
                "relation_type": rel_type,
                "relationship": rel_value,
            }
        )
    return out


def _select_evenly(items: List[int], count: int) -> List[int]:
    if count <= 0 or len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    last_idx = len(items) - 1
    indices = [round(i * last_idx / (count - 1)) for i in range(count)]
    return [items[i] for i in indices]


def _draw_label(frame: Any, bbox: List[float], label: str) -> bool:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    draw_x1 = max(0, min(width - 1, int(x1)))
    draw_y1 = max(0, min(height - 1, int(y1)))
    draw_x2 = max(0, min(width - 1, int(x2)))
    draw_y2 = max(0, min(height - 1, int(y2)))
    if draw_x2 <= draw_x1 or draw_y2 <= draw_y1:
        return False

    cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 0, 255), 3)
    cx = int((draw_x1 + draw_x2) / 2)
    cy = int((draw_y1 + draw_y2) / 2)
    cv2.putText(
        frame,
        label,
        (cx, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return True


def _draw_box_only(frame: Any, bbox: List[float], color: Tuple[int, int, int] = (0, 0, 255)) -> bool:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    draw_x1 = max(0, min(width - 1, int(x1)))
    draw_y1 = max(0, min(height - 1, int(y1)))
    draw_x2 = max(0, min(width - 1, int(x2)))
    draw_y2 = max(0, min(height - 1, int(y2)))
    if draw_x2 <= draw_x1 or draw_y2 <= draw_y1:
        return False
    cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), color, 3)
    return True


def _collect_pair_crops(
    video_path: str,
    obj_a: Dict[str, Any],
    obj_b: Dict[str, Any],
    output_dir: str,
    keyframe_count: int,
    min_shared_frames: int,
) -> Tuple[List[int], List[str], Optional[int], Optional[int]]:
    bboxes_a = obj_a.get("bboxes", {})
    bboxes_b = obj_b.get("bboxes", {})
    if not bboxes_a or not bboxes_b:
        return [], []

    frames_a = {int(frame_idx): box for frame_idx, box in bboxes_a.items()}
    frames_b = {int(frame_idx): box for frame_idx, box in bboxes_b.items()}
    overlap_frames = sorted(set(frames_a.keys()) & set(frames_b.keys()))
    if not overlap_frames:
        return [], [], None, None
    if min_shared_frames > 1 and len(overlap_frames) < min_shared_frames:
        return [], [], None, None

    keyframes = _select_evenly(overlap_frames, keyframe_count)
    overlap_start_frame = overlap_frames[0]
    overlap_end_frame = overlap_frames[-1]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    pair_dir = Path(output_dir) / f"{obj_a['node_id']}__{obj_b['node_id']}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    crop_paths: List[str] = []
    used_frames: List[int] = []

    for frame_idx in keyframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        annotated = frame.copy()
        valid_a = _draw_label(annotated, frames_a[frame_idx], "1")
        valid_b = _draw_label(annotated, frames_b[frame_idx], "2")
        if not valid_a or not valid_b:
            continue

        path_img = pair_dir / f"frame_{frame_idx}.jpg"
        cv2.imwrite(str(path_img), annotated)
        crop_paths.append(str(path_img))
        used_frames.append(frame_idx)

    cap.release()
    return used_frames, crop_paths, overlap_start_frame, overlap_end_frame


def _select_object_keyframes(
    bboxes: Dict[str, List[float]],
    keyframe_count: int,
) -> List[int]:
    if not bboxes:
        return []
    frames = sorted(int(k) for k in bboxes.keys())
    if not frames:
        return []
    return _select_evenly(frames, keyframe_count)


def _collect_single_object_images(
    graph: Dict[str, Any],
    video_path: str,
    output_dir: str,
    keyframe_count: int,
) -> List[SingleFrameInfo]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    out: List[SingleFrameInfo] = []
    obj_dir = Path(output_dir)
    obj_dir.mkdir(parents=True, exist_ok=True)

    for obj in graph.get("object_nodes", []):
        bboxes = obj.get("bboxes", {})
        keyframes = _select_object_keyframes(bboxes, keyframe_count)
        if not keyframes:
            continue
        for frame_idx in keyframes:
            box = bboxes.get(str(frame_idx)) or bboxes.get(frame_idx)
            if box is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            annotated = frame.copy()
            if not _draw_box_only(annotated, box):
                continue

            image_path = obj_dir / f"{obj['node_id']}_frame_{frame_idx}.jpg"
            cv2.imwrite(str(image_path), annotated)
            out.append(
                SingleFrameInfo(
                    object_id=obj["node_id"],
                    object_class=obj.get("object_class", "object"),
                    object_start_frame=int(obj.get("start_frame", frame_idx)),
                    object_end_frame=int(obj.get("end_frame", frame_idx)),
                    frame_idx=frame_idx,
                    image_path=str(image_path),
                )
            )

    cap.release()
    return out


def build_prompts(
    graph: Dict[str, Any],
    video_path: str,
    crop_output_dir: str,
    keyframe_count: int,
    min_shared_frames: int,
) -> Tuple[List[Dict[str, Any]], List[PairFrameInfo]]:
    prompts: List[Dict[str, Any]] = []
    pair_infos: List[PairFrameInfo] = []

    video_name = Path(video_path).stem
    base_output = Path(crop_output_dir) / video_name
    base_output.mkdir(parents=True, exist_ok=True)

    object_nodes = graph.get("object_nodes", [])
    for i in range(len(object_nodes)):
        for j in range(i + 1, len(object_nodes)):
            obj_a = object_nodes[i]
            obj_b = object_nodes[j]
            keyframes, crops, overlap_start_frame, overlap_end_frame = _collect_pair_crops(
                video_path=video_path,
                obj_a=obj_a,
                obj_b=obj_b,
                output_dir=str(base_output),
                keyframe_count=keyframe_count,
                min_shared_frames=min_shared_frames,
            )
            if (
                not keyframes
                or not crops
                or overlap_start_frame is None
                or overlap_end_frame is None
            ):
                continue

            pair_id = f"{obj_a['node_id']}|{obj_b['node_id']}"
            pair_infos.append(
                PairFrameInfo(
                    pair_id=pair_id,
                    object_a=obj_a["node_id"],
                    object_b=obj_b["node_id"],
                    keyframes=keyframes,
                    overlap_start_frame=overlap_start_frame,
                    overlap_end_frame=overlap_end_frame,
                )
            )

            for frame_idx, path_img in zip(keyframes, crops):
                prompt_id = f"{pair_id}|{frame_idx}"
                content = [
                    {"type": "image", "image": path_img},
                    {
                        "type": "text",
                        "text": PROMPT_TEMPLATE.format(
                            attention_list=", ".join(ATTENTION_OPTIONS),
                            spatial_list=", ".join(SPATIAL_OPTIONS),
                            contacting_list=", ".join(CONTACTING_OPTIONS),
                            class_a=obj_a.get("object_class", "object"),
                            class_b=obj_b.get("object_class", "object"),
                        ),
                    },
                ]
                prompts.append({"id": prompt_id, "prompt": content})

    return prompts, pair_infos


def _build_vision_test_prompts(
    prompts: List[Dict[str, Any]],
    max_prompts: int,
) -> List[Dict[str, Any]]:
    selected = prompts[:max_prompts]
    out: List[Dict[str, Any]] = []
    for item in selected:
        prompt_id = str(item.get("id", "unknown"))
        content = item.get("prompt", [])
        images = [x for x in content if x.get("type") == "image"]
        test_content: List[Dict[str, Any]] = list(images)
        test_content.append({"type": "text", "text": TEST_VISION_PROMPT})
        out.append({"id": prompt_id, "prompt": test_content})
    return out


def _build_implicit_relation_prompts(
    single_infos: List[SingleFrameInfo],
) -> Tuple[List[Dict[str, Any]], Dict[str, SingleFrameInfo]]:
    prompts: List[Dict[str, Any]] = []
    prompt_map: Dict[str, SingleFrameInfo] = {}
    for info in single_infos:
        prompt_id = f"{info.object_id}|{info.frame_idx}"
        prompt = [
            {"type": "image", "image": info.image_path},
            {
                "type": "text",
                "text": IMPLICIT_RELATION_PROMPT.format(
                    object_class=info.object_class,
                    attention_list=", ".join(ATTENTION_OPTIONS),
                    spatial_list=", ".join(SPATIAL_OPTIONS),
                    contacting_list=", ".join(CONTACTING_OPTIONS),
                ),
            },
        ]
        prompts.append({"id": prompt_id, "prompt": prompt})
        prompt_map[prompt_id] = info
    return prompts, prompt_map


async def _run_vision_input_test(
    model_name: str,
    api_keys: Optional[Union[str, Iterable[str]]],
    prompts: List[Dict[str, Any]],
    max_concurrent_per_key: int,
    max_retries: int,
) -> None:
    if not prompts:
        print("[test] no prompts to test")
        return
    generator = StreamGenerator(
        model_name=model_name,
        api_keys=_resolve_api_keys(api_keys),
        max_concurrent_per_key=max_concurrent_per_key,
        max_retries=max_retries,
        rational=False,
        with_unique_id=True,
    )
    print(f"[test] sending {len(prompts)} vision-test prompts")
    async for item in generator.generate_stream(prompts=prompts, system_prompt="You are a visual assistant."):
        if not item:
            continue
        print(f"[test] {item.get('id')} -> {item.get('result')}")


async def _generate_implicit_relations(
    model_name: str,
    api_keys: Optional[Union[str, Iterable[str]]],
    prompts: List[Dict[str, Any]],
    max_concurrent_per_key: int,
    max_retries: int,
    system_prompt: str,
) -> Dict[str, List[Dict[str, str]]]:
    generator = StreamGenerator(
        model_name=model_name,
        api_keys=_resolve_api_keys(api_keys),
        max_concurrent_per_key=max_concurrent_per_key,
        max_retries=max_retries,
        rational=False,
        with_unique_id=True,
    )
    results: Dict[str, List[Dict[str, str]]] = {}
    async for item in generator.generate_stream(
        prompts=prompts,
        system_prompt=system_prompt,
        validate_func=lambda response: result if (result := _extract_implicit_relations(response)) is not None else False,
    ):
        if not item:
            continue
        pid = item.get("id")
        rels = item.get("result")
        if pid is not None and isinstance(rels, list):
            results[pid] = rels
    return results


class RelationGenerator:
    def __init__(
        self,
        model_name: str,
        api_keys: Optional[Union[str, Iterable[str]]],
        max_concurrent_per_key: int = 100,
        max_retries: int = 5,
    ) -> None:
        api_keys = _resolve_api_keys(api_keys)
        self.generator = StreamGenerator(
            model_name=model_name,
            api_keys=api_keys,
            max_concurrent_per_key=max_concurrent_per_key,
            max_retries=max_retries,
            rational=False,
            with_unique_id=True,
        )

    async def generate_relations(
        self,
        prompts: List[Dict[str, Any]],
        system_prompt: str,
        verbose: bool = False,
    ) -> Dict[str, Dict[str, Optional[str]]]:
        results: Dict[str, Dict[str, Optional[str]]] = {}
        async for item in self.generator.generate_stream(
            prompts=prompts,
            system_prompt=system_prompt,
            validate_func=lambda response: result if (result := _extract_relations(response)) is not None else False,
        ):
            if not item:
                continue
            result_id = item.get("id")
            relation = item.get("result")
            if result_id and relation:
                results[result_id] = relation
                if verbose:
                    print(f"[relation] {result_id} -> {relation}")
        return results


def _aggregate_segments(
    keyframes: List[int],
    relations: List[str],
    first_segment_start: Optional[int] = None,
    last_segment_end: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    segments: List[Tuple[str, int, int]] = []
    if not keyframes or not relations:
        return segments
    for idx, relation in enumerate(relations):
        start_frame = keyframes[idx]
        if idx == 0 and first_segment_start is not None:
            start_frame = min(start_frame, first_segment_start)
        if idx + 1 < len(keyframes):
            end_frame = max(start_frame, keyframes[idx + 1] - 1)
        else:
            end_frame = start_frame
        if idx == len(keyframes) - 1 and last_segment_end is not None:
            end_frame = max(end_frame, last_segment_end)

        if segments and segments[-1][0] == relation and start_frame <= segments[-1][2] + 1:
            prev_relation, prev_start, _ = segments[-1]
            segments[-1] = (prev_relation, prev_start, end_frame)
        else:
            segments.append((relation, start_frame, end_frame))

    return segments


def add_relation_edges(
    graph: Dict[str, Any],
    pair_infos: List[PairFrameInfo],
    relation_results: Dict[str, Dict[str, Optional[str]]],
    overwrite: bool = True,
) -> Dict[str, Any]:
    edges = graph.get("edges", [])
    if overwrite:
        edges = [
            edge for edge in edges
            if not any(
                key in edge
                for key in (
                    "attention_relationship",
                    "spatial_relationship",
                    "contacting_relationship",
                )
            )
        ]

    edge_index = len(edges)

    for pair_info in pair_infos:
        relation_frames: Dict[str, List[Tuple[int, str]]] = {
            "attention_relationship": [],
            "spatial_relationship": [],
            "contacting_relationship": [],
        }
        for frame_idx in pair_info.keyframes:
            prompt_id = f"{pair_info.pair_id}|{frame_idx}"
            relation_bundle = relation_results.get(prompt_id)
            if not relation_bundle:
                continue
            for relation_type, relation_value in relation_bundle.items():
                if relation_value is None:
                    continue
                relation_frames[relation_type].append((frame_idx, relation_value))

        for relation_type, frame_values in relation_frames.items():
            if not frame_values:
                continue
            frame_values.sort(key=lambda item: item[0])
            keyframes = [item[0] for item in frame_values]
            relations = [item[1] for item in frame_values]
            segments = _aggregate_segments(
                keyframes,
                relations,
                first_segment_start=pair_info.overlap_start_frame,
                last_segment_end=pair_info.overlap_end_frame,
            )
            for relation, start_frame, end_frame in segments:
                edges.append({
                    "edge_id": f"rel_{edge_index}",
                    "source_id": pair_info.object_a,
                    "target_id": pair_info.object_b,
                    relation_type: relation,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                })
                edge_index += 1

    graph["edges"] = edges
    return graph


def add_implicit_relation_edges(
    graph: Dict[str, Any],
    prompt_map: Dict[str, SingleFrameInfo],
    implicit_results: Dict[str, List[Dict[str, str]]],
    overwrite: bool = True,
) -> Dict[str, Any]:
    edges = graph.get("edges", [])
    if overwrite:
        edges = [
            edge
            for edge in edges
            if edge.get("edge_scope") != "implicit" and "implicit_relationship" not in edge
        ]

    relation_frames: Dict[Tuple[str, str, str], List[Tuple[int, str, int, int]]] = {}
    for prompt_id, relations in implicit_results.items():
        info = prompt_map.get(prompt_id)
        if info is None:
            continue
        for rel in relations:
            relation_type = rel.get("relation_type")
            relation_value = rel.get("relationship")
            target_object = rel.get("other_object")
            if not relation_type or not relation_value or not target_object:
                continue
            key = (info.object_id, target_object, relation_type)
            relation_frames.setdefault(key, []).append(
                (
                    info.frame_idx,
                    relation_value,
                    info.object_start_frame,
                    info.object_end_frame,
                )
            )

    edge_index = len(edges)
    for (source_id, target_object, relation_type), frame_values in relation_frames.items():
        if not frame_values:
            continue
        frame_values.sort(key=lambda item: item[0])
        keyframes = [item[0] for item in frame_values]
        relations = [item[1] for item in frame_values]
        object_start_frame = min(item[2] for item in frame_values)
        object_end_frame = max(item[3] for item in frame_values)
        segments = _aggregate_segments(
            keyframes,
            relations,
            first_segment_start=object_start_frame,
            last_segment_end=object_end_frame,
        )
        for relation, start_frame, end_frame in segments:
            edges.append(
                {
                    "edge_id": f"rel_implicit_{edge_index}",
                    "source_id": source_id,
                    "target_object": target_object,
                    relation_type: relation,
                    "edge_scope": "implicit",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )
            edge_index += 1

    graph["edges"] = edges
    return graph


async def generate_relation_edges(
    graph: Dict[str, Any],
    video_path: str,
    model_name: str,
    api_keys: Optional[Union[str, Iterable[str]]],
    crop_output_dir: str,
    keyframe_count: int = 4,
    min_shared_frames: int = 3,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    system_prompt: str = "You are a visual relationship classifier. Follow the instructions exactly.",
    infer_implicit_relations: bool = True,
    implicit_output_dir: str = "output/relation_single_object",
    implicit_system_prompt: str = "You are a visual relationship inference assistant. Follow instructions exactly.",
    verbose: bool = False,
) -> Tuple[Dict[str, Any], List[PairFrameInfo]]:
    prompts, pair_infos = build_prompts(
        graph=graph,
        video_path=video_path,
        crop_output_dir=crop_output_dir,
        keyframe_count=keyframe_count,
        min_shared_frames=min_shared_frames,
    )
    if not prompts:
        return graph, []

    generator = RelationGenerator(
        model_name=model_name,
        api_keys=api_keys,
        max_concurrent_per_key=max_concurrent_per_key,
        max_retries=max_retries,
    )
    results = await generator.generate_relations(
        prompts,
        system_prompt=system_prompt,
        verbose=verbose,
    )
    graph = add_relation_edges(graph, pair_infos, results, overwrite=True)

    if infer_implicit_relations:
        single_infos = _collect_single_object_images(
            graph=graph,
            video_path=video_path,
            output_dir=implicit_output_dir,
            keyframe_count=keyframe_count,
        )
        implicit_prompts, prompt_map = _build_implicit_relation_prompts(single_infos)
        if implicit_prompts:
            implicit_results = await _generate_implicit_relations(
                model_name=model_name,
                api_keys=api_keys,
                prompts=implicit_prompts,
                max_concurrent_per_key=max_concurrent_per_key,
                max_retries=max_retries,
                system_prompt=implicit_system_prompt,
            )
            graph = add_implicit_relation_edges(
                graph=graph,
                prompt_map=prompt_map,
                implicit_results=implicit_results,
                overwrite=True,
            )
    return graph, pair_infos


def run(
    jsonl: str,
    video: str,
    model_name: str,
    api_keys: Optional[Union[str, Iterable[str]]] = None,
    output: Optional[str] = None,
    crop_output_dir: str = "output/relation_crops",
    keyframe_count: int = 4,
    min_shared_frames: int = 3,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    verbose: bool = False,
    test_image_inputs_with_api: bool = False,
    test_max_prompts: int = 3,
    infer_implicit_relations: bool = True,
    implicit_output_dir: str = "output/relation_single_object",
) -> None:
    graph = _load_graph(jsonl, video)
    if test_image_inputs_with_api:
        prompts, _ = build_prompts(
            graph=graph,
            video_path=video,
            crop_output_dir=crop_output_dir,
            keyframe_count=keyframe_count,
            min_shared_frames=min_shared_frames,
        )
        test_prompts = _build_vision_test_prompts(prompts, max_prompts=test_max_prompts)
        asyncio.run(
            _run_vision_input_test(
                model_name=model_name,
                api_keys=api_keys,
                prompts=test_prompts,
                max_concurrent_per_key=max_concurrent_per_key,
                max_retries=max_retries,
            )
        )
        return

    graph, _ = asyncio.run(
        generate_relation_edges(
            graph=graph,
            video_path=video,
            model_name=model_name,
            api_keys=api_keys,
            crop_output_dir=crop_output_dir,
            keyframe_count=keyframe_count,
            min_shared_frames=min_shared_frames,
            max_concurrent_per_key=max_concurrent_per_key,
            max_retries=max_retries,
            infer_implicit_relations=infer_implicit_relations,
            implicit_output_dir=implicit_output_dir,
            verbose=verbose,
        )
    )

    if output and output != jsonl:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(graph, ensure_ascii=False) + "\n")
    else:
        _update_jsonl_inplace(jsonl, video, graph)


if __name__ == "__main__":
    import fire

    fire.Fire(run)
