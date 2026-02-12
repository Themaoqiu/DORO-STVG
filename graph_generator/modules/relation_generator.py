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
    "You will see one cropped image from a video frame. "
    "Two objects are inside the crop, labeled with red numbers: 1 and 2. "
    "Decide their relationships and output all three types below. "
    "If you cannot determine a type, output null for that field. "
    "Attention options: {attention_list}. "
    "Spatial options: {spatial_list}. "
    "Contacting options: {contacting_list}. "
    "Only output strict JSON only:\n"
    "{{\"attention_relationship\": \"...\", "
    "\"spatial_relationship\": \"...\", "
    "\"contacting_relationship\": \"...\"}}.\n"
    "Object 1 class: {class_a}. Object 2 class: {class_b}."
)

ATTENTION_RELATIONS = set(ATTENTION_OPTIONS)
SPATIAL_RELATIONS = set(SPATIAL_OPTIONS)
CONTACTING_RELATIONS = set(CONTACTING_OPTIONS)


@dataclass
class PairFrameInfo:
    pair_id: str
    object_a: str
    object_b: str
    keyframes: List[int]
    crop_paths: List[str]


def _normalize_api_keys(api_keys: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(api_keys, str):
        return [key.strip() for key in api_keys.split(",") if key.strip()]
    return [str(key).strip() for key in api_keys if str(key).strip()]


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


def _relation_list_text(items: List[str]) -> str:
    return ", ".join(items)


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


def _validate_relation(response: str) -> Union[Dict[str, Optional[str]], bool]:
    relations = _extract_relations(response)
    if relations is None:
        return False
    return relations


def _relation_to_edge_type(relation: str) -> str:
    if relation in ATTENTION_RELATIONS:
        return "attention_relationship"
    if relation in SPATIAL_RELATIONS:
        return "spatial_relationship"
    return "contacting_relationship"


def _select_evenly(items: List[int], count: int) -> List[int]:
    if count <= 0 or len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    last_idx = len(items) - 1
    indices = [round(i * last_idx / (count - 1)) for i in range(count)]
    return [items[i] for i in indices]


def _crop_with_padding(
    frame: Any,
    bbox: List[float],
    padding_ratio: float,
) -> Optional[Any]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = box_w * padding_ratio
    pad_h = box_h * padding_ratio
    crop_x1 = max(0, int(x1 - pad_w))
    crop_y1 = max(0, int(y1 - pad_h))
    crop_x2 = min(width, int(x2 + pad_w))
    crop_y2 = min(height, int(y2 + pad_h))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    return frame[crop_y1:crop_y2, crop_x1:crop_x2]


def _crop_with_padding_and_offset(
    frame: Any,
    bbox: List[float],
    padding_ratio: float,
) -> Tuple[Optional[Any], Tuple[int, int]]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = box_w * padding_ratio
    pad_h = box_h * padding_ratio
    crop_x1 = max(0, int(x1 - pad_w))
    crop_y1 = max(0, int(y1 - pad_h))
    crop_x2 = min(width, int(x2 + pad_w))
    crop_y2 = min(height, int(y2 + pad_h))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None, (0, 0)
    return frame[crop_y1:crop_y2, crop_x1:crop_x2], (crop_x1, crop_y1)


def _union_bbox(bbox_a: List[float], bbox_b: List[float]) -> List[float]:
    return [
        min(bbox_a[0], bbox_b[0]),
        min(bbox_a[1], bbox_b[1]),
        max(bbox_a[2], bbox_b[2]),
        max(bbox_a[3], bbox_b[3]),
    ]


def _draw_label(frame: Any, bbox: List[float], label: str, offset: Tuple[int, int]) -> None:
    x1, y1, x2, y2 = bbox
    offset_x, offset_y = offset
    cx = int((x1 + x2) / 2) - offset_x
    cy = int((y1 + y2) / 2) - offset_y
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


def _build_prompt(object_class_a: str, object_class_b: str) -> str:
    return PROMPT_TEMPLATE.format(
        attention_list=_relation_list_text(ATTENTION_OPTIONS),
        spatial_list=_relation_list_text(SPATIAL_OPTIONS),
        contacting_list=_relation_list_text(CONTACTING_OPTIONS),
        class_a=object_class_a,
        class_b=object_class_b,
    )


def _collect_pair_crops(
    video_path: str,
    obj_a: Dict[str, Any],
    obj_b: Dict[str, Any],
    output_dir: str,
    keyframe_count: int,
    padding_ratio: float,
) -> Tuple[List[int], List[str]]:
    bboxes_a = obj_a.get("bboxes", {})
    bboxes_b = obj_b.get("bboxes", {})
    if not bboxes_a or not bboxes_b:
        return [], []

    frames_a = {int(frame_idx): box for frame_idx, box in bboxes_a.items()}
    frames_b = {int(frame_idx): box for frame_idx, box in bboxes_b.items()}
    overlap_frames = sorted(set(frames_a.keys()) & set(frames_b.keys()))
    if not overlap_frames:
        return [], []

    keyframes = _select_evenly(overlap_frames, keyframe_count)

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

        union_box = _union_bbox(frames_a[frame_idx], frames_b[frame_idx])
        crop, offset = _crop_with_padding_and_offset(frame, union_box, padding_ratio)
        if crop is None:
            continue
        _draw_label(crop, frames_a[frame_idx], "1", offset)
        _draw_label(crop, frames_b[frame_idx], "2", offset)

        path_img = pair_dir / f"frame_{frame_idx}.jpg"
        cv2.imwrite(str(path_img), crop)
        crop_paths.append(str(path_img))
        used_frames.append(frame_idx)

    cap.release()
    return used_frames, crop_paths


def build_prompts(
    graph: Dict[str, Any],
    video_path: str,
    crop_output_dir: str,
    keyframe_count: int,
    padding_ratio: float,
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
            keyframes, crops = _collect_pair_crops(
                video_path=video_path,
                obj_a=obj_a,
                obj_b=obj_b,
                output_dir=str(base_output),
                keyframe_count=keyframe_count,
                padding_ratio=padding_ratio,
            )
            if not keyframes or not crops:
                continue

            pair_id = f"{obj_a['node_id']}|{obj_b['node_id']}"
            pair_infos.append(
                PairFrameInfo(
                    pair_id=pair_id,
                    object_a=obj_a["node_id"],
                    object_b=obj_b["node_id"],
                    keyframes=keyframes,
                    crop_paths=crops,
                )
            )

            for frame_idx, path_img in zip(keyframes, crops):
                prompt_id = f"{pair_id}|{frame_idx}"
                content = [
                    {"type": "image", "image": path_img},
                    {
                        "type": "text",
                        "text": _build_prompt(
                            obj_a.get("object_class", "object"),
                            obj_b.get("object_class", "object"),
                        ),
                    },
                ]
                prompts.append({"id": prompt_id, "prompt": content})

    return prompts, pair_infos


class RelationGenerator:
    def __init__(
        self,
        model_name: str,
        api_keys: Union[str, Iterable[str]],
        max_concurrent_per_key: int = 100,
        max_retries: int = 5,
    ) -> None:
        api_keys = _normalize_api_keys(api_keys)
        if not api_keys:
            raise ValueError("api_keys is required")
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
            validate_func=_validate_relation,
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
) -> List[Tuple[str, int, int]]:
    segments: List[Tuple[str, int, int]] = []
    if not keyframes or not relations:
        return segments
    for idx, relation in enumerate(relations):
        start_frame = keyframes[idx]
        if idx + 1 < len(keyframes):
            end_frame = max(start_frame, keyframes[idx + 1] - 1)
        else:
            end_frame = start_frame

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
            segments = _aggregate_segments(keyframes, relations)
            for relation, start_frame, end_frame in segments:
                edges.append({
                    "edge_id": f"edge_rel_{edge_index}",
                    "source_id": pair_info.object_a,
                    "target_id": pair_info.object_b,
                    relation_type: relation,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                })
                edge_index += 1

    graph["edges"] = edges
    return graph


async def generate_relation_edges(
    graph: Dict[str, Any],
    video_path: str,
    model_name: str,
    api_keys: Union[str, Iterable[str]],
    crop_output_dir: str,
    keyframe_count: int = 4,
    padding_ratio: float = 0.1,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    system_prompt: str = "You are a visual relationship classifier. Follow the instructions exactly.",
    verbose: bool = False,
) -> Tuple[Dict[str, Any], List[PairFrameInfo]]:
    prompts, pair_infos = build_prompts(
        graph=graph,
        video_path=video_path,
        crop_output_dir=crop_output_dir,
        keyframe_count=keyframe_count,
        padding_ratio=padding_ratio,
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
    return graph, pair_infos


def run(
    jsonl: str,
    video: str,
    model_name: str,
    api_keys: Union[str, Iterable[str]],
    output: Optional[str] = None,
    crop_output_dir: str = "output/relation_crops",
    keyframe_count: int = 4,
    padding_ratio: float = 0.1,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    verbose: bool = False,
) -> None:
    graph = _load_graph(jsonl, video)

    graph, _ = asyncio.run(
        generate_relation_edges(
            graph=graph,
            video_path=video,
            model_name=model_name,
            api_keys=api_keys,
            crop_output_dir=crop_output_dir,
            keyframe_count=keyframe_count,
            padding_ratio=padding_ratio,
            max_concurrent_per_key=max_concurrent_per_key,
            max_retries=max_retries,
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
