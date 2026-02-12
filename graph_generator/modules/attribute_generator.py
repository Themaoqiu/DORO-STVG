import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2

from api_sync.api import StreamGenerator
from api_sync.utils.parser import JSONParser
from modules.keyframe_clustering import KeyframeClustering


PROMPT_TEMPLATE = (
    "You will see multiple frames of the same object from a video at different times. "
    "Write a single-sentence English caption that comprehensively describes the object's appearance. "
    "Describe the object that is centered in the frame. "
    "If the object is a person, specify whether they are an adult or a child, "
    "and whether they are male or female, then describe their appearance. "
    "If the object is an animal or an item, describe its species/type, color, and other visible attributes. "
    "Do not describe actions or the background. "
    "Output strict JSON only: "
    "{{\"caption\": \"...\"}}. "
    "Object class: {object_class}. Number of images: {num_images}."
)


@dataclass
class AttributeCaption:
    object_node_id: str
    caption: str
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


def _build_prompt(object_class: str, num_images: int) -> str:
    return PROMPT_TEMPLATE.format(
        object_class=object_class,
        num_images=num_images,
    )


def _validate_caption(response: str) -> Union[str, bool]:
    parsed = JSONParser.parse(response)
    if not parsed or "caption" not in parsed:
        return False
    caption = parsed.get("caption")
    if not isinstance(caption, str):
        return False
    caption = caption.strip()
    if not caption:
        return False
    return caption


class AttributeCaptioner:
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

    async def generate_captions(
        self,
        prompts: List[Dict[str, Any]],
        system_prompt: str,
    ) -> Dict[str, Optional[str]]:
        results: Dict[str, Optional[str]] = {}
        async for item in self.generator.generate_stream(
            prompts=prompts,
            system_prompt=system_prompt,
            validate_func=_validate_caption,
        ):
            if not item:
                continue
            object_id = item.get("id")
            result = item.get("result")
            results[object_id] = result
        return results


def _collect_crops(
    video_path: str,
    obj_node: Dict[str, Any],
    output_dir: str,
    keyframe_count: int,
    padding_ratio: float,
    keyframe_selector: KeyframeClustering,
) -> Tuple[List[int], List[str]]:
    bboxes = obj_node.get("bboxes", {})
    if not bboxes:
        return [], []

    frames_dict = {
        int(frame_idx): {"box": box, "conf": 1.0}
        for frame_idx, box in bboxes.items()
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    keyframes = keyframe_selector.select_keyframes(frames_dict, video_width, video_height)
    keyframes = _select_evenly(keyframes, keyframe_count)

    crop_paths: List[str] = []
    for frame_idx in keyframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        bbox = frames_dict[frame_idx]["box"]
        crop = _crop_with_padding(frame, bbox, padding_ratio)
        if crop is None:
            continue

        obj_dir = Path(output_dir) / obj_node["node_id"]
        obj_dir.mkdir(parents=True, exist_ok=True)
        crop_path = obj_dir / f"frame_{frame_idx}.jpg"
        cv2.imwrite(str(crop_path), crop)
        crop_paths.append(str(crop_path))

    cap.release()
    return keyframes, crop_paths


def add_attribute_nodes(
    graph: Dict[str, Any],
    captions: Dict[str, Optional[str]],
    default_caption: str = "unknown",
    attribute_type: str = "appearance",
    overwrite: bool = True,
) -> Dict[str, Any]:
    attribute_nodes = graph.get("attribute_nodes", [])
    edges = graph.get("edges", [])

    if overwrite:
        removed_ids = {
            node.get("node_id")
            for node in attribute_nodes
            if node.get("attribute_type") == attribute_type
        }
        attribute_nodes = [
            node for node in attribute_nodes
            if node.get("attribute_type") != attribute_type
        ]
        if removed_ids:
            edges = [
                edge for edge in edges
                if edge.get("target_id") not in removed_ids
            ]

    for obj_node in graph.get("object_nodes", []):
        object_id = obj_node.get("node_id")
        caption = captions.get(object_id) or default_caption
        attr_id = f"attr_{object_id}"

        attribute_nodes.append({
            "node_id": attr_id,
            "object_node_id": object_id,
            "attribute_type": attribute_type,
            "description": caption,
        })
        edges.append({
            "edge_id": f"edge_has_attribute_{object_id}",
            "source_id": object_id,
            "target_id": attr_id,
            "edge_type": "has_attribute",
            "relation": attribute_type,
        })
        obj_node["appearance"] = caption

    graph["attribute_nodes"] = attribute_nodes
    graph["edges"] = edges
    return graph


def build_prompts(
    graph: Dict[str, Any],
    video_path: str,
    crop_output_dir: str,
    keyframe_count: int,
    padding_ratio: float,
    keyframe_selector: KeyframeClustering,
) -> Tuple[List[Dict[str, Any]], List[AttributeCaption]]:
    prompts: List[Dict[str, Any]] = []
    captures: List[AttributeCaption] = []

    video_name = Path(video_path).stem
    base_output = Path(crop_output_dir) / video_name
    base_output.mkdir(parents=True, exist_ok=True)

    for obj_node in graph.get("object_nodes", []):
        keyframes, crop_paths = _collect_crops(
            video_path=video_path,
            obj_node=obj_node,
            output_dir=str(base_output),
            keyframe_count=keyframe_count,
            padding_ratio=padding_ratio,
            keyframe_selector=keyframe_selector,
        )

        captures.append(
            AttributeCaption(
                object_node_id=obj_node["node_id"],
                caption="",
                keyframes=keyframes,
                crop_paths=crop_paths,
            )
        )

        if not crop_paths:
            continue

        content: List[Dict[str, Any]] = [
            {"type": "image", "image": path}
            for path in crop_paths
        ]
        content.append({
            "type": "text",
            "text": _build_prompt(obj_node.get("object_class", "object"), len(crop_paths)),
        })
        prompts.append({
            "id": obj_node["node_id"],
            "prompt": content,
        })

    return prompts, captures


async def generate_attribute_captions(
    graph: Dict[str, Any],
    video_path: str,
    model_name: str,
    api_keys: Union[str, Iterable[str]],
    crop_output_dir: str,
    keyframe_count: int = 4,
    padding_ratio: float = 0.1,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    system_prompt: str = "You are a professional visual description assistant. Please output JSON strictly in accordance with the requirements.",
) -> Dict[str, Optional[str]]:
    keyframe_selector = KeyframeClustering()
    prompts, _ = build_prompts(
        graph=graph,
        video_path=video_path,
        crop_output_dir=crop_output_dir,
        keyframe_count=keyframe_count,
        padding_ratio=padding_ratio,
        keyframe_selector=keyframe_selector,
    )
    if not prompts:
        return {}

    captioner = AttributeCaptioner(
        model_name=model_name,
        api_keys=api_keys,
        max_concurrent_per_key=max_concurrent_per_key,
        max_retries=max_retries,
    )
    return await captioner.generate_captions(prompts, system_prompt=system_prompt)


def run(
    jsonl: str,
    video: str,
    model_name: str,
    api_keys: Union[str, Iterable[str]],
    output: Optional[str] = None,
    crop_output_dir: str = "output/attribute_crops",
    keyframe_count: int = 4,
    padding_ratio: float = 0.1,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    default_caption: str = "unknown",
    overwrite: bool = True,
) -> None:
    graph = _load_graph(jsonl, video)

    captions = asyncio.run(
        generate_attribute_captions(
            graph=graph,
            video_path=video,
            model_name=model_name,
            api_keys=api_keys,
            crop_output_dir=crop_output_dir,
            keyframe_count=keyframe_count,
            padding_ratio=padding_ratio,
            max_concurrent_per_key=max_concurrent_per_key,
            max_retries=max_retries,
        )
    )

    graph = add_attribute_nodes(
        graph=graph,
        captions=captions,
        default_caption=default_caption,
        overwrite=overwrite,
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
