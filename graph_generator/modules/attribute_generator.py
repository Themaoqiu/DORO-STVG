import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import fire
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from api_sync.api import StreamGenerator
from api_sync.utils.parser import JSONParser
from dependence.dam import DescribeAnythingModel, disable_torch_init


VIDEO_QUERY_TEMPLATE = (
    "Video: {image_tokens}\n"
    "Given the video in the form of a sequence of frames above, describe the object in the masked region in the video in detail."
)

PROMPT_MODES = {
    "focal_prompt": "full+focal_crop",
}

STRUCTURED_EXTRACTION_PROMPT = """## You are an expert in scene understanding. I will give you a short paragraph that describes a video clip.
### Your task is to extract structured information about a single object described in the paragraph. Be careful not to omit any representative object information.
### Please return a JSON with the following fields:
-"object": The main object being described (e.g.,"person","dog", "car"). If the
inference about the object is uncertain based on the description, add "(uncertain)"
after the object name.
- "attributes": A list of ONLY the visual/physical attributes that can be directly
observed about the object itself. Include only:
* Visual appearance: color, shape, size, texture, pattern, material appearance, style, the clothing and appearance of the person.
* Physical properties: state, transparency, reflectiveness, orientation, material
* Design elements: stripes, dots, logos, decorative features. DO NOT include: implied
states, inferred conditions, functional descriptions, or anything that describes the
object’s interaction with its environment.
- "relationships": A list of relationships between this object and other entities (but not clothes and actions) or the environment (e.g., "on top of table", "next to person", "inside container", "facing camera", "part of group", "Leaning against the wall", "carrying a briefcase").
- "actions": A list of actions that the object is performing or movements it is making
(e.g., "rotating", "moving", "falling", "bouncing", "sliding", "right arm extended outward"). Including the subtle movements of the person.
### Important distinctions:
- Attributes = What the object looks like (visual only). Please use ADJECTIVE
form. If you are extracting a person's clothing, put it in attributes.
- Relationships = How the object relates to other things spatially, functionally, or contextually. If you are extracting interactions between objects and other objects or the environment, put them in relationships.
- Actions = What the object is doing or how it‘s moving
- However, please note that the attributes of clothing on a person should not be directly stored as attributes of the person. If the description mentions a brown hat, it should be stored as wear a brown hat rather than just brown.
Now process the following description:
{description}"""


@dataclass
class ObjectDescription:
    object_node_id: str
    global_track_id: int
    raw_description: str
    category: str
    attributes: List[str]
    relationships: List[str]
    actions: List[str]


def _load_graph(jsonl_path: str, video_path: str) -> Dict[str, Any]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            graph_video = data.get("video_path", "")
            if graph_video == video_path or Path(graph_video).stem == Path(video_path).stem:
                return data
    raise ValueError(f"No graph found for video {video_path}")


def _update_jsonl_inplace(jsonl_path: str, video_path: str, graph: Dict[str, Any]) -> None:
    input_path = Path(jsonl_path)
    temp_path = input_path.with_suffix(input_path.suffix + ".tmp")

    replaced = False
    with open(input_path, "r", encoding="utf-8") as f_in, open(temp_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            graph_video = data.get("video_path", "")
            if graph_video == video_path or Path(graph_video).stem == Path(video_path).stem:
                f_out.write(json.dumps(graph, ensure_ascii=False) + "\n")
                replaced = True
            else:
                f_out.write(line)

    if not replaced:
        with open(temp_path, "a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(graph, ensure_ascii=False) + "\n")

    temp_path.replace(input_path)


def _select_evenly(items: List[int], count: int) -> List[int]:
    if count <= 0 or len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    last_idx = len(items) - 1
    indices = [round(i * last_idx / (count - 1)) for i in range(count)]
    return [items[i] for i in indices]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \n\t,.;:[]{}\"'")


def _normalize_string_list(raw_value: Any) -> List[str]:
    if isinstance(raw_value, str):
        raw_items = [item.strip() for item in raw_value.split(",")]
    elif isinstance(raw_value, list):
        raw_items = [str(item) for item in raw_value]
    else:
        raw_items = []

    out: List[str] = []
    seen = set()
    for item in raw_items:
        normalized = _normalize_text(item)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _strip_uncertainty_suffix(category: str) -> str:
    return _normalize_text(re.sub(r"\s*\(uncertain\)\s*$", "", category, flags=re.IGNORECASE))


def _parse_extraction(
    raw_output: str,
    default_category: str,
) -> Tuple[str, List[str], List[str], List[str]]:
    parsed = JSONParser.parse(raw_output)
    if parsed is None:
        match = re.search(r"\{.*\}", raw_output.strip(), flags=re.DOTALL)
        if match is not None:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        parts = [_normalize_text(part) for part in raw_output.split(",")]
        parts = [part for part in parts if part]
        if not parts:
            return default_category, [], [], []

        category = parts[0] or default_category
        attributes: List[str] = []
        seen = {category.lower()}
        for part in parts[1:]:
            key = part.lower()
            if key in seen:
                continue
            seen.add(key)
            attributes.append(part)
        return category, attributes, [], []

    category = _normalize_text(
        str(
            parsed.get("object")
            or parsed.get("category")
            or default_category
        )
    )
    attributes = _normalize_string_list(parsed.get("attributes", []))
    relationships = _normalize_string_list(parsed.get("relationships", []))
    actions = _normalize_string_list(parsed.get("actions", []))

    category_key = category.lower()
    attributes = [attr for attr in attributes if attr.lower() != category_key]
    return category or default_category, attributes, relationships, actions


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
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() != "API_KEYS":
                continue
            value = value.strip().strip('"').strip("'")
            return value
    return ""


def _resolve_api_keys() -> List[str]:
    api_keys = os.getenv("API_KEYS", "")
    if not api_keys:
        api_keys = _load_api_keys_from_project_env()
    keys = _normalize_api_keys(api_keys)
    if not keys:
        raise ValueError("API_KEYS is required in env or graph_generator/.env")
    return keys


def _parse_structured_response(response: str) -> Optional[Dict[str, Any]]:
    parsed = JSONParser.parse(response)
    if parsed is None:
        match = re.search(r"\{.*\}", response.strip(), flags=re.DOTALL)
        if match is None:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    return {
        "object": _normalize_text(str(parsed.get("object") or parsed.get("category") or "")),
        "attributes": _normalize_string_list(parsed.get("attributes", [])),
        "relationships": _normalize_string_list(parsed.get("relationships", [])),
        "actions": _normalize_string_list(parsed.get("actions", [])),
    }


def _load_indexed_masks(mask_json_path: Path) -> Dict[int, Dict[int, np.ndarray]]:
    with open(mask_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    frame_masks: Dict[int, Dict[int, np.ndarray]] = {}
    for frame_idx, entries in enumerate(payload):
        if not isinstance(entries, list):
            raise ValueError(f"Invalid mask file format at frame {frame_idx}: expected a list.")
        frame_map: Dict[int, np.ndarray] = {}
        for entry in entries:
            if "global_track_id" not in entry or "segmentation" not in entry:
                raise ValueError(
                    f"Mask file {mask_json_path} is legacy format without object ids. "
                    "Please rerun main.py with --groundedsam2_mask_output_dir to produce "
                    "*_sam2_masks_indexed.json."
                )
            global_track_id = int(entry["global_track_id"])
            mask = mask_utils.decode(entry["segmentation"])
            if mask.ndim == 3:
                mask = mask[..., 0]
            frame_map[global_track_id] = mask.astype(np.uint8)
        if frame_map:
            frame_masks[frame_idx] = frame_map
    return frame_masks


def _collect_object_views(
    video_path: str,
    global_track_id: int,
    frame_masks: Dict[int, Dict[int, np.ndarray]],
    max_frames: int,
) -> Tuple[List[int], List[Image.Image], List[Image.Image]]:
    candidate_frames = sorted(
        frame_idx
        for frame_idx, objects in frame_masks.items()
        if global_track_id in objects and np.any(objects[global_track_id] > 0)
    )
    sampled_frames = _select_evenly(candidate_frames, max_frames)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    masks: List[Image.Image] = []
    valid_frames: List[int] = []
    valid_images: List[Image.Image] = []
    for frame_idx in sampled_frames:
        mask_np = frame_masks[frame_idx][global_track_id]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        valid_frames.append(frame_idx)
        valid_images.append(Image.fromarray(frame_rgb))
        masks.append(Image.fromarray((mask_np > 0).astype(np.uint8) * 255))
    cap.release()
    return valid_frames, valid_images, masks


class DAMAttributeGenerator:
    def __init__(
        self,
        model_path: str,
        prompt_mode: str = "focal_prompt",
        conv_mode: str = "v1",
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
    ) -> None:
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        disable_torch_init()
        self.model = DescribeAnythingModel(
            model_path=model_path,
            conv_mode=conv_mode,
            prompt_mode=PROMPT_MODES.get(prompt_mode, prompt_mode),
        ).to(self.device)
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    def describe_object(
        self,
        images: List[Image.Image],
        masks: List[Image.Image],
    ) -> str:
        image_tokens = "<image>" * len(images)
        raw_description = _normalize_text(self.model.get_description(
            images,
            masks,
            VIDEO_QUERY_TEMPLATE.format(image_tokens=image_tokens),
            streaming=False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
        ))
        # print(f"[attribute_generator][dam_raw] {raw_description}")
        return raw_description


async def _extract_structured_descriptions(
    model_name: str,
    prompts: List[Dict[str, Any]],
    max_concurrent_per_key: int,
    max_retries: int,
) -> Dict[str, Dict[str, Any]]:
    if not prompts:
        return {}

    generator = StreamGenerator(
        model_name=model_name,
        api_keys=_resolve_api_keys(),
        max_concurrent_per_key=max_concurrent_per_key,
        max_retries=max_retries,
        rational=False,
        with_unique_id=True,
    )
    results: Dict[str, Dict[str, Any]] = {}
    async for item in generator.generate_stream(
        prompts=prompts,
        system_prompt="You are a precise information extraction assistant.",
        validate_func=lambda response: result if (result := _parse_structured_response(response)) is not None else False,
    ):
        if not item:
            continue
        object_id = item.get("id")
        result = item.get("result")
        if object_id is not None and isinstance(result, dict):
            results[object_id] = result
    return results


def apply_attributes_to_object_nodes(
    graph: Dict[str, Any],
    descriptions: Dict[str, ObjectDescription],
    overwrite_object_class: bool = False,
) -> Dict[str, Any]:
    graph.pop("attribute_nodes", None)
    if "edges" in graph:
        graph["edges"] = [
            edge
            for edge in graph.get("edges", [])
            if edge.get("edge_type") != "has_attribute"
            and not str(edge.get("edge_id", "")).startswith("edge_has_attribute_")
        ]

    for obj_node in graph.get("object_nodes", []):
        obj_node.pop("appearance", None)
        obj_node.pop("attribute_source", None)
        obj_node.pop("attribute_frame_indices", None)

        object_id = obj_node.get("node_id")
        info = descriptions.get(object_id)
        if info is None:
            continue

        obj_node["dam_category"] = info.category
        obj_node["attributes"] = info.attributes
        obj_node["relationships"] = info.relationships
        obj_node["actions"] = info.actions
        resolved_category = _strip_uncertainty_suffix(info.category)
        current_class = _normalize_text(str(obj_node.get("object_class", "")))
        if resolved_category and (
            overwrite_object_class or resolved_category.lower() != current_class.lower()
        ):
            obj_node["object_class"] = resolved_category
    return graph


def run(
    jsonl: str,
    video: str,
    model_name: str,
    masks_json: Optional[str] = None,
    model_path: str = "nvidia/DAM-3B-Video",
    output: Optional[str] = None,
    prompt_mode: str = "focal_prompt",
    conv_mode: str = "v1",
    max_frames: int = 8,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    overwrite_object_class: bool = True,
) -> None:
    graph = _load_graph(jsonl, video)
    mask_json_path = (
        Path(masks_json)
        if masks_json
        else Path(__file__).resolve().parents[1] / "output" / "sam2_masks" / f"{Path(video).stem}_sam2_masks_indexed.json"
    )
    if not mask_json_path.exists():
        raise FileNotFoundError(f"Mask json not found: {mask_json_path}")

    frame_masks = _load_indexed_masks(mask_json_path)
    generator = DAMAttributeGenerator(
        model_path=model_path,
        prompt_mode=prompt_mode,
        conv_mode=conv_mode,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    descriptions: Dict[str, ObjectDescription] = {}
    structured_prompts: List[Dict[str, Any]] = []
    pending_objects: Dict[str, Dict[str, Any]] = {}
    for obj_node in graph.get("object_nodes", []):
        object_id = obj_node.get("node_id")
        global_track_id = int(obj_node.get("global_track_id"))
        frame_indices, images, masks = _collect_object_views(
            video_path=video,
            global_track_id=global_track_id,
            frame_masks=frame_masks,
            max_frames=max_frames,
        )
        if not images or not masks:
            print(f"[attribute_generator] skip {object_id}: no valid SAM2 masks found")
            continue

        raw_description = generator.describe_object(
            images=images,
            masks=masks,
        )
        default_category = _normalize_text(str(obj_node.get("object_class", "object"))) or "object"
        structured_prompts.append(
            {
                "id": object_id,
                "prompt": STRUCTURED_EXTRACTION_PROMPT.format(description=raw_description),
            }
        )
        pending_objects[object_id] = {
            "global_track_id": global_track_id,
            "raw_description": raw_description,
            "default_category": default_category,
        }

    structured_results = asyncio.run(
        _extract_structured_descriptions(
            model_name=model_name,
            prompts=structured_prompts,
            max_concurrent_per_key=max_concurrent_per_key,
            max_retries=max_retries,
        )
    )

    for object_id, info in pending_objects.items():
        structured = structured_results.get(object_id)
        raw_description = info["raw_description"]
        default_category = info["default_category"]
        if structured is None:
            category, attributes, relationships, actions = _parse_extraction(
                raw_description,
                default_category=default_category,
            )
        else:
            category = _normalize_text(str(structured.get("object") or default_category)) or default_category
            attributes = _normalize_string_list(structured.get("attributes", []))
            relationships = _normalize_string_list(structured.get("relationships", []))
            actions = _normalize_string_list(structured.get("actions", []))

        descriptions[object_id] = ObjectDescription(
            object_node_id=object_id,
            global_track_id=int(info["global_track_id"]),
            raw_description=raw_description,
            category=category,
            attributes=attributes,
            relationships=relationships,
            actions=actions,
        )
        print(
            f"[attribute_generator] {object_id} -> "
            f"category={category}; attributes={attributes}; "
            f"relationships={relationships}; actions={actions}"
        )

    graph = apply_attributes_to_object_nodes(
        graph=graph,
        descriptions=descriptions,
        overwrite_object_class=overwrite_object_class,
    )

    if output and output != jsonl:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(graph, ensure_ascii=False) + "\n")
    else:
        _update_jsonl_inplace(jsonl, video, graph)


if __name__ == "__main__":
    fire.Fire(run)
