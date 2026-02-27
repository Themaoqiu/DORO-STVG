import asyncio
import json
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2

from api_sync.api import StreamGenerator
from api_sync.utils.parser import JSONParser


PROMPT_TEMPLATE = (
    "You will see two objects from different shots of the same video. "
    "Images are ordered as: full frame A, crop A, full frame B, crop B. "
    "Use both the images and the appearance descriptions to decide if they are the same real-world entity. "
    "If you are unsure, output unsure. "
    "Return strict JSON only:\n"
    "{{\"same_entity\": \"yes|no|unsure\", \"confidence\": 0-1}}.\n"
    "Object A class: {class_a}. Appearance A: {appearance_a}. "
    "Object B class: {class_b}. Appearance B: {appearance_b}."
)


@dataclass
class ObjectView:
    frame_idx: int
    shot_id: int
    full_path: str
    crop_path: str


@dataclass
class ObjectBundle:
    node_id: str
    object_class: str
    appearance: str
    shot_ids: List[int]
    views: List[ObjectView]


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


def _appearance_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()


def _select_frame_in_shot(
    shot: Dict[str, Any],
    bboxes: Dict[str, List[float]],
) -> Optional[int]:
    start = shot["start_frame"]
    end = shot["end_frame"]
    frames = sorted(int(k) for k in bboxes.keys() if start <= int(k) <= end)
    if not frames:
        return None
    return frames[len(frames) // 2]


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


def _build_object_bundles(
    graph: Dict[str, Any],
    video_path: str,
    output_dir: str,
    max_views_per_object: Optional[int],
    padding_ratio: float,
) -> List[ObjectBundle]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    temporal_nodes = {clip["clip_id"]: clip for clip in graph.get("temporal_nodes", [])}
    video_name = Path(video_path).stem
    base_output = Path(output_dir) / video_name
    base_output.mkdir(parents=True, exist_ok=True)

    bundles: List[ObjectBundle] = []
    view_limit = None
    if isinstance(max_views_per_object, int) and max_views_per_object > 0:
        view_limit = max_views_per_object

    for obj in graph.get("object_nodes", []):
        bboxes = obj.get("bboxes", {})
        if not bboxes:
            continue
        views: List[ObjectView] = []
        for shot_id in obj.get("shot_ids", []):
            if shot_id not in temporal_nodes:
                continue
            shot = temporal_nodes[shot_id]
            frame_idx = _select_frame_in_shot(shot, bboxes)
            if frame_idx is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            bbox = bboxes.get(str(frame_idx))
            if bbox is None:
                continue
            crop = _crop_with_padding(frame, bbox, padding_ratio)
            if crop is None:
                continue

            obj_dir = base_output / obj["node_id"]
            obj_dir.mkdir(parents=True, exist_ok=True)
            full_path = obj_dir / f"shot_{shot_id}_frame_{frame_idx}_full.jpg"
            crop_path = obj_dir / f"shot_{shot_id}_frame_{frame_idx}_crop.jpg"
            cv2.imwrite(str(full_path), frame)
            cv2.imwrite(str(crop_path), crop)
            views.append(
                ObjectView(
                    frame_idx=frame_idx,
                    shot_id=shot_id,
                    full_path=str(full_path),
                    crop_path=str(crop_path),
                )
            )
            if view_limit is not None and len(views) >= view_limit:
                break

        if not views:
            continue

        bundles.append(
            ObjectBundle(
                node_id=obj["node_id"],
                object_class=obj.get("object_class", "object"),
                appearance=obj.get("appearance", ""),
                shot_ids=obj.get("shot_ids", []),
                views=views,
            )
        )

    cap.release()
    return bundles


def _build_prompt(bundle_a: ObjectBundle, bundle_b: ObjectBundle) -> str:
    return PROMPT_TEMPLATE.format(
        class_a=bundle_a.object_class,
        class_b=bundle_b.object_class,
        appearance_a=bundle_a.appearance or "unknown",
        appearance_b=bundle_b.appearance or "unknown",
    )


def _extract_same_entity(response: str) -> Optional[Dict[str, Any]]:
    parsed = JSONParser.parse(response)
    if not isinstance(parsed, dict):
        return None
    raw_value = str(parsed.get("same_entity", "")).strip().lower()
    if raw_value in {"yes", "true", "same", "1"}:
        same = True
    elif raw_value in {"no", "false", "different", "0"}:
        same = False
    elif raw_value in {"unsure", "unknown", "uncertain"}:
        same = None
    else:
        return None
    confidence = parsed.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence = max(0.0, min(float(confidence), 1.0))
    else:
        confidence = None
    return {"same": same, "confidence": confidence}


def _validate_response(response: str) -> Union[Dict[str, Any], bool]:
    result = _extract_same_entity(response)
    if result is None:
        return False
    return result


class ReferenceEdgeGenerator:
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

    async def generate(
        self,
        prompts: List[Dict[str, Any]],
        system_prompt: str,
        verbose: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        async for item in self.generator.generate_stream(
            prompts=prompts,
            system_prompt=system_prompt,
            validate_func=_validate_response,
        ):
            if not item:
                continue
            result_id = item.get("id")
            result = item.get("result")
            if result_id and result:
                results[result_id] = result
                if verbose:
                    print(f"[reference] {result_id} -> {result}")
        return results


def _build_prompts(
    bundles: List[ObjectBundle],
    max_pairs_per_object: int,
    similarity_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[str, ObjectBundle, ObjectBundle]]]:
    prompts: List[Dict[str, Any]] = []
    pair_map: Dict[str, Tuple[str, ObjectBundle, ObjectBundle]] = {}

    for i, bundle_a in enumerate(bundles):
        candidates: List[Tuple[float, ObjectBundle]] = []
        for j, bundle_b in enumerate(bundles):
            if j <= i:
                continue
            if bundle_a.object_class != bundle_b.object_class:
                continue
            if set(bundle_a.shot_ids) & set(bundle_b.shot_ids):
                continue
            similarity = _appearance_similarity(bundle_a.appearance, bundle_b.appearance)
            if similarity < similarity_threshold:
                continue
            candidates.append((similarity, bundle_b))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for similarity, bundle_b in candidates[:max_pairs_per_object]:
            pair_key = f"{bundle_a.node_id}|{bundle_b.node_id}"
            for view_a in bundle_a.views:
                for view_b in bundle_b.views:
                    prompt_id = (
                        f"{pair_key}|"
                        f"{view_a.shot_id}:{view_a.frame_idx}|"
                        f"{view_b.shot_id}:{view_b.frame_idx}"
                    )
                    content: List[Dict[str, Any]] = []
                    content.append({"type": "image", "image": view_a.full_path})
                    content.append({"type": "image", "image": view_a.crop_path})
                    content.append({"type": "image", "image": view_b.full_path})
                    content.append({"type": "image", "image": view_b.crop_path})
                    content.append(
                        {
                            "type": "text",
                            "text": _build_prompt(bundle_a, bundle_b),
                        }
                    )
                    prompts.append({"id": prompt_id, "prompt": content})
                    pair_map[prompt_id] = (pair_key, bundle_a, bundle_b)

    return prompts, pair_map


def _confidence_score(value: Optional[float]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return -1.0


def add_reference_edges(
    graph: Dict[str, Any],
    pair_map: Dict[str, Tuple[str, ObjectBundle, ObjectBundle]],
    results: Dict[str, Dict[str, Any]],
    overwrite: bool = True,
) -> Dict[str, Any]:
    edges = graph.get("edges", [])
    if overwrite:
        edges = [edge for edge in edges if "reference_relationship" not in edge]

    best_by_pair: Dict[str, Dict[str, Any]] = {}
    for prompt_id, result in results.items():
        info = pair_map.get(prompt_id)
        if not info:
            continue
        pair_key, bundle_a, bundle_b = info
        if result.get("same") is not True:
            continue
        confidence = result.get("confidence")
        current = best_by_pair.get(pair_key)
        if current is None or _confidence_score(confidence) > _confidence_score(current.get("confidence")):
            best_by_pair[pair_key] = {
                "bundle_a": bundle_a,
                "bundle_b": bundle_b,
                "confidence": confidence,
            }

    edge_index = len(edges)
    for pair_key in sorted(best_by_pair.keys()):
        entry = best_by_pair[pair_key]
        edge = {
            "edge_id": f"edge_ref_{edge_index}",
            "source_id": entry["bundle_a"].node_id,
            "target_id": entry["bundle_b"].node_id,
            "reference_relationship": "same_entity",
        }
        confidence = entry.get("confidence")
        if confidence is not None:
            edge["confidence"] = confidence
        edges.append(edge)
        edge_index += 1

    graph["edges"] = edges
    return graph


async def generate_reference_edges(
    graph: Dict[str, Any],
    video_path: str,
    model_name: str,
    api_keys: Optional[Union[str, Iterable[str]]],
    crop_output_dir: str,
    max_views_per_object: Optional[int] = None,
    max_pairs_per_object: int = 3,
    similarity_threshold: float = 0.35,
    padding_ratio: float = 0.1,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    system_prompt: str = "You are a visual re-identification assistant. Follow the instructions exactly.",
    verbose: bool = False,
) -> Dict[str, Any]:
    bundles = _build_object_bundles(
        graph=graph,
        video_path=video_path,
        output_dir=crop_output_dir,
        max_views_per_object=max_views_per_object,
        padding_ratio=padding_ratio,
    )
    prompts, pair_map = _build_prompts(
        bundles=bundles,
        max_pairs_per_object=max_pairs_per_object,
        similarity_threshold=similarity_threshold,
    )
    if not prompts:
        return graph

    generator = ReferenceEdgeGenerator(
        model_name=model_name,
        api_keys=api_keys,
        max_concurrent_per_key=max_concurrent_per_key,
        max_retries=max_retries,
    )
    results = await generator.generate(prompts, system_prompt=system_prompt, verbose=verbose)
    graph = add_reference_edges(graph, pair_map, results, overwrite=True)
    return graph


def run(
    jsonl: str,
    video: str,
    model_name: str,
    api_keys: Optional[Union[str, Iterable[str]]] = None,
    output: Optional[str] = None,
    crop_output_dir: str = "output/reference_crops",
    max_views_per_object: Optional[int] = None,
    max_pairs_per_object: int = 3,
    similarity_threshold: float = 0.35,
    padding_ratio: float = 0.1,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    verbose: bool = False,
) -> None:
    graph = _load_graph(jsonl, video)

    graph = asyncio.run(
        generate_reference_edges(
            graph=graph,
            video_path=video,
            model_name=model_name,
            api_keys=api_keys,
            crop_output_dir=crop_output_dir,
            max_views_per_object=max_views_per_object,
            max_pairs_per_object=max_pairs_per_object,
            similarity_threshold=similarity_threshold,
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
