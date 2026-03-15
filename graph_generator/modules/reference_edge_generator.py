import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2

from api_sync.api import StreamGenerator
from api_sync.utils.parser import JSONParser


PROMPT_TEMPLATE = (
    "You will see two objects from different shots of the same video. "
    "Images are ordered as: all boxed frames of object A first, then all boxed frames of object B. "
    "In each image, the target object is highlighted by a green bounding box. "
    "Use both the images and the appearance descriptions to decide if they are the same real-world entity. "
    "If you are unsure, output unsure. "
    "Return strict JSON only:\n"
    "{{\"same_entity\": \"yes|no|unsure\", \"confidence\": 0-1}}.\n"
    "Object A class: {class_a}. Appearance A: {appearance_a}. "
    "Object B class: {class_b}. Appearance B: {appearance_b}."
)
TEST_VISION_PROMPT = (
    "Please describe what you see. "
)


@dataclass
class ObjectView:
    frame_idx: int
    shot_id: int
    boxed_path: str


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


def _select_frames_in_shot(
    shot: Dict[str, Any],
    bboxes: Dict[str, List[float]],
    views_per_shot: int,
) -> List[int]:
    start = shot["start_frame"]
    end = shot["end_frame"]
    frames = sorted(int(k) for k in bboxes.keys() if start <= int(k) <= end)
    if not frames:
        return []
    if views_per_shot <= 1 or len(frames) <= 1:
        return [frames[len(frames) // 2]]
    if len(frames) <= views_per_shot:
        return frames

    last = len(frames) - 1
    indices = [round(i * last / (views_per_shot - 1)) for i in range(views_per_shot)]
    return [frames[i] for i in sorted(set(indices))]


def _draw_box_on_frame(
    frame: Any,
    bbox: List[float],
    object_class: str,
) -> Optional[Any]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    draw_x1 = max(0, min(width - 1, int(x1)))
    draw_y1 = max(0, min(height - 1, int(y1)))
    draw_x2 = max(0, min(width - 1, int(x2)))
    draw_y2 = max(0, min(height - 1, int(y2)))
    if draw_x2 <= draw_x1 or draw_y2 <= draw_y1:
        return None
    annotated = frame.copy()
    cv2.rectangle(annotated, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 3)
    if object_class:
        text_org = (draw_x1, max(20, draw_y1 - 8))
        cv2.putText(
            annotated,
            object_class,
            text_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _build_object_bundles(
    graph: Dict[str, Any],
    video_path: str,
    output_dir: str,
    max_views_per_object: Optional[int],
    padding_ratio: float,
    views_per_shot: int,
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
            frame_indices = _select_frames_in_shot(shot, bboxes, views_per_shot)
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                bbox = bboxes.get(str(frame_idx))
                if bbox is None:
                    continue
                boxed = _draw_box_on_frame(
                    frame=frame,
                    bbox=bbox,
                    object_class=obj.get("object_class", "object"),
                )
                if boxed is None:
                    continue

                obj_dir = base_output / obj["node_id"]
                obj_dir.mkdir(parents=True, exist_ok=True)
                boxed_path = obj_dir / f"shot_{shot_id}_frame_{frame_idx}_boxed.jpg"
                cv2.imwrite(str(boxed_path), boxed)
                views.append(
                    ObjectView(
                        frame_idx=frame_idx,
                        shot_id=shot_id,
                        boxed_path=str(boxed_path),
                    )
                )
                if view_limit is not None and len(views) >= view_limit:
                    break
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


def _node_id_category(node_id: str) -> str:
    node = str(node_id).strip().lower()
    if not node:
        return ""
    if "_" in node:
        return node.split("_", 1)[0]
    return node


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
        candidates: List[ObjectBundle] = []
        cat_a = _node_id_category(bundle_a.node_id)
        for j, bundle_b in enumerate(bundles):
            if j <= i:
                continue
            cat_b = _node_id_category(bundle_b.node_id)
            if not cat_a or not cat_b or cat_a != cat_b:
                continue
            if set(bundle_a.shot_ids) & set(bundle_b.shot_ids):
                continue
            candidates.append(bundle_b)

        for bundle_b in candidates[:max_pairs_per_object]:
            pair_key = f"{bundle_a.node_id}|{bundle_b.node_id}"
            prompt_id = pair_key
            content: List[Dict[str, Any]] = []
            for view_a in bundle_a.views:
                content.append({"type": "image", "image": view_a.boxed_path})
            for view_b in bundle_b.views:
                content.append({"type": "image", "image": view_b.boxed_path})
            content.append(
                {
                    "type": "text",
                    "text": _build_prompt(bundle_a, bundle_b),
                }
            )
            prompts.append({"id": prompt_id, "prompt": content})
            pair_map[prompt_id] = (pair_key, bundle_a, bundle_b)

    return prompts, pair_map


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
            "edge_id": f"ref_{edge_index}",
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
    views_per_shot: int = 3,
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
        views_per_shot=views_per_shot,
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
    views_per_shot: int = 3,
    max_pairs_per_object: int = 3,
    similarity_threshold: float = 0.35,
    padding_ratio: float = 0.1,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    verbose: bool = False,
    test_image_inputs_with_api: bool = False,
    test_max_prompts: int = 3,
) -> None:
    graph = _load_graph(jsonl, video)
    if test_image_inputs_with_api:
        bundles = _build_object_bundles(
            graph=graph,
            video_path=video,
            output_dir=crop_output_dir,
            max_views_per_object=max_views_per_object,
            padding_ratio=padding_ratio,
            views_per_shot=views_per_shot,
        )
        prompts, _ = _build_prompts(
            bundles=bundles,
            max_pairs_per_object=max_pairs_per_object,
            similarity_threshold=similarity_threshold,
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

    graph = asyncio.run(
        generate_reference_edges(
            graph=graph,
            video_path=video,
            model_name=model_name,
            api_keys=api_keys,
            crop_output_dir=crop_output_dir,
            max_views_per_object=max_views_per_object,
            views_per_shot=views_per_shot,
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
