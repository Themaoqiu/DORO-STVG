import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2

from api_sync.api import StreamGenerator
from api_sync.utils.parser import JSONParser
from modules.graph_filter import GraphFilter


PROMPT_TEMPLATE =  """## Role
You are a careful **cross-shot entity matching annotator**.
Your goal is to match ids in Shot A to ids in Shot B only when visual evidence is strong.

## Task Context
- You receive 3 paired images from the same video.
- In each paired image: left = Shot A, right = Shot B.
- Candidate objects are marked with red integer ids.
- Id numbers are local within each shot and may differ across shots.
- The same physical object should keep consistent appearance cues across the 3 pairs.
- Shot A candidate ids: {shot_a_ids}
- Shot B candidate ids: {shot_b_ids}

## Visibility Notes
{visibility_notes}

## Required Analysis Checklist
- Compare each proposed match across all pairs, not just one image.
- Check stable cues: body shape, clothing/texture, size, accessories, relative position/motion pattern.
- Reject pairs with weak or ambiguous evidence.
- Note that the input image pairs are sorted in time order, so there may not be matching objects in the image pairs. You must carefully observe the images and reason with temporal relationships before determining the matched object pairs. Do not force matches.

## Output Rules
- Return only confident same-entity matches as id pairs: [shot_a_id, shot_b_id].
- Do not include explanations.
- Use strict JSON only:
{{
  "matches": [[shot_a_id, shot_b_id], ...]
}}
- If no confident match exists, return: {{"matches": []}}
"""

SYSTEM_PROMPT = (
    "You are a rigorous visual verifier. Carefully inspect all provided paired images, "
    "cross-check consistency before deciding, and output strict JSON only."
)


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


def _node_category(node_id: str) -> str:
    node = str(node_id).strip().lower()
    return node.split("_", 1)[0] if "_" in node else node


def _node_numeric_id(node_id: str) -> Optional[int]:
    m = re.search(r"_(\d+)$", str(node_id).strip())
    if not m:
        return None
    return int(m.group(1))


def _collect_shot_objects(
    graph: Dict[str, Any],
    shot_a: int,
    shot_b: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    objs = graph.get("object_nodes", [])
    objs_a = [o for o in objs if shot_a in o.get("shot_ids", [])]
    objs_b = [o for o in objs if shot_b in o.get("shot_ids", [])]

    cats_a = {_node_category(str(o.get("node_id", ""))) for o in objs_a}
    cats_b = {_node_category(str(o.get("node_id", ""))) for o in objs_b}
    shared_cats = {c for c in cats_a & cats_b if c}

    objs_a = [o for o in objs_a if _node_category(str(o.get("node_id", ""))) in shared_cats]
    objs_b = [o for o in objs_b if _node_category(str(o.get("node_id", ""))) in shared_cats]
    return objs_a, objs_b


def _pick_frames_for_shot(
    shot: Dict[str, Any],
    objects: List[Dict[str, Any]],
    num_frames: int = 3,
    boundary_margin_frames: int = 2,
) -> List[int]:
    start = int(shot["start_frame"])
    end = int(shot["end_frame"])
    boundary_margin = max(0, int(boundary_margin_frames))
    effective_start = start + boundary_margin
    effective_end = end - boundary_margin

    frame_to_count: Dict[int, int] = {}
    frame_to_nodes: Dict[int, set[str]] = {}
    for obj in objects:
        node_id = str(obj.get("node_id", ""))
        for k in obj.get("bboxes", {}).keys():
            fi = int(k)
            if start <= fi <= end:
                frame_to_count[fi] = frame_to_count.get(fi, 0) + 1
                frame_to_nodes.setdefault(fi, set()).add(node_id)

    if not frame_to_count:
        return []

    in_margin = [f for f in frame_to_count.keys() if effective_start <= f <= effective_end]
    candidate_frames = in_margin if in_margin else list(frame_to_count.keys())

    shot_center = (start + end) / 2.0
    ranked = sorted(
        candidate_frames,
        key=lambda f: (
            frame_to_count.get(f, 0),
            -abs(f - shot_center),
            -f,
        ),
        reverse=True,
    )

    selected: List[int] = []
    all_nodes = {str(o.get("node_id", "")) for o in objects if str(o.get("node_id", ""))}
    uncovered = set(all_nodes)
    min_gap = 2
    while uncovered and len(selected) < num_frames:
        best_f = None
        best_gain = -1
        for f in ranked:
            if f in selected:
                continue
            if selected and not all(abs(f - s) >= min_gap for s in selected):
                continue
            gain = len(frame_to_nodes.get(f, set()) & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_f = f
        if best_f is None or best_gain <= 0:
            break
        selected.append(best_f)
        uncovered -= frame_to_nodes.get(best_f, set())

    for f in ranked:
        if all(abs(f - s) >= min_gap for s in selected):
            selected.append(f)
        if len(selected) >= num_frames:
            break

    if len(selected) < num_frames:
        for f in ranked:
            if f not in selected:
                selected.append(f)
            if len(selected) >= num_frames:
                break

    return sorted(selected[:num_frames])


def _visible_ids_in_frame(
    frame_idx: int,
    objects: List[Dict[str, Any]],
    id_map: Dict[str, int],
) -> List[int]:
    ids: List[int] = []
    for obj in objects:
        node_id = str(obj.get("node_id", ""))
        if node_id not in id_map:
            continue
        if str(frame_idx) in obj.get("bboxes", {}):
            ids.append(int(id_map[node_id]))
    return sorted(set(ids))


def _draw_objects_with_ids(
    frame: Any,
    frame_idx: int,
    objects: List[Dict[str, Any]],
    id_map: Dict[str, int],
) -> Any:
    out = frame.copy()
    h, w = out.shape[:2]
    for obj in objects:
        node_id = str(obj.get("node_id", ""))
        if node_id not in id_map:
            continue
        bbox = obj.get("bboxes", {}).get(str(frame_idx))
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(float(x)) for x in bbox]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.putText(
            out,
            str(id_map[node_id]),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def _resize_to_height(img: Any, target_h: int) -> Any:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = max(1, int(round(w * (target_h / h))))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _concat_pair(left: Any, right: Any) -> Any:
    target_h = min(left.shape[0], right.shape[0])
    left_r = _resize_to_height(left, target_h)
    right_r = _resize_to_height(right, target_h)
    return cv2.hconcat([left_r, right_r])


def _build_pair_images(
    graph: Dict[str, Any],
    video_path: str,
    shot_a: int,
    shot_b: int,
    output_dir: str,
    frames_per_shot: int = 3,
    boundary_margin_frames: int = 2,
) -> Tuple[List[str], Dict[str, int], Dict[str, int], List[Dict[str, Any]]]:
    temporal = {int(x["clip_id"]): x for x in graph.get("temporal_nodes", [])}
    if shot_a not in temporal or shot_b not in temporal:
        raise ValueError(f"shot ids not found: shot_a={shot_a}, shot_b={shot_b}")

    objs_a, objs_b = _collect_shot_objects(graph, shot_a, shot_b)
    if not objs_a or not objs_b:
        raise ValueError("No same-category objects found across selected shots.")

    id_map_a: Dict[str, int] = {}
    for o in sorted(objs_a, key=lambda x: str(x["node_id"])):
        node_id = str(o["node_id"])
        nid = _node_numeric_id(node_id)
        if nid is None:
            raise ValueError(f"node_id has no numeric suffix: {node_id}")
        id_map_a[node_id] = nid

    id_map_b: Dict[str, int] = {}
    for o in sorted(objs_b, key=lambda x: str(x["node_id"])):
        node_id = str(o["node_id"])
        nid = _node_numeric_id(node_id)
        if nid is None:
            raise ValueError(f"node_id has no numeric suffix: {node_id}")
        id_map_b[node_id] = nid

    frames_a = _pick_frames_for_shot(
        temporal[shot_a],
        objs_a,
        num_frames=frames_per_shot,
        boundary_margin_frames=boundary_margin_frames,
    )
    frames_b = _pick_frames_for_shot(
        temporal[shot_b],
        objs_b,
        num_frames=frames_per_shot,
        boundary_margin_frames=boundary_margin_frames,
    )
    pair_count = min(len(frames_a), len(frames_b), frames_per_shot)
    if pair_count <= 0:
        raise ValueError("No valid frame pairs found to render.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[str] = []
    pair_meta: List[Dict[str, Any]] = []

    for i in range(pair_count):
        fa = int(frames_a[i])
        fb = int(frames_b[i])

        cap.set(cv2.CAP_PROP_POS_FRAMES, fa)
        ok_a, img_a = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, fb)
        ok_b, img_b = cap.read()
        if not ok_a or not ok_b:
            continue

        ann_a = _draw_objects_with_ids(img_a, fa, objs_a, id_map_a)
        ann_b = _draw_objects_with_ids(img_b, fb, objs_b, id_map_b)
        pair_img = _concat_pair(ann_a, ann_b)
        save_path = out_dir / f"pair_{i}_A{fa}_B{fb}.jpg"
        cv2.imwrite(str(save_path), pair_img)
        out_paths.append(str(save_path))
        pair_meta.append(
            {
                "pair_index": i,
                "frame_a": fa,
                "frame_b": fb,
                "visible_a_ids": _visible_ids_in_frame(fa, objs_a, id_map_a),
                "visible_b_ids": _visible_ids_in_frame(fb, objs_b, id_map_b),
            }
        )

    cap.release()
    if not out_paths:
        raise ValueError("Failed to render paired images.")
    return out_paths, id_map_a, id_map_b, pair_meta


def _build_visibility_notes(pair_meta: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in pair_meta:
        lines.append(
            f"- Pair {item['pair_index']}: "
            f"ShotA(frame {item['frame_a']}) visible ids={item['visible_a_ids']}; "
            f"ShotB(frame {item['frame_b']}) visible ids={item['visible_b_ids']}"
        )
    return "\n".join(lines) if lines else "- No visibility metadata."


def _extract_matches(response: str) -> Optional[Dict[str, Any]]:
    parsed = JSONParser.parse(response)
    if not isinstance(parsed, dict):
        return None
    matches = parsed.get("matches")
    if not isinstance(matches, list):
        return None
    norm: List[List[int]] = []
    for item in matches:
        if not isinstance(item, list) or len(item) != 2:
            continue
        try:
            a = int(item[0])
            b = int(item[1])
        except Exception:
            continue
        norm.append([a, b])
    return {"matches": norm}


async def _run_model(
    model_name: str,
    api_keys: Optional[Union[str, Iterable[str]]],
    image_paths: List[str],
    prompt_text: str,
    max_concurrent_per_key: int = 20,
    max_retries: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    generator = StreamGenerator(
        model_name=model_name,
        api_keys=_resolve_api_keys(api_keys),
        max_concurrent_per_key=max_concurrent_per_key,
        max_retries=max_retries,
        rational=False,
        with_unique_id=True,
    )
    prompt = [{"type": "image", "image": p} for p in image_paths]
    prompt.append({"type": "text", "text": prompt_text})
    prompts = [{"id": "pair_match_test", "prompt": prompt}]

    result_out: Optional[Dict[str, Any]] = None
    async for item in generator.generate_stream(
        prompts=prompts,
        system_prompt=SYSTEM_PROMPT,
        validate_func=lambda resp: out if (out := _extract_matches(resp)) is not None else False,
    ):
        if not item:
            continue
        result_out = item.get("result")
        if verbose:
            print(f"[reference] model result: {result_out}")
    if result_out is None:
        raise RuntimeError("No valid model output received.")
    return result_out


def _apply_matches_to_graph(
    graph: Dict[str, Any],
    matches: List[List[int]],
    id_map_a: Dict[str, int],
    id_map_b: Dict[str, int],
    overwrite_reference_edges: bool = True,
) -> Dict[str, Any]:
    edges = graph.get("edges", [])
    if overwrite_reference_edges:
        edges = [edge for edge in edges if "reference_relationship" not in edge]

    inv_a = {int(v): str(k) for k, v in id_map_a.items()}
    inv_b = {int(v): str(k) for k, v in id_map_b.items()}

    existing_pairs = {
        (str(e.get("source_id", "")), str(e.get("target_id", "")))
        for e in edges
        if isinstance(e, dict)
    }

    edge_index = len(edges)
    for pair in matches:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        try:
            ida = int(pair[0])
            idb = int(pair[1])
        except Exception:
            continue
        src = inv_a.get(ida)
        tgt = inv_b.get(idb)
        if not src or not tgt:
            continue
        if (src, tgt) in existing_pairs:
            continue
        edges.append(
            {
                "edge_id": f"ref_{edge_index}",
                "source_id": src,
                "target_id": tgt,
                "reference_relationship": "same_entity",
            }
        )
        existing_pairs.add((src, tgt))
        edge_index += 1

    graph["edges"] = edges
    return graph


def run(
    jsonl: str,
    video: str,
    model_name: str,
    api_keys: Optional[Union[str, Iterable[str]]] = None,
    output: Optional[str] = None,
    crop_output_dir: str = "output/reference_id_match_test",
    save_intermediate_frames: bool = False,
    shot_a: int = 0,
    shot_b: int = 1,
    frames_per_shot: int = 3,
    boundary_margin_frames: int = 2,
    output_json: Optional[str] = None,
    max_concurrent_per_key: int = 20,
    max_retries: int = 5,
    verbose: bool = False,
    overwrite_reference_edges: bool = True,
    # Kept for CLI compatibility with older calls; currently unused.
    max_views_per_object: Optional[int] = None,
    views_per_shot: int = 3,
    max_pairs_per_object: int = 3,
    repeats_per_pair: int = 3,
    similarity_threshold: float = 0.35,
    padding_ratio: float = 0.1,
    test_image_inputs_with_api: bool = False,
    test_max_prompts: int = 3,
) -> None:
    del max_views_per_object, views_per_shot, max_pairs_per_object, repeats_per_pair, similarity_threshold
    del padding_ratio, test_image_inputs_with_api, test_max_prompts

    graph = _load_graph(jsonl, video)

    tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    effective_crop_dir = crop_output_dir
    if not save_intermediate_frames:
        tmp_dir = tempfile.TemporaryDirectory(prefix="reference_id_match_")
        effective_crop_dir = tmp_dir.name

    try:
        image_paths, id_map_a, id_map_b, pair_meta = _build_pair_images(
            graph=graph,
            video_path=video,
            shot_a=shot_a,
            shot_b=shot_b,
            output_dir=effective_crop_dir,
            frames_per_shot=frames_per_shot,
            boundary_margin_frames=boundary_margin_frames,
        )

        prompt_text = PROMPT_TEMPLATE.format(
            shot_a_ids=sorted(id_map_a.values()),
            shot_b_ids=sorted(id_map_b.values()),
            visibility_notes=_build_visibility_notes(pair_meta),
        )

        result = asyncio.run(
            _run_model(
                model_name=model_name,
                api_keys=api_keys,
                image_paths=image_paths,
                prompt_text=prompt_text,
                max_concurrent_per_key=max_concurrent_per_key,
                max_retries=max_retries,
                verbose=verbose,
            )
        )
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()

    matches = result.get("matches", []) if isinstance(result, dict) else []
    graph = _apply_matches_to_graph(
        graph,
        matches=matches,
        id_map_a=id_map_a,
        id_map_b=id_map_b,
        overwrite_reference_edges=overwrite_reference_edges,
    )
    graph = GraphFilter().normalize_graph(graph, filter_objects=False)

    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "video": video,
                    "shot_a": shot_a,
                    "shot_b": shot_b,
                    "pair_images": image_paths,
                    "shot_a_id_map": {str(v): k for k, v in id_map_a.items()},
                    "shot_b_id_map": {str(v): k for k, v in id_map_b.items()},
                    "model_output": result,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    if output and output != jsonl:
        output_file = Path(output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(graph, ensure_ascii=False) + "\n")
    else:
        _update_jsonl_inplace(jsonl, video, graph)

    if save_intermediate_frames:
        print(f"[reference] pair images: {len(image_paths)} -> {effective_crop_dir}")
    else:
        print(f"[reference] pair images: {len(image_paths)} (temporary, auto-cleaned)")
    print(f"[reference] model matches: {matches}")
    if output_json:
        print(f"[reference] saved result: {output_json}")


if __name__ == "__main__":
    import fire

    fire.Fire(run)
