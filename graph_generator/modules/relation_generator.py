import asyncio
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2

from api_sync.api import StreamGenerator
from api_sync.utils.parser import JSONParser
from modules.graph_filter import GraphFilter


SPATIAL_PROMPT_TEMPLATE = """## Role
You are a detail-oriented Video Relationship Annotator responsible for reviewing
sequences of sampled video frames and extracting a comprehensive set of spatial
relationships. All relationships must be visually grounded, type-consistent, and
strictly follow the defined schema.
## Task Context
- Videos are sampled at 2 fps.
- Each frame already has the two target objects marked by red integer IDs.
- The red number at the lower-right corner of each frame is the frame index. You must use these frame indices to determine the time spans of relationships.
- You must use the marked IDs directly and must not detect new objects.
- Your analysis should consider all provided frames jointly.
## Input Format
- Input is an ordered sequence of frames from one video.
- In each frame, exactly two target objects are marked:
  - Object A: id={id_a}, class={class_a}
  - Object B: id={id_b}, class={class_b}
- Frame indices in this sequence: {frame_ids}
## Guidelines
- Extract only purely spatial (physical or geometric) relationships visible in the
3D scene. Do not extract any relationships that indicate state, function, action,
or purpose.
- Exclude all temporal, social, functional, or attentional relationships, as well as
any stateful or action-based verbs.
- Each relationship must be visually supported by the visual frames.
- Ensure logical consistency with common sense and real-world physics; do NOT
output implausible or unsupported relationships.
- Think in terms of 3D spatial layout by using depth information derived from
world knowledge and visual cues, not just 2D image positions. Do NOT rely
solely on 2D bounding box coordinates. Do NOT output 'left of' or 'right of'.
- Use precise, explicit, and non-redundant verbs.
- Do NOT miss any clear and valid spatial relationships between objects.
### Temporal Grounding
- Output relationships with one or more time spans ([[start_frame, end_frame], ...]), as continuous intervals where the relationship is visually supported and both objects are present.
## Output Format
Return a single valid JSON object:
{{
  "relationships": [
    [subject_id, predicate_verb, object_id, [[start_frame, end_frame], ...]]
  ]
}}
## Output Specifications
- subject_id/object_id must be integers and should be either {id_a} or {id_b}.
- If there are no valid relationships, output: {{"relationships": []}}.
- Output strict valid JSON only, with key "relationships".
"""

NON_SPATIAL_PROMPT_TEMPLATE = """## Role
You are a detail-oriented **Video Relationship Annotator** tasked with reviewing
sequences of sampled video frames and extracting a comprehensive set of
**temporal (non-spatial) relationships**. Ensure all extracted relationships are
visually grounded, type-consistent, and strictly follow the defined schema.
You are analyzing videos sampled at 2 fps. Each frame contains detected objects
with bounding boxes. Analyze all frames jointly and output temporal relationships.
## Task Context
- Videos are sampled at 2 fps.
- Each frame already has the two target objects marked by red integer IDs.
- The red number at the lower-right corner of each frame is the frame index. You must use these frame indices to determine the time spans of relationships.
- Two target objects are already marked in red:
  - Object A: id={id_a}, class={class_a}
  - Object B: id={id_b}, class={class_b}
- Frame indices in this sequence: {frame_ids}
## Relationship Taxonomy: Classify each relationship into exactly ONE category:
1. **Functional — Contact / Manipulation**
- Direct physical interaction where an animate subject alters or uses the state of
another object.
- Subject: animate | Object: animate or inanimate
- Exclude: Pure motion or gaze without contact.
2. **Stateful — Attachment / Possession-like**
- Visually grounded, time-persistent attachment or carrying relationships that indicate sustained physical association rather than instantaneous action.
- Subject: animate or inanimate | Object: animate or inanimate
- Exclude: Abstract ownership or purely spatial layout.
3. **Motion — Relative Movement**
- Temporal changes in relative position or movement trajectory between entities.
- Subject: movable (animate or movable inanimate) | Object: animate or inanimate
- Exclude: Static layout or manipulation actions.
4. **Social — Animate-to-Animate Interaction**
- Communication, coordination, or interpersonal acts between animate agents.
- Subject/Object: animate
- Exclude: One-sided attention or non-social contact.
5. **Attentional — Gaze / Focus (includes Camera)**
- Visual attention or camera focus directed at another object or agent.
- Subject: animate or camera (with object_id = -1) | Object: animate or inanimate
- Exclude: Communication or manipulation.
6. **Event-Level — Goal-Directed Multi-Step Activity**
- Higher-level, time-extended actions combining multiple functional, causality or
motion relations into a single purposeful event.
- Subject: animate or inanimate | Object: animate or inanimate
- Exclude: Single short actions or ungrounded intent.
## Core Analysis Logic & Constraints
1. **Object Typing:** - Animate: humans, animals, humanoid robots - Inanimate:
cars, tools, furniture, etc. - Camera: unseen observer/recorder, always object_id = -1
2. **Typing Rules:** - Functional / Social: Subject must be animate - Motion:
Subject must be movable - Attentional: Subject must be animate - Social: Both subject and object must be animate
3. **Extraction Basis:**
- All relationships must be visually supported and logically consistent with common sense. Do **NOT** infer relationships not visually evidenced or that contradict common sense.
- Relationships must have temporal grounding: define one or more time spans [[start_frame, end_frame], ...] as continuous frame intervals supported by visual evidence. If an object in a labeled relationship disappears from subsequent frames, end the relationship at the object's last visible frame. Do not continue relationships if objects become occluded or are missing.
- Do NOT create self-relations (subject_id == object_id), e.g. (i, verb, i, ...).
## Output Format
Return one valid JSON object:
{{
  "relationships": [
    [subject_id, predicate_verb, object_id, [[start_frame, end_frame], ...], relationship_type]
  ]
}}
## Output Details
- relationship_type must be one of:
  functional, stateful, motion, social, attentional, event_level
- If no valid relation exists, output: {{"relationships": []}}
- Output strict valid JSON only, with key "relationships".
"""

@dataclass
class PairFrameInfo:
    pair_id: str
    object_a: str
    object_b: str
    object_a_gid: int
    object_b_gid: int
    frames: List[int]
    image_paths: List[str]
    class_a: str
    class_b: str


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


def _draw_label(frame: Any, bbox: List[float], label: str) -> bool:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    draw_x1 = max(0, min(width - 1, int(x1)))
    draw_y1 = max(0, min(height - 1, int(y1)))
    draw_x2 = max(0, min(width - 1, int(x2)))
    draw_y2 = max(0, min(height - 1, int(y2)))
    if draw_x2 <= draw_x1 or draw_y2 <= draw_y1:
        return False

    cx = int((draw_x1 + draw_x2) / 2)
    cy = int((draw_y1 + draw_y2) / 2)
    cv2.putText(
        frame,
        label,
        (cx, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return True


def _draw_frame_index_bottom_right(frame: Any, frame_idx: int) -> None:
    h, w = frame.shape[:2]
    text = f"frame: {int(frame_idx)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    margin = 10
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(0, w - tw - margin)
    y = max(th + baseline, h - margin)
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def _collect_pair_frames(
    video_path: str,
    obj_a: Dict[str, Any],
    obj_b: Dict[str, Any],
    output_dir: str,
    min_shared_frames: int,
) -> Tuple[List[int], List[str]]:
    bboxes_a = obj_a.get("bboxes", {})
    bboxes_b = obj_b.get("bboxes", {})
    if not bboxes_a or not bboxes_b:
        return [], []

    frames_a = {int(frame_idx): box for frame_idx, box in bboxes_a.items()}
    frames_b = {int(frame_idx): box for frame_idx, box in bboxes_b.items()}
    overlap_frames = sorted(set(frames_a.keys()) & set(frames_b.keys()))
    if min_shared_frames > 0 and len(overlap_frames) < min_shared_frames:
        return [], []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    gid_a = int(obj_a.get("global_track_id", -1))
    gid_b = int(obj_b.get("global_track_id", -1))
    pair_dir = Path(output_dir) / f"{obj_a['node_id']}__{obj_b['node_id']}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    out_paths: List[str] = []
    used_frames: List[int] = []

    for frame_idx in overlap_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        annotated = frame.copy()
        ok_a = _draw_label(annotated, frames_a[frame_idx], str(gid_a))
        ok_b = _draw_label(annotated, frames_b[frame_idx], str(gid_b))
        if not ok_a or not ok_b:
            continue
        _draw_frame_index_bottom_right(annotated, frame_idx)

        path_img = pair_dir / f"frame_{frame_idx}.jpg"
        cv2.imwrite(str(path_img), annotated)
        out_paths.append(str(path_img))
        used_frames.append(frame_idx)

    cap.release()
    return used_frames, out_paths


def _extract_spatial_relationships(response: str) -> Optional[List[List[Any]]]:
    parsed = JSONParser.parse(response)
    if not isinstance(parsed, dict):
        return None
    rels = parsed.get("relationships")
    if not isinstance(rels, list):
        return None

    out: List[List[Any]] = []
    for rel in rels:
        if not isinstance(rel, list) or len(rel) != 4:
            continue

        subject_id, predicate, object_id, spans = rel
        try:
            sid = int(subject_id)
            oid = int(object_id)
        except Exception:
            continue

        pred = str(predicate).strip()
        if not pred:
            continue

        if not isinstance(spans, list):
            continue
        norm_spans: List[List[int]] = []
        for span in spans:
            if not isinstance(span, list) or len(span) != 2:
                continue
            try:
                s = int(span[0])
                e = int(span[1])
            except Exception:
                continue
            if e < s:
                s, e = e, s
            norm_spans.append([s, e])
        if not norm_spans:
            continue

        out.append([sid, pred, oid, norm_spans])

    return out


def _extract_contacting_relationships(response: str) -> Optional[List[List[Any]]]:
    parsed = JSONParser.parse(response)
    if not isinstance(parsed, dict):
        return None
    rels = parsed.get("relationships")
    if not isinstance(rels, list):
        return None

    allowed_types = {"functional", "stateful", "motion", "social", "attentional", "event_level"}
    out: List[List[Any]] = []
    for rel in rels:
        if not isinstance(rel, list) or len(rel) != 5:
            continue

        subject_id, predicate, object_id, spans, rel_type = rel
        try:
            sid = int(subject_id)
            oid = int(object_id)
        except Exception:
            continue

        pred = str(predicate).strip()
        rtype = str(rel_type).strip().lower()
        if not pred or rtype not in allowed_types:
            continue

        if not isinstance(spans, list):
            continue
        norm_spans: List[List[int]] = []
        for span in spans:
            if not isinstance(span, list) or len(span) != 2:
                continue
            try:
                s = int(span[0])
                e = int(span[1])
            except Exception:
                continue
            if e < s:
                s, e = e, s
            norm_spans.append([s, e])
        if not norm_spans:
            continue

        out.append([sid, pred, oid, norm_spans, rtype])

    return out


def _merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    if not intervals:
        return []
    intervals = sorted([[int(a), int(b)] for a, b in intervals], key=lambda x: (x[0], x[1]))
    merged: List[List[int]] = [intervals[0]]
    for s, e in intervals[1:]:
        last = merged[-1]
        if s <= last[1] + 1:
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])
    return merged


def collect_pair_infos(
    graph: Dict[str, Any],
    video_path: str,
    crop_output_dir: str,
    min_shared_frames: int,
) -> List[PairFrameInfo]:
    pair_infos: List[PairFrameInfo] = []

    video_name = Path(video_path).stem
    base_output = Path(crop_output_dir) / video_name
    base_output.mkdir(parents=True, exist_ok=True)

    object_nodes = graph.get("object_nodes", [])
    for i in range(len(object_nodes)):
        for j in range(i + 1, len(object_nodes)):
            obj_a = object_nodes[i]
            obj_b = object_nodes[j]

            frames, images = _collect_pair_frames(
                video_path=video_path,
                obj_a=obj_a,
                obj_b=obj_b,
                output_dir=str(base_output),
                min_shared_frames=min_shared_frames,
            )
            if not frames or not images:
                continue

            gid_a = int(obj_a.get("global_track_id", -1))
            gid_b = int(obj_b.get("global_track_id", -1))
            pair_id = f"{obj_a['node_id']}|{obj_b['node_id']}"
            pair_infos.append(
                PairFrameInfo(
                    pair_id=pair_id,
                    object_a=obj_a["node_id"],
                    object_b=obj_b["node_id"],
                    object_a_gid=gid_a,
                    object_b_gid=gid_b,
                    frames=frames,
                    image_paths=images,
                    class_a=obj_a.get("object_class", "object"),
                    class_b=obj_b.get("object_class", "object"),
                )
            )
    return pair_infos


def build_prompts_from_pair_infos(
    pair_infos: List[PairFrameInfo],
    prompt_template: str,
) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    for pair_info in pair_infos:
        content: List[Dict[str, Any]] = [{"type": "image", "image": p} for p in pair_info.image_paths]
        content.append(
            {
                "type": "text",
                "text": prompt_template.format(
                    id_a=pair_info.object_a_gid,
                    id_b=pair_info.object_b_gid,
                    class_a=pair_info.class_a,
                    class_b=pair_info.class_b,
                    frame_ids=", ".join(str(x) for x in pair_info.frames),
                ),
            }
        )
        prompts.append({"id": pair_info.pair_id, "prompt": content})
    return prompts


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
        validate_func,
        tag: str = "relation",
        verbose: bool = False,
    ) -> Dict[str, List[List[Any]]]:
        results: Dict[str, List[List[Any]]] = {}
        async for item in self.generator.generate_stream(
            prompts=prompts,
            system_prompt=system_prompt,
            validate_func=validate_func,
        ):
            if not item:
                continue
            pid = str(item.get("id"))
            rels = item.get("result")
            if isinstance(rels, list):
                results[pid] = rels
                if verbose:
                    print(f"[relation][{tag}] {pid}: {rels}")
        return results


def build_grouped_relation_edges(
    pair_infos: List[PairFrameInfo],
    relation_results: Dict[str, List[List[Any]]],
    relation_arity: int,
    verbose: bool = False,
    tag: str = "relation",
) -> List[Dict[str, Any]]:
    pair_map = {p.pair_id: p for p in pair_infos}

    edge_to_spans: Dict[Tuple[int, str, int], List[List[int]]] = {}

    for pair_id, rels in relation_results.items():
        pair_info = pair_map.get(pair_id)
        if pair_info is None:
            continue
        valid_ids = {int(pair_info.object_a_gid), int(pair_info.object_b_gid)}

        for rel in rels:
            if not isinstance(rel, list) or len(rel) != relation_arity:
                continue
            sid, pred, oid, spans = rel[:4]
            sid = int(sid)
            oid = int(oid)
            pred = str(pred).strip()
            if sid == oid or not pred:
                continue
            if sid not in valid_ids or oid not in valid_ids:
                continue
            if not isinstance(spans, list):
                continue

            norm_spans = _merge_intervals([[int(s), int(e)] for s, e in spans])
            if not norm_spans:
                continue

            if relation_arity == 5:
                rtype = str(rel[4]).strip().lower()
                edge_to_spans.setdefault((sid, pred, oid, rtype), []).extend(norm_spans)
            else:
                edge_to_spans.setdefault((sid, pred, oid), []).extend(norm_spans)
        if verbose:
            print(f"[relation][{tag}] accepted pair={pair_id} raw={len(rels)}")

    new_edges: List[List[Any]] = []
    for key, spans in sorted(edge_to_spans.items(), key=lambda x: x[0]):
        merged = _merge_intervals(spans)
        if relation_arity == 5:
            sid, pred, oid, rtype = key
            new_edges.append([sid, pred, oid, merged, rtype])
        else:
            sid, pred, oid = key
            new_edges.append([sid, pred, oid, merged])

    grouped: Dict[int, List[List[Any]]] = {}
    for edge in new_edges:
        sid = int(edge[0])
        grouped.setdefault(sid, []).append(edge)
    grouped_edges: List[Dict[str, Any]] = []
    for sid in sorted(grouped.keys()):
        rels = grouped[sid]
        rels.sort(key=lambda x: (x[1], x[2]))
        grouped_edges.append({"subject_id": sid, "relationships": rels})
    if verbose:
        rel_cnt = sum(len(x.get("relationships", [])) for x in grouped_edges)
        print(f"[relation][{tag}] grouped subjects={len(grouped_edges)} relations={rel_cnt}")

    return grouped_edges


def merge_spatial_and_contacting_edges(
    spatial_grouped: List[Dict[str, Any]],
    contacting_grouped: List[Dict[str, Any]],
    gid_to_node_id: Dict[int, str],
) -> List[Dict[str, Any]]:
    merged: Dict[str, List[Dict[str, Any]]] = {}

    def _to_time_spans(spans: Any) -> List[Dict[str, int]]:
        out: List[Dict[str, int]] = []
        if not isinstance(spans, list):
            return out
        for seg in spans:
            if not isinstance(seg, list) or len(seg) != 2:
                continue
            try:
                s = int(seg[0])
                e = int(seg[1])
            except Exception:
                continue
            if e < s:
                s, e = e, s
            out.append({"start_frame": s, "end_frame": e})
        return out

    for item in spatial_grouped:
        try:
            sid = int(item.get("subject_id"))
        except Exception:
            continue
        subject_node_id = gid_to_node_id.get(sid)
        if not subject_node_id:
            continue
        for rel in item.get("relationships", []):
            if not isinstance(rel, list) or len(rel) != 4:
                continue
            _, pred, oid, spans = rel
            try:
                object_gid = int(oid)
            except Exception:
                continue
            object_node_id = gid_to_node_id.get(object_gid)
            if not object_node_id or object_node_id == subject_node_id:
                continue
            norm_spans = _to_time_spans(spans)
            if not norm_spans:
                continue
            merged.setdefault(subject_node_id, []).append(
                {
                    "object_id": object_node_id,
                    "predicate_verb": str(pred),
                    "time_spans": norm_spans,
                    "edge_type": "spatial",
                }
            )

    for item in contacting_grouped:
        try:
            sid = int(item.get("subject_id"))
        except Exception:
            continue
        subject_node_id = gid_to_node_id.get(sid)
        if not subject_node_id:
            continue
        for rel in item.get("relationships", []):
            if not isinstance(rel, list) or len(rel) != 5:
                continue
            _, pred, oid, spans, rtype = rel
            try:
                object_gid = int(oid)
            except Exception:
                continue
            object_node_id = gid_to_node_id.get(object_gid)
            if not object_node_id or object_node_id == subject_node_id:
                continue
            norm_spans = _to_time_spans(spans)
            if not norm_spans:
                continue
            merged.setdefault(subject_node_id, []).append(
                {
                    "object_id": object_node_id,
                    "predicate_verb": str(pred),
                    "time_spans": norm_spans,
                    "edge_type": "contacting",
                    "relationship_type": str(rtype),
                }
            )

    out: List[Dict[str, Any]] = []
    for sid in sorted(merged.keys()):
        rels = merged[sid]
        rels.sort(key=lambda x: (x["edge_type"], x["predicate_verb"], x["object_id"]))
        out.append({"subject_id": sid, "relationships": rels})
    return out


async def generate_relation_edges(
    graph: Dict[str, Any],
    video_path: str,
    model_name: str,
    api_keys: Optional[Union[str, Iterable[str]]],
    crop_output_dir: str,
    min_shared_frames: int = 1,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    spatial_system_prompt: str = "",
    contacting_system_prompt: str = "",
    verbose: bool = False,
) -> Tuple[Dict[str, Any], List[PairFrameInfo]]:
    pair_infos = collect_pair_infos(
        graph=graph,
        video_path=video_path,
        crop_output_dir=crop_output_dir,
        min_shared_frames=min_shared_frames,
    )
    if verbose:
        print(f"[relation] collected pairs: {len(pair_infos)}")
        for p in pair_infos[:10]:
            print(
                f"[relation][pair] {p.pair_id} ids=({p.object_a_gid},{p.object_b_gid}) "
                f"frames={len(p.frames)} range={p.frames[0]}-{p.frames[-1]}"
            )
        if len(pair_infos) > 10:
            print(f"[relation] ... ({len(pair_infos) - 10} more pairs)")
    spatial_prompts = build_prompts_from_pair_infos(pair_infos, SPATIAL_PROMPT_TEMPLATE)
    contacting_prompts = build_prompts_from_pair_infos(pair_infos, NON_SPATIAL_PROMPT_TEMPLATE)
    if verbose:
        print(f"[relation] spatial prompts: {len(spatial_prompts)}")
        print(f"[relation] contacting prompts: {len(contacting_prompts)}")

    if not spatial_prompts and not contacting_prompts:
        graph["edges"] = []
        return graph, []

    generator = RelationGenerator(
        model_name=model_name,
        api_keys=api_keys,
        max_concurrent_per_key=max_concurrent_per_key,
        max_retries=max_retries,
    )
    spatial_results = await generator.generate_relations(
        spatial_prompts,
        system_prompt=spatial_system_prompt,
        validate_func=lambda response: rels if (rels := _extract_spatial_relationships(response)) is not None else False,
        tag="spatial",
        verbose=verbose,
    )
    contacting_results = await generator.generate_relations(
        contacting_prompts,
        system_prompt=contacting_system_prompt,
        validate_func=lambda response: rels if (rels := _extract_contacting_relationships(response)) is not None else False,
        tag="contacting",
        verbose=verbose,
    )
    if verbose:
        print(f"[relation] spatial responses: {len(spatial_results)}")
        print(f"[relation] contacting responses: {len(contacting_results)}")

    spatial_grouped = build_grouped_relation_edges(
        pair_infos,
        spatial_results,
        relation_arity=4,
        verbose=verbose,
        tag="spatial",
    )
    contacting_grouped = build_grouped_relation_edges(
        pair_infos,
        contacting_results,
        relation_arity=5,
        verbose=verbose,
        tag="contacting",
    )
    if verbose:
        spatial_rel_cnt = sum(len(x.get("relationships", [])) for x in spatial_grouped if isinstance(x, dict))
        contacting_rel_cnt = sum(len(x.get("relationships", [])) for x in contacting_grouped if isinstance(x, dict))
        print(f"[relation] spatial grouped subjects={len(spatial_grouped)} relations={spatial_rel_cnt}")
        print(f"[relation] contacting grouped subjects={len(contacting_grouped)} relations={contacting_rel_cnt}")

    gid_to_node_id: Dict[int, str] = {}
    for obj in graph.get("object_nodes") or []:
        node_id = str(obj.get("node_id", "")).strip()
        if not node_id:
            continue
        try:
            gid = int(obj.get("global_track_id"))
        except Exception:
            continue
        gid_to_node_id[gid] = node_id

    graph.pop("spatial_edges", None)
    graph.pop("contacting_edges", None)
    graph["edges"] = merge_spatial_and_contacting_edges(
        spatial_grouped,
        contacting_grouped,
        gid_to_node_id=gid_to_node_id,
    )
    graph = GraphFilter().normalize_graph(graph, filter_objects=False)
    if verbose:
        merged_rel_cnt = sum(len(x.get("relationships", [])) for x in graph["edges"] if isinstance(x, dict))
        print(f"[relation] merged subjects={len(graph['edges'])} relations={merged_rel_cnt}")
    return graph, pair_infos


def run(
    jsonl: str,
    video: str,
    model_name: str,
    api_keys: Optional[Union[str, Iterable[str]]] = None,
    output: Optional[str] = None,
    crop_output_dir: str = "output/relation_crops",
    save_intermediate_frames: bool = False,
    min_shared_frames: int = 1,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    spatial_system_prompt: str = "",
    contacting_system_prompt: str = "",
    verbose: bool = False,
) -> None:
    graph = _load_graph(jsonl, video)
    tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    try:
        effective_crop_dir = crop_output_dir
        if not save_intermediate_frames:
            tmp_dir = tempfile.TemporaryDirectory(prefix="relation_crops_")
            effective_crop_dir = tmp_dir.name

        graph, _ = asyncio.run(
            generate_relation_edges(
                graph=graph,
                video_path=video,
                model_name=model_name,
                api_keys=api_keys,
                crop_output_dir=effective_crop_dir,
                min_shared_frames=min_shared_frames,
                max_concurrent_per_key=max_concurrent_per_key,
                max_retries=max_retries,
                spatial_system_prompt=spatial_system_prompt,
                contacting_system_prompt=contacting_system_prompt,
                verbose=verbose,
            )
        )
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()

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
