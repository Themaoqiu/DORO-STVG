import asyncio
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2

from api_sync.api import StreamGenerator
from api_sync.utils.parser import JSONParser


SPATIAL_PROMPT_TEMPLATE = """## Role
You are a detail-oriented **Video Relationship Annotator** responsible for reviewing
sequences of sampled video frames and extracting a comprehensive set of **spatial
relationships**. All relationships must be visually grounded, type-consistent, and
strictly follow the defined schema.
Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual,
not implementation-level.
## Task Context
- Videos are sampled at 2 fps.
- Each frame already has the two target objects marked by red boxes and red integer IDs.
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
- Each relationship must be visually supported by the frames.
- Ensure logical consistency with common sense and real-world physics; do NOT
output implausible or unsupported relationships.
- Think in terms of 3D spatial layout by using depth information derived from
world knowledge and visual cues, not just 2D image positions. Do NOT rely
solely on 2D bounding box coordinates. Do NOT output ’left of’ or ’right of’.
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
## Pair Context
- Input is an ordered sequence of frames from one video.
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
- Relationships must have temporal grounding: define one or more time spans [[start_frame, end_frame], ...] as continuous frame intervals supported by visual evidence. If an object in a labeled relationship disappears from subsequent frames, end the relationship at the object’s last visible frame. Do not continue relationships if objects become occluded or are missing.
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


def _normalize_text(text: str) -> str:
    cleaned = str(text).strip().lower().replace("_", " ").replace("-", " ")
    out_chars: List[str] = []
    for ch in cleaned:
        out_chars.append(ch if ch.isalnum() or ch.isspace() else " ")
    return " ".join("".join(out_chars).split())


# Action labels that are clear relation candidates (from AVA label_map),
# mapped to one or multiple normalized relation keys.
ACTION_LABEL_TO_REL_KEYS: Dict[str, List[str]] = {
    "answer phone": ["answer phone", "talk to", "listen to"],
    "carry hold an object": ["carry hold", "hold", "carry"],
    "hit an object": ["hit", "touch"],
    "point to an object": ["point to"],
    "pull an object": ["pull"],
    "push an object": ["push"],
    "ride e g a bike a car a horse": ["ride"],
    "take a photo": ["take a photo"],
    "text on look at a cellphone": ["watch", "look at"],
    "touch an object": ["touch"],
    "watch e g tv": ["watch", "look at"],
    "work on a computer": ["work on", "use"],
    "fight hit a person": ["fight", "hit"],
    "give serve an object to a person": ["give", "serve", "hand"],
    "grab a person": ["grab"],
    "hand shake": ["hand shake", "shake hands", "shake"],
    "hug a person": ["hug"],
    "kiss a person": ["kiss"],
    "lift a person": ["lift"],
    "listen to a person": ["listen to"],
    "push another person": ["push"],
    "sing to e g self a person a group": ["sing to"],
    "take an object from a person": ["take from", "grab from", "take"],
    "talk to e g self a person a group": ["talk to"],
    "watch a person": ["watch", "look at"],
}


PREDICATE_KEY_ALIASES: Dict[str, List[str]] = {
    "watch": ["watch", "look at", "looking at", "look"],
    "listen to": ["listen to", "listening to", "hear"],
    "talk to": ["talk to", "talking to", "speak to", "chat with"],
    "carry hold": ["carry hold", "carry", "hold", "holding", "carrying"],
    "touch": ["touch", "touching"],
    "push": ["push", "pushing"],
    "pull": ["pull", "pulling"],
    "point to": ["point to", "point at"],
    "hit": ["hit", "hitting", "strike"],
    "fight": ["fight", "fighting"],
    "give": ["give", "giving", "serve"],
    "hand": ["hand", "hand over"],
    "grab": ["grab", "grabbing"],
    "hand shake": ["hand shake", "shake hands", "shake"],
    "hug": ["hug", "hugging"],
    "kiss": ["kiss", "kissing"],
    "lift": ["lift", "lifting", "pick up"],
    "sing to": ["sing to", "singing to"],
    "take from": ["take from", "taking from"],
    "ride": ["ride", "riding"],
    "take a photo": ["take a photo", "photograph"],
    "work on": ["work on", "using"],
    "use": ["use", "using"],
    "answer phone": ["answer phone"],
    "look at": ["look at", "looking at", "watch"],
}


REL_KEY_REGEX_PATTERNS: Dict[str, List[str]] = {
    "watch": [
        r"\bwatch(?:ing)?\b",
        r"\blook(?:ing)?\s*at\b",
        r"lookat",
    ],
    "listen to": [r"\blisten(?:ing)?\s*to\b"],
    "talk to": [r"\btalk(?:ing)?\s*to\b", r"\bspeak(?:ing)?\s*to\b", r"\bchat(?:ting)?\s*with\b"],
    "carry hold": [r"\bcarry(?:ing)?\b", r"\bhold(?:ing)?\b"],
    "touch": [r"\btouch(?:ing)?\b"],
    "push": [r"\bpush(?:ing)?\b"],
    "pull": [r"\bpull(?:ing)?\b"],
    "point to": [r"\bpoint(?:ing)?\s*(?:to|at)\b"],
    "hit": [r"\bhit(?:ting)?\b", r"\bstrike\b"],
    "fight": [r"\bfight(?:ing)?\b"],
    "give": [r"\bgive\b", r"\bgiving\b", r"\bserve\b"],
    "hand": [r"\bhand\b", r"\bhand\s*over\b"],
    "grab": [r"\bgrab(?:bing)?\b"],
    "hand shake": [r"\bhand\s*shake\b", r"\bshake\s*hands?\b"],
    "hug": [r"\bhug(?:ging)?\b"],
    "kiss": [r"\bkiss(?:ing)?\b"],
    "lift": [r"\blift(?:ing)?\b", r"\bpick\s*up\b"],
    "sing to": [r"\bsing(?:ing)?\s*to\b"],
    "take from": [r"\btak(?:e|ing)\s+.*\s+from\b"],
    "ride": [r"\bride\b", r"\briding\b"],
    "take a photo": [r"\btake\b.*\bphoto\b", r"\bphotograph\b"],
    "work on": [r"\bwork(?:ing)?\s*on\b"],
    "use": [r"\buse\b", r"\busing\b"],
    "answer phone": [r"\banswer(?:ing)?\s*phone\b"],
    "look at": [r"\blook(?:ing)?\s*at\b", r"lookat"],
}


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _match_rel_keys_by_regex(text: str) -> List[str]:
    norm = _normalize_text(text)
    compact = _compact_text(text)
    matched: List[str] = []
    for key, patterns in REL_KEY_REGEX_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, norm) or re.search(pat, compact):
                matched.append(key)
                break
    return matched


def _to_relation_key_from_action_label(action_label: str) -> List[str]:
    norm = _normalize_text(action_label)
    keys = list(ACTION_LABEL_TO_REL_KEYS.get(norm, []))
    for key in _match_rel_keys_by_regex(action_label):
        if key == "look at":
            key = "watch"
        if key not in keys:
            keys.append(key)
    return keys


def _to_relation_key_from_predicate(predicate: str) -> Optional[str]:
    norm = _normalize_text(predicate)
    if not norm:
        return None

    # Regex-first to cover variants like look_at / look-at / lookat / looking at.
    regex_keys = _match_rel_keys_by_regex(predicate)
    if regex_keys:
        key = regex_keys[0]
        if key == "look at":
            return "watch"
        return key

    for key, aliases in PREDICATE_KEY_ALIASES.items():
        if norm == key:
            return key
        if norm in aliases:
            return key
    # Fallback: if predicate already equals one of the action relation keys, keep it.
    all_action_keys = {k for keys in ACTION_LABEL_TO_REL_KEYS.values() for k in keys}
    if norm in all_action_keys:
        return norm
    return None


def _has_time_overlap(edge_spans: Any, start_frame: int, end_frame: int) -> bool:
    if not isinstance(edge_spans, list):
        return False
    for span in edge_spans:
        if not isinstance(span, list) or len(span) != 2:
            continue
        try:
            s = int(span[0])
            e = int(span[1])
        except Exception:
            continue
        if e < s:
            s, e = e, s
        if max(s, start_frame) <= min(e, end_frame):
            return True
    return False


def merge_contacting_edges_into_actions(graph: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    object_nodes = graph.get("object_nodes", [])
    gid_to_node_id: Dict[int, str] = {}
    node_id_to_gid: Dict[str, int] = {}
    for obj in object_nodes:
        try:
            gid = int(obj.get("global_track_id"))
        except Exception:
            continue
        node_id = str(obj.get("node_id", ""))
        if node_id:
            gid_to_node_id[gid] = node_id
            node_id_to_gid[node_id] = gid

    action_nodes = graph.get("action_nodes", [])
    action_index: Dict[int, List[Dict[str, Any]]] = {}
    for action_group in action_nodes:
        obj_node_id = str(action_group.get("object_node_id", ""))
        if not obj_node_id:
            continue
        subject_gid = node_id_to_gid.get(obj_node_id)
        if subject_gid is None:
            continue
        for action in action_group.get("actions", []):
            if not isinstance(action, dict):
                continue
            keys = _to_relation_key_from_action_label(str(action.get("action_label", "")))
            if not keys:
                continue
            try:
                start_frame = int(action.get("start_frame"))
                end_frame = int(action.get("end_frame"))
            except Exception:
                continue
            action_index.setdefault(subject_gid, []).append(
                {
                    "action": action,
                    "keys": set(keys),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )

    new_edge_groups: List[Dict[str, Any]] = []
    removed_edges = 0
    for edge_group in graph.get("edges", []):
        try:
            subject_id = int(edge_group.get("subject_id"))
        except Exception:
            continue
        subject_actions = action_index.get(subject_id, [])
        kept_rels: List[Dict[str, Any]] = []
        for rel in edge_group.get("relationships", []):
            if not isinstance(rel, dict):
                continue
            if rel.get("edge_type") != "contacting":
                kept_rels.append(rel)
                continue
            pred_key = _to_relation_key_from_predicate(str(rel.get("predicate_verb", "")))
            if pred_key is None or not subject_actions:
                kept_rels.append(rel)
                continue

            try:
                target_gid = int(rel.get("object_id"))
            except Exception:
                kept_rels.append(rel)
                continue

            matched = False
            for action_item in subject_actions:
                if pred_key not in action_item["keys"]:
                    continue
                if not _has_time_overlap(rel.get("time_frames"), action_item["start_frame"], action_item["end_frame"]):
                    continue
                action_dict = action_item["action"]
                targets = action_dict.get("target_object_ids")
                if not isinstance(targets, list):
                    targets = []
                if target_gid not in targets:
                    targets.append(target_gid)
                    targets.sort()
                action_dict["target_object_ids"] = targets
                matched = True

            if matched:
                removed_edges += 1
            else:
                kept_rels.append(rel)

        if kept_rels:
            def _safe_int(v: Any) -> int:
                try:
                    return int(v)
                except Exception:
                    return -1

            kept_rels.sort(
                key=lambda x: (
                    x.get("edge_type", ""),
                    x.get("predicate_verb", ""),
                    _safe_int(x.get("object_id", -1)),
                )
            )
            new_edge_groups.append({"subject_id": subject_id, "relationships": kept_rels})

    graph["edges"] = new_edge_groups
    graph["action_nodes"] = action_nodes
    if verbose:
        print(f"[relation][post] merged contacting edges into actions: removed={removed_edges}")
    return graph


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
) -> List[Dict[str, Any]]:
    merged: Dict[int, List[Dict[str, Any]]] = {}

    for item in spatial_grouped:
        sid = int(item.get("subject_id"))
        for rel in item.get("relationships", []):
            if not isinstance(rel, list) or len(rel) != 4:
                continue
            _, pred, oid, spans = rel
            merged.setdefault(sid, []).append(
                {
                    "object_id": int(oid),
                    "predicate_verb": str(pred),
                    "time_frames": spans,
                    "edge_type": "spatial",
                }
            )

    for item in contacting_grouped:
        sid = int(item.get("subject_id"))
        for rel in item.get("relationships", []):
            if not isinstance(rel, list) or len(rel) != 5:
                continue
            _, pred, oid, spans, rtype = rel
            merged.setdefault(sid, []).append(
                {
                    "object_id": int(oid),
                    "predicate_verb": str(pred),
                    "time_frames": spans,
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

    graph.pop("spatial_edges", None)
    graph.pop("contacting_edges", None)
    graph["edges"] = merge_spatial_and_contacting_edges(spatial_grouped, contacting_grouped)
    graph = merge_contacting_edges_into_actions(graph, verbose=verbose)
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
