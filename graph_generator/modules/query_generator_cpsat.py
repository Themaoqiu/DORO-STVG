from __future__ import annotations

import asyncio
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import fire

from ortools.sat.python import cp_model
from api_sync.api import StreamGenerator
from api_sync.utils.parser import JSONParser


CLUE_TYPES = ("cls", "app", "act", "seq", "spa", "env", "int")

RELATION_SYNONYMS: Dict[str, str] = {}
ATTRIBUTE_SYNONYMS: Dict[str, str] = {}

DECORATION_POOL: Dict[str, Tuple[str, ...]] = {
    "temporal": (
        "within this time window",
        "across this segment",
    ),
    "action": (
        "following its visible behavior flow",
        "under consistent motion context",
    ),
    "spatial": (
        "with stable relative positioning cues",
        "against nearby object context",
    ),
    "appearance": (
        "with visually consistent appearance",
        "using lightweight attribute hints",
    ),
    "neutral": (
        "as the referred instance",
    ),
}

QUERY_POLISH_SYSTEM_PROMPT = """## Role
You are a precise Query Writer for spatiotemporal video grounding. Your task is
to generate one natural, concise grounding query from structured clues while preserving
all grounding semantics.
## Core Principle
- Preserve semantics exactly. Do not add, remove, or alter target identity clues.
- Keep temporal meaning faithful to the original clue set.
- Use clear and fluent wording suitable for human annotation.
- Also produce one standalone natural-language target description for each target object.
## Output Rule
Return strict JSON only:
{
  "query": "<polished full query>",
  "target_queries": {
    "target 1": "<standalone natural-language description for target 1>",
    "target 2": "<standalone natural-language description for target 2>"
  }
}
"""

QUERY_POLISH_PROMPT_TEMPLATE = """## Task
You will receive visual cues for constructing query text for video localization tasks. You need to correctly use all the cues to generate a single query for locating the target object. Ensure the description is fluent and natural without losing any information.
## Input Format
- target_classes_json: target index to class mapping.
- target_intervals_json: target index to interval mapping.
- clues_per_target_json: clues grouped by target and category.
- category_legend_json: category name mapping.
## The information you need to use: 
1. target object class for localization:
{target_classes_text}

2. Visual cues used to localize each object:
{clues_per_target_text}
## Guidelines
- Carefully examine the content in the section Visual cues used to localize each object.
  - If have only one object: output a semantically clear query using all the provided clue information to locate this object.
  - If have multi-target: Using all the provided clues, generate one semantically clear query that can locate both objects simultaneously. You may describe multiple objects separately, such as "the person in black clothing and the person leaning against the wall"; you may also fuse the descriptions of multiple objects, such as "the person in black clothing and the cat at his feet".
- Prefer concise natural English; avoid redundant filler phrases.
- The clues may not be naturally phrased, and the generated query should not follow the language style of the clues.
- In addition to the full query, also output one simple general natural-language target description for each target separately. And keep it brief—just enough to distinguish the target objects.
- Each standalone target description should be a natural referring expression rather than a keyword list.
## Output Format
Return one valid JSON object:
{{
  "query": "...",
  "target_queries": {{
    "target 1": "...",
    "target 2": "..."
  }}
}}
## Output Specifications
- Output strict JSON only.
- "query" must be non-empty and in English.
- "target_queries" must be an object.
- Each target listed in the input must have exactly one entry in "target_queries".
- Each target description must be non-empty and in English.
"""

PROMPT_CATEGORY_TEXT: Dict[str, str] = {
    "cls": "class",
    "app": "appearance",
    "env": "Interaction with the environment",
    "act": "action",
    "seq": "Sequence of actions and spatial position changes",
    "spa": "Spatial position",
    "int": "Interaction with other objects",
}

PROMPT_CATEGORY_ORDER: Tuple[str, ...] = ("cls", "app", "env", "act", "seq", "spa", "int")


def _norm(text: Any) -> str:
    return " ".join(str(text).strip().split())


def _canon_token(token: str) -> str:
    t = token.strip().lower()
    irregular = {
        "is": "be",
        "are": "be",
        "was": "be",
        "were": "be",
        "has": "have",
        "had": "have",
        "looks": "look",
        "looking": "look",
        "looked": "look",
        "moves": "move",
        "moving": "move",
        "moved": "move",
        "walks": "walk",
        "walking": "walk",
        "walked": "walk",
        "talks": "talk",
        "talking": "talk",
        "talked": "talk",
        "touching": "touch",
        "touched": "touch",
    }
    if t in irregular:
        return irregular[t]
    if len(t) > 4 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 3 and t.endswith("ed"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def _normalize_raw_phrase(text: Any) -> str:
    return _norm(str(text).lower().replace("_", " ").replace("-", " "))


def _canon_relation_phrase(text: str) -> str:
    s = _normalize_raw_phrase(text)
    if not s:
        return ""
    tokens = s.split()
    if not tokens:
        return ""
    tokens[0] = _canon_token(tokens[0])
    if len(tokens) > 1 and tokens[0] == "be":
        tokens = tokens[1:]
    return _norm(" ".join(tokens))


def _canon_attribute_phrase(text: str) -> str:
    s = _normalize_raw_phrase(text)
    if not s:
        return ""
    tokens = s.split()
    if not tokens:
        return ""
    tokens[0] = _canon_token(tokens[0])
    if tokens and tokens[0] == "be":
        tokens = tokens[1:]
    return _norm(" ".join(tokens))


def _choose_group_representative(values: List[str], counts: Dict[str, int]) -> str:
    if not values:
        return ""
    return sorted(values, key=lambda x: (-counts.get(x, 0), len(x), x))[0]


def _build_synonym_map_from_values(
    values: List[str],
    canon_fn: Callable[[str], str],
) -> Dict[str, str]:
    norm_counts: Dict[str, int] = defaultdict(int)
    groups: Dict[str, Set[str]] = defaultdict(set)
    for raw in values:
        s = _normalize_raw_phrase(raw)
        if not s:
            continue
        norm_counts[s] += 1
        c = canon_fn(s)
        if not c:
            continue
        groups[c].add(s)

    mapping: Dict[str, str] = {}
    for _, variants in groups.items():
        if len(variants) <= 1:
            continue
        rep = _choose_group_representative(sorted(variants), norm_counts)
        if not rep:
            continue
        for v in variants:
            if v != rep:
                mapping[v] = rep
    return mapping


def _summarize_synonyms_from_graph_file(input_path: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int]]:
    relation_values: List[str] = []
    attribute_values: List[str] = []
    scanned_graphs = 0
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            scanned_graphs += 1
            graph = json.loads(line)
            for obj in graph.get("object_nodes") or []:
                attribute_values.extend(obj.get("attributes") or [])
            for edge in graph.get("edges") or []:
                for rel in edge.get("relationships") or []:
                    relation_values.append(rel.get("predicate_verb", ""))

    rel_map = _build_synonym_map_from_values(relation_values, _canon_relation_phrase)
    attr_map = _build_synonym_map_from_values(attribute_values, _canon_attribute_phrase)
    stats = {
        "graphs": scanned_graphs,
        "relation_values": len(relation_values),
        "attribute_values": len(attribute_values),
        "relation_synonyms": len(rel_map),
        "attribute_synonyms": len(attr_map),
    }
    return rel_map, attr_map, stats


def _normalize_phrase(text: Any, kind: str) -> str:
    s = _norm(text).lower().replace("_", " ").replace("-", " ")
    s = _norm(s)
    if not s:
        return ""
    if kind == "act":
        return s
    elif kind == "rel":
        s = RELATION_SYNONYMS.get(s, s)
    elif kind == "attr":
        s = ATTRIBUTE_SYNONYMS.get(s, s)
    return s


def _normalized_values(values: Iterable[Any], kind: str) -> List[str]:
    out: List[str] = []
    for v in _uniq(values):
        n = _normalize_phrase(v, kind)
        if n:
            out.append(n)
    return out


def _display_phrase(text: str) -> str:
    return _norm(str(text).replace("_", " "))


def _parse_seq_token(token: str) -> Tuple[str, str, str]:
    if token.startswith("act:"):
        return "act", token[4:], ""
    if token.startswith("rel:"):
        body = token[4:]
        if ":" in body:
            pred, ref = body.split(":", 1)
        else:
            pred, ref = body, "object"
        return "rel", pred, ref
    return "raw", token, ""


def _format_seq_part(token: str) -> str:
    kind, a, b = _parse_seq_token(token)
    if kind == "act":
        return _display_phrase(a)
    if kind == "rel":
        if b:
            return _norm(f"{_display_phrase(a)} {_display_phrase(b)}")
        return _display_phrase(a)
    return _display_phrase(a)


def _enumerate_ordered_chains(
    events: List[Tuple[int, int, str]],
    max_chain_len: int,
    max_chains: int = 128,
) -> Set[Tuple[str, ...]]:
    if max_chain_len < 2 or len(events) < 2:
        return set()
    sorted_events = sorted(events, key=lambda x: (x[0], x[1], x[2]))
    out: Set[Tuple[str, ...]] = set()

    def dfs(last_idx: int, chain: List[str]) -> None:
        if len(out) >= max_chains:
            return
        if 2 <= len(chain) <= max_chain_len:
            out.add(tuple(chain))
        if len(chain) >= max_chain_len:
            return
        last_end = sorted_events[last_idx][1]
        for j in range(last_idx + 1, len(sorted_events)):
            s, e, token = sorted_events[j]
            if s <= last_end:
                continue
            if token == chain[-1]:
                continue
            chain.append(token)
            dfs(j, chain)
            chain.pop()
            if len(out) >= max_chains:
                return

    for i in range(len(sorted_events)):
        dfs(i, [sorted_events[i][2]])
        if len(out) >= max_chains:
            break
    return out


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _uniq(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in values:
        s = _norm(item)
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _normalize_api_keys(api_keys: Optional[Union[str, Iterable[str]]]) -> List[str]:
    if isinstance(api_keys, str):
        return [key.strip() for key in api_keys.split(",") if key.strip()]
    if api_keys is None:
        return []
    return [str(key).strip() for key in api_keys if str(key).strip()]


def _load_env_var_from_project_env(var_name: str) -> str:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return ""
    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() != var_name:
                continue
            return value.strip().strip('"').strip("'")
    return ""


def _clue_role_text(clue_type: str) -> str:
    role_map = {
        "cls": "class anchor: define object category",
        "app": "appearance cue: visual attribute disambiguation",
        "env": "context cue: scene or temporal-segment grounding",
        "act": "action cue: dynamic behavior evidence",
        "seq": "sequence cue: temporal order constraint",
        "spa": "spatial cue: geometric/positional constraint",
        "int": "interaction cue: relational interaction constraint",
    }
    return role_map.get(str(clue_type), "generic grounding cue")




def _sample_class(obj: Dict[str, Any], node_id: str) -> str:
    cls = _norm(obj.get("dam_category") or obj.get("object_class") or "object").lower()
    if cls == "person" and str(node_id).lower().startswith("man"):
        return "person"
    return cls or "object"


def _frames_from_obj(obj: Dict[str, Any]) -> Set[int]:
    bboxes = obj.get("bboxes") or {}
    frames = {_to_int(k, -1) for k in bboxes.keys()}
    frames.discard(-1)
    if frames:
        return frames
    s = _to_int(obj.get("start_frame"), 0)
    e = _to_int(obj.get("end_frame"), s)
    if e < s:
        s, e = e, s
    return set(range(s, e + 1))


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    if e < s:
        return None
    return s, e


def _interval_tiou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    ov = _overlap(a, b)
    if not ov:
        return 0.0
    inter = ov[1] - ov[0] + 1
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _segments_from_frames(frames: Set[int]) -> List[Tuple[int, int]]:
    if not frames:
        return []
    arr = sorted(frames)
    out: List[Tuple[int, int]] = []
    s = e = arr[0]
    for x in arr[1:]:
        if x == e + 1:
            e = x
        else:
            out.append((s, e))
            s = e = x
    out.append((s, e))
    return out


def _extract_relation_segments(rel: Dict[str, Any]) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    for seg in rel.get("time_spans") or []:
        if isinstance(seg, dict):
            a = _to_int(seg.get("start_frame"), 0)
            b = _to_int(seg.get("end_frame"), a)
            if b < a:
                a, b = b, a
            segments.append((a, b))
    if (not segments) and ("start_frame" in rel or "end_frame" in rel):
        a = _to_int(rel.get("start_frame"), 0)
        b = _to_int(rel.get("end_frame"), a)
        if b < a:
            a, b = b, a
        segments.append((a, b))
    return sorted(set(segments))


def _relation_interval(rel: Dict[str, Any]) -> Tuple[int, int]:
    start = _to_int(rel.get("start"), 0)
    end = _to_int(rel.get("end"), start)
    if end < start:
        start, end = end, start
    return start, end


def _normalized_relation_pair(rel: Dict[str, Any]) -> Tuple[str, str]:
    pred = _normalize_phrase(rel.get("predicate", ""), "rel")
    ref_cls = _normalize_phrase(rel.get("ref_class", ""), "attr") or _normalize_phrase(rel.get("ref_class", ""), "rel")
    return pred, ref_cls


def _add_overlap_boundaries(
    boundaries: Set[int],
    span: Tuple[int, int],
    full_span: Tuple[int, int],
) -> None:
    ov = _overlap(span, full_span)
    if not ov:
        return
    boundaries.add(ov[0])
    boundaries.add(ov[1] + 1)


@dataclass(frozen=True)
class CandidateMember:
    object_id: str
    start: int
    end: int


@dataclass
class CandidateTarget:
    candidate_id: str
    members: List[CandidateMember]
    difficulty: float = 0.0
    difficulty_bucket: str = "medium"
    sampling_bucket: str = "easy"
    D_t: float = 0.0
    D_s: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def arity(self) -> int:
        return len(self.members)

    @property
    def interval(self) -> Tuple[int, int]:
        s = min(m.start for m in self.members)
        e = max(m.end for m in self.members)
        return s, e

    @property
    def primary_object_id(self) -> str:
        return self.members[0].object_id


@dataclass(frozen=True)
class AtomicClue:
    clue_id: str
    clue_type: str
    text: str
    signature: Tuple[Any, ...]
    member_indices: Tuple[int, ...]
    is_temporal_evidence: bool
    chain_len: int
    cost: int = 1


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    bucket: str
    type_min: Dict[str, int]
    type_max: Dict[str, int]
    k_min: int
    k_max: int
    q_min: int
    require_seq: int = 0
    seq_min_chain_len: int = 0
    seq_max_chain_len: int = 0
    min_long_seq_len: int = 0
    min_seq_chain_sum: int = 0
    require_time_uniqueness: int = 0
    weight: float = 1.0
    arity_min: int = 1
    arity_max: int = 1


@dataclass(frozen=True)
class DifficultyWeights:
    eps: float = 0.01
    lambda_weight: float = 0.5


@dataclass
class GraphIndex:
    objects: Dict[str, Dict[str, Any]]
    frames: Dict[str, Set[int]]
    actions_by_obj: Dict[str, List[Dict[str, Any]]]
    relations_by_subj: Dict[str, List[Dict[str, Any]]]
    temporal_spans: List[Dict[str, int]]


@dataclass
class CandidateProfile:
    candidate_id: str
    arity: int
    classes: List[str]
    attrs: List[Set[str]]
    env: List[Set[str]]
    temporal_tags: List[Set[str]]
    actions: List[Set[str]]
    sequences: List[Set[Tuple[str, ...]]]
    spa: List[Set[Tuple[str, str]]]
    inter: List[Set[Tuple[str, str]]]


def _object_desc_for_action_target(
    index: GraphIndex,
    object_id: str,
    target_index_by_object_id: Optional[Dict[str, int]] = None,
) -> str:
    if target_index_by_object_id and object_id in target_index_by_object_id:
        return f"target {target_index_by_object_id[object_id]}"
    obj = index.objects.get(object_id)
    if obj is None:
        return "object"
    cls = _display_phrase(_sample_class(obj, object_id))
    attrs = _normalized_values(obj.get("attributes") or [], "attr")
    envs = _normalized_values(obj.get("environment") or [], "rel")
    if attrs:
        return _norm(f"{cls} {attrs[0]}")
    if envs:
        return _norm(f"{cls} {envs[0]}")
    return cls


def _action_target_desc_map(
    index: GraphIndex,
    member: CandidateMember,
    target_index_by_object_id: Optional[Dict[str, int]] = None,
) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    interval = (member.start, member.end)
    for item in index.actions_by_obj.get(member.object_id, []):
        if not _overlap(interval, (item["start"], item["end"])):
            continue
        action = _normalize_phrase(item["label"], "act")
        if not action:
            continue
        for tid in item.get("targets", []):
            if tid not in index.objects:
                continue
            out[action].add(_object_desc_for_action_target(index, tid, target_index_by_object_id))
    return out


def _relation_target_desc_maps(
    index: GraphIndex,
    member: CandidateMember,
    target_index_by_object_id: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[Tuple[str, str], Set[str]], Dict[Tuple[str, str], Set[str]]]:
    spatial: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    interacting: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    interval = (member.start, member.end)
    for rel in index.relations_by_subj.get(member.object_id, []):
        if not _overlap(interval, (rel["start"], rel["end"])):
            continue
        pred, ref_cls = _normalized_relation_pair(rel)
        oid = str(rel.get("object_id", "")).strip()
        if not pred or not oid or oid not in index.objects:
            continue
        desc = _object_desc_for_action_target(index, oid, target_index_by_object_id)
        if not desc:
            continue
        key = (pred, ref_cls)
        if str(rel.get("edge_type", "")).strip().lower() == "spatial":
            spatial[key].add(desc)
        else:
            interacting[key].add(desc)
    return spatial, interacting


def _member_full_track_interval(index: GraphIndex, member: CandidateMember) -> Tuple[int, int]:
    frames = index.frames.get(member.object_id, set())
    if not frames:
        return member.start, member.end
    return min(frames), max(frames)


def _is_local_interval(span: Tuple[int, int], full_track: Tuple[int, int]) -> bool:
    return span[0] > full_track[0] or span[1] < full_track[1]


def _has_local_dynamic_relation_pair(
    index: GraphIndex,
    member: CandidateMember,
    pred: str,
    ref_cls: str,
    *,
    require_spatial: Optional[bool],
) -> bool:
    full_track = _member_full_track_interval(index, member)
    member_interval = (member.start, member.end)
    for rel in index.relations_by_subj.get(member.object_id, []):
        edge_type = str(rel.get("edge_type", "")).strip().lower()
        if require_spatial is True and edge_type != "spatial":
            continue
        if require_spatial is False and edge_type == "spatial":
            continue
        rel_pred, rel_ref_cls = _normalized_relation_pair(rel)
        if rel_pred != pred or rel_ref_cls != ref_cls:
            continue
        rel_interval = _relation_interval(rel)
        ov_member = _overlap(member_interval, rel_interval)
        if not ov_member:
            continue
        ov_full = _overlap(rel_interval, full_track)
        if ov_full and _is_local_interval(ov_full, full_track):
            return True
    return False


def _has_local_dynamic_action_target_pair(
    index: GraphIndex,
    member: CandidateMember,
    pred: str,
    ref_cls: str,
) -> bool:
    full_track = _member_full_track_interval(index, member)
    member_interval = (member.start, member.end)
    for item in index.actions_by_obj.get(member.object_id, []):
        ov_member = _overlap(member_interval, (item["start"], item["end"]))
        if not ov_member:
            continue
        action = _normalize_phrase(item.get("label", ""), "act")
        if action != pred:
            continue
        has_target = False
        for tid in item.get("targets", []):
            if tid not in index.objects:
                continue
            tcls = _normalize_phrase(_sample_class(index.objects[tid], tid), "attr")
            if tcls == ref_cls:
                has_target = True
                break
        if not has_target:
            continue
        ov_full = _overlap((item["start"], item["end"]), full_track)
        if ov_full and _is_local_interval(ov_full, full_track):
            return True
    return False


def _action_has_local_targets(
    index: GraphIndex,
    member: CandidateMember,
    action: str,
) -> bool:
    interval = (member.start, member.end)
    for item in index.actions_by_obj.get(member.object_id, []):
        if not _overlap(interval, (item["start"], item["end"])):
            continue
        label = _normalize_phrase(item.get("label", ""), "act")
        if label != action:
            continue
        if any(tid in index.objects for tid in item.get("targets", [])):
            return True
    return False


def build_graph_index(graph: Dict[str, Any]) -> GraphIndex:
    object_nodes = graph.get("object_nodes") or []
    objects: Dict[str, Dict[str, Any]] = {}
    frames: Dict[str, Set[int]] = {}

    for obj in object_nodes:
        node_id = str(obj.get("node_id", "")).strip()
        if not node_id:
            continue
        objects[node_id] = obj
        frames[node_id] = _frames_from_obj(obj)

    def _resolve_node_id(raw_id: Any) -> str:
        text = str(raw_id).strip()
        return text if text in objects else ""

    actions_by_obj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in graph.get("action_nodes") or []:
        owner = _resolve_node_id(node.get("object_id"))
        if owner not in objects:
            continue
        for item in node.get("actions") or []:
            label = _normalize_phrase(item.get("action_label", ""), "act")
            if not label:
                continue
            s = _to_int(item.get("start_frame"), 0)
            e = _to_int(item.get("end_frame"), s)
            if e < s:
                s, e = e, s
            ov = _overlap((s, e), (min(frames[owner]), max(frames[owner])))
            if not ov:
                continue
            actions_by_obj[owner].append(
                {
                    "label": label,
                    "start": ov[0],
                    "end": ov[1],
                    "targets": [
                        tid
                        for tid in (_resolve_node_id(t) for t in (item.get("target_object_ids") or []))
                        if tid in objects and tid != owner
                    ],
                }
            )

    relations_by_subj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge in graph.get("edges") or []:
        if "subject_id" not in edge or "relationships" not in edge:
            continue
        sid = _resolve_node_id(edge.get("subject_id"))
        if sid not in objects:
            continue
        for rel in edge.get("relationships") or []:
            oid = _resolve_node_id(rel.get("object_id"))
            if oid not in objects:
                continue
            pred = _normalize_phrase(rel.get("predicate_verb", ""), "rel")
            if not pred:
                continue
            segments = _extract_relation_segments(rel)
            if not segments:
                shared = frames[sid] & frames[oid]
                segments = _segments_from_frames(shared)
            edge_type = _norm(rel.get("edge_type", "")).lower() or "spatial"
            rel_type = _norm(rel.get("relationship_type", "")).lower()
            ref_cls = _sample_class(objects[oid], oid)
            for s, e in segments:
                ov = _overlap((s, e), (min(frames[sid]), max(frames[sid])))
                if not ov:
                    continue
                relations_by_subj[sid].append(
                    {
                        "predicate": pred,
                        "start": ov[0],
                        "end": ov[1],
                        "object_id": oid,
                        "ref_class": ref_cls,
                        "edge_type": edge_type,
                        "relationship_type": rel_type,
                    }
                )

    temporal_spans: List[Dict[str, int]] = [
        {
            "clip_id": _to_int(tn.get("clip_id"), 0),
            "start": _to_int(tn.get("start_frame"), 0),
            "end": _to_int(tn.get("end_frame"), 0),
        }
        for tn in (graph.get("temporal_nodes") or [])
    ]

    return GraphIndex(
        objects=objects,
        frames=frames,
        actions_by_obj=actions_by_obj,
        relations_by_subj=relations_by_subj,
        temporal_spans=temporal_spans,
    )


def _downsample_intervals(intervals: Set[Tuple[int, int]], limit: int) -> List[Tuple[int, int]]:
    ordered = sorted(intervals, key=lambda x: (x[0], x[1]))
    if limit <= 0 or len(ordered) <= limit:
        return ordered
    keep = [ordered[0], ordered[-1]] if len(ordered) >= 2 else ordered
    step = max(1, len(ordered) // max(1, limit - len(keep)))
    keep.extend(ordered[1:-1:step])
    return sorted(set(keep))[:limit]


def _ranked_multi_object_ids(index: GraphIndex, max_objects: int = 18) -> List[str]:
    object_ids = sorted(index.objects.keys())
    if len(object_ids) <= max_objects:
        return object_ids
    # Prefer objects with longer tracks when multi-target combinations must be capped.
    return sorted(object_ids, key=lambda oid: (-len(index.frames.get(oid, set())), oid))[:max_objects]


def _interval_group_gap(intervals: Tuple[Tuple[int, int], ...]) -> int:
    starts = [s for s, _ in intervals]
    ends = [e for _, e in intervals]
    return max(0, max(starts) - min(ends))


def _interval_group_span(intervals: Tuple[Tuple[int, int], ...]) -> int:
    starts = [s for s, _ in intervals]
    ends = [e for _, e in intervals]
    return max(ends) - min(starts)


def _multi_interval_choices_for_combo(
    combo: Tuple[str, ...],
    object_intervals: Dict[str, List[Tuple[int, int]]],
    limit: int,
) -> List[Tuple[Tuple[int, int], ...]]:
    if limit <= 0:
        return []
    per_obj_cap = max(1, min(3, limit))
    interval_lists: List[List[Tuple[int, int]]] = []
    for oid in combo:
        intervals = object_intervals.get(oid, [])
        if not intervals:
            return []
        interval_lists.append(intervals[:per_obj_cap])

    products = list(product(*interval_lists))
    products = [tuple(p) for p in products]
    products.sort(key=lambda p: (_interval_group_gap(p), _interval_group_span(p), p))
    return products[:limit]


def build_candidate_intervals(
    graph: Dict[str, Any],
    index: GraphIndex,
    min_interval_len: int = 3,
    max_intervals_per_object: int = 12,
    include_full_track: bool = True,
    max_target_arity: int = 3,
    max_multi_intervals_per_group: int = 8,
    max_multi_candidates_total: int = 240,
) -> List[CandidateTarget]:
    del graph
    candidates: List[CandidateTarget] = []
    object_intervals_by_obj: Dict[str, List[Tuple[int, int]]] = {}

    # Single-target candidates (existing behavior).
    for object_id, obj in index.objects.items():
        obj_frames = index.frames.get(object_id, set())
        if not obj_frames:
            continue
        full_s, full_e = min(obj_frames), max(obj_frames)
        boundaries: Set[int] = {full_s, full_e + 1}

        full_span = (full_s, full_e)
        for item in index.actions_by_obj.get(object_id, []):
            _add_overlap_boundaries(boundaries, (item["start"], item["end"]), full_span)
        for rel in index.relations_by_subj.get(object_id, []):
            _add_overlap_boundaries(boundaries, (rel["start"], rel["end"]), full_span)
        for tn in index.temporal_spans:
            _add_overlap_boundaries(boundaries, (tn["start"], tn["end"]), full_span)

        intervals: Set[Tuple[int, int]] = set()
        if include_full_track and full_e - full_s + 1 >= min_interval_len:
            intervals.add((full_s, full_e))

        sorted_bounds = sorted(boundaries)
        for i in range(len(sorted_bounds) - 1):
            s = sorted_bounds[i]
            e = sorted_bounds[i + 1] - 1
            if e < s:
                continue
            seg_frames = [f for f in range(s, e + 1) if f in obj_frames]
            if not seg_frames:
                continue
            ss, ee = min(seg_frames), max(seg_frames)
            if ee - ss + 1 >= min_interval_len:
                intervals.add((ss, ee))

        sampled_intervals = _downsample_intervals(intervals, max_intervals_per_object)
        object_intervals_by_obj[object_id] = sampled_intervals
        for s, e in sampled_intervals:
            candidates.append(
                CandidateTarget(
                    candidate_id=f"{object_id}@{s}-{e}",
                    members=[CandidateMember(object_id=object_id, start=s, end=e)],
                    meta={
                        "object_id": object_id,
                        "target_class": _sample_class(obj, object_id),
                        "is_full_track_interval": int(s == full_s and e == full_e),
                    },
                )
            )

    # Optional multi-target candidates. Keep compact to avoid temporal-combination explosion.
    max_target_arity = min(3, max(1, int(max_target_arity)))
    if max_target_arity <= 1:
        return candidates

    object_ids = [oid for oid in _ranked_multi_object_ids(index, max_objects=18) if object_intervals_by_obj.get(oid)]
    multi_count = 0
    for arity in range(2, max_target_arity + 1):
        for combo in combinations(object_ids, arity):
            interval_choices = _multi_interval_choices_for_combo(
                combo=combo,
                object_intervals=object_intervals_by_obj,
                limit=max_multi_intervals_per_group,
            )
            if not interval_choices:
                continue

            for interval_tuple in interval_choices:
                members = [
                    CandidateMember(object_id=oid, start=s, end=e)
                    for oid, (s, e) in zip(combo, interval_tuple)
                ]
                member_parts = [f"{oid}@{s}-{e}" for oid, (s, e) in zip(combo, interval_tuple)]
                candidates.append(
                    CandidateTarget(
                        candidate_id="|".join(member_parts),
                        members=members,
                        meta={
                            "object_ids": list(combo),
                            "arity": arity,
                            "is_shared_interval": int(len({(m.start, m.end) for m in members}) == 1),
                        },
                    )
                )
                multi_count += 1
                if multi_count >= max_multi_candidates_total:
                    return candidates

    return candidates


def _profile_for_candidate(index: GraphIndex, candidate: CandidateTarget, max_chain_len: int = 4) -> CandidateProfile:
    classes: List[str] = []
    attrs: List[Set[str]] = []
    env: List[Set[str]] = []
    temporal_tags: List[Set[str]] = []
    actions: List[Set[str]] = []
    sequences: List[Set[Tuple[str, ...]]] = []
    spa: List[Set[Tuple[str, str]]] = []
    inter: List[Set[Tuple[str, str]]] = []

    for member in candidate.members:
        obj = index.objects[member.object_id]
        classes.append(_sample_class(obj, member.object_id))
        attrs.append(set(_normalized_values(obj.get("attributes") or [], "attr")))
        env.append(set(_normalized_values(obj.get("environment") or [], "rel")))

        tags: Set[str] = set()
        for tn in index.temporal_spans:
            if _overlap((member.start, member.end), (tn["start"], tn["end"])):
                tags.add(f"clip_{tn['clip_id']}")
        temporal_tags.append(tags)

        act_events = []
        for item in index.actions_by_obj.get(member.object_id, []):
            if not _overlap((member.start, member.end), (item["start"], item["end"])):
                continue
            label = _normalize_phrase(item["label"], "act")
            if not label:
                continue
            act_events.append(
                {
                    "label": label,
                    "start": item["start"],
                    "end": item["end"],
                    "targets": list(item.get("targets", [])),
                }
            )
        act_events.sort(key=lambda x: (x["start"], x["end"], x["label"]))

        act_labels = {a["label"] for a in act_events}
        actions.append(act_labels)

        rel_events = []
        for rel in index.relations_by_subj.get(member.object_id, []):
            if not _overlap((member.start, member.end), (rel["start"], rel["end"])):
                continue
            pred = _normalize_phrase(rel["predicate"], "rel")
            ref_cls = _normalize_phrase(rel["ref_class"], "attr") or _normalize_phrase(rel["ref_class"], "rel")
            if not pred or not ref_cls:
                continue
            rel_events.append(
                {
                    "predicate": pred,
                    "ref_class": ref_cls,
                    "edge_type": rel["edge_type"],
                    "start": rel["start"],
                    "end": rel["end"],
                }
            )
        rel_events.sort(key=lambda x: (x["start"], x["end"], x["predicate"], x["ref_class"]))

        spa_set: Set[Tuple[str, str]] = set()
        int_set: Set[Tuple[str, str]] = set()
        for rel in rel_events:
            pair = (rel["predicate"], rel["ref_class"])
            if rel["edge_type"] == "spatial":
                spa_set.add(pair)
            else:
                int_set.add(pair)
        for a in act_events:
            for tid in a["targets"]:
                tcls = _normalize_phrase(_sample_class(index.objects[tid], tid), "attr")
                if tcls:
                    int_set.add((a["label"], tcls))
        spa.append(spa_set)
        inter.append(int_set)

        action_events_for_chain = [(a["start"], a["end"], f"act:{a['label']}") for a in act_events]
        relation_events_for_chain = [
            (r["start"], r["end"], f"rel:{r['predicate']}:{r['ref_class']}")
            for r in rel_events
        ]
        act_chains = _enumerate_ordered_chains(action_events_for_chain, max_chain_len=max_chain_len)
        rel_chains = _enumerate_ordered_chains(relation_events_for_chain, max_chain_len=max_chain_len)
        mixed_chains = {
            chain
            for chain in _enumerate_ordered_chains(action_events_for_chain + relation_events_for_chain, max_chain_len=max_chain_len)
            if any(tok.startswith("act:") for tok in chain) and any(tok.startswith("rel:") for tok in chain)
        }
        sequences.append(act_chains | rel_chains | mixed_chains)

    return CandidateProfile(
        candidate_id=candidate.candidate_id,
        arity=candidate.arity,
        classes=classes,
        attrs=attrs,
        env=env,
        temporal_tags=temporal_tags,
        actions=actions,
        sequences=sequences,
        spa=spa,
        inter=inter,
    )


def build_atomic_clues(index: GraphIndex, candidate: CandidateTarget, max_chain_len: int = 4) -> List[AtomicClue]:
    profile = _profile_for_candidate(index, candidate, max_chain_len=max_chain_len)
    clues: List[AtomicClue] = []
    seen: Set[Tuple[Any, ...]] = set()
    target_index_by_object_id = {
        member.object_id: idx + 1
        for idx, member in enumerate(candidate.members)
    }

    def add(
        clue_type: str,
        text: str,
        signature: Tuple[Any, ...],
        member_indices: Tuple[int, ...],
        is_temporal_evidence: bool,
        chain_len: int,
    ) -> None:
        if signature in seen:
            return
        seen.add(signature)
        clues.append(
            AtomicClue(
                clue_id=f"c{len(clues)}",
                clue_type=clue_type,
                text=text,
                signature=signature,
                member_indices=member_indices,
                is_temporal_evidence=is_temporal_evidence,
                chain_len=chain_len,
                cost=1,
            )
        )

    for k in range(profile.arity):
        cls = profile.classes[k]
        member = candidate.members[k]
        member_idx = (k,)
        action_target_descs = _action_target_desc_map(index, member, target_index_by_object_id)
        spa_target_descs, int_target_descs = _relation_target_desc_maps(index, member, target_index_by_object_id)
        add(
            "cls",
            cls,
            ("cls", k, cls),
            member_idx,
            False,
            0,
        )

        for attr in sorted(profile.attrs[k]):
            add(
                "app",
                attr,
                ("app", k, attr),
                member_idx,
                False,
                0,
            )

        for phrase in sorted(profile.env[k]):
            add(
                "env",
                phrase,
                ("env", "general", k, phrase),
                member_idx,
                False,
                0,
            )

        for tag in sorted(profile.temporal_tags[k]):
            add(
                "env",
                tag,
                ("env", "temporal", k, tag),
                member_idx,
                True,
                0,
            )

        for action in sorted(profile.actions[k]):
            target_descs = sorted(action_target_descs.get(action, set()))
            has_local_targets = _action_has_local_targets(index, member, action)
            if has_local_targets:
                # If an action has concrete target objects, the clue must mention target info.
                if not target_descs:
                    continue
                action_text = f"{action} {target_descs[0]}"
            elif target_descs:
                action_text = f"{action} {target_descs[0]}"
            else:
                action_text = action
            add(
                "act",
                action_text,
                ("act", k, action),
                member_idx,
                True,
                1,
            )

        for chain in sorted(profile.sequences[k]):
            if len(chain) < 2:
                continue
            chain_text = " then ".join(_format_seq_part(tok) for tok in chain)
            add(
                "seq",
                chain_text,
                ("seq", k, *chain),
                member_idx,
                True,
                len(chain),
            )

        for pred, ref_cls in sorted(profile.spa[k]):
            target_descs = sorted(spa_target_descs.get((pred, ref_cls), set()))
            # Spatial relations always come with an object_id in graph edges;
            # require target information in text when such clues are used.
            if not target_descs:
                continue
            spa_text = f"{pred} {target_descs[0]}"
            spa_temporal = _has_local_dynamic_relation_pair(
                index,
                member,
                pred,
                ref_cls,
                require_spatial=True,
            )
            add(
                "spa",
                spa_text,
                ("spa", k, pred, ref_cls),
                member_idx,
                spa_temporal,
                0,
            )

        for pred, ref_cls in sorted(profile.inter[k]):
            target_descs = sorted(int_target_descs.get((pred, ref_cls), set()))
            if not target_descs:
                target_descs = sorted(action_target_descs.get(pred, set()))
            # Interaction clues with target objects must expose target info in query text.
            if not target_descs:
                continue
            int_text = f"{pred} {target_descs[0]}"
            int_temporal = _has_local_dynamic_relation_pair(
                index,
                member,
                pred,
                ref_cls,
                require_spatial=False,
            ) or _has_local_dynamic_action_target_pair(
                index,
                member,
                pred,
                ref_cls,
            )
            add(
                "int",
                int_text,
                ("int", k, pred, ref_cls),
                member_idx,
                int_temporal,
                1,
            )

    return clues


def _clue_satisfied(profile: CandidateProfile, clue: AtomicClue) -> bool:
    ctype = clue.clue_type
    sig = clue.signature
    if ctype == "cls":
        _, k, cls = sig
        return 0 <= k < profile.arity and profile.classes[k] == cls
    if ctype == "app":
        _, k, attr = sig
        return 0 <= k < profile.arity and attr in profile.attrs[k]
    if ctype == "act":
        _, k, action = sig
        return 0 <= k < profile.arity and action in profile.actions[k]
    if ctype == "seq":
        _, k, *chain = sig
        return 0 <= k < profile.arity and tuple(chain) in profile.sequences[k]
    if ctype == "spa":
        _, k, pred, ref_cls = sig
        return 0 <= k < profile.arity and (pred, ref_cls) in profile.spa[k]
    if ctype == "int":
        _, k, pred, ref_cls = sig
        return 0 <= k < profile.arity and (pred, ref_cls) in profile.inter[k]
    if ctype == "env":
        _, kind, k, value = sig
        if not (0 <= k < profile.arity):
            return False
        if kind == "temporal":
            return value in profile.temporal_tags[k]
        return value in profile.env[k]
    return False


def build_exclusion_matrix(
    target: CandidateTarget,
    all_candidates: List[CandidateTarget],
    clues: List[AtomicClue],
    profile_map: Dict[str, CandidateProfile],
) -> Tuple[List[CandidateTarget], List[List[int]], List[List[int]]]:
    others = [c for c in all_candidates if c.candidate_id != target.candidate_id and c.arity == target.arity]
    target_obj_sig = tuple(m.object_id for m in target.members)
    object_rows: List[List[int]] = []
    time_rows: List[List[int]] = []
    for cand in others:
        prof = profile_map[cand.candidate_id]
        row = []
        for clue in clues:
            p = 1 if _clue_satisfied(prof, clue) else 0
            row.append(1 - p)
        cand_obj_sig = tuple(m.object_id for m in cand.members)
        if cand_obj_sig == target_obj_sig:
            time_rows.append(row)
        else:
            object_rows.append(row)
    return others, object_rows, time_rows


def solve_query_cpsat(
    target: CandidateTarget,
    template: TemplateSpec,
    clues: List[AtomicClue],
    object_exclusion_matrix: List[List[int]],
    time_exclusion_matrix: List[List[int]],
    enforce_time_uniqueness: bool = True,
    time_limit_sec: float = 2.0,
) -> Optional[Dict[str, Any]]:
    if cp_model is None:
        raise RuntimeError("ortools is required. Install it with: uv pip install ortools")

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{j}") for j in range(len(clues))]

    for row in object_exclusion_matrix:
        model.Add(sum(row[j] * x[j] for j in range(len(clues))) >= 1)

    if template.require_time_uniqueness and enforce_time_uniqueness:
        for row in time_exclusion_matrix:
            model.Add(sum(row[j] * x[j] for j in range(len(clues))) >= 1)

    # Keep member coverage constraints: each target member must be grounded by >=1 selected clue.
    # For multi-target this is essential to avoid "partially described" pair/triplet targets.
    for k in range(target.arity):
        idx = [j for j, clue in enumerate(clues) if k in clue.member_indices]
        if not idx:
            return None
        model.Add(sum(x[j] for j in idx) >= 1)
        if target.arity > 1:
            non_cls_idx = [
                j
                for j, clue in enumerate(clues)
                if clue.clue_type != "cls" and k in clue.member_indices
            ]
            if not non_cls_idx:
                return None
            model.Add(sum(x[j] for j in non_cls_idx) >= 1)

    for c in CLUE_TYPES:
        idx = [j for j, clue in enumerate(clues) if clue.clue_type == c]
        if idx:
            model.Add(sum(x[j] for j in idx) >= int(template.type_min.get(c, 0)))
            model.Add(sum(x[j] for j in idx) <= int(template.type_max.get(c, template.k_max)))
        elif template.type_min.get(c, 0) > 0:
            return None

    model.Add(sum(x) >= int(template.k_min))
    model.Add(sum(x) <= int(template.k_max))

    model.Add(sum(clue.chain_len * x[j] for j, clue in enumerate(clues)) >= int(template.q_min))

    seq_idx = [j for j, clue in enumerate(clues) if clue.clue_type == "seq"]
    seq_min_len = int(template.seq_min_chain_len)
    seq_max_len = int(template.seq_max_chain_len)
    seq_required_idx = [
        j
        for j in seq_idx
        if (seq_min_len <= 0 or clues[j].chain_len >= seq_min_len)
        and (seq_max_len <= 0 or clues[j].chain_len <= seq_max_len)
    ]

    if template.require_seq:
        if not seq_required_idx:
            return None
        model.Add(sum(x[j] for j in seq_required_idx) >= int(template.require_seq))

    if template.min_long_seq_len > 0 or template.min_seq_chain_sum > 0:
        if not seq_idx:
            return None
        long_seq_idx = [
            j
            for j in seq_idx
            if clues[j].chain_len >= int(template.min_long_seq_len)
        ]
        seq_chain_sum = sum(clues[j].chain_len * x[j] for j in seq_idx)
        if template.min_long_seq_len > 0 and template.min_seq_chain_sum > 0:
            enough_seq_sum = model.NewBoolVar("enough_seq_chain_sum")
            model.Add(seq_chain_sum >= int(template.min_seq_chain_sum)).OnlyEnforceIf(enough_seq_sum)
            model.Add(seq_chain_sum < int(template.min_seq_chain_sum)).OnlyEnforceIf(enough_seq_sum.Not())
            disjuncts = [enough_seq_sum]
            disjuncts.extend(x[j] for j in long_seq_idx)
            model.AddBoolOr(disjuncts)
        elif template.min_seq_chain_sum > 0:
            model.Add(seq_chain_sum >= int(template.min_seq_chain_sum))
        elif template.min_long_seq_len > 0:
            if not long_seq_idx:
                return None
            model.Add(sum(x[j] for j in long_seq_idx) >= 1)

    # Keep objective as min sum(x_j) for both single- and multi-target.
    model.Minimize(sum(x))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    chosen = [j for j in range(len(clues)) if solver.Value(x[j]) == 1]
    return {
        "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        "selected_indices": chosen,
        "objective": int(round(solver.ObjectiveValue())),
    }


def _build_default_templates() -> List[TemplateSpec]:
    inf = 8
    return [
        # Single-target templates (preserve current behavior).
        TemplateSpec(
            name="easy_static_attr",
            bucket="easy",
            type_min={"app": 1},
            type_max={"cls": 1, "app": 2, "act": 1, "seq": 0, "spa": 1, "env": 2, "int": 0},
            k_min=1,
            k_max=3,
            q_min=0,
            require_seq=0,
            require_time_uniqueness=0,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        TemplateSpec(
            name="easy_single_act",
            bucket="easy",
            type_min={"act": 1},
            type_max={"cls": 1, "app": 1, "act": 1, "seq": 0, "spa": 1, "env": 1, "int": 0},
            k_min=1,
            k_max=3,
            q_min=1,
            require_seq=0,
            require_time_uniqueness=0,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        TemplateSpec(
            name="medium_short_seq",
            bucket="medium",
            type_min={"seq": 1},
            type_max={"cls": 1, "app": 1, "act": 1, "seq": 1, "spa": 1, "env": inf, "int": 1},
            k_min=1,
            k_max=2,
            q_min=2,
            require_seq=1,
            seq_min_chain_len=2,
            seq_max_chain_len=2,
            require_time_uniqueness=1,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        TemplateSpec(
            name="medium_short_seq_app",
            bucket="medium",
            type_min={"seq": 1, "app": 1},
            type_max={"cls": 1, "app": 2, "act": 1, "seq": 1, "spa": 1, "env": inf, "int": 1},
            k_min=2,
            k_max=3,
            q_min=2,
            require_seq=1,
            seq_min_chain_len=2,
            seq_max_chain_len=2,
            require_time_uniqueness=1,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        TemplateSpec(
            name="medium_short_seq_spa",
            bucket="medium",
            type_min={"seq": 1, "spa": 1},
            type_max={"cls": 1, "app": 1, "act": 1, "seq": 1, "spa": 2, "env": inf, "int": 1},
            k_min=2,
            k_max=3,
            q_min=2,
            require_seq=1,
            seq_min_chain_len=2,
            seq_max_chain_len=2,
            require_time_uniqueness=1,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        TemplateSpec(
            name="medium_short_seq_int",
            bucket="medium",
            type_min={"seq": 1, "int": 1},
            type_max={"cls": 1, "app": 1, "act": 1, "seq": 1, "spa": 1, "env": inf, "int": 2},
            k_min=2,
            k_max=4,
            q_min=2,
            require_seq=1,
            seq_min_chain_len=2,
            seq_max_chain_len=2,
            require_time_uniqueness=1,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        TemplateSpec(
            name="hard_long_seq_int",
            bucket="hard",
            type_min={"seq": 1, "int": 1},
            type_max={"cls": 1, "app": 2, "act": 3, "seq": 3, "spa": 2, "env": inf, "int": 2},
            k_min=4,
            k_max=7,
            q_min=5,
            require_seq=1,
            seq_min_chain_len=3,
            min_long_seq_len=3,
            min_seq_chain_sum=7,
            require_time_uniqueness=1,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        TemplateSpec(
            name="hard_long_seq_spa",
            bucket="hard",
            type_min={"act": 1, "seq": 1, "spa": 1},
            type_max={"cls": 1, "app": 2, "act": 3, "seq": 3, "spa": 2, "env": inf, "int": 1},
            k_min=5,
            k_max=8,
            q_min=6,
            require_seq=1,
            seq_min_chain_len=3,
            min_long_seq_len=3,
            min_seq_chain_sum=8,
            require_time_uniqueness=1,
            weight=1.0,
            arity_min=1,
            arity_max=1,
        ),
        # Multi-target templates (simple clues only; no long sequence constraints).
        TemplateSpec(
            name="easy_pair_attr_spa",
            bucket="easy",
            type_min={"app": 1, "spa": 1},
            type_max={"cls": 3, "app": 3, "act": 1, "seq": 0, "spa": 2, "env": 2, "int": 1},
            k_min=3,
            k_max=5,
            q_min=0,
            require_seq=0,
            require_time_uniqueness=0,
            weight=0.9,
            arity_min=2,
            arity_max=3,
        ),
        TemplateSpec(
            name="easy_pair_attr_env",
            bucket="easy",
            type_min={"app": 1, "env": 1},
            type_max={"cls": 3, "app": 3, "act": 2, "seq": 0, "spa": 1, "env": 2, "int": 1},
            k_min=3,
            k_max=5,
            q_min=0,
            require_seq=0,
            require_time_uniqueness=0,
            weight=0.9,
            arity_min=2,
            arity_max=3,
        ),
        TemplateSpec(
            name="medium_pair_dual_act",
            bucket="medium",
            type_min={"act": 2},
            type_max={"cls": 2, "app": 2, "act": 2, "seq": 0, "spa": 1, "env": 1, "int": 1},
            k_min=3,
            k_max=5,
            q_min=2,
            require_seq=0,
            require_time_uniqueness=0,
            weight=0.7,
            arity_min=2,
            arity_max=2,
        ),
        TemplateSpec(
            name="medium_triplet_simple",
            bucket="medium",
            type_min={"app": 1},
            type_max={"cls": 3, "app": 3, "act": 1, "seq": 0, "spa": 2, "env": 2, "int": 1},
            k_min=3,
            k_max=6,
            q_min=0,
            require_seq=0,
            require_time_uniqueness=0,
            weight=0.5,
            arity_min=3,
            arity_max=3,
        ),
    ]


def _compute_candidate_difficulty(
    candidate: CandidateTarget,
    index: GraphIndex,
    weights: DifficultyWeights,
) -> Tuple[float, float, float, Dict[str, Any]]:
    n_change = 0.0
    n_time = 0.0
    n_same = 0.0
    tiou_vals: List[float] = []

    for member in candidate.members:
        obj_id = member.object_id
        obj = index.objects[obj_id]
        cls = _sample_class(obj, obj_id)
        cur = (member.start, member.end)

        act_labels = {
            item["label"]
            for item in index.actions_by_obj.get(obj_id, [])
            if _overlap(cur, (item["start"], item["end"]))
        }
        rel_labels = {
            item["predicate"]
            for item in index.relations_by_subj.get(obj_id, [])
            if _overlap(cur, (item["start"], item["end"]))
        }
        n_change += max(0, len(act_labels) - 1)
        n_change += max(0, len(rel_labels) - 1)

        n_time += sum(1 for tn in index.temporal_spans if _overlap(cur, (tn["start"], tn["end"])))

        for other_id, other_obj in index.objects.items():
            if other_id == obj_id:
                continue
            if _sample_class(other_obj, other_id) != cls:
                continue
            other_range = (min(index.frames[other_id]), max(index.frames[other_id]))
            ov = _overlap(cur, other_range)
            if not ov:
                continue
            n_same += 1
            tiou_vals.append(_interval_tiou(cur, other_range))

    avg_tiou = (sum(tiou_vals) / len(tiou_vals)) if tiou_vals else 0.0
    # Simple arity normalization for multi-target candidates.
    arity_norm = float(max(1, candidate.arity))
    n_change_norm = n_change / arity_norm
    n_time_norm = n_time / arity_norm
    n_same_norm = n_same / arity_norm

    eps = max(0.0, float(weights.eps))
    lambda_weight = min(max(float(weights.lambda_weight), 0.0), 1.0)

    # Use geometric means to emphasize co-occurring difficulty factors rather than
    # letting a single large term dominate the score. A small epsilon prevents
    # one zero-valued factor from collapsing the whole component to zero.
    change_score = n_change_norm / (1.0 + n_change_norm)
    time_score = n_time_norm / (1.0 + n_time_norm)
    same_score = n_same_norm / (1.0 + n_same_norm)

    D_t = math.sqrt((eps + change_score) * (eps + time_score))
    D_s = math.sqrt((eps + same_score) * (eps + avg_tiou))
    D = lambda_weight * D_t + (1.0 - lambda_weight) * D_s

    return D_t, D_s, D, {
        "n_change": n_change,
        "n_time": n_time,
        "n_same": n_same,
        "n_change_norm": n_change_norm,
        "n_time_norm": n_time_norm,
        "n_same_norm": n_same_norm,
        "arity_norm": arity_norm,
        "avg_tiou": avg_tiou,
        "eps": eps,
        "lambda_weight": lambda_weight,
    }


def _assign_sampling_buckets(candidates: List[CandidateTarget]) -> None:
    if not candidates:
        return
    ordered = sorted(candidates, key=lambda c: (c.difficulty, c.interval[1] - c.interval[0] + 1))
    n = len(ordered)
    ratios = {"easy": 0.30, "medium": 0.45, "hard": 0.25}

    raw_counts = {k: ratios[k] * n for k in ratios}
    counts = {k: int(math.floor(raw_counts[k])) for k in ratios}
    remain = n - sum(counts.values())
    for k in sorted(ratios.keys(), key=lambda x: raw_counts[x] - counts[x], reverse=True):
        if remain <= 0:
            break
        counts[k] += 1
        remain -= 1

    if n >= 3:
        if counts["medium"] == 0:
            donor = "easy" if counts["easy"] > counts["hard"] else "hard"
            if counts[donor] > 1:
                counts[donor] -= 1
                counts["medium"] += 1
        if counts["easy"] == 0:
            donor = "medium" if counts["medium"] > counts["hard"] else "hard"
            if counts[donor] > 1:
                counts[donor] -= 1
                counts["easy"] += 1
        if counts["hard"] == 0:
            donor = "medium" if counts["medium"] > counts["easy"] else "easy"
            if counts[donor] > 1:
                counts[donor] -= 1
                counts["hard"] += 1

    easy_end = counts["easy"]
    medium_end = counts["easy"] + counts["medium"]
    for i, c in enumerate(ordered):
        if i < easy_end:
            c.sampling_bucket = "easy"
        elif i < medium_end:
            c.sampling_bucket = "medium"
        else:
            c.sampling_bucket = "hard"


def _assign_difficulty_buckets_5(candidates: List[CandidateTarget]) -> None:
    if not candidates:
        return
    for cand in candidates:
        score = float(cand.difficulty)
        if score < 0.2:
            cand.difficulty_bucket = "very_easy"
        elif score < 0.4:
            cand.difficulty_bucket = "easy"
        elif score < 0.6:
            cand.difficulty_bucket = "medium"
        elif score < 0.8:
            cand.difficulty_bucket = "hard"
        else:
            cand.difficulty_bucket = "very_hard"


def _render_query(profile: CandidateProfile, clues: List[AtomicClue]) -> str:
    cls_phrase = None
    temporal_phrases: List[str] = []
    other_phrases: List[str] = []

    for clue in clues:
        if clue.clue_type == "cls" and cls_phrase is None:
            cls_phrase = clue.text
            continue
        if clue.clue_type == "env" and clue.signature[1] == "temporal":
            temporal_phrases.append(clue.text)
            continue
        other_phrases.append(clue.text)

    if cls_phrase is None:
        cls_phrase = f"the {profile.classes[0] if profile.classes else 'object'}"

    parts = [cls_phrase]
    if temporal_phrases:
        parts.extend(temporal_phrases)
    if other_phrases:
        parts.append(" and ".join(other_phrases))
    return _norm(" ".join(parts))


def decorate_query(core_clues: List[AtomicClue], profile: CandidateProfile) -> Tuple[str, List[str]]:
    if profile.arity > 1 and len(core_clues) <= 2:
        return "", []
    clue_types = {c.clue_type for c in core_clues}
    bucket_keys: List[str] = []
    if "seq" in clue_types:
        bucket_keys.extend(["temporal", "action"])
    if "spa" in clue_types or "int" in clue_types:
        bucket_keys.append("spatial")
    if "app" in clue_types:
        bucket_keys.append("appearance")
    if not bucket_keys:
        bucket_keys.append("neutral")

    decorations: List[str] = []
    for key in bucket_keys:
        for phrase in DECORATION_POOL.get(key, ()):
            if phrase not in decorations:
                decorations.append(phrase)
                break
        if len(decorations) >= 2:
            break

    if not decorations:
        return "", []
    return _norm(", " + "; ".join(decorations)), decorations


def _query_core_from_clues(core_clues: List[AtomicClue], profile: CandidateProfile) -> str:
    if profile.arity <= 1:
        parts = [_norm(c.text) for c in core_clues if _norm(c.text)]
        if not parts:
            return ""
        return ", ".join(parts)

    member_cls: Dict[int, str] = {}
    member_props: Dict[int, List[str]] = defaultdict(list)
    shared_parts: List[str] = []

    for clue in core_clues:
        text = _norm(clue.text)
        if not text:
            continue
        if len(clue.member_indices) != 1:
            shared_parts.append(text)
            continue
        k = clue.member_indices[0]
        if clue.clue_type == "cls" and k not in member_cls:
            member_cls[k] = text
            continue
        if clue.clue_type == "seq":
            continue
        s = _norm(text)
        for prefix in ("that is ", "that "):
            if s.startswith(prefix):
                s = _norm(s[len(prefix) :])
                break
        member_props[k].append(s)

    parts: List[str] = []
    for k in range(profile.arity):
        cls = member_cls.get(k) or f"the {profile.classes[k] if k < len(profile.classes) else 'object'}"
        qualifiers: List[str] = []
        seen: Set[str] = set()
        for q in member_props.get(k, []):
            if not q or q == cls or q in seen:
                continue
            qualifiers.append(q)
            seen.add(q)
        if qualifiers:
            parts.append(_norm(f"{cls} {' and '.join(qualifiers[:2])}"))
        else:
            parts.append(cls)

    core = _norm(" and ".join(parts))
    if shared_parts:
        core = _norm(core + " and " + " and ".join(shared_parts))
    return core


class CPSATQuerySampler:
    def __init__(
        self,
        min_interval_len: int = 3,
        max_intervals_per_object: int = 12,
        max_target_arity: int = 3,
        max_multi_intervals_per_group: int = 8,
        max_multi_candidates_total: int = 240,
        multi_queries_per_graph: Optional[int] = None,
        strict_time_uniqueness_multi_target: bool = False,
        max_chain_len: int = 4,
        queries_per_graph: int = 12,
        max_queries_per_candidate: int = 1,
        time_limit_sec: float = 2.0,
        seed: int = 7,
        weights: DifficultyWeights = DifficultyWeights(),
    ) -> None:
        self.min_interval_len = max(1, int(min_interval_len))
        self.max_intervals_per_object = max(1, int(max_intervals_per_object))
        self.max_target_arity = min(3, max(1, int(max_target_arity)))
        self.max_multi_intervals_per_group = max(1, int(max_multi_intervals_per_group))
        self.max_multi_candidates_total = max(1, int(max_multi_candidates_total))
        self.multi_queries_per_graph = (
            max(0, int(multi_queries_per_graph))
            if multi_queries_per_graph is not None
            else max(0, int(queries_per_graph) // 3)
        )
        self.strict_time_uniqueness_multi_target = bool(strict_time_uniqueness_multi_target)
        self.max_chain_len = max(2, int(max_chain_len))
        self.queries_per_graph = max(1, int(queries_per_graph))
        self.max_queries_per_candidate = max(1, int(max_queries_per_candidate))
        self.time_limit_sec = max(0.1, float(time_limit_sec))
        self.seed = int(seed)
        self.weights = weights
        self.templates = _build_default_templates()

    def _template_quotas(self, total: int, templates: List[TemplateSpec]) -> Dict[str, int]:
        if total <= 0 or not templates:
            return {}
        weights = [max(0.0, t.weight) for t in templates]
        total_w = sum(weights) or 1.0
        raw = [w / total_w * total for w in weights]
        quotas = [int(math.floor(x)) for x in raw]
        remain = total - sum(quotas)
        order = sorted(range(len(raw)), key=lambda i: raw[i] - quotas[i], reverse=True)
        for i in order[:remain]:
            quotas[i] += 1
        return {templates[i].name: quotas[i] for i in range(len(templates))}

    def _candidate_order(
        self,
        candidates: List[CandidateTarget],
        template: TemplateSpec,
        sampled_intervals_by_sig: Dict[Tuple[str, ...], Set[Tuple[int, int]]],
        candidate_use_count: Dict[str, int],
    ) -> List[CandidateTarget]:
        pool = [
            c
            for c in candidates
            if c.sampling_bucket == template.bucket and template.arity_min <= c.arity <= template.arity_max
        ]

        def rank(c: CandidateTarget) -> Tuple[int, int, float, int]:
            sig = tuple(m.object_id for m in c.members)
            interval = c.interval
            sampled_for_sig = sampled_intervals_by_sig.get(sig, set())
            if sampled_for_sig and interval not in sampled_for_sig:
                mode = 0
            elif not sampled_for_sig:
                mode = 1
            else:
                mode = 2
            return (
                mode,
                candidate_use_count.get(c.candidate_id, 0),
                c.difficulty,
                interval[1] - interval[0] + 1,
            )

        return sorted(pool, key=rank)

    def _solve_for_candidate_template(
        self,
        candidate: CandidateTarget,
        all_candidates: List[CandidateTarget],
        profile_map: Dict[str, CandidateProfile],
        template: TemplateSpec,
    ) -> Optional[Dict[str, Any]]:
        clues = build_atomic_clues(self._index, candidate, max_chain_len=self.max_chain_len)
        if not clues:
            return None

        _, object_E, time_E = build_exclusion_matrix(candidate, all_candidates, clues, profile_map)
        solved = solve_query_cpsat(
            target=candidate,
            template=template,
            clues=clues,
            object_exclusion_matrix=object_E,
            time_exclusion_matrix=time_E,
            enforce_time_uniqueness=(candidate.arity == 1 or self.strict_time_uniqueness_multi_target),
            time_limit_sec=self.time_limit_sec,
        )
        if not solved:
            return None

        selected = [clues[i] for i in solved["selected_indices"]]
        profile = profile_map[candidate.candidate_id]
        core_query = _query_core_from_clues(selected, profile) or _render_query(profile, selected)
        suffix, decorations = decorate_query(selected, profile)
        query = _norm(core_query + suffix)

        return {
            "candidate": candidate,
            "profile": profile,
            "selected_clues": selected,
            "solver": solved,
            "template": template,
            "core_query": core_query,
            "decorations": decorations,
            "query": query,
        }

    def _templates_for_arity(self, multi: bool) -> List[TemplateSpec]:
        if multi:
            return [t for t in self.templates if t.arity_min >= 2]
        return [t for t in self.templates if t.arity_min <= 1 <= t.arity_max and t.arity_max == 1]

    def _node_from_solution(self, picked: Dict[str, Any], query_index: int) -> Dict[str, Any]:
        cand: CandidateTarget = picked["candidate"]
        profile: CandidateProfile = picked["profile"]
        template: TemplateSpec = picked["template"]
        members = [
            {
                "object_id": m.object_id,
                "class": profile.classes[idx],
                "start_frame": m.start,
                "end_frame": m.end,
            }
            for idx, m in enumerate(cand.members)
        ]
        per_target_clues: List[Dict[str, Any]] = []
        for idx, member in enumerate(members, 1):
            grouped = []
            for c in picked["selected_clues"]:
                if tuple(c.member_indices) != (idx - 1,):
                    continue
                grouped.append(
                    {
                        "type": c.clue_type,
                        "text": c.text,
                        "chain_len": c.chain_len,
                        "is_temporal_evidence": c.is_temporal_evidence,
                    }
                )
            per_target_clues.append(
                {
                    "target_index": idx,
                    **member,
                    "clues": grouped,
                }
            )

        shared_clues = [
            {
                "type": c.clue_type,
                "text": c.text,
                "member_indices": list(c.member_indices),
                "chain_len": c.chain_len,
                "is_temporal_evidence": c.is_temporal_evidence,
            }
            for c in picked["selected_clues"]
            if len(c.member_indices) != 1
        ]

        return {
            "query_id": f"cpsat_{query_index}_{cand.candidate_id}",
            "query": picked["query"],
            "query_core": picked["core_query"],
            "query_decorations": picked["decorations"],
            "template": template.name,
            "difficulty_bucket": cand.difficulty_bucket,
            "D_t": cand.D_t,
            "D_s": cand.D_s,
            "D": cand.difficulty,
            "target": {
                "candidate_id": cand.candidate_id,
                "members": members,
            },
            "clues": {
                "per_target": per_target_clues,
                "shared": shared_clues,
            },
            "solver": picked["solver"],
        }

    def _sample_with_templates(
        self,
        *,
        candidates: List[CandidateTarget],
        profile_map: Dict[str, CandidateProfile],
        templates: List[TemplateSpec],
        target_count: int,
        start_query_index: int,
    ) -> List[Dict[str, Any]]:
        if target_count <= 0 or not candidates or not templates:
            return []

        quotas = self._template_quotas(target_count, templates)
        cur_counts: Dict[str, int] = defaultdict(int)
        candidate_use_count: Dict[str, int] = defaultdict(int)
        sampled_intervals_by_sig: Dict[Tuple[str, ...], Set[Tuple[int, int]]] = defaultdict(set)
        used_object_sigs: Set[Tuple[str, ...]] = set()
        solve_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
        blocked_templates: Set[str] = set()
        template_by_name = {t.name: t for t in templates}
        nodes: List[Dict[str, Any]] = []

        while len(nodes) < target_count:
            deficits: List[Tuple[int, str]] = []
            for t in templates:
                if t.name in blocked_templates:
                    continue
                gap = quotas.get(t.name, 0) - cur_counts.get(t.name, 0)
                if gap > 0:
                    deficits.append((gap, t.name))
            if not deficits:
                break
            deficits.sort(reverse=True)
            _, tname = deficits[0]
            template = template_by_name[tname]

            picked = None
            for cand in self._candidate_order(candidates, template, sampled_intervals_by_sig, candidate_use_count):
                obj_sig = tuple(m.object_id for m in cand.members)
                if obj_sig in used_object_sigs:
                    continue
                if candidate_use_count[cand.candidate_id] >= self.max_queries_per_candidate:
                    continue
                cache_key = (cand.candidate_id, template.name)
                if cache_key not in solve_cache:
                    solve_cache[cache_key] = self._solve_for_candidate_template(
                        candidate=cand,
                        all_candidates=candidates,
                        profile_map=profile_map,
                        template=template,
                    )
                solved = solve_cache[cache_key]
                if solved:
                    picked = solved
                    break

            if not picked:
                blocked_templates.add(template.name)
                continue

            cand: CandidateTarget = picked["candidate"]
            candidate_use_count[cand.candidate_id] += 1
            obj_sig = tuple(m.object_id for m in cand.members)
            sampled_intervals_by_sig[obj_sig].add(cand.interval)
            used_object_sigs.add(obj_sig)
            cur_counts[template.name] += 1
            nodes.append(self._node_from_solution(picked, start_query_index + len(nodes)))

        return nodes

    def generate_for_graph(self, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._index = build_graph_index(graph)
        candidates = build_candidate_intervals(
            graph=graph,
            index=self._index,
            min_interval_len=self.min_interval_len,
            max_intervals_per_object=self.max_intervals_per_object,
            include_full_track=True,
            max_target_arity=self.max_target_arity,
            max_multi_intervals_per_group=self.max_multi_intervals_per_group,
            max_multi_candidates_total=self.max_multi_candidates_total,
        )
        if not candidates:
            return []

        for c in candidates:
            D_t, D_s, D, meta = _compute_candidate_difficulty(c, self._index, self.weights)
            c.D_t = D_t
            c.D_s = D_s
            c.difficulty = D
            c.meta.update(meta)

        single_candidates = [c for c in candidates if c.arity == 1]
        multi_candidates = [c for c in candidates if c.arity >= 2]
        # Keep single-target difficulty distribution consistent with legacy behavior.
        _assign_sampling_buckets(single_candidates)
        _assign_difficulty_buckets_5(single_candidates)
        if multi_candidates:
            _assign_sampling_buckets(multi_candidates)
            _assign_difficulty_buckets_5(multi_candidates)
        profile_map = {
            c.candidate_id: _profile_for_candidate(self._index, c, max_chain_len=self.max_chain_len)
            for c in candidates
        }

        single_templates = self._templates_for_arity(multi=False)
        results = self._sample_with_templates(
            candidates=single_candidates,
            profile_map=profile_map,
            templates=single_templates,
            target_count=self.queries_per_graph,
            start_query_index=0,
        )

        multi_templates = self._templates_for_arity(multi=True)
        multi_count = self.multi_queries_per_graph if self.max_target_arity > 1 else 0
        if multi_count > 0:
            results.extend(
                self._sample_with_templates(
                    candidates=multi_candidates,
                    profile_map=profile_map,
                    templates=multi_templates,
                    target_count=multi_count,
                    start_query_index=len(results),
                )
            )

        return results

    async def polish_queries_with_llm(
        self,
        query_nodes: List[Dict[str, Any]],
        model_name: str,
        api_keys: Optional[Union[str, Iterable[str]]],
        max_concurrent_per_key: int = 100,
        max_retries: int = 5,
        system_prompt: str = QUERY_POLISH_SYSTEM_PROMPT,
    ) -> Dict[str, Dict[str, Any]]:
        if not query_nodes:
            return {}

        if api_keys is None:
            api_keys = os.getenv("API_KEYS", "")
        keys = _normalize_api_keys(api_keys)
        if not keys:
            env_path = Path(__file__).resolve().parents[1] / ".env"
            if env_path.exists():
                with env_path.open("r", encoding="utf-8") as f:
                    for raw_line in f:
                        line = raw_line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        if key.strip() == "API_KEYS":
                            keys = _normalize_api_keys(value.strip().strip('"').strip("'"))
                            if keys:
                                break
        if not keys:
            raise ValueError("API_KEYS is required when use_llm_polish=True (set env/.env or pass --api_keys).")

        # Align with other API modules: read base URL from env/.env chain.
        api_base_url = (
            os.getenv("MM_API_BASE_URL", "").strip()
            or os.getenv("VISION_API_BASE_URL", "").strip()
            or os.getenv("VIDEO_API_BASE_URL", "").strip()
            or _load_env_var_from_project_env("MM_API_BASE_URL")
            or _load_env_var_from_project_env("VISION_API_BASE_URL")
            or _load_env_var_from_project_env("VIDEO_API_BASE_URL")
        )
        if api_base_url:
            os.environ.setdefault("MM_API_BASE_URL", api_base_url)

        generator = StreamGenerator(
            model_name=model_name,
            api_keys=keys,
            max_concurrent_per_key=max_concurrent_per_key,
            max_retries=max_retries,
            rational=False,
            with_unique_id=True,
        )

        prompts: List[Dict[str, Any]] = []
        for idx, node in enumerate(query_nodes):
            query_id = str(node.get("query_id", f"cpsat_{idx}"))
            target = node.get("target") or {}
            members = target.get("members") or []
            clues = node.get("clues") or {}

            target_classes = []
            for i, m in enumerate(members, start=1):
                cls = _norm(m.get("class", "object")) or "object"
                s = _to_int(m.get("start_frame"), 0)
                e = _to_int(m.get("end_frame"), s)
                if e < s:
                    s, e = e, s
                target_classes.append({"target_index": i, "class": cls, "interval": [s, e]})
            if not target_classes:
                target_classes = [{"target_index": 1, "class": "object", "interval": ["unknown", "unknown"]}]

            target_lines: List[str] = []
            for item in target_classes:
                tid = _to_int(item.get("target_index"), -1)
                cls = _norm(item.get("class", "object")) or "object"
                if tid <= 0:
                    tid = len(target_lines) + 1
                target_lines.append(f"- target {tid}: {cls}")
            target_classes_text = "\n".join(target_lines) if target_lines else "- target 1: object"

            category_name = {
                "cls": "target identity/class",
                "app": "appearance",
                "act": "action",
                "seq": "action sequence",
                "spa": "spatial relation",
                "int": "interaction relation",
                "env": "environment/temporal context",
            }
            categories = ["cls", "app", "act", "seq", "spa", "int", "env"]
            per_target: Dict[str, Dict[str, List[str]]] = {
                str(i + 1): {c: [] for c in categories} for i in range(len(members))
            }
            shared: Dict[str, List[str]] = {c: [] for c in categories}

            def append_unique(bucket: Dict[str, List[str]], ctype: str, text: str) -> None:
                if not text:
                    return
                bucket.setdefault(ctype, [])
                if text not in bucket[ctype]:
                    bucket[ctype].append(text)

            for item in clues.get("per_target", []) or []:
                tid = _to_int(item.get("target_index"), -1)
                if not (1 <= tid <= len(members)):
                    continue
                for clue in item.get("clues", []) or []:
                    append_unique(
                        per_target[str(tid)],
                        str(clue.get("type", "")).strip().lower() or "unknown",
                        _norm(clue.get("text", "")),
                    )
            for clue in clues.get("shared", []) or []:
                append_unique(
                    shared,
                    str(clue.get("type", "")).strip().lower() or "unknown",
                    _norm(clue.get("text", "")),
                )

            if shared and target_classes:
                for t in target_classes:
                    tid = str(_to_int(t.get("target_index"), -1))
                    per_target.setdefault(tid, {})
                    for ctype, vals in shared.items():
                        if not isinstance(vals, list) or not vals:
                            continue
                        existing = per_target[tid].get(ctype, [])
                        merged = list(existing)
                        for v in vals:
                            vv = _norm(v)
                            if vv and vv not in merged:
                                merged.append(vv)
                        per_target[tid][ctype] = merged

            def sort_key(tid: str) -> Tuple[int, str]:
                return (_to_int(tid, 10**9), str(tid))

            clue_lines: List[str] = []
            for tid in sorted(per_target.keys(), key=sort_key):
                clue_lines.append(f"  target {tid}:")
                buckets = per_target.get(tid, {})
                if not isinstance(buckets, dict):
                    clue_lines.append("    - xxx")
                    continue
                wrote = False
                for ctype in PROMPT_CATEGORY_ORDER:
                    vals = buckets.get(ctype, [])
                    if not isinstance(vals, list) or not vals:
                        continue
                    clean_vals = []
                    for v in vals:
                        vv = _norm(v)
                        lower = vv.lower()
                        for p in ("that is ", "that ", "the "):
                            if lower.startswith(p):
                                vv = _norm(vv[len(p):])
                                lower = vv.lower()
                        if vv:
                            clean_vals.append(vv)
                    if not clean_vals:
                        continue
                    label = PROMPT_CATEGORY_TEXT.get(ctype, ctype)
                    clue_lines.append(f"    - {label}: {', '.join(clean_vals)}")
                    wrote = True
                if not wrote:
                    clue_lines.append("    - xxx")
                clue_lines.append("")
            while clue_lines and not clue_lines[-1].strip():
                clue_lines.pop()
            clues_per_target_text = "\n".join(clue_lines) if clue_lines else "  target 1:\n    - xxx"

            prompt_text = QUERY_POLISH_PROMPT_TEMPLATE.format(
                target_classes_text=target_classes_text,
                clues_per_target_text=clues_per_target_text,
            )
            prompts.append({"id": query_id, "prompt": prompt_text})

        print(
            f"[query_cpsat] llm_polish_start: model={model_name} prompts={len(prompts)} keys={len(keys)} "
            f"base_url={'set' if api_base_url else 'default'}",
            flush=True,
        )
        polished_map: Dict[str, Dict[str, Any]] = {}
        completed = 0

        def extract_polished_query(response: str) -> Optional[Dict[str, Any]]:
            parsed = JSONParser.parse(response)
            if not isinstance(parsed, dict):
                return None
            query = parsed.get("query")
            if not isinstance(query, str):
                return None
            query = _norm(query)
            if not query:
                return None
            target_queries_raw = parsed.get("target_queries")
            if not isinstance(target_queries_raw, dict):
                return None
            target_queries: Dict[str, str] = {}
            for key, value in target_queries_raw.items():
                kk = _norm(key)
                vv = _norm(value)
                if not kk or not vv:
                    continue
                target_queries[kk] = vv
            if not target_queries:
                return None
            return {"query": query, "target_queries": target_queries}

        async for item in generator.generate_stream(
            prompts=prompts,
            system_prompt=system_prompt,
            validate_func=lambda resp: polished if (polished := extract_polished_query(resp)) is not None else False,
        ):
            result = item.get("result") if item else None
            if item and isinstance(result, dict):
                polished_map[str(item["id"])] = result
                completed += 1
                print(f"[query_cpsat] llm_polish_progress: {completed}/{len(prompts)}", flush=True)
        print(f"[query_cpsat] llm_polish_done: polished={len(polished_map)}", flush=True)
        return polished_map

    def _build_minimal_records(self, graph: Dict[str, Any], query_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        objects = {
            str(obj.get("node_id")): obj
            for obj in (graph.get("object_nodes") or [])
            if str(obj.get("node_id", "")).strip()
        }
        records: List[Dict[str, Any]] = []
        for q in query_nodes:
            target = q.get("target") or {}
            members = target.get("members") or []
            if not members:
                continue

            member_payload: List[Dict[str, Any]] = []
            for m in members:
                oid = str(m.get("object_id", "")).strip()
                if not oid or oid not in objects:
                    continue
                start = _to_int(m.get("start_frame"), 0)
                end = _to_int(m.get("end_frame"), start)
                if end < start:
                    start, end = end, start
                raw_boxes = objects[oid].get("bboxes") or {}
                boxes: Dict[str, Any] = {}
                for frame, box in raw_boxes.items():
                    frame_int = _to_int(frame, -1)
                    if frame_int < 0:
                        continue
                    if start <= frame_int <= end:
                        boxes[str(frame_int)] = box
                if boxes:
                    boxes = {str(f): boxes[str(f)] for f in sorted(int(k) for k in boxes.keys())}
                if not boxes:
                    continue
                member_payload.append(
                    {
                        "object_id": oid,
                        "start_frame": start,
                        "end_frame": end,
                        "target_index": len(member_payload) + 1,
                        "boxes": boxes,
                    }
                )
            if not member_payload:
                continue

            records.append(
                {
                    "video_path": graph.get("video_path"),
                    "video_width": graph.get("video_width"),
                    "video_height": graph.get("video_height"),
                    "query_id": q.get("query_id"),
                    "query": q.get("query"),
                    "llm_polished": q.get("llm_polished", False),
                    "query_core": q.get("query_core"),
                    "template": q.get("template"),
                    "difficulty_bucket": q.get("difficulty_bucket"),
                    "D_t": q.get("D_t"),
                    "D_s": q.get("D_s"),
                    "D": q.get("D"),
                    "target_arity": len(member_payload),
                    "target_members": member_payload,
                    "per_target_queries": q.get("per_target_queries", {}),
                    "clues": q.get("clues", {}),
                    "solver": q.get("solver"),
                }
            )
        return records

    def process_jsonl(
        self,
        input_path: str,
        output_path: str,
        use_llm_polish: bool = False,
        polish_model_name: str = "gpt-4.1-mini",
        api_keys: Optional[Union[str, Iterable[str]]] = None,
        max_concurrent_per_key: int = 100,
        max_retries: int = 5,
    ) -> None:
        global RELATION_SYNONYMS, ATTRIBUTE_SYNONYMS
        in_path = Path(input_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rel_map, attr_map, stats = _summarize_synonyms_from_graph_file(in_path)
        RELATION_SYNONYMS = rel_map
        ATTRIBUTE_SYNONYMS = attr_map
        print(
            "[query_cpsat] synonym_summary: "
            f"graphs={stats['graphs']} "
            f"rel_values={stats['relation_values']} attr_values={stats['attribute_values']} "
            f"rel_synonyms={stats['relation_synonyms']} attr_synonyms={stats['attribute_synonyms']}",
            flush=True,
        )

        graph_count = 0
        record_count = 0
        with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                graph = json.loads(line)
                query_nodes = self.generate_for_graph(graph)
                if use_llm_polish and query_nodes:
                    polished_map = asyncio.run(
                        self.polish_queries_with_llm(
                            query_nodes=query_nodes,
                            model_name=polish_model_name,
                            api_keys=api_keys,
                            max_concurrent_per_key=max_concurrent_per_key,
                            max_retries=max_retries,
                        )
                    )
                    kept_nodes: List[Dict[str, Any]] = []
                    dropped = 0
                    for q in query_nodes:
                        qid = str(q.get("query_id", ""))
                        polished = polished_map.get(qid) or {}
                        polished_query = _norm(polished.get("query", ""))
                        if polished_query:
                            q["query"] = polished_query
                            q["per_target_queries"] = polished.get("target_queries", {})
                            q["llm_polished"] = True
                            kept_nodes.append(q)
                        else:
                            dropped += 1
                    query_nodes = kept_nodes
                    if dropped > 0:
                        print(f"[query_cpsat] llm_generation_drop: dropped={dropped}", flush=True)
                records = self._build_minimal_records(graph, query_nodes)
                for record in records:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                graph_count += 1
                record_count += len(records)

        print(f"[query_cpsat] done. graphs={graph_count}, records={record_count}, output={out_path}", flush=True)


def main(
    input_path: str,
    output_path: str,
    queries_per_graph: int = 12,
    multi_queries_per_graph: Optional[int] = None,
    min_interval_len: int = 3,
    max_intervals_per_object: int = 12,
    max_target_arity: int = 3,
    max_multi_intervals_per_group: int = 6,
    max_multi_candidates_total: int = 240,
    strict_time_uniqueness_multi_target: bool = False,
    max_chain_len: int = 6,
    max_queries_per_candidate: int = 1,
    time_limit_sec: float = 2.0,
    seed: int = 7,
    eps: float = 0.01,
    lambda_weight: float = 0.5,
    use_llm_polish: bool = False,
    polish_model_name: str = "gpt-4.1-mini",
    api_keys: Optional[Union[str, Iterable[str]]] = None,
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
) -> None:
    sampler = CPSATQuerySampler(
        min_interval_len=min_interval_len,
        max_intervals_per_object=max_intervals_per_object,
        max_target_arity=max_target_arity,
        max_multi_intervals_per_group=max_multi_intervals_per_group,
        max_multi_candidates_total=max_multi_candidates_total,
        multi_queries_per_graph=multi_queries_per_graph,
        strict_time_uniqueness_multi_target=strict_time_uniqueness_multi_target,
        max_chain_len=max_chain_len,
        queries_per_graph=queries_per_graph,
        max_queries_per_candidate=max_queries_per_candidate,
        time_limit_sec=time_limit_sec,
        seed=seed,
        weights=DifficultyWeights(
            eps=eps,
            lambda_weight=lambda_weight,
        ),
    )
    sampler.process_jsonl(
        input_path=input_path,
        output_path=output_path,
        use_llm_polish=use_llm_polish,
        polish_model_name=polish_model_name,
        api_keys=api_keys,
        max_concurrent_per_key=max_concurrent_per_key,
        max_retries=max_retries,
    )


if __name__ == "__main__":
    fire.Fire(main)
