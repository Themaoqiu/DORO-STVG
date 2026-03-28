from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import fire

from ortools.sat.python import cp_model


CLUE_TYPES = ("cls", "app", "act", "seq", "spa", "env", "int")


def _norm(text: Any) -> str:
    return " ".join(str(text).strip().split())


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


def _bbox_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


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
    difficulty_bucket: str = "easy"
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
    time_anchor_indices: Tuple[int, ...]
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
    weight: float = 1.0


@dataclass(frozen=True)
class DifficultyWeights:
    alpha_change: float = 1.0
    alpha_time: float = 1.0
    beta_same: float = 1.0
    beta_sep: float = 1.0


@dataclass
class GraphIndex:
    objects: Dict[str, Dict[str, Any]]
    idx_to_node: Dict[int, str]
    frames: Dict[str, Set[int]]
    centers: Dict[str, Dict[int, Tuple[float, float]]]
    actions_by_obj: Dict[str, List[Dict[str, Any]]]
    relations_by_subj: Dict[str, List[Dict[str, Any]]]
    temporal_spans: List[Dict[str, int]]
    frame_diag: float


@dataclass
class CandidateProfile:
    candidate_id: str
    arity: int
    classes: List[str]
    attrs: List[Set[str]]
    env: List[Set[str]]
    temporal_tags: List[Set[str]]
    actions: List[Set[str]]
    sequences: List[Set[Tuple[str, str]]]
    spa: List[Set[Tuple[str, str]]]
    inter: List[Set[Tuple[str, str]]]


def build_graph_index(graph: Dict[str, Any]) -> GraphIndex:
    object_nodes = graph.get("object_nodes") or []
    objects: Dict[str, Dict[str, Any]] = {}
    idx_to_node: Dict[int, str] = {}
    frames: Dict[str, Set[int]] = {}
    centers: Dict[str, Dict[int, Tuple[float, float]]] = {}
    max_x, max_y = 1.0, 1.0

    for i, obj in enumerate(object_nodes):
        node_id = str(obj.get("node_id", "")).strip()
        if not node_id:
            continue
        objects[node_id] = obj
        idx_to_node[i] = node_id
        frames[node_id] = _frames_from_obj(obj)
        centers[node_id] = {}
        for fk, box in (obj.get("bboxes") or {}).items():
            fidx = _to_int(fk, -1)
            if fidx < 0 or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            centers[node_id][fidx] = _bbox_center(box)
            try:
                max_x = max(max_x, float(box[2]))
                max_y = max(max_y, float(box[3]))
            except Exception:
                pass

    actions_by_obj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in graph.get("action_nodes") or []:
        owner = str(node.get("object_id", "")).strip()
        if owner not in objects:
            continue
        for item in node.get("actions") or []:
            label = _norm(item.get("action_label", "")).lower()
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
                    "targets": [str(t).strip() for t in (item.get("target_object_ids") or []) if str(t).strip() in objects],
                }
            )

    relations_by_subj: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge in graph.get("edges") or []:
        if "subject_id" not in edge or "relationships" not in edge:
            continue
        sid = idx_to_node.get(_to_int(edge.get("subject_id"), -1), "")
        if sid not in objects:
            continue
        for rel in edge.get("relationships") or []:
            oid = idx_to_node.get(_to_int(rel.get("object_id"), -1), "")
            if oid not in objects:
                continue
            pred = _norm(rel.get("predicate_verb", "")).lower()
            if not pred:
                continue
            tf = rel.get("time_frames") or []
            segments: List[Tuple[int, int]] = []
            for seg in tf:
                if isinstance(seg, (list, tuple)) and len(seg) == 2:
                    a = _to_int(seg[0], 0)
                    b = _to_int(seg[1], a)
                    if b < a:
                        a, b = b, a
                    segments.append((a, b))
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

    temporal_spans: List[Dict[str, int]] = []
    for tn in graph.get("temporal_nodes") or []:
        temporal_spans.append(
            {
                "clip_id": _to_int(tn.get("clip_id"), 0),
                "start": _to_int(tn.get("start_frame"), 0),
                "end": _to_int(tn.get("end_frame"), 0),
            }
        )

    frame_diag = math.hypot(max_x, max_y)
    if frame_diag <= 0:
        frame_diag = 1.0

    return GraphIndex(
        objects=objects,
        idx_to_node=idx_to_node,
        frames=frames,
        centers=centers,
        actions_by_obj=actions_by_obj,
        relations_by_subj=relations_by_subj,
        temporal_spans=temporal_spans,
        frame_diag=frame_diag,
    )


def build_candidate_intervals(
    graph: Dict[str, Any],
    index: GraphIndex,
    min_interval_len: int = 3,
    max_intervals_per_object: int = 12,
    include_full_track: bool = True,
) -> List[CandidateTarget]:
    del graph
    candidates: List[CandidateTarget] = []

    for object_id, obj in index.objects.items():
        obj_frames = index.frames.get(object_id, set())
        if not obj_frames:
            continue
        full_s, full_e = min(obj_frames), max(obj_frames)
        boundaries: Set[int] = {full_s, full_e + 1}

        for item in index.actions_by_obj.get(object_id, []):
            ov = _overlap((item["start"], item["end"]), (full_s, full_e))
            if ov:
                boundaries.add(ov[0])
                boundaries.add(ov[1] + 1)

        for rel in index.relations_by_subj.get(object_id, []):
            ov = _overlap((rel["start"], rel["end"]), (full_s, full_e))
            if ov:
                boundaries.add(ov[0])
                boundaries.add(ov[1] + 1)

        for tn in index.temporal_spans:
            ov = _overlap((tn["start"], tn["end"]), (full_s, full_e))
            if ov:
                boundaries.add(ov[0])
                boundaries.add(ov[1] + 1)

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

        ordered = sorted(intervals, key=lambda x: (x[0], x[1]))
        if len(ordered) > max_intervals_per_object:
            keep = [ordered[0], ordered[-1]] if len(ordered) >= 2 else ordered
            step = max(1, len(ordered) // max(1, max_intervals_per_object - len(keep)))
            keep.extend(ordered[1:-1:step])
            ordered = sorted(set(keep))[:max_intervals_per_object]

        for s, e in ordered:
            candidates.append(
                CandidateTarget(
                    candidate_id=f"{object_id}@{s}-{e}",
                    members=[CandidateMember(object_id=object_id, start=s, end=e)],
                    meta={"object_id": object_id, "target_class": _sample_class(obj, object_id)},
                )
            )

    return candidates


def _profile_for_candidate(index: GraphIndex, candidate: CandidateTarget) -> CandidateProfile:
    classes: List[str] = []
    attrs: List[Set[str]] = []
    env: List[Set[str]] = []
    temporal_tags: List[Set[str]] = []
    actions: List[Set[str]] = []
    sequences: List[Set[Tuple[str, str]]] = []
    spa: List[Set[Tuple[str, str]]] = []
    inter: List[Set[Tuple[str, str]]] = []

    for member in candidate.members:
        obj = index.objects[member.object_id]
        classes.append(_sample_class(obj, member.object_id))
        attrs.append({x.lower() for x in _uniq(obj.get("attributes") or [])})
        env.append({x.lower() for x in _uniq(obj.get("relationships") or [])})

        tags: Set[str] = set()
        for tn in index.temporal_spans:
            if _overlap((member.start, member.end), (tn["start"], tn["end"])):
                tags.add(f"clip_{tn['clip_id']}")
        temporal_tags.append(tags)

        act_events = [
            a
            for a in index.actions_by_obj.get(member.object_id, [])
            if _overlap((member.start, member.end), (a["start"], a["end"]))
        ]
        act_events.sort(key=lambda x: (x["start"], x["end"], x["label"]))

        act_labels = {a["label"] for a in act_events}
        actions.append(act_labels)

        seq_set: Set[Tuple[str, str]] = set()
        for i in range(len(act_events)):
            for j in range(i + 1, len(act_events)):
                if act_events[i]["end"] < act_events[j]["start"] and act_events[i]["label"] != act_events[j]["label"]:
                    seq_set.add((act_events[i]["label"], act_events[j]["label"]))
        sequences.append(seq_set)

        spa_set: Set[Tuple[str, str]] = set()
        int_set: Set[Tuple[str, str]] = set()
        for rel in index.relations_by_subj.get(member.object_id, []):
            if not _overlap((member.start, member.end), (rel["start"], rel["end"])):
                continue
            pair = (rel["predicate"], rel["ref_class"])
            if rel["edge_type"] == "spatial":
                spa_set.add(pair)
            else:
                int_set.add(pair)
        for a in act_events:
            for tid in a["targets"]:
                tcls = _sample_class(index.objects[tid], tid)
                int_set.add((a["label"], tcls))
        spa.append(spa_set)
        inter.append(int_set)

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


def build_atomic_clues(index: GraphIndex, candidate: CandidateTarget) -> List[AtomicClue]:
    profile = _profile_for_candidate(index, candidate)
    clues: List[AtomicClue] = []
    seen: Set[Tuple[Any, ...]] = set()

    def add(
        clue_type: str,
        text: str,
        signature: Tuple[Any, ...],
        member_indices: Tuple[int, ...],
        time_anchor_indices: Tuple[int, ...],
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
                time_anchor_indices=time_anchor_indices,
                chain_len=chain_len,
                cost=1,
            )
        )

    for k in range(profile.arity):
        cls = profile.classes[k]
        add(
            "cls",
            f"the {cls}",
            ("cls", k, cls),
            (k,),
            tuple(),
            0,
        )

        for attr in sorted(profile.attrs[k]):
            add(
                "app",
                f"with {attr}",
                ("app", k, attr),
                (k,),
                tuple(),
                0,
            )

        for phrase in sorted(profile.env[k]):
            add(
                "env",
                f"that is {phrase}",
                ("env", "general", k, phrase),
                (k,),
                tuple(),
                0,
            )

        for tag in sorted(profile.temporal_tags[k]):
            clip_id = tag.split("_", 1)[-1]
            add(
                "env",
                f"during temporal segment {clip_id}",
                ("env", "temporal", k, tag),
                (k,),
                (k,),
                0,
            )

        for action in sorted(profile.actions[k]):
            add(
                "act",
                f"that is {action}",
                ("act", k, action),
                (k,),
                (k,),
                1,
            )

        for a1, a2 in sorted(profile.sequences[k]):
            add(
                "seq",
                f"that {a1} before {a2}",
                ("seq", k, a1, a2),
                (k,),
                (k,),
                2,
            )

        for pred, ref_cls in sorted(profile.spa[k]):
            add(
                "spa",
                f"that is {pred} the {ref_cls}",
                ("spa", k, pred, ref_cls),
                (k,),
                (k,),
                0,
            )

        for pred, ref_cls in sorted(profile.inter[k]):
            add(
                "int",
                f"that {pred} the {ref_cls}",
                ("int", k, pred, ref_cls),
                (k,),
                (k,),
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
        _, k, a1, a2 = sig
        return 0 <= k < profile.arity and (a1, a2) in profile.sequences[k]
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
) -> Tuple[List[CandidateTarget], List[List[int]]]:
    others = [c for c in all_candidates if c.candidate_id != target.candidate_id and c.arity == target.arity]
    rows: List[List[int]] = []
    for cand in others:
        prof = profile_map[cand.candidate_id]
        row = []
        for clue in clues:
            p = 1 if _clue_satisfied(prof, clue) else 0
            row.append(1 - p)
        rows.append(row)
    return others, rows


def solve_query_cpsat(
    target: CandidateTarget,
    template: TemplateSpec,
    clues: List[AtomicClue],
    exclusion_matrix: List[List[int]],
    time_limit_sec: float = 2.0,
) -> Optional[Dict[str, Any]]:
    if cp_model is None:
        raise RuntimeError("ortools is required. Install it with: uv pip install ortools")

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{j}") for j in range(len(clues))]

    for row in exclusion_matrix:
        model.Add(sum(row[j] * x[j] for j in range(len(clues))) >= 1)

    for k in range(target.arity):
        model.Add(
            sum(x[j] for j, clue in enumerate(clues) if k in clue.member_indices) >= 1
        )
        model.Add(
            sum(x[j] for j, clue in enumerate(clues) if k in clue.time_anchor_indices) >= 1
        )

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

    if template.require_seq:
        seq_idx = [j for j, clue in enumerate(clues) if clue.clue_type == "seq"]
        if not seq_idx:
            return None
        model.Add(sum(x[j] for j in seq_idx) >= int(template.require_seq))

    model.Minimize(sum(clue.cost * x[j] for j, clue in enumerate(clues)))

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
    inf = 6
    return [
        TemplateSpec(
            name="easy_app_spa",
            bucket="easy",
            type_min={"app": 1, "spa": 1},
            type_max={"cls": 1, "app": 2, "act": 1, "seq": 0, "spa": 2, "env": 1, "int": 0},
            k_min=2,
            k_max=4,
            q_min=0,
            require_seq=0,
            weight=1.0,
        ),
        TemplateSpec(
            name="easy_act",
            bucket="easy",
            type_min={"act": 1},
            type_max={"cls": 1, "app": 1, "act": 2, "seq": 0, "spa": 1, "env": inf, "int": 0},
            k_min=1,
            k_max=3,
            q_min=1,
            require_seq=0,
            weight=1.0,
        ),
        TemplateSpec(
            name="medium_act_spa",
            bucket="medium",
            type_min={"act": 1, "spa": 1},
            type_max={"cls": 1, "app": 2, "act": 2, "seq": 1, "spa": 2, "env": inf, "int": 1},
            k_min=2,
            k_max=4,
            q_min=1,
            require_seq=0,
            weight=1.0,
        ),
        TemplateSpec(
            name="medium_app_int",
            bucket="medium",
            type_min={"app": 1, "int": 1},
            type_max={"cls": 1, "app": 2, "act": 2, "seq": 1, "spa": 1, "env": inf, "int": 2},
            k_min=2,
            k_max=4,
            q_min=1,
            require_seq=0,
            weight=1.0,
        ),
        TemplateSpec(
            name="hard_seq_int",
            bucket="hard",
            type_min={"seq": 1, "int": 1},
            type_max={"cls": 1, "app": 2, "act": 2, "seq": 2, "spa": 2, "env": inf, "int": 2},
            k_min=3,
            k_max=6,
            q_min=2,
            require_seq=1,
            weight=1.0,
        ),
        TemplateSpec(
            name="hard_act_seq_spa",
            bucket="hard",
            type_min={"act": 1, "seq": 1, "spa": 1},
            type_max={"cls": 1, "app": 2, "act": 2, "seq": 2, "spa": 2, "env": inf, "int": 1},
            k_min=3,
            k_max=6,
            q_min=2,
            require_seq=1,
            weight=1.0,
        ),
    ]


def _quantile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    q = min(max(q, 0.0), 1.0)
    idx = q * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _compute_candidate_difficulty(
    candidate: CandidateTarget,
    index: GraphIndex,
    weights: DifficultyWeights,
) -> Tuple[float, float, float, Dict[str, Any]]:
    n_change = 0.0
    n_time = 0.0
    n_same = 0.0
    sep_vals: List[float] = []

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

        same_ids = []
        for other_id, other_obj in index.objects.items():
            if other_id == obj_id:
                continue
            if _sample_class(other_obj, other_id) != cls:
                continue
            other_range = (min(index.frames[other_id]), max(index.frames[other_id]))
            ov = _overlap(cur, other_range)
            if not ov:
                continue
            same_ids.append((other_id, ov))
        n_same += len(same_ids)

        c_map = index.centers.get(obj_id, {})
        for other_id, ov in same_ids:
            o_map = index.centers.get(other_id, {})
            dists: List[float] = []
            for f in range(ov[0], ov[1] + 1):
                if f not in c_map or f not in o_map:
                    continue
                dx = c_map[f][0] - o_map[f][0]
                dy = c_map[f][1] - o_map[f][1]
                d = math.hypot(dx, dy) / index.frame_diag
                dists.append(min(max(d, 0.0), 1.0))
            if dists:
                sep_vals.append(sum(dists) / len(dists))

    avg_sep = (sum(sep_vals) / len(sep_vals)) if sep_vals else 1.0
    sep_term = 1.0 - avg_sep

    D_t = weights.alpha_change * n_change + weights.alpha_time * n_time
    D_s = weights.beta_same * n_same + weights.beta_sep * sep_term
    D = D_t + D_s

    return D_t, D_s, D, {
        "n_change": n_change,
        "n_time": n_time,
        "n_same": n_same,
        "avg_sep": avg_sep,
        "sep_term": sep_term,
    }


def _assign_buckets(candidates: List[CandidateTarget]) -> None:
    scores = sorted(c.difficulty for c in candidates)
    t1 = _quantile(scores, 0.33)
    t2 = _quantile(scores, 0.66)
    for c in candidates:
        if c.difficulty <= t1:
            c.difficulty_bucket = "easy"
        elif c.difficulty <= t2:
            c.difficulty_bucket = "medium"
        else:
            c.difficulty_bucket = "hard"


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


class CPSATQuerySampler:
    def __init__(
        self,
        min_interval_len: int = 3,
        max_intervals_per_object: int = 12,
        queries_per_graph: int = 12,
        max_queries_per_candidate: int = 1,
        time_limit_sec: float = 2.0,
        seed: int = 7,
        weights: DifficultyWeights = DifficultyWeights(),
    ) -> None:
        self.min_interval_len = max(1, int(min_interval_len))
        self.max_intervals_per_object = max(1, int(max_intervals_per_object))
        self.queries_per_graph = max(1, int(queries_per_graph))
        self.max_queries_per_candidate = max(1, int(max_queries_per_candidate))
        self.time_limit_sec = max(0.1, float(time_limit_sec))
        self.rng = random.Random(seed)
        self.weights = weights
        self.templates = _build_default_templates()

    def _template_quotas(self, total: int) -> Dict[str, int]:
        weights = [max(0.0, t.weight) for t in self.templates]
        total_w = sum(weights) or 1.0
        raw = [w / total_w * total for w in weights]
        quotas = [int(math.floor(x)) for x in raw]
        remain = total - sum(quotas)
        order = sorted(range(len(raw)), key=lambda i: raw[i] - quotas[i], reverse=True)
        for i in order[:remain]:
            quotas[i] += 1
        return {self.templates[i].name: quotas[i] for i in range(len(self.templates))}

    def _candidate_order(
        self,
        candidates: List[CandidateTarget],
        template_bucket: str,
        sampled_intervals_by_obj: Dict[str, Set[Tuple[int, int]]],
        candidate_use_count: Dict[str, int],
    ) -> List[CandidateTarget]:
        pool = [c for c in candidates if c.difficulty_bucket == template_bucket]

        def rank(c: CandidateTarget) -> Tuple[int, int, float, int]:
            oid = c.primary_object_id
            interval = c.interval
            sampled_for_obj = sampled_intervals_by_obj.get(oid, set())
            if sampled_for_obj and interval not in sampled_for_obj:
                mode = 0
            elif not sampled_for_obj:
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
        clues = build_atomic_clues(self._index, candidate)
        if not clues:
            return None

        _, E = build_exclusion_matrix(candidate, all_candidates, clues, profile_map)
        solved = solve_query_cpsat(
            target=candidate,
            template=template,
            clues=clues,
            exclusion_matrix=E,
            time_limit_sec=self.time_limit_sec,
        )
        if not solved:
            return None

        selected = [clues[i] for i in solved["selected_indices"]]
        profile = profile_map[candidate.candidate_id]
        query = _render_query(profile, selected)

        return {
            "candidate": candidate,
            "profile": profile,
            "selected_clues": selected,
            "solver": solved,
            "template": template,
            "query": query,
        }

    def generate_for_graph(self, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._index = build_graph_index(graph)
        candidates = build_candidate_intervals(
            graph=graph,
            index=self._index,
            min_interval_len=self.min_interval_len,
            max_intervals_per_object=self.max_intervals_per_object,
            include_full_track=True,
        )
        if not candidates:
            return []

        for c in candidates:
            D_t, D_s, D, meta = _compute_candidate_difficulty(c, self._index, self.weights)
            c.D_t = D_t
            c.D_s = D_s
            c.difficulty = D
            c.meta.update(meta)

        _assign_buckets(candidates)
        profile_map = {c.candidate_id: _profile_for_candidate(self._index, c) for c in candidates}

        quotas = self._template_quotas(self.queries_per_graph)
        cur_counts: Dict[str, int] = defaultdict(int)
        candidate_use_count: Dict[str, int] = defaultdict(int)
        sampled_intervals_by_obj: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)
        solve_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
        blocked_templates: Set[str] = set()

        template_by_name = {t.name: t for t in self.templates}
        results: List[Dict[str, Any]] = []

        while len(results) < self.queries_per_graph:
            deficits = []
            for t in self.templates:
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
            for cand in self._candidate_order(candidates, template.bucket, sampled_intervals_by_obj, candidate_use_count):
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
            template = picked["template"]
            candidate_use_count[cand.candidate_id] += 1
            sampled_intervals_by_obj[cand.primary_object_id].add(cand.interval)
            cur_counts[template.name] += 1

            results.append(
                {
                    "query_id": f"cpsat_{len(results)}_{cand.candidate_id}",
                    "query": picked["query"],
                    "template": template.name,
                    "difficulty_bucket": cand.difficulty_bucket,
                    "D_t": cand.D_t,
                    "D_s": cand.D_s,
                    "D": cand.difficulty,
                    "target": {
                        "candidate_id": cand.candidate_id,
                        "members": [
                            {
                                "object_id": m.object_id,
                                "start_frame": m.start,
                                "end_frame": m.end,
                            }
                            for m in cand.members
                        ],
                    },
                    "clues": [
                        {
                            "type": c.clue_type,
                            "text": c.text,
                            "chain_len": c.chain_len,
                            "time_anchor_indices": list(c.time_anchor_indices),
                        }
                        for c in picked["selected_clues"]
                    ],
                    "solver": picked["solver"],
                }
            )

        return results

    def process_jsonl(self, input_path: str, output_path: str) -> None:
        in_path = Path(input_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wrote = 0
        with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                graph = json.loads(line)
                query_nodes = self.generate_for_graph(graph)
                out_graph = dict(graph)
                out_graph["query_nodes_cpsat"] = query_nodes
                fout.write(json.dumps(out_graph, ensure_ascii=False) + "\n")
                wrote += 1

        print(f"[query_cpsat] done. graphs={wrote}, output={out_path}", flush=True)


def main(
    input_path: str,
    output_path: str,
    queries_per_graph: int = 12,
    min_interval_len: int = 3,
    max_intervals_per_object: int = 12,
    max_queries_per_candidate: int = 1,
    time_limit_sec: float = 2.0,
    seed: int = 7,
    alpha_change: float = 1.0,
    alpha_time: float = 1.0,
    beta_same: float = 1.0,
    beta_sep: float = 1.0,
) -> None:
    sampler = CPSATQuerySampler(
        min_interval_len=min_interval_len,
        max_intervals_per_object=max_intervals_per_object,
        queries_per_graph=queries_per_graph,
        max_queries_per_candidate=max_queries_per_candidate,
        time_limit_sec=time_limit_sec,
        seed=seed,
        weights=DifficultyWeights(
            alpha_change=alpha_change,
            alpha_time=alpha_time,
            beta_same=beta_same,
            beta_sep=beta_sep,
        ),
    )
    sampler.process_jsonl(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    fire.Fire(main)
