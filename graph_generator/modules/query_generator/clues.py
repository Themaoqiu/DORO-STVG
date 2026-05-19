from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .data_models import (
    AtomicClue,
    CandidateMember,
    CandidateProfile,
    CandidateTarget,
    DifficultyWeights,
    GraphIndex,
    TemplateSpec,
)
from .indexing import (
    _enumerate_ordered_chains,
    _format_seq_part,
    _interval_tiou,
    _normalized_relation_pair,
    _overlap,
    _relation_interval,
    _sample_class,
)
from .text_utils import _display_phrase, _norm, _normalize_phrase, _normalized_values


def _object_desc_for_action_target(
    index: GraphIndex,
    object_id: str,
    target_desc_by_object_id: Optional[Dict[str, str]] = None,
) -> str:
    if target_desc_by_object_id:
        target_desc = _norm(target_desc_by_object_id.get(object_id, ""))
        if target_desc:
            return target_desc
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
    target_desc_by_object_id: Optional[Dict[str, str]] = None,
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
            out[action].add(_object_desc_for_action_target(index, tid, target_desc_by_object_id))
    return out


def _relation_target_desc_maps(
    index: GraphIndex,
    member: CandidateMember,
    target_desc_by_object_id: Optional[Dict[str, str]] = None,
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
        desc = _object_desc_for_action_target(index, oid, target_desc_by_object_id)
        if not desc:
            continue
        key = (pred, ref_cls)
        if str(rel.get("edge_type", "")).strip().lower() == "spatial":
            spatial[key].add(desc)
        else:
            interacting[key].add(desc)
    return spatial, interacting


def _candidate_member_reference_options(
    index: GraphIndex,
    profile: CandidateProfile,
    member_index: int,
    member: CandidateMember,
) -> List[str]:
    cls = _norm(profile.classes[member_index]) or _display_phrase(
        _sample_class(index.objects.get(member.object_id, {}), member.object_id)
    )
    options: List[str] = []

    def append_option(text: str) -> None:
        clean = _norm(text)
        if clean and clean not in options:
            options.append(clean)

    for attr in sorted(profile.attrs[member_index]):
        append_option(f"{cls} {attr}")
    for env in sorted(profile.env[member_index]):
        append_option(f"{cls} {env}")
    for action in sorted(profile.actions[member_index]):
        append_option(f"{cls} {action}")
    append_option(cls or "object")
    return options


def _candidate_target_desc_map(
    index: GraphIndex,
    candidate: CandidateTarget,
    profile: CandidateProfile,
) -> Dict[str, str]:
    options_by_object_id: Dict[str, List[str]] = {}
    counts: Dict[str, int] = defaultdict(int)
    for member_index, member in enumerate(candidate.members):
        options = _candidate_member_reference_options(index, profile, member_index, member)
        options_by_object_id[member.object_id] = options
        for option in options:
            counts[option] += 1

    desc_by_object_id: Dict[str, str] = {}
    for member in candidate.members:
        options = options_by_object_id.get(member.object_id, [])
        chosen = next((option for option in options if counts[option] == 1), "")
        if not chosen and options:
            chosen = options[0]
        desc_by_object_id[member.object_id] = chosen or "object"
    return desc_by_object_id


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


def _member_temporal_tags(index: GraphIndex, member: CandidateMember) -> Set[str]:
    tags: Set[str] = set()
    member_interval = (member.start, member.end)
    for temporal_node in index.temporal_spans:
        if _overlap(member_interval, (temporal_node["start"], temporal_node["end"])):
            tags.add(f"clip_{temporal_node['clip_id']}")
    return tags


def _member_action_events(index: GraphIndex, member: CandidateMember) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    member_interval = (member.start, member.end)
    for item in index.actions_by_obj.get(member.object_id, []):
        if not _overlap(member_interval, (item["start"], item["end"])):
            continue
        label = _normalize_phrase(item["label"], "act")
        if not label:
            continue
        events.append(
            {
                "label": label,
                "start": item["start"],
                "end": item["end"],
                "targets": list(item.get("targets", [])),
            }
        )
    events.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return events


def _member_relation_events(index: GraphIndex, member: CandidateMember) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    member_interval = (member.start, member.end)
    for rel in index.relations_by_subj.get(member.object_id, []):
        if not _overlap(member_interval, (rel["start"], rel["end"])):
            continue
        pred = _normalize_phrase(rel["predicate"], "rel")
        ref_cls = _normalize_phrase(rel["ref_class"], "attr") or _normalize_phrase(rel["ref_class"], "rel")
        if not pred or not ref_cls:
            continue
        events.append(
            {
                "predicate": pred,
                "ref_class": ref_cls,
                "edge_type": rel["edge_type"],
                "start": rel["start"],
                "end": rel["end"],
            }
        )
    events.sort(key=lambda x: (x["start"], x["end"], x["predicate"], x["ref_class"]))
    return events


def _relation_pair_sets(
    relation_events: List[Dict[str, Any]],
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    spatial_pairs: Set[Tuple[str, str]] = set()
    interaction_pairs: Set[Tuple[str, str]] = set()
    for rel in relation_events:
        pair = (rel["predicate"], rel["ref_class"])
        if rel["edge_type"] == "spatial":
            spatial_pairs.add(pair)
        else:
            interaction_pairs.add(pair)
    return spatial_pairs, interaction_pairs


def _interaction_pairs_from_action_targets(
    index: GraphIndex,
    action_events: List[Dict[str, Any]],
) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for action in action_events:
        for target_id in action["targets"]:
            if target_id not in index.objects:
                continue
            target_class = _normalize_phrase(_sample_class(index.objects[target_id], target_id), "attr")
            if target_class:
                pairs.add((action["label"], target_class))
    return pairs


def _sequence_set_for_member(
    action_events: List[Dict[str, Any]],
    relation_events: List[Dict[str, Any]],
    max_chain_len: int,
) -> Set[Tuple[str, ...]]:
    action_chain_events = [(a["start"], a["end"], f"act:{a['label']}") for a in action_events]
    relation_chain_events = [
        (rel["start"], rel["end"], f"rel:{rel['predicate']}:{rel['ref_class']}")
        for rel in relation_events
    ]
    mixed_chains = {
        chain
        for chain in _enumerate_ordered_chains(action_chain_events + relation_chain_events, max_chain_len=max_chain_len)
        if any(token.startswith("act:") for token in chain) and any(token.startswith("rel:") for token in chain)
    }
    return (
        _enumerate_ordered_chains(action_chain_events, max_chain_len=max_chain_len)
        | _enumerate_ordered_chains(relation_chain_events, max_chain_len=max_chain_len)
        | mixed_chains
    )


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
        temporal_tags.append(_member_temporal_tags(index, member))

        act_events = _member_action_events(index, member)
        act_labels = {a["label"] for a in act_events}
        actions.append(act_labels)

        rel_events = _member_relation_events(index, member)
        spa_set, int_set = _relation_pair_sets(rel_events)
        spa.append(spa_set)
        inter.append(int_set | _interaction_pairs_from_action_targets(index, act_events))

        sequences.append(_sequence_set_for_member(act_events, rel_events, max_chain_len))

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


AddClueFn = Callable[[str, str, Tuple[Any, ...], Tuple[int, ...], bool, int], None]


def _append_attribute_and_env_clues(
    profile: CandidateProfile,
    member_index: int,
    member_indices: Tuple[int, ...],
    add_clue: AddClueFn,
) -> None:
    for attr in sorted(profile.attrs[member_index]):
        add_clue("app", attr, ("app", member_index, attr), member_indices, False, 0)
    for phrase in sorted(profile.env[member_index]):
        add_clue("env", phrase, ("env", "general", member_index, phrase), member_indices, False, 0)
    for tag in sorted(profile.temporal_tags[member_index]):
        add_clue("env", tag, ("env", "temporal", member_index, tag), member_indices, True, 0)


def _append_action_clues(
    index: GraphIndex,
    profile: CandidateProfile,
    member: CandidateMember,
    member_index: int,
    member_indices: Tuple[int, ...],
    action_target_descs: Dict[str, Set[str]],
    add_clue: AddClueFn,
) -> None:
    for action in sorted(profile.actions[member_index]):
        target_descs = sorted(action_target_descs.get(action, set()))
        has_local_targets = _action_has_local_targets(index, member, action)
        if has_local_targets and not target_descs:
            continue
        action_text = action if not target_descs else f"{action} {target_descs[0]}"
        add_clue("act", action_text, ("act", member_index, action), member_indices, True, 1)


def _append_sequence_clues(
    profile: CandidateProfile,
    member_index: int,
    member_indices: Tuple[int, ...],
    add_clue: AddClueFn,
) -> None:
    for chain in sorted(profile.sequences[member_index]):
        if len(chain) < 2:
            continue
        chain_text = " then ".join(_format_seq_part(token) for token in chain)
        add_clue("seq", chain_text, ("seq", member_index, *chain), member_indices, True, len(chain))


def _append_spatial_clues(
    index: GraphIndex,
    profile: CandidateProfile,
    member: CandidateMember,
    member_index: int,
    member_indices: Tuple[int, ...],
    spatial_target_descs: Dict[Tuple[str, str], Set[str]],
    add_clue: AddClueFn,
) -> None:
    for pred, ref_cls in sorted(profile.spa[member_index]):
        target_descs = sorted(spatial_target_descs.get((pred, ref_cls), set()))
        if not target_descs:
            continue
        add_clue(
            "spa",
            f"{pred} {target_descs[0]}",
            ("spa", member_index, pred, ref_cls),
            member_indices,
            _has_local_dynamic_relation_pair(index, member, pred, ref_cls, require_spatial=True),
            0,
        )


def _append_interaction_clues(
    index: GraphIndex,
    profile: CandidateProfile,
    member: CandidateMember,
    member_index: int,
    member_indices: Tuple[int, ...],
    interaction_target_descs: Dict[Tuple[str, str], Set[str]],
    action_target_descs: Dict[str, Set[str]],
    add_clue: AddClueFn,
) -> None:
    for pred, ref_cls in sorted(profile.inter[member_index]):
        target_descs = sorted(interaction_target_descs.get((pred, ref_cls), set()))
        if not target_descs:
            target_descs = sorted(action_target_descs.get(pred, set()))
        if not target_descs:
            continue
        is_temporal = _has_local_dynamic_relation_pair(
            index,
            member,
            pred,
            ref_cls,
            require_spatial=False,
        ) or _has_local_dynamic_action_target_pair(index, member, pred, ref_cls)
        add_clue(
            "int",
            f"{pred} {target_descs[0]}",
            ("int", member_index, pred, ref_cls),
            member_indices,
            is_temporal,
            1,
        )


def build_atomic_clues(index: GraphIndex, candidate: CandidateTarget, max_chain_len: int = 4) -> List[AtomicClue]:
    profile = _profile_for_candidate(index, candidate, max_chain_len=max_chain_len)
    clues: List[AtomicClue] = []
    seen: Set[Tuple[Any, ...]] = set()
    target_desc_by_object_id = _candidate_target_desc_map(index, candidate, profile)

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
        action_target_descs = _action_target_desc_map(index, member, target_desc_by_object_id)
        spa_target_descs, int_target_descs = _relation_target_desc_maps(index, member, target_desc_by_object_id)
        add("cls", cls, ("cls", k, cls), member_idx, False, 0)
        _append_attribute_and_env_clues(profile, k, member_idx, add)
        _append_action_clues(index, profile, member, k, member_idx, action_target_descs, add)
        _append_sequence_clues(profile, k, member_idx, add)
        _append_spatial_clues(index, profile, member, k, member_idx, spa_target_descs, add)
        _append_interaction_clues(index, profile, member, k, member_idx, int_target_descs, action_target_descs, add)

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


# ---------------------------------------------------------------------------
# Difficulty scoring (uses build_atomic_clues + build_exclusion_matrix above)
# ---------------------------------------------------------------------------


_DIFFICULTY_BUCKET_ORDER = {
    "very_easy": 0,
    "easy": 1,
    "medium": 2,
    "hard": 3,
    "very_hard": 4,
}

_DIFFICULTY_BUCKET_THRESHOLDS: Tuple[Tuple[float, str], ...] = (
    (0.20, "very_easy"),
    (0.40, "easy"),
    (0.60, "medium"),
    (0.80, "hard"),
    (1.01, "very_hard"),
)

_TEMPLATE_BUCKET_DIFFICULTY_LABELS = {
    "easy": ("very_easy", "easy"),
    "medium": ("medium",),
    "hard": ("hard", "very_hard"),
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _member_object_signature(candidate: CandidateTarget) -> Tuple[str, ...]:
    return tuple(member.object_id for member in candidate.members)


def _profile_temporal_tokens(profile: CandidateProfile) -> Set[str]:
    tokens: Set[str] = set()
    for idx in range(profile.arity):
        for tag in profile.temporal_tags[idx]:
            tokens.add(f"m{idx}:tag:{tag}")
        for action in profile.actions[idx]:
            tokens.add(f"m{idx}:act:{action}")
        for chain in profile.sequences[idx]:
            tokens.add(f"m{idx}:seq:{'|'.join(chain)}")
    return tokens


def _profile_spatial_tokens(profile: CandidateProfile) -> Set[str]:
    tokens: Set[str] = set()
    for idx in range(profile.arity):
        tokens.add(f"m{idx}:cls:{profile.classes[idx]}")
        for attr in profile.attrs[idx]:
            tokens.add(f"m{idx}:attr:{attr}")
        for env in profile.env[idx]:
            tokens.add(f"m{idx}:env:{env}")
        for pred, ref_cls in profile.spa[idx]:
            tokens.add(f"m{idx}:spa:{pred}:{ref_cls}")
        for pred, ref_cls in profile.inter[idx]:
            tokens.add(f"m{idx}:int:{pred}:{ref_cls}")
    return tokens


def _jaccard_similarity(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right)) / float(len(union))


def _candidate_interval_similarity(left: CandidateTarget, right: CandidateTarget) -> float:
    left_intervals = [(member.start, member.end) for member in left.members]
    right_intervals = [(member.start, member.end) for member in right.members]
    if len(left_intervals) != len(right_intervals) or not left_intervals:
        return 0.0
    scores = [_interval_tiou(a, b) for a, b in zip(left_intervals, right_intervals)]
    return float(sum(scores) / len(scores))


def _candidate_spatial_confusability(
    candidate: CandidateTarget,
    all_candidates: List[CandidateTarget],
    profile_map: Dict[str, CandidateProfile],
) -> Tuple[float, Dict[str, Any]]:
    SPATIAL_EXPONENT = 2.0
    target_signature = _member_object_signature(candidate)
    target_tokens = _profile_spatial_tokens(profile_map[candidate.candidate_id])
    mass = 0.0
    best_pair = (0.0, "", 0.0, 0.0)
    competitor_count = 0

    for other in all_candidates:
        if other.candidate_id == candidate.candidate_id or other.arity != candidate.arity:
            continue
        if _member_object_signature(other) == target_signature:
            continue
        overlap = _interval_tiou(candidate.interval, other.interval)
        if overlap <= 0.0:
            continue
        competitor_count += 1
        similarity = _jaccard_similarity(target_tokens, _profile_spatial_tokens(profile_map[other.candidate_id]))
        contribution = similarity * overlap
        if contribution <= 0.0:
            continue
        mass += contribution ** SPATIAL_EXPONENT
        if contribution > best_pair[0]:
            best_pair = (contribution, other.candidate_id, similarity, overlap)

    d_s = mass / (1.0 + mass)
    return _clamp01(d_s), {
        "spatial_competitors": competitor_count,
        "spatial_confusion_mass": mass,
        "spatial_best_candidate_id": best_pair[1],
        "spatial_best_similarity": best_pair[2],
        "spatial_best_interval_overlap": best_pair[3],
        "spatial_token_count": len(target_tokens),
    }


def _best_exclusion_difficulty(
    rows: List[List[int]],
    clues: List[AtomicClue],
    *,
    allow_temporal: bool,
) -> Tuple[float, Dict[str, Any]]:
    if not rows:
        return 0.0, {
            "competitor_rows": 0,
            "eligible_clues": 0,
            "best_clue_id": "",
            "best_clue_type": "",
            "best_exclusion_rate": 1.0,
        }

    eligible = [
        clue
        for clue in clues
        if clue.is_temporal_evidence == allow_temporal
    ]
    if not eligible:
        return 1.0, {
            "competitor_rows": len(rows),
            "eligible_clues": 0,
            "best_clue_id": "",
            "best_clue_type": "",
            "best_exclusion_rate": 0.0,
        }

    best_clue = None
    best_rate = -1.0
    for clue in eligible:
        clue_idx = int(clue.clue_id[1:])
        rate = float(sum(row[clue_idx] for row in rows)) / float(len(rows))
        if rate > best_rate:
            best_rate = rate
            best_clue = clue

    return _clamp01(1.0 - best_rate), {
        "competitor_rows": len(rows),
        "eligible_clues": len(eligible),
        "best_clue_id": "" if best_clue is None else best_clue.clue_id,
        "best_clue_type": "" if best_clue is None else best_clue.clue_type,
        "best_exclusion_rate": _clamp01(best_rate),
    }


def _compute_candidate_difficulty(
    candidate: CandidateTarget,
    index: GraphIndex,
    all_candidates: List[CandidateTarget],
    profile_map: Dict[str, CandidateProfile],
    weights: DifficultyWeights,
) -> Tuple[float, float, float, Dict[str, Any]]:
    lambda_weight = min(max(float(weights.lambda_weight), 0.0), 1.0)
    clues = build_atomic_clues(index, candidate)
    _, _, time_rows = build_exclusion_matrix(candidate, all_candidates, clues, profile_map)
    D_t, temporal_meta = _best_exclusion_difficulty(time_rows, clues, allow_temporal=True)
    D_s, spatial_meta = _candidate_spatial_confusability(candidate, all_candidates, profile_map)
    D = _clamp01(lambda_weight * D_t + (1.0 - lambda_weight) * D_s)

    meta = {
        "lambda_weight": lambda_weight,
        "arity": candidate.arity,
        "difficulty_model": "hybrid_confusability_v2_spx2",
        "atomic_clue_count": len(clues),
    }
    meta.update(temporal_meta)
    meta.update(spatial_meta)
    return D_t, D_s, D, meta


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


def _difficulty_bucket_from_score(score: float) -> str:
    clamped = _clamp01(score)
    for upper_bound, label in _DIFFICULTY_BUCKET_THRESHOLDS:
        if clamped < upper_bound:
            return label
    return "very_hard"


def _assign_difficulty_buckets_5(candidates: List[CandidateTarget]) -> None:
    for cand in candidates:
        cand.difficulty_bucket = _difficulty_bucket_from_score(cand.difficulty)


def _template_candidate_labels(template_bucket: str) -> Tuple[str, ...]:
    return _TEMPLATE_BUCKET_DIFFICULTY_LABELS.get(template_bucket, ("medium",))


def _template_target_labels(template: TemplateSpec) -> Tuple[str, ...]:
    labels = _template_candidate_labels(template.bucket)
    only_app_min = template.type_min == {"app": 1}
    no_temporal_or_relational_min = (
        template.type_min.get("act", 0) == 0
        and template.type_min.get("seq", 0) == 0
        and template.type_min.get("spa", 0) == 0
        and template.type_min.get("int", 0) == 0
    )
    if template.arity_max == 1 and template.bucket == "easy" and only_app_min and no_temporal_or_relational_min:
        return ("very_easy",)
    return labels


def _template_difficulty_rank(candidate: CandidateTarget, template: TemplateSpec) -> Tuple[int, float]:
    preferred_labels = _template_target_labels(template)
    if candidate.difficulty_bucket in preferred_labels:
        bucket_distance = 0
    else:
        candidate_rank = _DIFFICULTY_BUCKET_ORDER.get(candidate.difficulty_bucket, _DIFFICULTY_BUCKET_ORDER["medium"])
        preferred_ranks = [_DIFFICULTY_BUCKET_ORDER[label] for label in preferred_labels]
        bucket_distance = min(abs(candidate_rank - rank) for rank in preferred_ranks)

    if template.bucket == "hard":
        difficulty_order = -candidate.difficulty
    else:
        difficulty_order = candidate.difficulty
    return bucket_distance, difficulty_order
