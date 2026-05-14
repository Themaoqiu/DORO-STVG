from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .data_models import AtomicClue, CandidateMember, CandidateProfile, CandidateTarget, GraphIndex
from .intervals import (
    _enumerate_ordered_chains,
    _format_seq_part,
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
