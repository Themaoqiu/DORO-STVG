from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .data_models import AtomicClue, CandidateProfile
from .text_utils import _norm


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
    return "", []


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
                s = _norm(s[len(prefix):])
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
