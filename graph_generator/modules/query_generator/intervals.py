from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from .text_utils import _normalize_phrase, _to_int


def _sample_class(obj: Dict[str, Any], node_id: str) -> str:
    from .text_utils import _norm

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
    from .text_utils import _display_phrase, _norm

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
