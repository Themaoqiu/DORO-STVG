import json
import re
from typing import Any, Dict, Optional, Tuple


TEMPORAL_FPS = 2.0

TEMPORAL_SYSTEM_PROMPT = (
    "You are an expert in temporal video grounding. Locate when the queried event "
    "or object appears in the video."
)

TEMPORAL_USER_PROMPT = """Where does {query} occur in the video?

Only identify the temporal range. Videos are sampled at 2 fps.

Output a strict JSON object and do not include any explanation:
{{"temporal_span": [start_frame, end_frame]}}
"""


def format_temporal_prompt(query: str) -> str:
    return TEMPORAL_USER_PROMPT.format(query=query)


def _extract_json_candidate(response_text: str) -> Optional[str]:
    text = str(response_text or "").strip()
    if not text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1].strip()


def _normalize_span(start: Any, end: Any) -> Optional[Tuple[int, int]]:
    try:
        span_start = int(round(float(start)))
        span_end = int(round(float(end)))
    except (TypeError, ValueError):
        return None
    if span_end < span_start:
        span_start, span_end = span_end, span_start
    return span_start, span_end


def _parse_json_span(response_text: str) -> Optional[Tuple[int, int]]:
    candidate = _extract_json_candidate(response_text)
    if not candidate:
        return None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    span = payload.get("temporal_span") or payload.get("span") or payload.get("time_span")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        return _normalize_span(span[0], span[1])

    for start_key, end_key in [
        ("start_frame", "end_frame"),
        ("start", "end"),
        ("begin", "finish"),
    ]:
        if start_key in payload and end_key in payload:
            return _normalize_span(payload[start_key], payload[end_key])

    return None


def _parse_text_span(response_text: str) -> Optional[Tuple[int, int]]:
    text = str(response_text or "")
    patterns = [
        (r"([0-9]+(?:\.[0-9]+)?)\s*s(?:ec(?:ond)?s?)?\s*(?:to|-|~)\s*([0-9]+(?:\.[0-9]+)?)\s*s", True),
        (r"time\s*range\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:to|-|~)\s*([0-9]+(?:\.[0-9]+)?)", True),
        (r"start(?:_frame)?\s*[:=]\s*([0-9]+)\D+end(?:_frame)?\s*[:=]\s*([0-9]+)", False),
        (r"\b([0-9]+)\s*(?:to|-|~)\s*([0-9]+)\b", False),
    ]

    for pattern, seconds in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        start = float(match.group(1))
        end = float(match.group(2))
        if seconds:
            start *= TEMPORAL_FPS
            end *= TEMPORAL_FPS
        return _normalize_span(start, end)
    return None


def parse_temporal_response(response_text: str) -> Dict[str, Optional[Tuple[int, int]]]:
    return {
        "temporal_span": _parse_json_span(response_text) or _parse_text_span(response_text),
    }
