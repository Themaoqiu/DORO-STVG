from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from api_sync.api import StreamGenerator
from api_sync.utils.parser import JSONParser

from .text_utils import (
    _load_env_var_from_project_env,
    _norm,
    _normalize_api_keys,
    _to_int,
)


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


def _resolve_api_keys(api_keys: Optional[Union[str, Iterable[str]]]) -> List[str]:
    if api_keys is None:
        api_keys = os.getenv("API_KEYS", "")
    keys = _normalize_api_keys(api_keys)
    if keys:
        return keys
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
                        return keys
    return []


def _resolve_api_base_url() -> str:
    return (
        os.getenv("MM_API_BASE_URL", "").strip()
        or os.getenv("VISION_API_BASE_URL", "").strip()
        or os.getenv("VIDEO_API_BASE_URL", "").strip()
        or _load_env_var_from_project_env("MM_API_BASE_URL")
        or _load_env_var_from_project_env("VISION_API_BASE_URL")
        or _load_env_var_from_project_env("VIDEO_API_BASE_URL")
    )


def _build_polish_prompt(node: Dict[str, Any], idx: int) -> Dict[str, Any]:
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
    return {"id": query_id, "prompt": prompt_text}


def _extract_polished_query(response: str) -> Optional[Dict[str, Any]]:
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
        vv = _norm(value).rstrip(".").strip()
        if not kk or not vv:
            continue
        target_queries[kk] = vv
    if not target_queries:
        return None
    return {"query": query, "target_queries": target_queries}


async def polish_queries_with_llm(
    query_nodes: List[Dict[str, Any]],
    *,
    model_name: str,
    api_keys: Optional[Union[str, Iterable[str]]],
    max_concurrent_per_key: int = 100,
    max_retries: int = 5,
    system_prompt: str = QUERY_POLISH_SYSTEM_PROMPT,
) -> Dict[str, Dict[str, Any]]:
    if not query_nodes:
        return {}

    keys = _resolve_api_keys(api_keys)
    if not keys:
        raise ValueError("API_KEYS is required when use_llm_polish=True (set env/.env or pass --api_keys).")

    api_base_url = _resolve_api_base_url()
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

    prompts: List[Dict[str, Any]] = [_build_polish_prompt(node, idx) for idx, node in enumerate(query_nodes)]

    print(
        f"[query_cpsat] llm_polish_start: model={model_name} prompts={len(prompts)} keys={len(keys)} "
        f"base_url={'set' if api_base_url else 'default'}",
        flush=True,
    )
    polished_map: Dict[str, Dict[str, Any]] = {}
    completed = 0

    async for item in generator.generate_stream(
        prompts=prompts,
        system_prompt=system_prompt,
        validate_func=lambda resp: polished if (polished := _extract_polished_query(resp)) is not None else False,
    ):
        result = item.get("result") if item else None
        if item and isinstance(result, dict):
            polished_map[str(item["id"])] = result
            completed += 1
            print(f"[query_cpsat] llm_polish_progress: {completed}/{len(prompts)}", flush=True)
    print(f"[query_cpsat] llm_polish_done: polished={len(polished_map)}", flush=True)
    return polished_map
