from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union


RELATION_SYNONYMS: Dict[str, str] = {}
ATTRIBUTE_SYNONYMS: Dict[str, str] = {}


def set_synonyms(rel_map: Dict[str, str], attr_map: Dict[str, str]) -> None:
    global RELATION_SYNONYMS, ATTRIBUTE_SYNONYMS
    RELATION_SYNONYMS = rel_map
    ATTRIBUTE_SYNONYMS = attr_map


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
