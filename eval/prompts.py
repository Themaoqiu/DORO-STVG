import json
import re
from typing import Any, Dict, List, Optional, Tuple


SYSTEM_PROMPT = """You are an expert in spatiotemporal video grounding tasked with precisely locating objects/subjects in videos. You should output the space-time tube for each object the user intends to find."""


USER_PROMPT = """Where does {query} occur in the video? Please find the location of the corresponding subject/object in this video. Give the corresponding bounding boxes for the object(s) in each corresponding frame.\n\n

Guidelines:\n
- Videos are sampled at 2 fps.\n
- Use normalized box coordinates in [0, 1].\n
- Do not output explanations.\n
- You must follow the exact output format below, and only output the format without any additional text or explanation.\n
- Output a strict JSON object whose keys are target descriptions.\n
- Each target description maps to a JSON object whose keys are frame indices as strings and whose values are normalized boxes [x1, y1, x2, y2].\n
- Put frame indices first. Do not output timestamps.\n
- Example for one target:\n
  {{"chair": {{"12": [0.10, 0.20, 0.30, 0.40], "13": [0.11, 0.20, 0.31, 0.40]}}}}\n
- Example for multiple targets:\n
  {{"chair": {{"12": [0.10, 0.20, 0.30, 0.40]}}, "table": {{"12": [0.40, 0.50, 0.70, 0.90]}}}}\n
"""

LLAVA_ST_USER_PROMPT = """Where does {query} occur in the video? Please find the location of the corresponding subject/object in this video. Give the corresponding bounding boxes for the object(s) in each corresponding frame.\n\n

Guidelines:\n
- Videos are sampled at 2 fps.\n
- Use normalized box coordinates in [0, 1].\n
- Do not output explanations.\n
- You must follow the exact output format below, and only output the format without any additional text or explanation.\n
- Please firstly give the timestamps, and then give the spatial bounding box corresponding to each timestamp in the time period.\n
"""


def format_prompt(query: str, prompt_style: str = "json") -> str:
    if prompt_style == "llava_st":
        return LLAVA_ST_USER_PROMPT.format(query=query)
    return USER_PROMPT.format(query=query)


def _extract_json_candidate(response_text: str) -> Optional[str]:
    text = response_text.strip()
    if not text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1].strip()


def _parse_objects_from_json(response_text: str) -> Optional[Dict[str, Any]]:
    json_candidate = _extract_json_candidate(response_text)
    if not json_candidate:
        return None

    try:
        payload = json.loads(json_candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    result = {
        'temporal_span': None,
        'spatial_bboxes': {},
        'objects': [],
    }

    for description, frame_map in payload.items():
        if not isinstance(frame_map, dict):
            continue

        spatial_bboxes = {}
        for frame_idx_text, coords in frame_map.items():
            try:
                frame_idx = int(str(frame_idx_text).strip())
            except ValueError:
                continue

            if not isinstance(coords, list) or len(coords) != 4:
                continue

            try:
                norm_coords = [max(0.0, min(1.0, float(c))) for c in coords]
            except (TypeError, ValueError):
                continue
            spatial_bboxes[frame_idx] = norm_coords

        temporal_span = None
        if spatial_bboxes:
            frames = sorted(spatial_bboxes.keys())
            temporal_span = (frames[0], frames[-1])

        result['objects'].append(
            {
                'description': str(description).strip(),
                'temporal_span': temporal_span,
                'spatial_bboxes': spatial_bboxes,
            }
        )

    if len(result['objects']) == 1:
        result['temporal_span'] = result['objects'][0]['temporal_span']
        result['spatial_bboxes'] = result['objects'][0]['spatial_bboxes']

    return result


def _empty_parse_result() -> Dict[str, Any]:
    return {
        'temporal_span': None,
        'spatial_bboxes': {},
        'objects': [],
    }


def _sample_position_to_frame(position: int, sampled_indices: Optional[List[int]]) -> Optional[int]:
    if sampled_indices is None:
        return position
    if not sampled_indices:
        return None
    if position < 0 or position >= len(sampled_indices):
        return None
    return int(sampled_indices[position])


def _parse_llava_st_box_tokens(response_text: str, sampled_indices: Optional[List[int]]) -> Dict[int, List[float]]:
    frame_map: Dict[int, List[float]] = {}
    matches = re.findall(
        r"<TEMP-(\d{3})>\s*:\s*\[<WIDTH-(\d{3})><HEIGHT-(\d{3})><WIDTH-(\d{3})><HEIGHT-(\d{3})>\]",
        response_text,
    )
    for temp_idx, x1, y1, x2, y2 in matches:
        frame_idx = _sample_position_to_frame(int(temp_idx), sampled_indices)
        if frame_idx is None:
            continue
        frame_map[frame_idx] = [
            int(x1) / 99.0,
            int(y1) / 99.0,
            int(x2) / 99.0,
            int(y2) / 99.0,
        ]
    return frame_map


def _parse_llava_st_temporal_span(response_text: str, sampled_indices: Optional[List[int]]) -> Optional[Tuple[int, int]]:
    match = re.search(r"\{\s*<TEMP-(\d{3})>\s*<TEMP-(\d{3})>\s*\}", response_text)
    if not match:
        return None
    start = _sample_position_to_frame(int(match.group(1)), sampled_indices)
    end = _sample_position_to_frame(int(match.group(2)), sampled_indices)
    if start is None or end is None:
        return None
    return (start, end)


def _parse_llava_st_response(
    response_text: str,
    query: Optional[str],
    sampled_indices: Optional[List[int]],
) -> Dict[str, Any]:
    result = _empty_parse_result()
    spatial_bboxes = _parse_llava_st_box_tokens(response_text, sampled_indices)
    if not spatial_bboxes:
        return result

    temporal_span = _parse_llava_st_temporal_span(response_text, sampled_indices)
    if temporal_span is None:
        frames = sorted(spatial_bboxes.keys())
        temporal_span = (frames[0], frames[-1])

    obj = {
        'description': str(query or "target").strip() or "target",
        'temporal_span': temporal_span,
        'spatial_bboxes': spatial_bboxes,
    }
    result['temporal_span'] = temporal_span
    result['spatial_bboxes'] = spatial_bboxes
    result['objects'] = [obj]
    return result


def parse_response(
    response_text: str,
    query: Optional[str] = None,
    sampled_indices: Optional[List[int]] = None,
    prompt_style: str = "json",
) -> Dict[str, Any]:
    if prompt_style == "llava_st":
        llava_result = _parse_llava_st_response(response_text, query, sampled_indices)
        if llava_result.get("objects"):
            return llava_result

    result = {
        'temporal_span': None,
        'spatial_bboxes': {},
        'objects': [],
    }

    json_result = _parse_objects_from_json(response_text)
    if json_result is not None and json_result.get("objects"):
        return json_result

    object_matches = re.findall(
        r'["\']?([^"\':{}]+?)["\']?\s*:\s*\{([^{}]*)\}',
        response_text,
        flags=re.DOTALL,
    )

    if object_matches:
        for description, frame_map_text in object_matches:
            spatial_bboxes = {}
            frame_matches = re.findall(
                r'["\']?(\d+)["\']?\s*:\s*\[\s*([^\]]+?)\s*\]',
                frame_map_text,
            )
            for frame_idx_text, coords_text in frame_matches:
                try:
                    frame_idx = int(frame_idx_text)
                    coords = [float(x.strip()) for x in coords_text.split(",")]
                    if len(coords) != 4:
                        continue
                    spatial_bboxes[frame_idx] = [max(0.0, min(1.0, c)) for c in coords]
                except (ValueError, IndexError):
                    continue
            temporal_span = None
            if spatial_bboxes:
                frames = sorted(spatial_bboxes.keys())
                temporal_span = (frames[0], frames[-1])
            result['objects'].append(
                {
                    'description': description.strip(),
                    'temporal_span': temporal_span,
                    'spatial_bboxes': spatial_bboxes,
                }
            )

        if len(result['objects']) == 1:
            result['temporal_span'] = result['objects'][0]['temporal_span']
            result['spatial_bboxes'] = result['objects'][0]['spatial_bboxes']
        return result

    return result
