import re
from typing import Dict, Any, Optional


SYSTEM_PROMPT = """You are an expert in spatiotemporal video grounding tasked with precisely locating objects/subjects in videos. When localizing the query in the video, you should watch the entire video carefully before producing an answer. For every subject/object you are asked to find, observe the video carefully first. Then provide the most plausible time interval in which the target appears, along with its bounding-box coordinates in every relevant frame."""


USER_PROMPT = """Localize the referred target or targets for the query "{query}" in the video.

Answer requirements:
- Use normalized box coordinates in [0, 1].
- Do not output explanations.
- Keep each trajectory in this exact coordinate format: <frame_idx, time_sec, x1, y1, x2, y2; ... />
- If there is one target, output: The object box is: <... />
- If there are multiple targets, output: target description 1: <...>; target description 2: <...>
"""


def format_prompt(query: str) -> str:
    return USER_PROMPT.format(query=query)


def parse_response(response_text: str) -> Dict[str, Any]:
    result = {
        'temporal_span': None,
        'spatial_bboxes': {},
        'objects': [],
    }

    object_matches = re.findall(r'([^:;<>]+?)\s*:\s*(<[^>]*?/>)', response_text, flags=re.IGNORECASE)
    if not object_matches:
        single_box_match = re.search(r'The object box is:\s*(<[^>]*?/>)', response_text, flags=re.IGNORECASE)
        if single_box_match:
            object_matches = [('object', single_box_match.group(1))]

    if object_matches:
        for description, box_text in object_matches:
            spatial_bboxes = {}
            inner = box_text.strip()
            if inner.startswith("<"):
                inner = inner[1:]
            if inner.endswith("/>"):
                inner = inner[:-2]
            inner = inner.strip()
            for chunk in inner.split(";"):
                parts = [part.strip() for part in chunk.split(",")]
                if len(parts) != 6:
                    continue
                try:
                    frame_idx = int(round(float(parts[0])))
                    coords = [float(x) for x in parts[2:]]
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

    temporal_pattern = r'[\{\[](\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)[\}\]]'
    temporal_match = re.search(temporal_pattern, response_text)

    if temporal_match:
        start = int(round(float(temporal_match.group(1))))
        end = int(round(float(temporal_match.group(2))))
        result['temporal_span'] = (start, end)

    spatial_pattern = r'(\d+(?:\.\d+)?)\s*:\s*\[([^\]]+)\]'
    spatial_matches = re.findall(spatial_pattern, response_text)

    for frame_str, bbox_str in spatial_matches:
        try:
            frame_idx = int(round(float(frame_str)))
            coords = [float(x.strip()) for x in bbox_str.split(',')]

            if len(coords) == 4:
                coords = [max(0.0, min(1.0, c)) for c in coords]
                result['spatial_bboxes'][frame_idx] = coords

        except (ValueError, IndexError):
            continue

    return result
