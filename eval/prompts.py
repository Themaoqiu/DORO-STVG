import re
from typing import Dict, Any, Optional


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

def format_prompt(query: str) -> str:
    return USER_PROMPT.format(query=query)


def parse_response(response_text: str) -> Dict[str, Any]:
    result = {
        'temporal_span': None,
        'spatial_bboxes': {},
        'objects': [],
    }

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
