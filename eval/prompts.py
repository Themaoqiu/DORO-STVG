import re
from typing import Dict, Any, Optional


SYSTEM_PROMPT = """You are an expert in spatiotemporal video grounding tasked with precisely locating objects/subjects in videos. When localizing the query in the video, you should watch the entire video carefully before producing an answer. For every subject/object you are asked to find, observe the video carefully first. The red numbers overlaid on each frame indicate the frame index. Then provide the most plausible time interval in which the target appears, along with its bounding-box coordinates in every relevant frame."""


USER_PROMPT = """At which time interval in the video can we see {query}? Please describe the location of the corresponding subject/object in this video. Firstly give the timestamps, and then give the spatial bounding box corresponding to each timestamp in the time period.

Answer format: During the span of {{start_frame, end_frame}}, start_frame: [x1, y1, x2, y2], start_frame+1: [x1, y1, x2, y2], ..., end_frame: [x1, y1, x2, y2].
Note: 
- Frame indices are in range [0, 99] (100 frames sampled uniformly)
- Bounding boxes positions are normalized coordinates in [0, 1]
"""


def format_prompt(query: str) -> str:
    return USER_PROMPT.format(query=query)


def parse_response(response_text: str) -> Dict[str, Any]:
    result = {
        'temporal_span': None,
        'spatial_bboxes': {}
    }
    
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
