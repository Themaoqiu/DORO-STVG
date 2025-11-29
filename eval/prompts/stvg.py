import random
from typing import Any, Dict, Optional

class STVGPromptTemplate:
    
    TVG_TEMPLATES = [
        "During the span of",
        "In the time range",
        "During the period",
        "Within the time frame of",
        "In the time period",
        "During",
        "Between",
        "At"
    ]
    
    SYSTEM_PROMPT = """You are an expert in video spatiotemporal grounding.
Given a video and a text query, you need to:
1. Identify when the event occurs (temporal localization)
2. Provide bounding boxes for each frame in that period (spatial localization)

Output format:
During the span of {start_frame, end_frame} start_frame: [x1, y1, x2, y2], start_frame+1: [x1, y1, x2, y2], ...

Notes:
- Frame indices are in range [0, 99] representing 100 sampled frames
- Bounding boxes are normalized to [0, 1] as [x1, y1, x2, y2]
"""
    
    @staticmethod
    def format_grounding_query(query: str, use_random_template: bool = False) -> str:
        if use_random_template:
            tvg_prefix = random.choice(STVGPromptTemplate.TVG_TEMPLATES)
        else:
            tvg_prefix = "During the span of"
        
        prompt = f"""When does "{query}" occur in the video?
Please describe the location of the corresponding subject/object in this video.

Please firstly give the frame indices, and then give the spatial bounding box corresponding to each frame in the time period.

Answer format: {tvg_prefix} {{start, end}} start: [x1, y1, x2, y2], start+1: [x1, y1, x2, y2], ..., end: [x1, y1, x2, y2].

Note: 
- Frame indices are in range [0, 99] (100 frames sampled uniformly)
- Bounding boxes are normalized coordinates in [0, 1]
"""
        return prompt
    
    @staticmethod
    def parse_stvg_output(output_text: str, video_metadata: Optional[Dict[str, Any]] = None) -> dict:
        import re
        
        result = {
            'temporal_span': None,
            'spatial_bboxes': {}
        }
        
        temporal_pattern = r'\{(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\}'
        temporal_match = re.search(temporal_pattern, output_text)
        
        if temporal_match:
            start = int(temporal_match.group(1))
            end = int(temporal_match.group(2))
            result['temporal_span'] = (start, end)
        
        spatial_pattern = r'(\d+(?:\.\d+)?)\s*:\s*\[([^\]]+)\]'
        spatial_matches = re.findall(spatial_pattern, output_text)
        
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