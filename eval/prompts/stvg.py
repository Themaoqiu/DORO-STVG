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
2. Provide bounding boxes for each timestamp in that period (spatial localization)

Output format:
During the span of {start_time, end_time} start_time: [x1, y1, x2, y2], start_time+1: [x1, y1, x2, y2], ...

Notes:
- Times are normalized to [0, 99] representing 100 sampled frames
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

Please firstly give the timestamps, and then give the spatial bounding box corresponding to each timestamp in the time period.

Answer format: {tvg_prefix} {{start, end}} start: [x1, y1, x2, y2], start+1: [x1, y1, x2, y2], ..., end: [x1, y1, x2, y2].

Note: 
- Timestamps are in range [0, 99] (100 frames sampled uniformly)
- Bounding boxes are normalized coordinates in [0, 1]
"""
        return prompt
    
    @staticmethod
    def parse_stvg_output(output_text: str, video_metadata: Optional[Dict[str, Any]] = None) -> dict:
        import re
        
        fps = video_metadata['fps'] if video_metadata else 30.0
        result = {
            'temporal_span': None,
            'temporal_span_seconds': None,
            'spatial_bboxes': {}
        }
        
        temporal_pattern = r'\{(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\}'
        temporal_match = re.search(temporal_pattern, output_text)
        
        if temporal_match:
            start = float(temporal_match.group(1))
            end = float(temporal_match.group(2))
            result['temporal_span_seconds'] = (int(round(start)), int(round(end)))

            start_frame = int(round(start * fps))
            end_frame = int(round(end * fps))
            result['temporal_span'] = (start_frame, end_frame)
        
        spatial_pattern = r'(\d+(?:\.\d+)?)\s*:\s*\[([^\]]+)\]'
        spatial_matches = re.findall(spatial_pattern, output_text)
        
        for time_str, bbox_str in spatial_matches:
            try:
                frame_idx = int(round(time_str * fps))

                coords = [float(x.strip()) for x in bbox_str.split(',')]
                
                if len(coords) == 4:
                    coords = [max(0.0, min(1.0, c)) for c in coords]
                    result['spatial_bboxes'][frame_idx] = coords
                    
            except (ValueError, IndexError) as e:
                continue
        
        return result