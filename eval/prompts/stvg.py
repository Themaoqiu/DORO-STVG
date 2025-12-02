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

    SYSTEM_PROMPT = """You are an expert in spatiotemporal video grounding tasked with precisely locating objects/subjects in videos. When localizing the query in the video, you should watch the entire clip carefully before producing an answer. For every subject/object you are asked to find, observe the video carefully first. The red numbers overlaid on each frame indicate the frame index. Then provide the most plausible time interval in which the target appears, along with its bounding-box coordinates in every relevant frame.
    """
    
    @staticmethod
    def format_grounding_query(query: str, use_random_template: bool = False) -> str:
        if use_random_template:
            tvg_prefix = random.choice(STVGPromptTemplate.TVG_TEMPLATES)
        else:
            tvg_prefix = "During the span of"

        prompt = f"""At which time interval in the video can we see {query}? Please describe the location of the corresponding subject/object in this video. Firstly give the timestamps, and then give the spatial bounding box corresponding to each timestamp in the time period.\n
        Answer format: {tvg_prefix} {{start frame, end frame}} frame1: [x1, y1, x2, y2], frame2: [x1, y1, x2, y2], ..., frameN: [x1, y1, x2, y2].
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