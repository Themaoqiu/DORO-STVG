"""STVG任务提示词模板"""


class STVGPromptTemplate:
    """STVG任务的提示词模板管理"""
    
    SYSTEM_PROMPT = """You are an expert in video spatiotemporal grounding.
Given a video and a text query describing an object or relation, you need to:
1. Identify the temporal boundary (start frame, end frame) where the described content appears
2. Localize the object with bounding boxes for key frames

Provide your answer in the following format:
Temporal Boundary: (start_frame, end_frame)
Bounding Boxes: {frame_id: [x1, y1, x2, y2], ...}
"""
    
    @staticmethod
    def format_grounding_query(query: str) -> str:
        """
        格式化定位查询
        
        Args:
            query: 原始查询文本
            
        Returns:
            格式化后的查询
        """
        return f"""Please locate the following in the video:
Query: {query}

Provide the temporal boundary and bounding boxes for the described object/relation.
Your answer should include:
1. Temporal Boundary: (start_frame, end_frame)
2. Key Frames with Bounding Boxes: {{frame_id: [x1, y1, x2, y2], ...}}

Answer:"""
    
    @staticmethod
    def format_relation_query(subject: str, predicate: str, object_name: str) -> str:
        """
        格式化关系查询(用于VidSTG)
        
        Args:
            subject: 主体
            predicate: 谓词
            object_name: 客体
            
        Returns:
            格式化后的关系查询
        """
        return f"{subject} {predicate} {object_name}"