"""Qwen系列模型实现"""
from typing import List
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from .base_model import BaseSTVGModel
from ..core.schema import STVGSample, PredictionResult
from ..prompts.stvg import STVGPromptTemplate


class Qwen3(BaseSTVGModel):
    
    def __init__(
        self, 
        model_name: str,
        model_path: str,
        batch_size: int = 4,
        nframes: int = 16,
        max_tokens: int = 512,
        max_model_len: int = 8192,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.model_path = model_path
        self.batch_size = batch_size
        self.nframes = nframes
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        # 采样参数
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.001,
            max_tokens=self.max_tokens,
            stop_token_ids=[],
        )
        
        self.llm = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """加载Qwen模型"""
        print(f"[Qwen] Loading model from: {self.model_path}")
        
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1, "video": 1},
            trust_remote_code=True,
            dtype="auto",
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        print(f"[Qwen] Model loaded successfully")
    
    def predict_batch(
        self, 
        samples: List[STVGSample],
        **kwargs
    ) -> List[PredictionResult]:
        """批量预测"""
        # 1. 构建输入
        prompts = []
        for sample in samples:
            conversation = self._build_conversation(sample)
            prompt = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
        
        mm_data = []
        for sample in samples:
            messages = self._build_conversation(sample)
            video_inputs = process_vision_info(messages)
            mm_data.append(video_inputs[0])
        
        llm_inputs = [
            {"prompt": prompt, "multi_modal_data": {"video": mm}}
            for prompt, mm in zip(prompts, mm_data)
        ]
        
        outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
        
        predictions = []
        for sample, output in zip(samples, outputs):
            pred_text = output.outputs[0].text
            pred_result = self.postprocess_output(pred_text, sample)
            predictions.append(pred_result)
        
        return predictions
    
    def _build_conversation(self, sample: STVGSample) -> List[dict]:
        query_text = STVGPromptTemplate.format_grounding_query(sample.query)

        fps = sample.metadata.get('fps', 30.0)
        frame_count = sample.metadata.get('frame_count', 300)
        duration = round(frame_count / fps, 1)

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": sample.video_path,
                        "fps": 1.0,
                        "max_pixels": 360 * 420
                    },
                    {
                        "type": "text",
                        "text": query_text
                    }
                ]
            }
        ]
    
    def postprocess_output(self, pred_text: str, sample: STVGSample) -> PredictionResult:
        """
        解析模型输出
        
        期望格式: "During the span of {10, 50} 10: [0.1,0.2,0.3,0.4], 11: [0.1,0.2,0.3,0.4], ..."
        """
        from ..prompts.stvg import STVGPromptTemplate
        
        parsed = STVGPromptTemplate.parse_stvg_output(pred_text)
        
        pred_temporal = parsed.get('temporal_span', (0, 99))
        pred_bboxes = {}
        
        # 转换为标准格式: {frame_idx: [[x1,y1,x2,y2]]}
        for frame_idx, bbox in parsed['spatial_bboxes'].items():
            pred_bboxes[frame_idx] = [bbox] 
        
        return PredictionResult(
            item_id=sample.item_id,
            pred_temporal_bound=pred_temporal,
            pred_bboxes=pred_bboxes,
            metadata={
                'raw_output': pred_text,
                'parsed': parsed
            }
        )