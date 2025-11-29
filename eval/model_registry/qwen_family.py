from typing import List
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

from .base_model import BaseSTVGModel
from ..core.schema import STVGSample, Result
from ..prompts.stvg import STVGPromptTemplate


class Qwen3(BaseSTVGModel):
    
    def __init__(
        self, 
        model_name: str,
        model_path: str,
        batch_size: int = 4,
        nframes: int = 100,
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
    ) -> List[Result]:
        from utils.stvg_video_utils import process_video
        from qwen_vl_utils import process_vision_info
        
        output_folder = Path(kwargs.get('output_folder', './annotated_videos'))
        
        prompts = []
        mm_data = []
        
        for sample in samples:
            video_inputs, video_metadata = process_video(
                video_path=sample.video_path,
                output_folder=str(output_folder),
                num_frames=self.nframes,
                annotate_frames=True,
                max_pixels=360 * 420
            )
            
            sample.video_metadata = video_metadata
            
            query_text = STVGPromptTemplate.format_grounding_query(sample.query)
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query_text}
                    ]
                }
            ]
            
            prompt = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
            mm_data.append(video_inputs)
        
        llm_inputs = [{
            "prompt": prompt, 
            "multi_modal_data": {"video": mm}
            }
            for prompt, mm in zip(prompts, mm_data)
        ]
        
        outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
        
        predictions = []
        for sample, output in zip(samples, outputs):
            pred_text = output.outputs[0].text
            pred_result = self.postprocess_output(pred_text, sample)
            predictions.append(pred_result)
        
        return predictions
    
    def postprocess_output(self, pred_text: str, sample: STVGSample) -> Result:
        parsed = STVGPromptTemplate.parse_stvg_output(pred_text)
        
        pred_temporal = parsed.get('temporal_span', (0, 99))
        pred_bboxes = parsed['spatial_bboxes']
        
        return Result(
            item_id=sample.item_id,
            pred_temporal_bound=pred_temporal,
            pred_bboxes=pred_bboxes,
            video_metadata=sample.video_metadata,
            metadata={
                'raw_output': pred_text,
                'parsed': parsed
            }
        )