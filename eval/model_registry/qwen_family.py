from typing import List
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import os

from .base_model import BaseModel
from ..core.schema import STVGSample, Result
from ..prompts.stvg import STVGPromptTemplate

os.environ["DECORD_EOF_RETRY_MAX"] = "20480"

class Qwen2_5VL(BaseModel):
    
    def __init__(
        self, 
        model_name: str,
        model_path: str,
        batch_size: int,
        nframes: int,
        max_tokens: int,
        max_model_len: int,
        temperature: float,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
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
        
        batch_messages = []
        
        for sample in samples:
            annotated_video_path, video_metadata = process_video(
                video_path=sample.video_path,
                output_folder=str(output_folder),
                num_frames=self.nframes,
                annotate_frames=True
            )
            
            sample.video_metadata = video_metadata
            
            query_text = STVGPromptTemplate.format_grounding_query(sample.query)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": annotated_video_path,
                            "nframes": self.nframes,
                            "max_pixels": 1280*28*28,
                        },
                        {
                            "type": "text",
                            "text": query_text
                        }
                    ]
                }
            ]
            batch_messages.append(messages)
        
        prompts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            batch_messages,
            return_video_kwargs=True
        )
        
        llm_inputs = []
        for idx, prompt in enumerate(prompts):
            sample_mm_data = {"video": video_inputs[idx]}
            sample_video_kw = {}
            for key, value in video_kwargs.items():
                if isinstance(value, (list, tuple)):
                    sample_video_kw[key] = value[idx]
                else:
                    sample_video_kw[key] = value
            
            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": sample_mm_data,
                "mm_processor_kwargs": sample_video_kw,
            })
        
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


class Qwen3VL(BaseModel):
    
    def __init__(
        self, 
        model_name: str,
        model_path: str,
        batch_size: int,
        nframes: int,
        max_tokens: int,
        max_model_len: int,
        temperature: float,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
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
        print(f"[Qwen3VL] Loading model from: {self.model_path}")
        
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            limit_mm_per_prompt={"image": 1, "video": 1},
            trust_remote_code=True,
            dtype="auto",
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        print(f"[Qwen3VL] Model loaded successfully")
    
    def predict_batch(
        self, 
        samples: List[STVGSample],
        **kwargs
    ) -> List[Result]:
        from utils.stvg_video_utils import process_video
        from qwen_vl_utils import process_vision_info
        
        output_folder = Path(kwargs.get('output_folder', './annotated_videos'))
        
        batch_messages = []
        
        for sample in samples:
            annotated_video_path, video_metadata = process_video(
                video_path=sample.video_path,
                output_folder=str(output_folder),
                num_frames=self.nframes,
                annotate_frames=True
            )
            
            sample.video_metadata = video_metadata
            
            query_text = STVGPromptTemplate.format_grounding_query(sample.query)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": annotated_video_path,
                            "nframes": self.nframes,
                            "max_pixels": 1280 * 28 * 28,
                        },
                        {
                            "type": "text",
                            "text": query_text
                        }
                    ]
                }
            ]
            batch_messages.append(messages)
        
        # 按照官方代码构建输入
        llm_inputs = []
        for messages in batch_messages:
            # 生成 prompt
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 使用 process_vision_info 处理视频 - 关键:添加 return_video_metadata=True
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=self.processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True  # 关键参数!
            )
            
            # 构建 multi_modal_data
            mm_data = {}
            if image_inputs is not None:
                mm_data['image'] = image_inputs
            if video_inputs is not None:
                mm_data['video'] = video_inputs
            
            llm_inputs.append({
                'prompt': text,
                'multi_modal_data': mm_data,
                'mm_processor_kwargs': video_kwargs
            })
        
        # 生成预测
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