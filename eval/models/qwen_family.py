import os
from typing import List, Dict, Any
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoProcessor

os.environ["DECORD_EOF_RETRY_MAX"] = "20480"


class QwenVLBase:
    """Base class for Qwen-VL models."""
    
    def __init__(
        self, 
        model_path: str,
        batch_size: int = 1,
        nframes: int = 100,
        max_tokens: int = 512,
        max_model_len: int = 8192,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.nframes = nframes
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

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
        """To be implemented by subclass."""
        raise NotImplementedError
    
    def prepare_messages(
        self, 
        query: str, 
        annotated_video_path: str,
        system_prompt: str
    ) -> List[Dict[str, Any]]:
        """Prepare messages for model input."""
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
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
                        "text": query
                    }
                ]
            }
        ]
        return messages
    
    def predict_batch(
        self, 
        queries: List[str],
        annotated_video_paths: List[str],
        system_prompt: str
    ) -> List[str]:
        """
        Batch inference.
        
        Returns:
            List of model response strings (full responses).
        """
        raise NotImplementedError


class Qwen2_5VL(QwenVLBase):
    """Qwen2.5-VL model."""

    def load_model(self):
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
    
    def predict_batch(
        self, 
        queries: List[str],
        annotated_video_paths: List[str],
        system_prompt: str
    ) -> List[str]:
        from qwen_vl_utils import process_vision_info
        
        batch_messages = []
        for query, video_path in zip(queries, annotated_video_paths):
            messages = self.prepare_messages(query, video_path, system_prompt)
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
        
        responses = [output.outputs[0].text for output in outputs]
        return responses


class Qwen3VL(QwenVLBase):
    """Qwen3-VL model."""
    
    def load_model(self):
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
    
    def predict_batch(
        self, 
        queries: List[str],
        annotated_video_paths: List[str],
        system_prompt: str
    ) -> List[str]:
        from qwen_vl_utils import process_vision_info
        
        llm_inputs = []
        for query, video_path in zip(queries, annotated_video_paths):
            messages = self.prepare_messages(query, video_path, system_prompt)
            
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=self.processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True
            )
            
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
        
        outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
        
        responses = [output.outputs[0].text for output in outputs]
        return responses
