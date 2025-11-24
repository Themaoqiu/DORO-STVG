import os
import io
import requests
import json
import base64
from tqdm import tqdm
import re
import random
import time
import argparse
import torch
import hashlib
from qwen_vl_utils import process_vision_info, fetch_video
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from functools import wraps
from typing import Optional, List, Dict, Any, Callable
from torchvision import transforms
import yaml
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor, LlavaNextVideoProcessor

os.environ["DECORD_EOF_RETRY_MAX"] = "20480"

SYSTEM_PROMPT = (
    "You are an expert video analyst tasked with solving problems based on video content. "
    "When answering a question about a video, you should carefully observe and analyze important visual clues from the videos to answer. "
    "For each important segment you notice, first observe the key visual elements, then analyze their significance using the following format: "
    "specify the time range with `<time>start_time-end_time</time>`, "
    "describe the key visual clues with `<caption>Description of key visual clues</caption>`, "
    "and provide your analysis about what this means with `<think>Your analysis and thoughts about this segment</think>`. "
    "Throughout your analysis, think about the question as if you were a human pondering deeply, "
    "engaging in an internal dialogue using natural thought expressions such as such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. "
    "After examining the key visual clues, continue with deeper reasoning that connects your observations to the answer. "
    "Self-reflection or verification in your reasoning process is encouraged when necessary, "
    "though if the answer is straightforward, you may proceed directly to the conclusion. "
    "Finally, conclude by placing your final answer in `<answer> </answer>` tags."
)

QUESTION_TEMPLATE = (
    "{Question}\n\n"
    "Please analyze the video carefully by identifying key segments and their important visual clues within "
    "`<time> </time>`, `<caption> </caption>`, `<think> </think>` tags "
    "then conduct deep analysis and reasoning to arrive at your answer to the question, "
    "finally provide only the single option letter (e.g., A, B, C, D, E, F etc.) within the `<answer> </answer>` tags."
    "Follow the format specified in the instructions."
    # "RULES:\n"
    # "- Use ONLY the exact tags shown above: <time>, <caption>, <think>, <answer>\n"
    # "- Time format MUST be MM:SS-MM:SS (e.g., 01:23-01:45)"
)


def retry_on_failure(max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 2.0):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    # Exponential backoff with jitter
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator

class VideoQA:
    """
    Video Question Answering class with built-in API endpoints and retry functionality
    """
    
    # Built-in API endpoint configurations
    ENDPOINTS = {
        "openai": {
            "url": "https://runway.devops.xiaohongshu.com/openai/chat/completions",
            "api_version": "2024-12-01-preview"
        },
        "claude": {
            "url": "https://runway.devops.rednote.life/openai/bedrock_runtime/model/invoke"
        },
        "gemini": {
            "url": "https://runway.devops.rednote.life/openai/google/v1:generateContent"
        }
    }
    
    def __init__(self, api_key: str, model_type: str = "openai", max_retries: int = 5):
        """
        Initialize VideoQA instance
        
        Args:
            api_key: API key for authentication
            model_type: Type of model to use ('openai' or 'claude')
            max_retries: Maximum number of retry attempts for API calls
        """
        self.api_key = api_key
        self.model_type = model_type.lower()
        self.max_retries = max_retries
        
        if self.model_type not in self.ENDPOINTS:
            raise ValueError(f"model_type must be one of {list(self.ENDPOINTS.keys())}")
        
        self.endpoint_config = self.ENDPOINTS[self.model_type]
        self._setup_headers()
    
    def _setup_headers(self):
        """Setup HTTP headers based on model type"""
        if self.model_type == "openai":
            self.headers = {
                'api-key': self.api_key,
                'Content-Type': 'application/json'
            }
        elif self.model_type == "claude":
            self.headers = {
                'token': self.api_key,
                'Content-Type': 'application/json'
            }
        elif self.model_type == "gemini":
            self.headers = {
                'api-key': self.api_key,
                'Content-Type': 'application/json'
            }
    
    def extract_frames(self, 
                      video_path: str,
                      nframes: int = 16,
                      min_pixels: int = 128*28*28,
                      max_pixels: int = 128*28*28,
                      video_start: Optional[float] = None,
                      video_end: Optional[float] = None) -> List[str]:
        """
        Extract frames from video and convert to base64 format
        
        Args:
            video_path: Path to the video file
            nframes: Number of frames to extract
            min_pixels: Minimum pixel count for frames
            max_pixels: Maximum pixel count for frames
            video_start: Start time in seconds (optional)
            video_end: End time in seconds (optional)
            
        Returns:
            List of base64 encoded frame strings
        """
        # Configure video extraction parameters
        video_config = {
            "video": video_path,
            "nframes": nframes,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels
        }
        
        if video_start is not None:
            video_config["video_start"] = video_start
        if video_end is not None:
            video_config["video_end"] = video_end
        
        # Extract video frames
        video_frames = fetch_video(video_config)
        base64_frames = []
        
        # Process frames based on their type
        if isinstance(video_frames, list):
            # Handle PIL Image list
            for pil_image in video_frames:
                buffer = io.BytesIO()
                pil_image.save(buffer, format="JPEG", quality=90)
                base64_frame = base64.b64encode(buffer.getvalue()).decode("utf-8")
                base64_frames.append(base64_frame)
        else:
            # Handle tensor format
            for i in range(video_frames.shape[0]):
                frame_tensor = video_frames[i]
                if frame_tensor.dtype != torch.uint8:
                    frame_tensor = (frame_tensor * 255).byte()
                
                pil_image = transforms.ToPILImage()(frame_tensor)
                buffer = io.BytesIO()
                pil_image.save(buffer, format="JPEG", quality=90)
                base64_frame = base64.b64encode(buffer.getvalue()).decode("utf-8")
                base64_frames.append(base64_frame)
        
        print(f"Extracted {len(base64_frames)} frames from {video_path}")
        return base64_frames
    
    def _prepare_request_data(self, frames: List[str], question: str) -> tuple:
        """
        Prepare request data for different API types
        
        Args:
            frames: List of base64 encoded frames
            question: Question to ask about the video
            
        Returns:
            Tuple of (url, data) for the API request
        """
        if self.model_type == "openai":
            # Prepare OpenAI format
            content = [{"type": "text", "text": question}]
            for frame in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
                })
            
            data = {
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.1,
                "top_p": 0.001,
                "max_tokens": 1024,
            }
            url = f"{self.endpoint_config['url']}?api-version={self.endpoint_config['api_version']}"
            
        elif self.model_type == "claude":
            # Prepare Claude format
            content = [{"type": "text", "text": question}]
            for frame in frames:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame
                    }
                })
            
            data = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.1,
                "top_p": 0.001,
            }
            url = self.endpoint_config['url']
        
        elif self.model_type == "gemini":
            parts = []
    
            for frame in frames:
                parts.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",  
                        "data": frame  
                    }
                })
            
            parts.append({"text": question})
            
            data = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1024,
                    "responseModalities": ["TEXT"],
                    "topP": 0.001
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"}
                ]
            }
            url = self.endpoint_config['url']
        
        return url, data
    
    def _parse_response(self, response_json: Dict[str, Any]) -> str:
        """
        Parse API response based on model type
        
        Args:
            response_json: JSON response from API
            
        Returns:
            Extracted answer text
        """
        if self.model_type == "openai":
            return response_json['choices'][0]['message']['content']
        elif self.model_type == "claude":
            return response_json['content'][0]['text']
        elif self.model_type == "gemini":
            return response_json['candidates'][0]['content']['parts'][0]['text']
    
    @retry_on_failure(max_retries=5)
    def _make_api_request(self, url: str, data: Dict[str, Any]) -> str:
        """
        Make API request with retry functionality
        
        Args:
            url: API endpoint URL
            data: Request payload
            
        Returns:
            Parsed response text
            
        Raises:
            Exception: If all retry attempts fail
        """
        response = requests.post(url, headers=self.headers, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return self._parse_response(result)
        else:
    
            raise Exception(f"API Request Failed: {response.status_code}, {response.text}")
    
    def qa(self, 
           video_path: str, 
           question: str,
           nframes: int = 16,
           min_pixels: int = 128*28*28,
           max_pixels: int = 128*28*28,
           video_start: Optional[float] = None,
           video_end: Optional[float] = None) -> str:
        """
        Perform question answering on a single video
        
        Args:
            video_path: Path to the video file
            question: Question to ask about the video
            nframes: Number of frames to extract
            min_pixels: Minimum pixel count for frames
            max_pixels: Maximum pixel count for frames
            video_start: Start time in seconds (optional)
            video_end: End time in seconds (optional)
            
        Returns:
            Answer to the question
        """
        frames = self.extract_frames(
            video_path=video_path,
            nframes=nframes,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            video_start=video_start,
            video_end=video_end
        )
        
        url, data = self._prepare_request_data(frames, question)
        return self._make_api_request(url, data)
    
    def _process_single_video(self, video_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single video task
        
        Args:
            video_task: Dictionary containing video_path, question, and optional kwargs
            
        Returns:
            Result dictionary with status and answer/error information
        """
        try:
            video_path = video_task['video_path']
            question = video_task['question']
            kwargs = video_task.get('kwargs', {})
            
            answer = self.qa(video_path, question, **kwargs)
            
            return {
                'video_path': video_path,
                'question': question,
                'answer': answer,
                'status': 'success',
                'error': None
            }
        except Exception as e:
            return {
                'video_path': video_task['video_path'],
                'question': video_task['question'],
                'answer': None,
                'status': 'failed',
                'error': str(e)
            }
    
    def batch_qa(self, 
                 video_tasks: List[Dict[str, Any]], 
                 max_workers: int = 4,
                 progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Perform batch video question answering with multithreading
        
        Args:
            video_tasks: List of video tasks, each containing 'video_path', 'question', and optional 'kwargs'
            max_workers: Maximum number of worker threads
            progress_callback: Optional callback function that receives (completed, total) parameters
            
        Returns:
            List of results, each containing video_path, question, answer, status, and error
        """
        results = []
        completed = 0
        total = len(video_tasks)
        
        print(f"Starting batch processing of {total} videos with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_single_video, task): task 
                for task in video_tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
                completed += 1
                
                # Report progress
                if progress_callback:
                    progress_callback(completed, total)
                else:
                    print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        
        print(f"Batch processing completed. Success: {sum(1 for r in results if r['status'] == 'success')}, Failed: {sum(1 for r in results if r['status'] == 'failed')}")
        return results

def prepare_api_client(model_name, api_key):
    if "openai" in model_name.lower():
        model_type = "openai"
    elif "claude" in model_name.lower():
        model_type = "claude"
    elif "gemini" in model_name.lower():
        model_type = "gemini"
    else:
        model_type = "openai"
    
    print(f"Initializing API client for model_type: {model_type}")
    # 注意：这里的 api_key 需要获取
    api_client = VideoQA(
        api_key=api_key,
        model_type=model_type,
        max_retries=5
    )
    return api_client

def prepare_Internvl_Family(model_name, model_path):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    if model_name == "Internvl-3-8B":
        MODEL_PATH = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/task/base_models/InternVL3-8B"
    elif model_name == "Internvl-2.5-8B":
        MODEL_PATH = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/task/base_models/InternVL2_5-8B"
    
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        max_model_len=64000, 
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 256}, 
    )
    
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        max_tokens=1024,
        stop_token_ids=[],
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return llm, tokenizer, sampling_params

def prepare_LLaVA_Family(model_name, model_path):

    if model_name == "LLaVA-NeXT-Video":
        MODEL_PATH = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/task/base_models/LLaVA-NeXT-Video-7B-hf"
    elif model_name == "Video-UTR":
        pass
        MODEL_PATH = ""
    elif model_name == "Video-LLaVA":
        pass
        MODEL_PATH = ""
    
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        trust_remote_code=True,
        # max_model_len=None,
        gpu_memory_utilization=0.8,
        max_num_seqs=2,
        limit_mm_per_prompt={"video": 1}
    )
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        max_tokens=1024,
        stop_token_ids=[],
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    return llm, processor, sampling_params

def prepare_Qwen_Family(model_name, model_path):

    if model_name == "Qwen2.5-VL-7B-Instruct":
        MODEL_PATH = "/home/wangxingjian/model/Qwen3-VL-2B-Instruct"
    elif model_name == "Qwen2.5-VL-32B-Instruct":
        MODEL_PATH = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/task/base_models/Qwen2.5-VL-32B-Instruct"
    elif model_name == "VideoChat-R1-7B":
        MODEL_PATH = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/task/base_models/VideoChat-R1_7B"
    elif model_name == "VideoChat-R1-Thinking-7B":
        MODEL_PATH = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/task/base_models/VideoChat-R1-thinking_7B"
    elif model_name == "GRPO-CARE":
        MODEL_PATH = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/task/base_models/GRPO-CARE/Qwen2.5-VL-7B-GRPO-CARE-Margin0.01-SEED-Bench-R1/checkpoint-1500"
    elif model_name == "Time-R1-7B":
        MODEL_PATH = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/task/base_models/Time-R1-7B"
    elif model_name == "Video-R1-7B":
        MODEL_PATH = "/mnt/tidal-alsh01/dataset/video_agent_data/ckpt/open/Video-R1-7B"
    elif model_name == "Temporal-R1-7B":
        MODEL_PATH = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/task/base_models/temporal-r1-7b-base"
    elif model_name == "Qwen2.5-Omni-7B":
        MODEL_PATH = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/task/base_models/Qwen2.5-Omni-7B"
    elif model_name == "Open-R1-Video-7B":
        MODEL_PATH = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/task/base_models/Open-R1-Video-7B"
    elif model_name == "TW-GRPO":
        MODEL_PATH = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/task/base_models/TW-GRPO"
    elif model_name == "LongVILA-R1-7B":
        MODEL_PATH = "/home/wangxingjian/model/LongVILA-R1-7B"
    elif model_name == "MiMo-VL-7B-RL-2508":
        MODEL_PATH = "/home/wangxingjian/model/MiMo-VL-7B-RL-2508"
    elif "Video-Thinker" in model_name:
        MODEL_PATH = model_path
    else:
        MODEL_PATH = model_path
    

    if model_name == "Qwen2.5-Omni-7B" or model_name == "Open-R1-Video-7B":
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,
            max_model_len=32768,
            gpu_memory_utilization=0.8,
            limit_mm_per_prompt={"image": 1, "video": 1},
            trust_remote_code=True,
            dtype="auto",
        )
    else:
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,
            max_model_len=64000,
            gpu_memory_utilization=0.8,
            limit_mm_per_prompt={"image": 1, "video": 1},
            trust_remote_code=True,
            dtype="auto",
        )
    
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        max_tokens=1024,
        stop_token_ids=[],
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    return llm, processor, sampling_params

def create_question_hash(item):
    if isinstance(item, str):
        stripped = item.strip()
        if not stripped:
            return None
        item = json.loads(stripped)
    """创建问题的唯一标识符"""
    content = item['problem'] + str(item['options']) + item['path']
    return hashlib.md5(content.encode()).hexdigest()

def get_video_path(video_path_config, data_source, default_source='default'):
    """
    根据video_path配置和data_source获取对应的视频路径
    
    Args:
        video_path_config: 可以是字符串路径或字典
        data_source: 数据源名称
        default_source: 默认数据源名称
    
    Returns:
        对应的视频文件夹路径
    """
    if isinstance(video_path_config, dict):
        # 如果是字典，根据data_source选择路径
        if data_source in video_path_config:
            return video_path_config[data_source]
        elif default_source in video_path_config:
            print(f"Warning: Data source '{data_source}' not found, using default '{default_source}'")
            return video_path_config[default_source]
        else:
            # 如果都没找到，使用第一个可用的路径
            first_key = list(video_path_config.keys())[0]
            print(f"Warning: Data source '{data_source}' and default '{default_source}' not found, using '{first_key}'")
            return video_path_config[first_key]
    else:
        # 如果是字符串，直接返回
        return video_path_config

def predict_LLaVA_Family_batch(llm, processor, sampling_params, examples, video_path_config, model_name, nframes=16):
    llm_inputs = []
    
    for i, example in enumerate(examples):
        data_source = example.get('data_source', 'default')
        video_base_path = get_video_path(video_path_config, data_source)
        video_full_path = os.path.join(video_base_path, example['path'])
        try:
            video_elements = {"video": video_full_path, "nframes": nframes}
            video_tensor = fetch_video(video_elements)
            video_data = video_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        except Exception as e:
            print(f"无法读取视频： '{video_full_path}'。错误信息：{e}。跳過此樣本。")
            continue

        if example["problem_type"] == 'multiple choice':
            question_with_options = example['problem'] + "\nOptions:\n"
            for op in example["options"]:
                question_with_options += op + "\n"
        else:
            question_with_options = example['problem']
            
        final_user_prompt = QUESTION_TEMPLATE.format(Question=question_with_options) 
        
        full_instruction = f"{SYSTEM_PROMPT}\n\n{final_user_prompt}"
        
        prompt = f"USER: <video>\n{full_instruction} ASSISTANT:"
        
        llm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"video": video_data}
        })

    if not llm_inputs:
        print("警告：批次中沒有任何可處理的影片，將返回空結果。")
        return []
    # 批量生成
    outputs = llm.generate(llm_inputs, sampling_params)
    batch_output_text = [out.outputs[0].text for out in outputs]
    return batch_output_text

def predict_Qwen_Family_batch(llm, processor, sampling_params, examples, video_path_config, nframes=16):
    # 为每个example构建messages
    batch_messages = []
    for example in examples:
        # 获取数据源对应的视频路径
        data_source = example.get('data_source', 'default')
        video_base_path = get_video_path(video_path_config, data_source)
        
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "\nOptions:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": example['data_type'],
                        "video": os.path.join(video_base_path, example['path']),
                        "nframes": nframes,
                        "max_pixels": 128 * 28 * 28,
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=question)
                    }
                ]
            }
        ]
        batch_messages.append(messages)

    # 批量生成prompts
    prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]

    # 批量处理视频输入
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        batch_messages, 
        return_video_kwargs=True,
    )

    # 构建批量LLM输入
    llm_inputs = []
    for idx, prompt in enumerate(prompts):
        sample_mm_data = {"video": video_inputs[idx]}
        sample_video_kw = {}
        for key, value in video_kwargs.items():
            sample_video_kw[key] = value[idx]
                    
        llm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": sample_mm_data,
            "mm_processor_kwargs": sample_video_kw,
        })

    # 批量生成
    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
    batch_output_text = [out.outputs[0].text for out in outputs]
    
    return batch_output_text

def predict_Internvl_Family_batch(llm, tokenizer, sampling_params, examples, video_path_config, nframes=16):
    """为 InternVL 系列模型设计的批量预测函数"""
    llm_inputs = []

    # 为每个样本准备输入
    for example in examples:
        data_source = example.get('data_source', 'default')
        video_base_path = get_video_path(video_path_config, data_source)
        video_path = os.path.join(video_base_path, example['path'])

        # 1. 加载视频帧
        video_element = {"video": video_path, "nframes": nframes}
        video_data = fetch_video(video_element)
        
        frames = []
        if isinstance(video_data, torch.Tensor):
            for i in range(video_data.shape[0]):
                frame_tensor = video_data[i].permute(1, 2, 0)
                frame_numpy = (frame_tensor.cpu().numpy() * 255).astype('uint8')
                frames.append(Image.fromarray(frame_numpy))
        else:
            frames = video_data

        # 2. 构造问题文本
        if example["problem_type"] == 'multiple choice':
            question_with_options = example['problem'] + "\nOptions:\n"
            for op in example["options"]:
                question_with_options += op + "\n"
        else:
            question_with_options = example['problem']

        # 2. 使用全局模板
        final_user_prompt = QUESTION_TEMPLATE.format(Question=question_with_options)
        
        # 3. 组合成 InternVL 的最终 prompt 字符串
        image_placeholders = '<image>\n' * len(frames)
        # 将 system prompt 和 user prompt 合并作为总指令
        full_instruction = f"{SYSTEM_PROMPT}\n\n{final_user_prompt}"
        final_prompt_content = image_placeholders + full_instruction

        messages = [{"role": "user", "content": final_prompt_content}]
        
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 4. 准备 vLLM 输入
        llm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": frames}
        })

    # 批量生成
    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
    batch_output_text = [out.outputs[0].text for out in outputs]
    
    return batch_output_text

def predict_api_batch(api_client, examples, video_path_config, nframes=16, max_workers=4):
    """使用 VideoQA API 进行批量预测"""
    video_tasks = []
    for item in examples:
        # 构造每个视频的处理任务
        data_source = item.get('data_source', 'default')
        video_base_path = get_video_path(video_path_config, data_source)
        full_video_path = os.path.join(video_base_path, item['path'])

        # 组合问题和选项
        if item["problem_type"] == 'multiple choice':
            question_with_options = item['problem'] + "\nOptions:\n" + "\n".join(item["options"])
        else:
            question_with_options = item['problem']

        # 2. 使用全局模板构建完整的 prompt
        final_user_prompt = QUESTION_TEMPLATE.format(Question=question_with_options)
        full_prompt_for_api = f"{SYSTEM_PROMPT}\n\n{final_user_prompt}"

        video_tasks.append({
            'video_path': full_video_path,
            'question': full_prompt_for_api,
            'kwargs': {'nframes': nframes}
        })

    # 调用 API 的批量处理方法
    # 注意：API 的 batch_qa 方法返回的结果顺序可能与输入不一致
    api_results = api_client.batch_qa(video_tasks, max_workers=max_workers)

    # 由于 batch_qa 使用 as_completed，结果是无序的，需要重新排序
    # 创建一个查找表，用 (视频路径, 问题)作为键
    results_map = {
        (res['video_path'], res['question']): res 
        for res in api_results
    }
    
    # 按照原始顺序重新构建结果列表
    ordered_responses = []
    for task in video_tasks:
        result = results_map.get((task['video_path'], task['question']))
        if result and result['status'] == 'success':
            ordered_responses.append(result['answer'])
        else:
            # 如果失败，则返回一个错误标识
            error_msg = result['error'] if result else "Unknown Error"
            print(f"API call failed for {task['video_path']}: {error_msg}")
            ordered_responses.append("Error: API request failed")

    return ordered_responses

def extract_answer_from_solution(solution_text):
    pattern = r'<answer>\s*([A-Z])\s*</answer>'
    match = re.search(pattern, solution_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def extract_predicted_answer(predicted_text):
    text_to_search = ""

    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    matches = re.findall(answer_pattern, predicted_text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        text_to_search = matches[-1].strip()
    else:
        final_answer_pattern = r'Final answer:'
        match = re.search(final_answer_pattern, predicted_text, re.IGNORECASE)
        
        if match:
            text_to_search = predicted_text[match.end():].strip()
        else:
            text_to_search = predicted_text

    match = re.match(r'\s*([A-Z])\b', text_to_search, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    for option in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']:
        if f'{option}:' in text_to_search or f'[{option}' in text_to_search or f'{option} ' in text_to_search:
            return option
            
    return 'WRONG'

def extract_thinking_process(predicted_response):
    """提取<answer>标签前的所有内容作为思考过程"""
    answer_pattern = r'<answer>\s*.*?\s*</answer>'
    answer_match = re.search(answer_pattern, predicted_response, re.IGNORECASE | re.DOTALL)
    
    if answer_match:
        thinking = predicted_response[:answer_match.start()]
    else:
        thinking = predicted_response
    
    return thinking.strip()

def load_data(file_path):
    try:
        # 先尝试JSON格式
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # 如果失败，尝试JSONL格式
        print("JSON parsing failed, trying JSONL format...")
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
        return data

def parse_video_path_config(video_path_arg):
    """
    解析video_path参数，支持字符串路径或JSON字符串格式的字典
    
    Args:
        video_path_arg: 命令行传入的video_path参数
    
    Returns:
        解析后的路径配置（字符串或字典）
    """
    if not video_path_arg:
        return 'videos/'
    
    # 尝试解析为JSON字典
    if video_path_arg.startswith('{') and video_path_arg.endswith('}'):
        try:
            return json.loads(video_path_arg)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse video_path as JSON, treating as string path: {video_path_arg}")
            return video_path_arg
    else:
        return video_path_arg

def get_api_key_from_file(model_name: str, key_file: str) -> str:
    """
    从 YAML 文件中读取配置，并根据模型名称匹配 API 密钥。

    Args:
        model_name: 模型名称 (e.g., "openai-gpt4o", "claude-3-opus").
        key_file: 密钥 YAML 文件的路径.

    Returns:
        匹配到的 API 密钥字符串.
        
    Raises:
        FileNotFoundError: 如果 key_file 不存在.
        ValueError: 如果 YAML 文件中没有 api_keys 部分或找不到匹配的密钥.
    """
    try:
        with open(key_file, 'r') as f:
            keys_config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"API 密钥文件未找到: {key_file}")
    
    if 'api_keys' not in keys_config:
        raise ValueError(f"密钥文件 {key_file} 中缺少 'api_keys' 顶级键。")

    api_keys = keys_config['api_keys']
    
    # 智能匹配逻辑：检查模型名称是否包含 YAML 中的某个键
    normalized_model_name = model_name.lower()
    for key_name, api_key in api_keys.items():
        if key_name.lower() in normalized_model_name:
            print(f"找到匹配的密钥，用于 '{key_name}'。")
            return api_key
            
    raise ValueError(f"在 {key_file} 中找不到与模型 '{model_name}' 匹配的 API 密钥。")

def evaluate(video_path_config, json_file_path, output_path, model_name, model_path, batch_size=4, nframes=16, api_key=None):
    api_models = ["openai", "claude"] # 定义哪些模型名称属于API调用
    is_api_model = any(api_model in model_name.lower() for api_model in api_models)
    if is_api_model:
        if not api_key:
            raise ValueError("API key must be provided for API models.")
        api_client = prepare_api_client(model_name, api_key)
    elif "Internvl" in model_name:  
        llm, tokenizer, sampling_params = prepare_Internvl_Family(model_name, model_path)
    elif "LLaVA" in model_name or "UTR" in model_name:
        llm, processor, sampling_params = prepare_LLaVA_Family(model_name, model_path)
    else:
        llm, processor, sampling_params = prepare_Qwen_Family(model_name, model_path)

    os.makedirs(output_path, exist_ok=True)
    
    data = load_data(json_file_path)

    output_process = []
    json_file_output = os.path.join(output_path, f"Results-{model_name}.json")

    # 如果已有结果文件，加载已处理的问题hash
    processed_hashes = set()
    if os.path.exists(json_file_output):
        with open(json_file_output, "r", encoding="utf-8") as f:
            output_process = json.load(f)
            for item in output_process:
                question_hash = item.get("Question Hash")
                if question_hash:
                    processed_hashes.add(question_hash)

    correct_count = 0
    total_count = 0

    # 过滤出未处理的数据
    unprocessed_data = []
    for idx, item in enumerate(data):
        question_hash = create_question_hash(item)
        if question_hash not in processed_hashes:
            # 检查视频文件是否存在
            data_source = item.get('data_source', 'default')
            video_base_path = get_video_path(video_path_config, data_source)
            video_full_path = os.path.join(video_base_path, item.get('path'))
            
            if os.path.exists(video_full_path):
                unprocessed_data.append((idx, item))
            else:
                print(f"Warning: Video file not found: {video_full_path}")

    # 批量处理
    for i in tqdm(range(0, len(unprocessed_data), batch_size), desc="Processing batches"):
        batch_data = unprocessed_data[i:i + batch_size]
        batch_indices = [idx for idx, _ in batch_data]
        batch_items = [item for _, item in batch_data]
        
        # 批量预测
        if is_api_model:
            # 调用 API 的批量预测函数
            batch_responses = predict_api_batch(api_client, batch_items, video_path_config, nframes, max_workers=batch_size)
        elif "Internvl" in model_name:
            batch_responses = predict_Internvl_Family_batch(llm, tokenizer, sampling_params, batch_items, video_path_config, nframes)
        elif "LLaVA" in model_name or "UTR" in model_name:
            batch_responses = predict_LLaVA_Family_batch(llm, processor, sampling_params, batch_items, video_path_config, model_name, nframes)
        else:
            batch_responses = predict_Qwen_Family_batch(llm, processor, sampling_params, batch_items, video_path_config, nframes)
        
        # print(f"Error processing batch: {e}")
        # batch_responses = ["Error"] * len(batch_items)
        
        # 处理批量结果
        for idx, item, predicted_response in zip(batch_indices, batch_items, batch_responses):
            question_hash = create_question_hash(item)
            
            # 提取正确答案
            solution = item.get('solution')
            correct_answer = extract_answer_from_solution(solution)
            if not correct_answer:
                print(f"Warning: Could not extract answer from solution for question {idx}")
                continue
            
            # 提取预测答案
            try:
                predicted_answer = extract_predicted_answer(predicted_response)
            except Exception as e:
                predicted_answer = "WRONG"

            data_source = item.get('data_source', 'default')
            video_base_path = get_video_path(video_path_config, data_source)
            video_filename = item.get('path')
            
            print(f"Question {idx}: Predicted: {predicted_answer}, Correct: {correct_answer}")
            
            # 提取思考过程
            thinking = extract_thinking_process(predicted_response)

            is_correct = predicted_answer == correct_answer
            if is_correct:
                correct_count += 1
            total_count += 1

            output_process.append({
                "Question Index": idx,
                "Question Hash": question_hash,
                "Problem": item.get('problem'),
                "Problem Type": item.get('problem_type'),
                "Options": item.get('options'),
                "GT": correct_answer,
                "Predicted Answer": predicted_answer,
                "Full Response": predicted_response,
                "Thinking": thinking,
                "Correct": is_correct,
                "Video Path": video_filename,
                "Data Source": item.get('data_source', 'Unknown'),
                "Video Base Path": video_base_path
            })
        
        # 每处理一个批次就保存一次结果
        with open(json_file_output, "w", encoding="utf-8") as f:
            json.dump(output_process, f, indent=2, ensure_ascii=False)

    overall_accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"\nResults:")
    print(f"Correct: {correct_count}/{total_count}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    
    stats = {
        "model_name": model_name,
        "video_path_config": video_path_config,
        "total_questions": total_count,
        "correct_answers": correct_count,
        "accuracy": overall_accuracy
    }
    
    stats_file = os.path.join(output_path, f"Stats-{model_name}.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Evaluation")
    parser.add_argument('--model_name', default="Qwen2.5-VL-7B", type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--video_path', default='videos/', type=str, 
                        help='Video path config: can be a single path string or JSON string for multiple data sources, e.g., \'{"source1": "/path1", "source2": "/path2"}\'')
    parser.add_argument('--benchmark', default='train_data.json', type=str)
    parser.add_argument('--output_path', default='Results/', type=str)
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for processing')
    parser.add_argument('--nframes', default=16, type=int, help='Number of frames to extract from video')
    parser.add_argument('--key_file', default='keys.yaml', type=str, help='Path to the YAML file containing API keys.')
    args = parser.parse_args()
    
    video_path_config = parse_video_path_config(args.video_path)

    api_key_to_use = None
    api_models = ["openai", "claude", "gemini"] # 扩展需要API key的模型标识
    is_api_model = any(api_model in args.model_name.lower() for api_model in api_models)
    
    if is_api_model:
        try:
            api_key_to_use = get_api_key_from_file(args.model_name, args.key_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"错误: {e}")
            exit(1) 

    evaluate(video_path_config, args.benchmark, args.output_path, args.model_name, args.model_path, args.batch_size, args.nframes, api_key=api_key_to_use)