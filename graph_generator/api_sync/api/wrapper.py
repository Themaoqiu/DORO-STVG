from openai import AsyncOpenAI
from typing import Dict, Any, List
import logging
import asyncio
import os
import json
from .vision_utils import build_multimodal_message

logger = logging.getLogger(__name__)


class QAWrapper:
    """Asynchronous wrapper for LLM API client."""

    SUPPORTED_REASONING_MODELS = ["DeepSeek-R1", "gemini-3-flash"]

    def __init__(self, model_name: str, api_key: str, max_retries: int = 5):
        """
        Initialize an async API wrapper instance.

        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.model_name = model_name
        self.api_key = api_key
        self.max_retries = max_retries

        # Use multimodal Qwen/DashScope API endpoint
        api_base_url = (
            os.getenv("MM_API_BASE_URL")
            or os.getenv("VISION_API_BASE_URL")
            or os.getenv("VIDEO_API_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        logger.info(f"Initializing QAWrapper")
        logger.info(f"  API Base URL: {api_base_url}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  API Key: {api_key[:20]}...")
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base_url
        )

        self.stats = {
            "calls": 0,
            "errors": 0,
            "retries": 0
        }

    async def qa(self, system_prompt: str, user_prompt: str = "", rational: bool = False) -> Any:
        """
        Send a prompt to the model and get a response.

        Args:
            system_prompt: System message
            user_prompt: User query content or list of content dicts (multimodal)
            rational: Whether to enable deep reasoning mode

        Returns:
            If rational=True, returns dict with answer and reasoning.
            Otherwise, returns the answer string.

        Raises:
            ValueError: If reasoning is requested but not supported by the model
        """
        if rational and self.model_name not in self.SUPPORTED_REASONING_MODELS:
            raise ValueError(f"Model {self.model_name} does not support reasoning")

        for attempt in range(self.max_retries):
            try:
                # Determine if input is multimodal
                if isinstance(user_prompt, list):
                    return await self._qa_multimodal(system_prompt, user_prompt)
                else:
                    if rational:
                        return await self._qa_with_reasoning(system_prompt, user_prompt)
                    else:
                        return await self._qa_standard(system_prompt, user_prompt)

            except Exception as e:
                self.stats["errors"] += 1
                self.stats["retries"] += 1

                logger.error(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {type(e).__name__}")
                logger.debug(f"Exception: {str(e)}")

                if attempt == self.max_retries - 1:
                    raise

                # Exponential backoff before retrying
                retry_delay = 2 ** attempt
                logger.info(f"Retrying after {retry_delay}s...")
                await asyncio.sleep(retry_delay)

    async def _qa_standard(self, system_prompt: str, user_prompt: str) -> Dict[str, str]:
        """Execute a standard query without reasoning."""
        logger.debug(f"_qa_standard: Sending query to {self.model_name}")
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                temperature=1
            )

            self.stats["calls"] += 1
            answer = completion.choices[0].message.content
            
            return {
                "answer": answer,
                "rational": ""
            }
        except Exception as e:
            logger.error(f"Error in _qa_standard: {type(e).__name__}: {str(e)}")
            raise

    async def _qa_with_reasoning(self, system_prompt: str, user_prompt: str) -> Dict[str, str]:
        """Execute a query with reasoning enabled."""
        logger.debug(f"_qa_with_reasoning: Sending query to {self.model_name}")
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": "<think>\n"}
                ],
                stream=False,
                temperature=1
            )

            self.stats["calls"] += 1
            return {
                "answer": completion.choices[0].message.content,
                "rational": getattr(completion.choices[0].message, 'reasoning_content', '')
            }
        except Exception as e:
            logger.error(f"Error in _qa_with_reasoning: {type(e).__name__}: {str(e)}")
            raise
    
    async def _qa_multimodal(self, system_prompt: str, user_prompt: List[Dict[str, Any]]) -> Dict[str, str]:
        """Execute a multimodal query (image/video + text) - Qwen API compatible."""
        logger.debug(f"_qa_multimodal: Sending multimodal query to {self.model_name}")
        logger.debug(f"  Prompt structure: {[item.get('type', 'unknown') for item in user_prompt]}")
        
        try:
            # Build content list directly for Qwen API
            content = []
            
            for item in user_prompt:
                if item.get('type') == 'video':
                    # Qwen API video format support
                    video_obj = {
                        "type": "video",
                        "video": item.get('video')  # Local path or URL
                    }
                    # Add optional video parameters if present
                    if 'video_start' in item:
                        video_obj['video_start'] = item['video_start']
                    if 'video_end' in item:
                        video_obj['video_end'] = item['video_end']
                    if 'nframes' in item:
                        video_obj['nframes'] = item['nframes']
                    
                    logger.debug(f"  Video content: {video_obj}")
                    content.append(video_obj)
                
                elif item.get('type') == 'text':
                    text_obj = {
                        "type": "text",
                        "text": item.get('text')
                    }
                    logger.debug(f"  Text content: {text_obj['text'][:100]}...")
                    content.append(text_obj)
                
                elif item.get('type') == 'image':
                    image_value = (
                        item.get('image')
                        or item.get('image_path')
                        or item.get('path')
                    )
                    if image_value is None:
                        raise ValueError("Image item must include 'image' or 'image_path'")
                    image_obj = {
                        "type": "image",
                        "image": image_value  # Local path or URL
                    }
                    logger.debug(f"  Image content: {image_obj}")
                    content.append(image_obj)
            
            logger.info(f"Making API request:")
            logger.info(f"  Model: {self.model_name}")
            logger.info(f"  Content items: {len(content)}")
            logger.info(f"  API Base: {self.client.base_url}")
            
            # Create completion with multimodal content
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                stream=False,
                temperature=1
            )
            # print("RAW_COMPLETION:", type(completion), completion)
            self.stats["calls"] += 1
            answer = completion.choices[0].message.content
            
            logger.info(f"✅ API call successful, answer length: {len(str(answer))}")
            
            return {
                "answer": answer,
                "rational": ""
            }
            
        except Exception as e:
            logger.error(f"❌ Error in _qa_multimodal: {type(e).__name__}")
            logger.error(f"  Message: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"  Response: {e.response}")
            raise

    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics for this API instance."""
        return self.stats.copy()
