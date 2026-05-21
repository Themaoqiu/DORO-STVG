from openai import AsyncOpenAI
from typing import Dict, Any, List
import logging
import asyncio
import os
import json
import base64
import mimetypes
from pathlib import Path
from .vision_utils import build_multimodal_message

logger = logging.getLogger(__name__)


class QAWrapper:
    """Asynchronous wrapper for LLM API client."""

    SUPPORTED_REASONING_MODELS = ["DeepSeek-R1", "gemini-3-flash-preview", "qwen3-vl-flash"]

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

        default_headers = self._load_default_headers()

        logger.info(f"Initializing QAWrapper")
        logger.info(f"  API Base URL: {api_base_url}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  API Key: {api_key[:20]}...")
        if default_headers:
            logger.info(f"  Default headers: {list(default_headers.keys())}")

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "base_url": api_base_url,
        }
        if default_headers:
            client_kwargs["default_headers"] = default_headers

        self.client = AsyncOpenAI(**client_kwargs)

        self.stats = {
            "calls": 0,
            "errors": 0,
            "retries": 0
        }

    @staticmethod
    def _load_default_headers() -> Dict[str, str]:
        """Collect optional default HTTP headers from the environment.

        Supports two mechanisms, both opt-in (no effect when unset):
          * ``MM_DEFAULT_HEADERS`` — a JSON object of arbitrary headers.
          * ``MM_USER_EMAIL`` / ``MM_APP_ID`` — convenience vars for the
            MAAS-style ``x-maas-user-email`` / ``x-maas-app-id`` headers.
        """
        headers: Dict[str, str] = {}

        raw = os.getenv("MM_DEFAULT_HEADERS")
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    headers.update({str(k): str(v) for k, v in parsed.items()})
                else:
                    logger.warning("MM_DEFAULT_HEADERS must be a JSON object; ignored")
            except json.JSONDecodeError as exc:
                logger.warning(f"Failed to parse MM_DEFAULT_HEADERS as JSON: {exc}")

        user_email = os.getenv("MM_USER_EMAIL")
        if user_email:
            headers.setdefault("x-maas-user-email", user_email)

        app_id = os.getenv("MM_APP_ID")
        if app_id:
            headers.setdefault("x-maas-app-id", app_id)

        return headers

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
                error_text = str(e).lower()
                if "data_inspection_failed" in error_text or "datainspectionfailed" in error_text:
                    logger.error("Content inspection failed; skip this sample without retry")
                    return {
                        "answer": "__ERROR__:data_inspection_failed",
                        "rational": ""
                    }
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
                    image_str = str(image_value)
                    if image_str.startswith(("http://", "https://", "data:")):
                        image_url = image_str
                    else:
                        image_path = Path(image_str)
                        if not image_path.exists():
                            raise FileNotFoundError(f"Image file not found: {image_path}")
                        mime_type, _ = mimetypes.guess_type(str(image_path))
                        if not mime_type:
                            mime_type = "image/jpeg"
                        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
                        image_url = f"data:{mime_type};base64,{encoded}"

                    image_obj = {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                    logger.debug(f"  Image content: {image_obj}")
                    content.append(image_obj)
                
                elif item.get('type') == 'image_url':
                    image_obj = {
                        "type": "image_url",
                        "image_url": item.get("image_url"),
                    }
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
