from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


_DEFAULT_RUNWAY_ENDPOINT = (
    "https://runway.devops.rednote.life/openai/google/v1:generateContent"
)


class GeminiWrapper:
    """Async wrapper that talks to a Gemini-native ``:generateContent`` endpoint."""

    # Models that should produce a separate ``reasoning`` trace when ``rational=True``.
    SUPPORTED_REASONING_MODELS = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
    ]

    def __init__(self, model_name: str, api_key: str, max_retries: int = 5) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.max_retries = max_retries

        self.endpoint_template = (
            os.getenv("GEMINI_API_BASE_URL")
            or os.getenv("MM_API_BASE_URL")
            or _DEFAULT_RUNWAY_ENDPOINT
        )
        if ":generateContent" not in self.endpoint_template:
            self.endpoint_template = self.endpoint_template.rstrip("/") + ":generateContent"

        self.auth_header_name = os.getenv("GEMINI_AUTH_HEADER", "api-key")

        timeout_seconds = float(os.getenv("GEMINI_TIMEOUT", "120"))
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

        self.extra_headers = self._load_extra_headers()

        logger.info("Initializing GeminiWrapper")
        logger.info(f"  Endpoint template: {self.endpoint_template}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Auth header: {self.auth_header_name}")
        logger.info(f"  API Key: {api_key[:20]}...")
        if self.extra_headers:
            logger.info(f"  Extra headers: {list(self.extra_headers.keys())}")

        self.stats = {"calls": 0, "errors": 0, "retries": 0}

    @staticmethod
    def _load_extra_headers() -> Dict[str, str]:
        raw = os.getenv("GEMINI_DEFAULT_HEADERS") or os.getenv("MM_DEFAULT_HEADERS")
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse extra headers as JSON: {exc}")
            return {}
        if not isinstance(parsed, dict):
            logger.warning("Extra headers env must be a JSON object; ignored")
            return {}
        return {str(k): str(v) for k, v in parsed.items()}

    def _resolve_endpoint(self) -> str:
        return self.endpoint_template.replace("{model}", self.model_name)

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            self.auth_header_name: self.api_key,
        }
        headers.update(self.extra_headers)
        return headers

    @staticmethod
    def _image_to_inline_part(image_value: str) -> Dict[str, Any]:
        if image_value.startswith("data:"):
            # data:<mime>;base64,<payload>
            try:
                meta, encoded = image_value.split(",", 1)
                mime_type = meta.split(";")[0].removeprefix("data:") or "image/jpeg"
            except ValueError:
                raise ValueError(f"Malformed data URI for image: {image_value[:40]}")
            return {"inline_data": {"mime_type": mime_type, "data": encoded}}

        if image_value.startswith(("http://", "https://")):
            return {"file_data": {"file_uri": image_value, "mime_type": "image/*"}}

        image_path = Path(image_value)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        mime_type, _ = mimetypes.guess_type(str(image_path))
        mime_type = mime_type or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return {"inline_data": {"mime_type": mime_type, "data": encoded}}

    def _build_user_parts(self, user_prompt: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if isinstance(user_prompt, str):
            return [{"text": user_prompt}]

        parts: List[Dict[str, Any]] = []
        for item in user_prompt:
            item_type = item.get("type")
            if item_type == "text":
                parts.append({"text": str(item.get("text", ""))})
            elif item_type == "image":
                image_value = item.get("image") or item.get("image_path") or item.get("path")
                if not image_value:
                    raise ValueError("Image item must include 'image' or 'image_path'")
                parts.append(self._image_to_inline_part(str(image_value)))
            elif item_type == "image_url":
                url = (item.get("image_url") or {}).get("url") if isinstance(item.get("image_url"), dict) else item.get("image_url")
                if not url:
                    raise ValueError("image_url item missing url")
                parts.append(self._image_to_inline_part(str(url)))
            elif item_type == "video":
                video_value = item.get("video") or item.get("video_path") or item.get("path")
                if not video_value:
                    raise ValueError("Video item must include 'video'")
                video_str = str(video_value)
                if video_str.startswith(("http://", "https://")):
                    parts.append({"file_data": {"file_uri": video_str, "mime_type": "video/*"}})
                else:
                    video_path = Path(video_str)
                    if not video_path.exists():
                        raise FileNotFoundError(f"Video file not found: {video_path}")
                    mime_type, _ = mimetypes.guess_type(str(video_path))
                    mime_type = mime_type or "video/mp4"
                    encoded = base64.b64encode(video_path.read_bytes()).decode("utf-8")
                    parts.append({"inline_data": {"mime_type": mime_type, "data": encoded}})
            else:
                logger.warning(f"Unknown prompt part type '{item_type}', skipping")
        return parts

    def _build_generation_config(self, rational: bool) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "temperature": float(os.getenv("GEMINI_TEMPERATURE", "1")),
        }

        max_output = os.getenv("GEMINI_MAX_OUTPUT_TOKENS")
        if max_output:
            try:
                config["maxOutputTokens"] = int(max_output)
            except ValueError:
                logger.warning(f"Invalid GEMINI_MAX_OUTPUT_TOKENS={max_output}; ignored")

        top_p = os.getenv("GEMINI_TOP_P")
        if top_p:
            try:
                config["topP"] = float(top_p)
            except ValueError:
                logger.warning(f"Invalid GEMINI_TOP_P={top_p}; ignored")

        seed = os.getenv("GEMINI_SEED")
        if seed:
            try:
                config["seed"] = int(seed)
            except ValueError:
                logger.warning(f"Invalid GEMINI_SEED={seed}; ignored")

        if rational:
            thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "HIGH")
            config["thinkingConfig"] = {
                "thinkingLevel": thinking_level,
                "includeThoughts": True,
            }
        return config

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: Union[str, List[Dict[str, Any]]],
        rational: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": self._build_user_parts(user_prompt),
                }
            ],
            "generationConfig": self._build_generation_config(rational),
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    @staticmethod
    def _parse_response(data: Dict[str, Any]) -> Dict[str, str]:
        candidates = data.get("candidates") or []
        if not candidates:
            prompt_feedback = data.get("promptFeedback") or {}
            block_reason = prompt_feedback.get("blockReason")
            if block_reason:
                raise RuntimeError(f"Gemini blocked response: {block_reason}")
            raise RuntimeError("Gemini response contains no candidates")

        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        answer_chunks: List[str] = []
        reasoning_chunks: List[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if not isinstance(text, str):
                continue
            if part.get("thought"):
                reasoning_chunks.append(text)
            else:
                answer_chunks.append(text)

        return {
            "answer": "".join(answer_chunks),
            "rational": "".join(reasoning_chunks),
        }

    async def qa(
        self,
        system_prompt: str,
        user_prompt: Union[str, List[Dict[str, Any]]] = "",
        rational: bool = False,
    ) -> Dict[str, str]:
        if rational and self.model_name not in self.SUPPORTED_REASONING_MODELS:
            raise ValueError(f"Model {self.model_name} does not support reasoning")

        payload = self._build_payload(system_prompt, user_prompt, rational=rational)
        endpoint = self._resolve_endpoint()
        headers = self._build_headers()

        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(endpoint, headers=headers, json=payload)
                if response.status_code >= 400:
                    body_snippet = response.text[:500]
                    lowered = body_snippet.lower()
                    if "data_inspection_failed" in lowered or "datainspectionfailed" in lowered:
                        logger.error("Content inspection failed; skip this sample without retry")
                        return {"answer": "__ERROR__:data_inspection_failed", "rational": ""}
                    raise RuntimeError(
                        f"Gemini HTTP {response.status_code}: {body_snippet}"
                    )

                data = response.json()
                parsed = self._parse_response(data)
                self.stats["calls"] += 1
                return parsed

            except Exception as exc:  # noqa: BLE001 - network-ish, retried below
                last_exc = exc
                error_text = str(exc).lower()
                if "data_inspection_failed" in error_text or "datainspectionfailed" in error_text:
                    logger.error("Content inspection failed; skip this sample without retry")
                    return {"answer": "__ERROR__:data_inspection_failed", "rational": ""}

                self.stats["errors"] += 1
                self.stats["retries"] += 1
                logger.error(
                    f"Gemini API call failed (attempt {attempt + 1}/{self.max_retries}): "
                    f"{type(exc).__name__}"
                )
                logger.debug(f"Exception: {exc}")

                if attempt == self.max_retries - 1:
                    raise

                await asyncio.sleep(2 ** attempt)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Gemini API exhausted retries without raising")

    def get_stats(self) -> Dict[str, int]:
        return self.stats.copy()

    async def aclose(self) -> None:
        await self._client.aclose()
