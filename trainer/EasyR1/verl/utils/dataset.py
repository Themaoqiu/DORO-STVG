# 

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import json
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    # print(max_pixels)

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str,
    min_pixels: int = 4 * 32 * 32,
    max_pixels: int = 64 * 32 * 32,
    max_frames: int = 128,
    video_fps: float = 2.0,
    return_fps: bool = False,
    return_metadata: bool = False,
):
    vision_info = {
        "video": video,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_frames": max_frames,
        "fps": video_fps,
    }
    output = fetch_video(
        vision_info,
        image_patch_size=16,
        return_video_sample_fps=return_fps,
        return_video_metadata=return_metadata,
    )

    video_tensor = output
    video_metadata = None
    sample_fps = None

    if isinstance(output, tuple):
        if len(output) == 3:
            video_tensor, video_metadata, sample_fps = output
        elif len(output) == 2:
            video_tensor, second = output
            if isinstance(second, dict):
                video_metadata = second
            else:
                sample_fps = second

    # qwen-vl-utils may wrap (video, metadata) as the first return item.
    if (
        isinstance(video_tensor, tuple)
        and len(video_tensor) == 2
        and isinstance(video_tensor[1], dict)
    ):
        nested_video_tensor, nested_metadata = video_tensor
        video_tensor = nested_video_tensor
        if video_metadata is None:
            video_metadata = nested_metadata

    if return_fps and return_metadata:
        return video_tensor, sample_fps, video_metadata
    if return_fps:
        return video_tensor, sample_fps
    if return_metadata:
        return video_tensor, video_metadata
    return video_tensor


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_dir_map: Optional[dict[str, str]] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_dir_map = video_dir_map or {}
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _ensure_media_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _resolve_image_paths(self, images: list[Any]) -> list[Any]:
        if self.image_dir is None:
            return images
        if len(images) == 0 or not isinstance(images[0], str):
            return images
        return [image if os.path.isabs(image) else os.path.join(self.image_dir, image) for image in images]

    def _resolve_video_paths(self, videos: list[Any], example: dict[str, Any]) -> list[Any]:
        if len(videos) == 0 or not isinstance(videos[0], str):
            return videos

        base_dir = None
        data_source = example.get("data_source")
        if isinstance(data_source, str):
            base_dir = self.video_dir_map.get(data_source)
        if base_dir is None:
            base_dir = self.image_dir
        if base_dir is None:
            return videos
        return [video if os.path.isabs(video) else os.path.join(base_dir, video) for video in videos]


    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        base_prompt: str = example[self.prompt_key]
        prompt_str = base_prompt
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=base_prompt, example=example, **example)

        images = self._ensure_media_list(example.get(self.image_key))
        videos = self._ensure_media_list(example.get(self.video_key))

        # If jinja outputs structured JSON, respect it directly.
        if isinstance(prompt_str, str):
            try:
                parsed_prompt = json.loads(prompt_str)
            except Exception:
                parsed_prompt = None

            if isinstance(parsed_prompt, list):
                return parsed_prompt
            if isinstance(parsed_prompt, dict) and ("system" in parsed_prompt or "user" in parsed_prompt):
                messages = []
                system_text = parsed_prompt.get("system")
                user_text = parsed_prompt.get("user")
                if system_text:
                    messages.append({"role": "system", "content": str(system_text)})
                if user_text:
                    messages.append({"role": "user", "content": str(user_text)})
                if messages:
                    return messages

        if len(images) > 0:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            # print(content_list)

            return [{"role": "user", "content": content_list}]
        elif len(videos) > 0:
            if "<video>" not in prompt_str:
                prompt_str = "<video>\n" + prompt_str
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            # print(content_list)

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]


    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        images = self._ensure_media_list(example.get(self.image_key))
        videos = self._ensure_media_list(example.get(self.video_key))

        if len(images) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = self._resolve_image_paths(images)
            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif len(videos) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = self._resolve_video_paths(videos, example)

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            # print(videos, model_inputs["input_ids"].size(-1))
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)
        images = self._ensure_media_list(example.pop(self.image_key, None))
        videos = self._ensure_media_list(example.pop(self.video_key, None))

        if len(images) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = self._resolve_image_paths(images)

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif len(videos) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = self._resolve_video_paths(videos, example)
            processed_videos = []
            video_metadatas = []
            for video in videos:
                processed_video, _, video_metadata = process_video(
                    video, video_fps=self.video_fps, return_fps=True, return_metadata=True
                )
                processed_videos.append(processed_video)
                video_metadatas.append(video_metadata)
            video_kwargs = {"do_sample_frames": False}
            has_metadata = any(metadata is not None for metadata in video_metadatas)
            if has_metadata:
                model_inputs = self.processor(
                    text=[prompt],
                    videos=processed_videos,
                    add_special_tokens=False,
                    video_metadata=video_metadatas,
                    return_tensors="pt",
                    **video_kwargs,
                )
            else:
                model_inputs = self.processor(
                    text=[prompt],
                    videos=processed_videos,
                    add_special_tokens=False,
                    return_tensors="pt",
                    **video_kwargs,
                )

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
            if has_metadata:
                example["multi_modal_data"]["video_metadata"] = video_metadatas
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index



            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)

        # print(example)
        # print(input_ids.shape)


        # print(example)
        return example
