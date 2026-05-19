import argparse
import json
import os
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict


PROTOCOL_PREFIX = "__DORO_GROUNDED_VIDEO_LLM__ "


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default=os.getenv("GROUNDED_VIDEO_LLM_SOURCE_DIR", ""))

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    parser.add_argument("--model", type=str, default="llava_next_video", choices=["llava_next_video"])
    parser.add_argument("--llm", type=str, default="phi3.5", choices=["llama3", "vicuna", "phi3.5"])
    parser.add_argument("--stage", type=str, default="sft", choices=["pretrain", "grounded", "sft"])
    parser.add_argument("--max_txt_len", type=int, default=2048)
    parser.add_argument("--num_temporal_tokens", type=int, default=300)
    parser.add_argument("--num_frames", type=int, default=96)
    parser.add_argument("--num_segs", type=int, default=12)
    parser.add_argument("--lora", type=_parse_bool, default=True)
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "flash_attention_2"])

    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--pretrained_video_path", type=str, required=True)
    parser.add_argument("--pretrained_vision_proj_llm_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--do_sample", type=_parse_bool, default=True)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    return parser.parse_args()


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def emit(payload: Dict[str, Any]) -> None:
    print(PROTOCOL_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def load_external_modules(source_dir: str) -> None:
    if not source_dir:
        raise RuntimeError("--source_dir or GROUNDED_VIDEO_LLM_SOURCE_DIR is required")
    source_path = Path(source_dir).expanduser().resolve()
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

    global torch, np, cudnn
    global LLaMA3_Template, Vicuna_Template, Phi_3_5_Template, DEFAULT_IMAGE_TOKEN, GROUNDING_TOKEN
    global LLAVA_NEXT_VIDEO, read_frames_decord, frame_transform
    global INTERNVIDEO_MEAN, INTERNVIDEO_STD, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

    import numpy as imported_np
    import torch as imported_torch
    from torch.backends import cudnn as imported_cudnn
    from datasets.chat.base_template import (
        DEFAULT_IMAGE_TOKEN as imported_default_image_token,
        GROUNDING_TOKEN as imported_grounding_token,
        LLaMA3_Template as imported_llama3_template,
        Phi_3_5_Template as imported_phi_template,
        Vicuna_Template as imported_vicuna_template,
    )
    from mm_utils.utils import (
        INTERNVIDEO_MEAN as imported_internvideo_mean,
        INTERNVIDEO_STD as imported_internvideo_std,
        OPENAI_DATASET_MEAN as imported_openai_mean,
        OPENAI_DATASET_STD as imported_openai_std,
        frame_transform as imported_frame_transform,
    )
    from mm_utils.video_utils import read_frames_decord as imported_read_frames_decord
    from models.llava_next_video import LLAVA_NEXT_VIDEO as imported_llava_next_video

    torch = imported_torch
    np = imported_np
    cudnn = imported_cudnn
    LLaMA3_Template = imported_llama3_template
    Vicuna_Template = imported_vicuna_template
    Phi_3_5_Template = imported_phi_template
    DEFAULT_IMAGE_TOKEN = imported_default_image_token
    GROUNDING_TOKEN = imported_grounding_token
    LLAVA_NEXT_VIDEO = imported_llava_next_video
    read_frames_decord = imported_read_frames_decord
    frame_transform = imported_frame_transform
    INTERNVIDEO_MEAN = imported_internvideo_mean
    INTERNVIDEO_STD = imported_internvideo_std
    OPENAI_DATASET_MEAN = imported_openai_mean
    OPENAI_DATASET_STD = imported_openai_std


def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def dtype_from_name(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model(args):
    model = LLAVA_NEXT_VIDEO(
        dtype=dtype_from_name(args.dtype),
        stage=args.stage,
        max_txt_len=args.max_txt_len,
        num_frames=args.num_frames,
        num_segs=args.num_segs,
        num_temporal_tokens=args.num_temporal_tokens,
        lora=args.lora,
        llm=args.llm,
        attn_implementation=args.attn_implementation,
        config_path=args.config_path,
        tokenizer_path=args.tokenizer_path,
        pretrained_video_path=args.pretrained_video_path,
        pretrained_vision_proj_llm_path=args.pretrained_vision_proj_llm_path,
    )
    ckpt = torch.load(args.ckpt_path, map_location="cpu")["model"]
    if "multi_modal_projector" in ckpt:
        model.multi_modal_projector.load_state_dict(ckpt["multi_modal_projector"])
    if "video_projecter" in ckpt:
        model.video_projecter.load_state_dict(ckpt["video_projecter"])
    if "language_model" in ckpt:
        model.language_model.load_state_dict(ckpt["language_model"])
    model.eval()
    model.to(args.device)
    return model


def create_grounding_inputs(args, video_path: str, prompt_grounding: str):
    video_processor = frame_transform(image_size=224, mean=INTERNVIDEO_MEAN, std=INTERNVIDEO_STD)
    image_processor = frame_transform(image_size=336, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
    pixel_values, _frame_indices, _fps, _total_frame_num, duration = read_frames_decord(
        video_path=video_path,
        num_frames=args.num_frames,
        sample="middle",
    )

    temporal_pixel_values = []
    for i in range(pixel_values.shape[0]):
        temporal_pixel_values.append(video_processor(pixel_values[i]))
    temporal_pixel_values = torch.tensor(np.array(temporal_pixel_values)).unsqueeze(0)

    num_frames_per_seg = int(args.num_frames // args.num_segs)
    indices_spatial = [(i * num_frames_per_seg) + int(num_frames_per_seg / 2) for i in range(args.num_segs)]
    spatial_pixel_values = []
    for i_spatial in indices_spatial:
        spatial_pixel_values.append(image_processor(pixel_values[i_spatial]))
    spatial_pixel_values = torch.tensor(np.array(spatial_pixel_values)).unsqueeze(0)

    chat_template = {
        "phi3.5": Phi_3_5_Template(),
        "llama3": LLaMA3_Template(),
        "vicuna": Vicuna_Template(),
    }[args.llm]
    conv = [
        {"from": "human", "value": DEFAULT_IMAGE_TOKEN + " " + GROUNDING_TOKEN + "\n" + prompt_grounding},
        {"from": "gpt", "value": ""},
    ]
    _sep, eos = chat_template.separator.apply()
    prompt = chat_template.encode(conv).replace(eos, "")

    samples = {
        "video_ids": [video_path],
        "question_ids": [video_path],
        "prompts": [prompt],
        "temporal_pixel_values": temporal_pixel_values.to(args.device),
        "spatial_pixel_values": spatial_pixel_values.to(args.device),
    }
    return samples, duration


def parse_time_interval(text, duration, num_temporal_tokens=300, llm="phi3.5"):
    pattern = r"<(\d+)>"

    def replace_func(match):
        x = int(match.group(1))
        seconds = duration * x / num_temporal_tokens
        if llm == "phi3.5":
            return f" {seconds:.2f} seconds"
        if llm == "llama3":
            return f"{seconds:.2f} seconds"
        return f"{seconds:.2f} seconds"

    return re.sub(pattern, replace_func, text)


def generate_grounding(model, args, video_path: str, prompt: str) -> str:
    samples, duration = create_grounding_inputs(args, video_path, prompt)
    generate_kwargs = {
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    with torch.cuda.amp.autocast(enabled=True, dtype=model.dtype):
        with torch.inference_mode():
            pred_text = model.generate(samples, **generate_kwargs)[0]
    return parse_time_interval(pred_text, duration, args.num_temporal_tokens, args.llm)


def main():
    args = parse_args()
    try:
        load_external_modules(args.source_dir)
        init_seeds(args.seed)
        model = load_model(args)
        emit({"type": "ready"})
    except Exception as exc:
        emit({"type": "startup_error", "error": str(exc), "traceback": traceback.format_exc()})
        raise

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            request = json.loads(raw_line)
            if request.get("type") == "shutdown":
                emit({"type": "shutdown"})
                return
            request_id = request.get("id")
            output = generate_grounding(
                model,
                args,
                video_path=str(request["video_path"]),
                prompt=str(request["prompt_grounding"]),
            )
            emit({"type": "response", "id": request_id, "output": output})
        except Exception as exc:
            emit(
                {
                    "type": "error",
                    "id": request.get("id") if "request" in locals() else None,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )


if __name__ == "__main__":
    main()
