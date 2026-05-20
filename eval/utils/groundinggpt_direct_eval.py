import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List


EVAL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EVAL_DIR.parent
GROUNDINGGPT_ROOT = EVAL_DIR / "dependence" / "groundinggpt"
for path in (EVAL_DIR, GROUNDINGGPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pipelines.dorostvg import DOROSTVGPipeline


logger = logging.getLogger(__name__)

torch = None
conversation_lib = None
CONFIG = None
KeywordsStoppingCriteria = None
SeparatorStyle = None
load_pretrained_model = None
load_video = None
tokenizer_image_token = None
DEFAULT_VIDEO_END_TOKEN = None
DEFAULT_VIDEO_PATCH_TOKEN = None
DEFAULT_VIDEO_START_TOKEN = None
DEFAULT_VIDEO_TOKEN = None
IMAGE_TOKEN_INDEX = None


def _load_groundinggpt_modules() -> None:
    global torch
    global conversation_lib
    global CONFIG
    global KeywordsStoppingCriteria
    global SeparatorStyle
    global load_pretrained_model
    global load_video
    global tokenizer_image_token
    global DEFAULT_VIDEO_END_TOKEN
    global DEFAULT_VIDEO_PATCH_TOKEN
    global DEFAULT_VIDEO_START_TOKEN
    global DEFAULT_VIDEO_TOKEN
    global IMAGE_TOKEN_INDEX

    if torch is not None:
        return

    import torch as torch_module
    from lego import conversation as conversation_module
    from lego.constants import (
        DEFAULT_VIDEO_END_TOKEN as video_end_token,
        DEFAULT_VIDEO_PATCH_TOKEN as video_patch_token,
        DEFAULT_VIDEO_START_TOKEN as video_start_token,
        DEFAULT_VIDEO_TOKEN as video_token,
        IMAGE_TOKEN_INDEX as image_token_index,
    )
    from lego.conversation import SeparatorStyle as separator_style
    from lego.mm_utils import KeywordsStoppingCriteria as keywords_stopping_criteria
    from lego.mm_utils import tokenizer_image_token as tokenizer_image_token_func
    from lego.model.builder import CONFIG as config
    from lego.model.builder import load_pretrained_model as load_model
    from video_llama.processors.video_processor import load_video as load_video_func

    torch = torch_module
    conversation_lib = conversation_module
    CONFIG = config
    KeywordsStoppingCriteria = keywords_stopping_criteria
    SeparatorStyle = separator_style
    load_pretrained_model = load_model
    load_video = load_video_func
    tokenizer_image_token = tokenizer_image_token_func
    DEFAULT_VIDEO_END_TOKEN = video_end_token
    DEFAULT_VIDEO_PATCH_TOKEN = video_patch_token
    DEFAULT_VIDEO_START_TOKEN = video_start_token
    DEFAULT_VIDEO_TOKEN = video_token
    IMAGE_TOKEN_INDEX = image_token_index


def _extract_video_path(video_input: str) -> str:
    return str(video_input or "").split("::split=", 1)[0]


class DirectGroundingGPTModel:
    prompt_style = "json"

    def __init__(self, model_path: str, max_new_tokens: int, temperature: float):
        _load_groundinggpt_modules()

        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.last_raw_responses: List[str] = []
        self.last_user_prompts: List[str] = []

        logger.info("Loading GroundingGPT model from %s", model_path)
        self.model, self.tokenizer, self.image_processor, self.video_transform, self.context_len = load_pretrained_model(
            model_path
        )
        self.model.eval()
        logger.info("Loaded GroundingGPT direct model | context_len=%s", self.context_len)

    @staticmethod
    def _new_conversation(system_prompt: str):
        conv = conversation_lib.default_conversation.copy()
        conv.system = system_prompt
        return conv

    def _build_prompt(self, query: str, system_prompt: str) -> str:
        conv = self._new_conversation(system_prompt)
        if self.model.config.mm_use_im_start_end:
            user_message = (
                DEFAULT_VIDEO_START_TOKEN
                + DEFAULT_VIDEO_PATCH_TOKEN * CONFIG.video_token_len
                + DEFAULT_VIDEO_END_TOKEN
                + "\n"
                + query
            )
        else:
            user_message = DEFAULT_VIDEO_TOKEN + "\n" + query
        conv.append_message(conv.roles[0], user_message)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _predict_one(self, query: str, video_path: str, system_prompt: str) -> str:
        real_video_path = _extract_video_path(video_path)
        logger.info("Direct GroundingGPT inference on %s", Path(real_video_path).name)

        video = load_video(
            video_path=real_video_path,
            n_frms=self.model.config.max_frame,
            height=224,
            width=224,
            sampling="uniform",
            return_msg=False,
        )
        video_tensor = self.video_transform(video).unsqueeze(0).to(CONFIG.device, dtype=torch.bfloat16)

        prompt = self._build_prompt(query, system_prompt)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        input_ids = input_ids.to(CONFIG.device)

        conv = self._new_conversation(system_prompt)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                videos=video_tensor,
                sounds=None,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        output = self.tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        if output.endswith(stop_str):
            output = output[: -len(stop_str)].strip()
        return output

    def predict_batch(self, queries: List[str], video_paths: List[str], system_prompt: str) -> List[str]:
        self.last_user_prompts = list(queries)
        outputs = [self._predict_one(query, video_path, system_prompt) for query, video_path in zip(queries, video_paths)]
        self.last_raw_responses = list(outputs)
        return outputs

    def close(self) -> None:
        pass


class LimitedDOROSTVGPipeline(DOROSTVGPipeline):
    def __init__(self, *args, max_samples: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_samples = max_samples

    def load_data(self):
        samples = super().load_data()
        if self.max_samples > 0:
            return samples[: self.max_samples]
        return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GroundingGPT by importing official load_pretrained_model.")
    parser.add_argument("--model_path", default="/mnt/sdc/xingjianwang/yibowang/model_zoo/GroundingGPT")
    parser.add_argument("--annotation_path", default="/mnt/sdc/xingjianwang/data/vidstg/query_polished.jsonl")
    parser.add_argument("--video_dir", default="/mnt/sdc/xingjianwang/data/vidstg/video")
    parser.add_argument("--output_dir", default=str(EVAL_DIR / "res_groundinggpt_direct"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--max_samples", type=int, default=0, help="Use 0 for the full annotation file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    model = DirectGroundingGPTModel(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    pipeline = LimitedDOROSTVGPipeline(
        model=model,
        model_name="groundinggpt-direct",
        data_name="doro-stvg",
        annotation_path=args.annotation_path,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    metrics = pipeline.run_evaluation()
    logger.info("Direct GroundingGPT metrics: %s", metrics)


if __name__ == "__main__":
    main()
