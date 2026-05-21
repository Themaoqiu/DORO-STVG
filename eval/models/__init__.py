__all__ = [
    "Qwen2_5VL",
    "Qwen3VL",
    "LlavaSTQwen2",
    "VTimeLLMModel",
    "GroundedVideoLLMModel",
    "Llava16Model",
    "STVGR1",
    "GroundingGPTModel",
    "VideoMolmoModel",
    "CGSTVGModel",
    "TASTVGModel",
    "TubeDETRModel",
    "DeViLModel",
    "InternVL3",
    "InternVL3_5",
    "LlavaNextVideo",
    "LlavaOneVision1_5",
    "LlavaOneVision2",
]


def __getattr__(name):
    if name in {"Qwen2_5VL", "Qwen3VL"}:
        from .qwen_family import Qwen2_5VL, Qwen3VL

        return {"Qwen2_5VL": Qwen2_5VL, "Qwen3VL": Qwen3VL}[name]

    if name == "LlavaSTQwen2":
        from .llava_st import LlavaSTQwen2

        return LlavaSTQwen2

    if name == "VTimeLLMModel":
        from .vtimellm import VTimeLLMModel

        return VTimeLLMModel

    if name == "GroundedVideoLLMModel":
        from .grounded_video_llm import GroundedVideoLLMModel

        return GroundedVideoLLMModel

    if name == "Llava16Model":
        from .llava16 import Llava16Model

        return Llava16Model

    if name == "STVGR1":
        from .stvg_r1 import STVGR1

        return STVGR1

    if name == "GroundingGPTModel":
        from .groundinggpt import GroundingGPTModel

        return GroundingGPTModel

    if name == "VideoMolmoModel":
        from .videomolmo import VideoMolmoModel

        return VideoMolmoModel

    if name == "CGSTVGModel":
        from .cgstvg import CGSTVGModel

        return CGSTVGModel

    if name == "TASTVGModel":
        from .tastvg import TASTVGModel

        return TASTVGModel

    if name == "TubeDETRModel":
        from .tubedetr import TubeDETRModel

        return TubeDETRModel

    if name == "DeViLModel":
        from .devil import DeViLModel

        return DeViLModel

    if name in {"InternVL3", "InternVL3_5"}:
        from .internvl_family import InternVL3, InternVL3_5

        return {"InternVL3": InternVL3, "InternVL3_5": InternVL3_5}[name]

    if name in {"LlavaNextVideo", "LlavaOneVision1_5", "LlavaOneVision2"}:
        from .llava_family import LlavaNextVideo, LlavaOneVision1_5, LlavaOneVision2

        return {
            "LlavaNextVideo": LlavaNextVideo,
            "LlavaOneVision1_5": LlavaOneVision1_5,
            "LlavaOneVision2": LlavaOneVision2,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
