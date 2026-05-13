__all__ = [
    "Qwen2_5VL",
    "Qwen3VL",
    "LlavaSTQwen2",
    "VTimeLLMModel",
    "GroundedVideoLLMModel",
    "Llava16Model",
    "VideoChatR1",
    "STVGR1",
    "GroundingGPTModel",
    "VideoMolmoModel",
    "CGSTVGModel",
    "TASTVGModel",
    "TubeDETRModel",
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

    if name == "VideoChatR1":
        from .videochat_r1 import VideoChatR1

        return VideoChatR1

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

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
